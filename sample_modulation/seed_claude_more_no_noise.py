# -*- coding: utf-8 -*-
"""
seed_claude_more_no_noise.py

目的:
  seed_claude_more.py の量子ノイズなし版。
  ノイズなし条件（理想的な盗聴環境）での性能上限を測定し、
  ノイズあり版（seed_claude_more.py）との比較解析に使用する。
  - 3層Conv1D（マルチスケール）+ BiLSTM + 2層Multi-Head Attention
  - Warmup + CosineAnnealing スケジューラ
  - Gradient Accumulation
  - LFSR長に応じたバッチサイズ最適化
"""

import os
import sys
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time
import math
import yaml

# ── モデル保存ディレクトリ ──
MODEL_SAVE_DIR = 'saved_models_more_no_noise'

# ── 量子ノイズのスケール設定 ──
# ノイズなしバージョン: スケール = 0.0
QUANTUM_NOISE_SCALE = 0.0

# ── LFSRタップ位置の定義 ──
LFSR_TAPS = {
    4:  [3, 2],
    6:  [5, 4],
    8:  [7, 5, 4, 3],
    10: [9, 6],
    12: [11, 10, 9, 3],
    14: [13, 12, 11, 1],
    16: [15, 13, 12, 10]
}

# ── 強化版パラメータ設定 ──
# (d_model, lstm_hidden, nhead, lstm_layers, n_train, n_test, seq_len, epochs, batch_size, accum_steps)
CONFIG = {
    4:  (64,   64,   4, 1,    20_000,   5_000,    64,  80, 512, 1),
    6:  (64,   64,   4, 1,    30_000,   8_000,    80, 100, 512, 1),
    8:  (128,  128,  4, 2,    50_000,  10_000,   128, 120, 512, 1),
    10: (128,  128,  4, 2,   100_000,  15_000,   256, 200, 512, 1),
}


class SeedPredictorEnhanced(nn.Module):
    """
    強化版シード値推論モデル。
    1. 3層Conv1D（kernel=3,5,7）: マルチスケール局所特徴抽出
    2. BiLSTM: 双方向時系列依存性の学習
    3. 2層Multi-Head Attention: 長距離相関の深い捕捉
    4. MLP Head: シード各ビットの確率を出力
    """
    def __init__(self, out_dim, d_model=128, lstm_hidden=128, nhead=4,
                 lstm_layers=2, dropout=0.15):
        super().__init__()
        self.d_model = d_model
        ch1 = d_model // 4
        ch2 = d_model // 2

        # 1. マルチスケールConv1D
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, ch1, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(ch1),
            nn.Conv1d(ch1, ch2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(ch2),
            nn.Conv1d(ch2, d_model, kernel_size=7, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
        )

        # 2. 双方向LSTM
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # 3. 2層Multi-Head Attention
        lstm_out_dim = lstm_hidden * 2
        self.attn1 = nn.MultiheadAttention(
            embed_dim=lstm_out_dim, num_heads=nhead,
            dropout=dropout, batch_first=True)
        self.attn_norm1 = nn.LayerNorm(lstm_out_dim)
        self.attn2 = nn.MultiheadAttention(
            embed_dim=lstm_out_dim, num_heads=nhead,
            dropout=dropout, batch_first=True)
        self.attn_norm2 = nn.LayerNorm(lstm_out_dim)

        # 4. MLP Head
        self.fc_out = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = x.transpose(1, 2)
        h = self.conv_block(h)
        h = h.transpose(1, 2)
        h, _ = self.lstm(h)
        # Attention層1
        a1, _ = self.attn1(h, h, h)
        h = self.attn_norm1(h + a1)
        # Attention層2
        a2, _ = self.attn2(h, h, h)
        h = self.attn_norm2(h + a2)
        # 最終ステップの特徴量で推論
        return self.fc_out(h[:, -1, :])


def generate_seed_dataset(n_samples, lfsr_length, seq_len, noise_scale):
    """ランダムシードからLFSRを駆動し、観測系列とシードビット列を生成。
    ※ noise_scale=0.0 の場合、ノイズなしの理想的な観測系列となる。"""
    N = 12
    BNum = 2 ** N
    S_max = 10.0
    S_levels = np.linspace(0, S_max, BNum * 2)
    taps = LFSR_TAPS[lfsr_length]
    mask = (1 << lfsr_length) - 1
    rng = np.random.default_rng()
    seeds = rng.integers(1, 2**lfsr_length, size=n_samples, dtype=np.int64)

    Y = np.zeros((n_samples, lfsr_length), dtype=np.float32)
    for i in range(lfsr_length):
        Y[:, lfsr_length - 1 - i] = (seeds >> i) & 1

    regs = seeds.copy()
    total_bits = seq_len * N
    bits_all = np.empty((n_samples, total_bits), dtype=np.uint8)
    for k in range(total_bits):
        SR = np.zeros(n_samples, dtype=np.int64)
        for t in taps:
            SR ^= (regs >> t)
        SR &= 1
        bits_all[:, k] = (regs >> (lfsr_length - 1)) & 1
        regs = ((regs << 1) & mask) | SR

    weights = (1 << np.arange(N)).astype(np.int64)
    bits_reshaped = bits_all.reshape(n_samples, seq_len, N)
    base_ids = bits_reshaped.dot(weights)
    input_data = rng.integers(0, 2, size=(n_samples, seq_len), dtype=np.int64)
    mod_indices = (input_data + base_ids % 2) % 2
    output_levels = S_levels[base_ids + BNum * mod_indices]
    # 量子ノイズの付加（noise_scale=0.0 の場合はノイズなし）
    if noise_scale > 0:
        xs_all = rng.normal(loc=output_levels, scale=noise_scale).astype(np.float32)
    else:
        xs_all = output_levels.astype(np.float32)

    X = xs_all.reshape(n_samples, seq_len, 1)
    mean_x = np.mean(X)
    std_x = np.std(X)
    X = (X - mean_x) / (std_x + 1e-8)
    return X, Y, seeds, input_data, xs_all


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """Warmup付きCosineAnnealingスケジューラを返す。"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_and_evaluate(lfsr_len, device):
    """指定LFSR長でモデルを学習・評価し、結果を返す。"""
    d_model, lstm_hidden, nhead, lstm_layers, n_train, n_test, \
        seq_len, epochs, batch_size, accum_steps = CONFIG[lfsr_len]
    noise_scale = QUANTUM_NOISE_SCALE
    warmup_epochs = min(20, epochs // 10)

    print(f"  パラメータ: d_model={d_model}, lstm_hidden={lstm_hidden}, "
          f"nhead={nhead}, lstm_layers={lstm_layers}")
    print(f"  データ: 学習={n_train}, 評価={n_test}, 系列長={seq_len}, "
          f"エポック={epochs}")
    print(f"  バッチサイズ={batch_size}, 勾配蓄積={accum_steps}ステップ, "
          f"warmup={warmup_epochs}エポック")
    print(f"  量子ノイズスケール: {noise_scale}")

    # ── データ生成 ──
    print("  データ生成中...")
    X_train, Y_train, _, _, _ = generate_seed_dataset(
        n_train, lfsr_len, seq_len, noise_scale)
    X_test, Y_test, test_seeds, test_input_data, test_raw_obs = \
        generate_seed_dataset(n_test, lfsr_len, seq_len, noise_scale)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # ── モデル初期化 ──
    model = SeedPredictorEnhanced(
        out_dim=lfsr_len, d_model=d_model, lstm_hidden=lstm_hidden,
        nhead=nhead, lstm_layers=lstm_layers
    ).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs, epochs)

    # パラメータ数を表示
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  モデルパラメータ数: {n_params:,}")

    # ── Early Stopping ──
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    best_state = None
    train_losses = []
    val_losses = []

    # ── 学習ループ ──
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for step, (bx, by) in enumerate(train_loader):
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            preds = model(bx)
            loss = criterion(preds, by) / accum_steps
            loss.backward()
            total_loss += loss.item() * accum_steps * len(by)
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        scheduler.step()
        avg_train_loss = total_loss / n_train
        train_losses.append(avg_train_loss)

        # 検証
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                preds = model(bx)
                val_loss += criterion(preds, by).item() * len(by)
                val_count += len(by)
        avg_val_loss = val_loss / val_count
        val_losses.append(avg_val_loss)

        if epoch == 1 or epoch % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch [{epoch:03d}/{epochs}] "
                  f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
                  f"LR: {lr_now:.6f}")

        if avg_val_loss < best_val_loss - 1e-5:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early Stopping at epoch {epoch} (patience={patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # ── 最終評価 ──
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device, non_blocking=True)
            preds = model(bx)
            all_preds.append((preds >= 0.5).int().cpu().numpy())
            all_targets.append(by.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    ber = 1.0 - np.mean(all_preds == all_targets)
    exact_matches = np.all(all_preds == all_targets, axis=1)
    exact_match_rate = np.mean(exact_matches)
    hamming_dists = np.sum(all_preds != all_targets, axis=1)

    print(f"  => 最終推論BER: {ber:.4f}")
    print(f"  => シード完全一致率: {exact_match_rate:.4f} "
          f"({np.sum(exact_matches)}/{n_test})")
    print(f"  => 平均ハミング距離: {np.mean(hamming_dists):.2f} / {lfsr_len}")

    pred_seeds = np.zeros(len(all_preds), dtype=np.int64)
    for i in range(lfsr_len):
        pred_seeds += all_preds[:, lfsr_len - 1 - i].astype(np.int64) << i

    n_verify = min(100, len(all_preds))
    verify_data = {
        'true_seeds': test_seeds[:n_verify].tolist(),
        'predicted_seeds': pred_seeds[:n_verify].tolist(),
        'input_data': test_input_data[:n_verify],
        'raw_observations': test_raw_obs[:n_verify],
        'seq_len': seq_len,
    }
    return {
        'ber': ber, 'exact_match': exact_match_rate,
        'hamming_dists': hamming_dists,
        'train_losses': train_losses, 'val_losses': val_losses,
        'lfsr_len': lfsr_len, 'verify_data': verify_data,
        'best_state': best_state,
        'model_config': {
            'd_model': d_model, 'lstm_hidden': lstm_hidden,
            'nhead': nhead, 'lstm_layers': lstm_layers,
            'out_dim': lfsr_len,
        },
    }


def plot_results(results_list, lfsr_lengths):
    """全LFSR長の結果をまとめた統合グラフを生成・保存する。"""
    n = len(lfsr_lengths)
    fig = plt.figure(figsize=(18, 12))

    ax1 = fig.add_subplot(2, 2, 1)
    bers = [r['ber'] for r in results_list]
    ax1.plot(lfsr_lengths, bers, marker='o', linewidth=2, color='tab:blue', label='BER')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random (0.5)')
    ax1.set_xlabel('LFSR Bit Length', fontsize=12)
    ax1.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax1.set_title('Seed Prediction BER vs LFSR Length (Enhanced)', fontsize=14)
    ax1.set_xticks(lfsr_lengths)
    ax1.set_ylim(-0.05, 0.55)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    exact = [r['exact_match'] for r in results_list]
    ax2.plot(lfsr_lengths, exact, marker='s', linewidth=2, color='tab:red',
             label='Exact Match Rate')
    ax2.set_xlabel('LFSR Bit Length', fontsize=12)
    ax2.set_ylabel('Exact Match Rate', fontsize=12)
    ax2.set_title('Seed Exact Match Rate vs LFSR Length (Enhanced)', fontsize=14)
    ax2.set_xticks(lfsr_lengths)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    for i, r in enumerate(results_list):
        ep = range(1, len(r['train_losses']) + 1)
        ax3.plot(ep, r['train_losses'], color=colors[i], alpha=0.7,
                 label=f'{r["lfsr_len"]}bit Train')
        ax3.plot(ep, r['val_losses'], color=colors[i], linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss (BCE)', fontsize=12)
    ax3.set_title('Training / Validation Loss Curves', fontsize=14)
    ax3.grid(True, linestyle=':', alpha=0.7)
    ax3.legend(fontsize=8, ncol=2)

    ax4 = fig.add_subplot(2, 2, 4)
    hamming_data = [r['hamming_dists'] for r in results_list]
    bp = ax4.boxplot(hamming_data, labels=[str(l) for l in lfsr_lengths],
                     patch_artist=True)
    box_colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, n))
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_xlabel('LFSR Bit Length', fontsize=12)
    ax4.set_ylabel('Hamming Distance (bits)', fontsize=12)
    ax4.set_title('Hamming Distance Distribution', fontsize=14)
    ax4.grid(True, linestyle=':', alpha=0.7)

    fig.suptitle(f'Y00 Seed Prediction Attack — Enhanced Eavesdropper\n'
                 f'(No Quantum Noise)', fontsize=16, y=1.01)
    fig.tight_layout()
    output_filename = 'seed_prediction_results_more_no_noise.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"グラフを {output_filename} として保存しました。")


def main():
    lfsr_lengths = [4, 6, 8, 10]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"実行デバイス: {device}")
    print(f"量子ノイズスケール: {QUANTUM_NOISE_SCALE}（ノイズなし）")
    print(f"検証LFSR長: {lfsr_lengths}")
    print(f"※ 強化版・ノイズなし: 大規模データ・拡張モデル・長系列対応\n")

    # 推定実行時間の表示
    print("推定実行時間（目安）:")
    for l in lfsr_lengths:
        cfg = CONFIG[l]
        est_min = cfg[4] * cfg[7] / 500_000  # 大雑把な推定
        print(f"  LFSR {l:2d}bit: 約 {est_min:.0f} 分")
    print()

    # モデル保存ディレクトリの作成
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    print(f"モデル保存先: {MODEL_SAVE_DIR}/\n")

    results_list = []
    for l_len in lfsr_lengths:
        print(f"=========================================")
        print(f" LFSR長: {l_len} bit (周期: {(1 << l_len) - 1}) のシード値推論を開始")
        print(f"=========================================")
        t_start = time.time()
        result = train_and_evaluate(l_len, device)
        elapsed = time.time() - t_start
        print(f"  所要時間: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分)")

        # ── 学習済みモデルの保存 ──
        if result['best_state'] is not None:
            model_path = os.path.join(
                MODEL_SAVE_DIR, f'seed_predictor_{l_len}bit.pth')
            torch.save({
                'model_state_dict': result['best_state'],
                'model_config': result['model_config'],
                'lfsr_length': l_len,
                'ber': result['ber'],
                'exact_match': result['exact_match'],
                'quantum_noise_scale': QUANTUM_NOISE_SCALE,
            }, model_path)
            print(f"  モデルを {model_path} に保存しました。")

        results_list.append(result)
        print()

    # ── 結果サマリー ──
    print("=" * 50)
    print(" 全結果サマリー（強化版）")
    print("=" * 50)
    print(f"{'LFSR長':>8} | {'BER':>8} | {'完全一致率':>10} | {'平均ハミング距離':>14}")
    print("-" * 50)
    for r in results_list:
        print(f"{r['lfsr_len']:>6} bit | {r['ber']:>8.4f} | "
              f"{r['exact_match']:>10.4f} | "
              f"{np.mean(r['hamming_dists']):>12.2f}")

    # ── 検証データ保存 ──
    print("\n検証データを保存しています...")
    yaml_data = {
        'quantum_noise_scale': float(QUANTUM_NOISE_SCALE),
        'N': 12, 'S_max': 10.0, 'BNum': 2**12,
        'version': 'enhanced', 'experiments': {},
    }
    for r in results_list:
        l_len = r['lfsr_len']
        vd = r['verify_data']
        data_file = f'experiment_data_{l_len}bit_more_no_noise.npz'
        yaml_data['experiments'][l_len] = {
            'lfsr_length': l_len, 'taps': LFSR_TAPS[l_len],
            'seq_len': vd['seq_len'],
            'n_verify': len(vd['true_seeds']),
            'true_seeds': vd['true_seeds'],
            'predicted_seeds': vd['predicted_seeds'],
            'data_file': data_file,
        }
        np.savez(data_file, input_data=vd['input_data'],
                 observations=vd['raw_observations'])
        print(f"  {data_file} を保存しました。")
    with open('experiment_params_more_no_noise.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
    print("  experiment_params_more_no_noise.yaml を保存しました。")

    print("\nグラフを描画・保存します...")
    plot_results(results_list, lfsr_lengths)


class TeeLogger:
    """標準出力をコンソールとファイルの両方に書き出すロガー。"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


if __name__ == "__main__":
    # ログファイル名にタイムスタンプを付与
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'seed_claude_more_no_noise_log_{timestamp}.txt'
    tee = TeeLogger(log_filename)
    sys.stdout = tee

    print(f"ログファイル: {log_filename}")
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print(f"総実行時間: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分)")

    tee.close()
    print(f"ログを {log_filename} に保存しました。")
