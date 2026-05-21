# -*- coding: utf-8 -*-
"""
seed_claude.py

目的:
  Y00量子変調のLFSR初期状態（シード値）を推論する
  最適化ニューラルネット盗聴者（Eve）プログラム。
  Conv1D + BiLSTM + Multi-Head Attention のハイブリッドモデルを使用し、
  4, 6, 8, 12, 16 bit のLFSR長に対して検証を行う。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time
import math
import yaml

# ── 量子ノイズのスケール設定 ──
# 真空揺らぎに相当する標準偏差。変更時はprint文で出力される。
QUANTUM_NOISE_SCALE = 0.5

# ── LFSRタップ位置の定義 ──
# 最長周期 (2^n - 1) を生成するための原始多項式に基づくタップ
LFSR_TAPS = {
    4:  [3, 2],
    6:  [5, 4],
    8:  [7, 5, 4, 3],
    10: [9, 6],
    12: [11, 10, 9, 3],
    14: [13, 12, 11, 1],
    16: [15, 13, 12, 10]
}

# ── LFSR長に応じたモデル・学習パラメータの設定 ──
# (d_model, lstm_hidden, nhead, lstm_layers, n_train, n_test, seq_len, epochs)
CONFIG = {
    4:  (64,  64,  4, 1,  20_000,  5_000,  64, 80),
    6:  (64,  64,  4, 1,  30_000,  8_000,  80, 100),
    8:  (128, 128, 4, 2,  50_000, 10_000, 128, 120),
    10: (128, 128, 4, 2,  60_000, 12_000, 160, 130),
    12: (128, 128, 8, 2,  80_000, 15_000, 256, 150),
    14: (256, 256, 8, 3,  90_000, 18_000, 384, 180),
    16: (256, 256, 8, 3, 100_000, 20_000, 512, 200),
}


# ══════════════════════════════════════════════
# モデル定義: Conv1D + BiLSTM + Attention ハイブリッド
# ══════════════════════════════════════════════

class SeedPredictorHybrid(nn.Module):
    """
    盗聴者（Eve）のシード値推論モデル。
    1. Conv1D: 局所的な時系列パターンを特徴抽出
    2. BiLSTM: 双方向の時系列依存性（LFSR周期パターン）を学習
    3. Multi-Head Attention: 長距離の相関関係を捕捉
    4. MLP Head: シード各ビットの確率を出力
    """
    def __init__(self, out_dim, d_model=128, lstm_hidden=128, nhead=4,
                 lstm_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 1. 多層Conv1Dで局所特徴抽出
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # 2. 双方向LSTM
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # 3. Multi-Head Attention（BiLSTMの出力次元 = lstm_hidden * 2）
        lstm_out_dim = lstm_hidden * 2
        self.attn = nn.MultiheadAttention(
            embed_dim=lstm_out_dim, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(lstm_out_dim)

        # 4. MLP Head: シードビット推論
        self.fc_out = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, 1)
        # Conv1Dは (batch, channels, seq_len) を要求
        h = x.transpose(1, 2)
        h = self.conv_block(h)
        h = h.transpose(1, 2)  # (batch, seq_len, d_model)

        # BiLSTM
        h, _ = self.lstm(h)  # (batch, seq_len, lstm_hidden*2)

        # Self-Attention + 残差接続
        attn_out, _ = self.attn(h, h, h)
        h = self.attn_norm(h + attn_out)

        # 最終ステップの特徴量でシードを推論
        last = h[:, -1, :]
        return self.fc_out(last)


# ══════════════════════════════════════════════
# データ生成
# ══════════════════════════════════════════════

def generate_seed_dataset(n_samples, lfsr_length, seq_len, noise_scale):
    """
    ランダムなシードからLFSRを駆動し、量子ノイズ付き観測系列とシードビット列を生成。
    """
    N = 12  # 基底決定に用いるビット数
    BNum = 2 ** N
    S_max = 10.0
    S_levels = np.linspace(0, S_max, BNum * 2)

    taps = LFSR_TAPS[lfsr_length]
    mask = (1 << lfsr_length) - 1

    rng = np.random.default_rng()
    # シード値を 1 〜 (2^lfsr_length - 1) の範囲でランダム生成（0は除外）
    seeds = rng.integers(1, 2**lfsr_length, size=n_samples, dtype=np.int64)

    # 正解ラベル: シード値を2進数配列に変換
    Y = np.zeros((n_samples, lfsr_length), dtype=np.float32)
    for i in range(lfsr_length):
        Y[:, lfsr_length - 1 - i] = (seeds >> i) & 1

    regs = seeds.copy()
    total_bits = seq_len * N

    # LFSR出力ビット列を一括計算
    bits_all = np.empty((n_samples, total_bits), dtype=np.uint8)
    for k in range(total_bits):
        SR = np.zeros(n_samples, dtype=np.int64)
        for t in taps:
            SR ^= (regs >> t)
        SR &= 1
        bits_all[:, k] = (regs >> (lfsr_length - 1)) & 1
        regs = ((regs << 1) & mask) | SR

    # Nビットずつまとめて基底IDを算出
    weights = (1 << np.arange(N)).astype(np.int64)
    bits_reshaped = bits_all.reshape(n_samples, seq_len, N)
    base_ids = bits_reshaped.dot(weights)

    # 送信データをランダム生成
    input_data = rng.integers(0, 2, size=(n_samples, seq_len), dtype=np.int64)

    # Y00変調
    mod_indices = (input_data + base_ids % 2) % 2
    output_levels = S_levels[base_ids + BNum * mod_indices]

    # 量子ノイズの付加
    xs_all = rng.normal(loc=output_levels, scale=noise_scale).astype(np.float32)

    # (バッチ, 系列長, 1) に変形 + 標準化
    X = xs_all.reshape(n_samples, seq_len, 1)
    mean_x = np.mean(X)
    std_x = np.std(X)
    X = (X - mean_x) / (std_x + 1e-8)

    return X, Y, seeds, input_data, xs_all


# ══════════════════════════════════════════════
# 学習・評価ループ
# ══════════════════════════════════════════════

def train_and_evaluate(lfsr_len, device):
    """
    指定LFSR長でモデルを学習・評価し、結果を返す。
    """
    d_model, lstm_hidden, nhead, lstm_layers, n_train, n_test, seq_len, epochs = CONFIG[lfsr_len]
    batch_size = 512
    noise_scale = QUANTUM_NOISE_SCALE

    print(f"  パラメータ: d_model={d_model}, lstm_hidden={lstm_hidden}, nhead={nhead}, "
          f"lstm_layers={lstm_layers}")
    print(f"  データ: 学習={n_train}, 評価={n_test}, 系列長={seq_len}, エポック={epochs}")
    print(f"  量子ノイズスケール: {noise_scale}")

    # ── データ生成 ──
    print("  データ生成中...")
    X_train, Y_train, _, _, _ = generate_seed_dataset(n_train, lfsr_len, seq_len, noise_scale)
    X_test, Y_test, test_seeds, test_input_data, test_raw_obs = generate_seed_dataset(
        n_test, lfsr_len, seq_len, noise_scale)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # ── モデル初期化 ──
    model = SeedPredictorHybrid(
        out_dim=lfsr_len, d_model=d_model, lstm_hidden=lstm_hidden,
        nhead=nhead, lstm_layers=lstm_layers
    ).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # ── Early Stopping 用変数 ──
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_state = None

    train_losses = []
    val_losses = []

    # ── 学習ループ ──
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(by)
        scheduler.step()

        avg_train_loss = total_loss / n_train
        train_losses.append(avg_train_loss)

        # 検証Loss（テストデータの一部で簡易評価）
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
                preds = model(bx)
                val_loss += criterion(preds, by).item() * len(by)
                val_count += len(by)
        avg_val_loss = val_loss / val_count
        val_losses.append(avg_val_loss)

        if epoch == 1 or epoch % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch [{epoch:03d}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                  f"LR: {lr_now:.6f}")

        # Early Stopping 判定
        if avg_val_loss < best_val_loss - 1e-5:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early Stopping at epoch {epoch} (patience={patience})")
                break

    # ベストモデルをロード
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
            preds_bin = (preds >= 0.5).int().cpu().numpy()
            all_preds.append(preds_bin)
            all_targets.append(by.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # BER（ビットエラー率）
    bit_accuracy = np.mean(all_preds == all_targets)
    ber = 1.0 - bit_accuracy

    # シード完全一致率
    exact_matches = np.all(all_preds == all_targets, axis=1)
    exact_match_rate = np.mean(exact_matches)

    # ハミング距離の分布
    hamming_dists = np.sum(all_preds != all_targets, axis=1)

    print(f"  => 最終推論BER: {ber:.4f}")
    print(f"  => シード完全一致率: {exact_match_rate:.4f} "
          f"({np.sum(exact_matches)}/{n_test})")
    print(f"  => 平均ハミング距離: {np.mean(hamming_dists):.2f} / {lfsr_len}")

    # 推定seed値をビット列から整数に変換
    pred_seeds = np.zeros(len(all_preds), dtype=np.int64)
    for i in range(lfsr_len):
        pred_seeds += all_preds[:, lfsr_len - 1 - i].astype(np.int64) << i

    # 検証用100サンプルを収集
    n_verify = min(100, len(all_preds))
    verify_data = {
        'true_seeds': test_seeds[:n_verify].tolist(),
        'predicted_seeds': pred_seeds[:n_verify].tolist(),
        'input_data': test_input_data[:n_verify],
        'raw_observations': test_raw_obs[:n_verify],
        'seq_len': seq_len,
    }

    return {
        'ber': ber,
        'exact_match': exact_match_rate,
        'hamming_dists': hamming_dists,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'lfsr_len': lfsr_len,
        'verify_data': verify_data,
    }


# ══════════════════════════════════════════════
# 可視化
# ══════════════════════════════════════════════

def plot_results(results_list, lfsr_lengths):
    """
    全LFSR長の結果をまとめた統合グラフを生成・保存する。
    """
    n = len(lfsr_lengths)
    fig = plt.figure(figsize=(18, 12))

    # ── 上段左: BER vs LFSR長 ──
    ax1 = fig.add_subplot(2, 2, 1)
    bers = [r['ber'] for r in results_list]
    ax1.plot(lfsr_lengths, bers, marker='o', linewidth=2, color='tab:blue', label='BER')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random (0.5)')
    ax1.set_xlabel('LFSR Bit Length', fontsize=12)
    ax1.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax1.set_title('Seed Prediction BER vs LFSR Length', fontsize=14)
    ax1.set_xticks(lfsr_lengths)
    ax1.set_ylim(-0.05, 0.55)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend()

    # ── 上段右: シード完全一致率 vs LFSR長 ──
    ax2 = fig.add_subplot(2, 2, 2)
    exact = [r['exact_match'] for r in results_list]
    ax2.plot(lfsr_lengths, exact, marker='s', linewidth=2, color='tab:red', label='Exact Match Rate')
    ax2.set_xlabel('LFSR Bit Length', fontsize=12)
    ax2.set_ylabel('Exact Match Rate', fontsize=12)
    ax2.set_title('Seed Exact Match Rate vs LFSR Length', fontsize=14)
    ax2.set_xticks(lfsr_lengths)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()

    # ── 下段左: 学習曲線（各LFSR長を重ねて表示）──
    ax3 = fig.add_subplot(2, 2, 3)
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    for i, r in enumerate(results_list):
        epochs_range = range(1, len(r['train_losses']) + 1)
        ax3.plot(epochs_range, r['train_losses'], color=colors[i], alpha=0.7,
                 label=f'{r["lfsr_len"]}bit Train')
        ax3.plot(epochs_range, r['val_losses'], color=colors[i], linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss (BCE)', fontsize=12)
    ax3.set_title('Training / Validation Loss Curves', fontsize=14)
    ax3.grid(True, linestyle=':', alpha=0.7)
    ax3.legend(fontsize=8, ncol=2)

    # ── 下段右: ハミング距離の分布（箱ひげ図）──
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

    fig.suptitle(f'Y00 Seed Prediction Attack — Hybrid NN Eavesdropper\n'
                 f'(Noise Scale = {QUANTUM_NOISE_SCALE})', fontsize=16, y=1.01)
    fig.tight_layout()
    output_filename = 'seed_prediction_results.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"グラフを {output_filename} として保存しました。")


# ══════════════════════════════════════════════
# メイン処理
# ══════════════════════════════════════════════

def main():
    lfsr_lengths = [4, 6, 8, 10, 12, 14, 16]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"実行デバイス: {device}")
    print(f"量子ノイズスケール: {QUANTUM_NOISE_SCALE}")
    print(f"検証LFSR長: {lfsr_lengths}\n")

    results_list = []

    for l_len in lfsr_lengths:
        print(f"=========================================")
        print(f" LFSR長: {l_len} bit (周期: {(1 << l_len) - 1}) のシード値推論を開始")
        print(f"=========================================")

        result = train_and_evaluate(l_len, device)
        results_list.append(result)
        print()

    # ── 結果サマリー ──
    print("=" * 50)
    print(" 全結果サマリー")
    print("=" * 50)
    print(f"{'LFSR長':>8} | {'BER':>8} | {'完全一致率':>10} | {'平均ハミング距離':>14}")
    print("-" * 50)
    for r in results_list:
        print(f"{r['lfsr_len']:>6} bit | {r['ber']:>8.4f} | {r['exact_match']:>10.4f} | "
              f"{np.mean(r['hamming_dists']):>12.2f}")

    # ── 検証データ保存 ──
    print("\n検証データを保存しています...")
    yaml_data = {
        'quantum_noise_scale': float(QUANTUM_NOISE_SCALE),
        'N': 12,
        'S_max': 10.0,
        'BNum': 2**12,
        'experiments': {},
    }
    for r in results_list:
        l_len = r['lfsr_len']
        vd = r['verify_data']
        data_file = f'experiment_data_{l_len}bit.npz'
        yaml_data['experiments'][l_len] = {
            'lfsr_length': l_len,
            'taps': LFSR_TAPS[l_len],
            'seq_len': vd['seq_len'],
            'n_verify': len(vd['true_seeds']),
            'true_seeds': vd['true_seeds'],
            'predicted_seeds': vd['predicted_seeds'],
            'data_file': data_file,
        }
        np.savez(
            data_file,
            input_data=vd['input_data'],
            observations=vd['raw_observations'],
        )
        print(f"  {data_file} を保存しました。")
    with open('experiment_params.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
    print("  experiment_params.yaml を保存しました。")

    # ── グラフ描画 ──
    print("\nグラフを描画・保存します...")
    plot_results(results_list, lfsr_lengths)


if __name__ == "__main__":
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print(f"総実行時間: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分)")
