# -*- coding: utf-8 -*-
"""
seed_claude_more_smax.py

目的:
  seed_claude_more.py の S_max 掃引版。
  S_max を 1〜10 まで変化させて学習・評価を行い、
  信号レベル間隔がシード推論性能に与える影響を検証する。
  ※ 変更点は generate_seed_dataset に s_max 引数を追加し、
    main で S_max の外側ループを追加しただけの最小変更。
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
MODEL_SAVE_DIR = 'saved_models_more_smax'

# ── 量子ノイズのスケール設定 ──
QUANTUM_NOISE_SCALE = 0.5

# ── S_max 掃引値 ──
SMAX_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

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
    """強化版シード値推論モデル（seed_claude_more.py と同一構造）。"""
    def __init__(self, out_dim, d_model=128, lstm_hidden=128, nhead=4,
                 lstm_layers=2, dropout=0.15):
        super().__init__()
        self.d_model = d_model
        ch1 = d_model // 4
        ch2 = d_model // 2

        self.conv_block = nn.Sequential(
            nn.Conv1d(1, ch1, kernel_size=3, padding=1), nn.GELU(), nn.BatchNorm1d(ch1),
            nn.Conv1d(ch1, ch2, kernel_size=5, padding=2), nn.GELU(), nn.BatchNorm1d(ch2),
            nn.Conv1d(ch2, d_model, kernel_size=7, padding=3), nn.GELU(), nn.BatchNorm1d(d_model),
        )
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        lstm_out_dim = lstm_hidden * 2
        self.attn1 = nn.MultiheadAttention(embed_dim=lstm_out_dim, num_heads=nhead,
                                           dropout=dropout, batch_first=True)
        self.attn_norm1 = nn.LayerNorm(lstm_out_dim)
        self.attn2 = nn.MultiheadAttention(embed_dim=lstm_out_dim, num_heads=nhead,
                                           dropout=dropout, batch_first=True)
        self.attn_norm2 = nn.LayerNorm(lstm_out_dim)
        self.fc_out = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, lstm_out_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, out_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        h = x.transpose(1, 2)
        h = self.conv_block(h)
        h = h.transpose(1, 2)
        h, _ = self.lstm(h)
        a1, _ = self.attn1(h, h, h)
        h = self.attn_norm1(h + a1)
        a2, _ = self.attn2(h, h, h)
        h = self.attn_norm2(h + a2)
        return self.fc_out(h[:, -1, :])


def generate_seed_dataset(n_samples, lfsr_length, seq_len, noise_scale, s_max=10.0):
    """ランダムシードからLFSRを駆動し、量子ノイズ付き観測系列とシードビット列を生成。
    ※ s_max: 信号レベルの最大値（変更可能）。"""
    N = 12
    BNum = 2 ** N
    S_levels = np.linspace(0, s_max, BNum * 2)
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
    xs_all = rng.normal(loc=output_levels, scale=noise_scale).astype(np.float32)

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


def train_and_evaluate(lfsr_len, device, s_max=10.0):
    """指定LFSR長・S_maxでモデルを学習・評価し、結果を返す。"""
    d_model, lstm_hidden, nhead, lstm_layers, n_train, n_test, \
        seq_len, epochs, batch_size, accum_steps = CONFIG[lfsr_len]
    noise_scale = QUANTUM_NOISE_SCALE
    warmup_epochs = min(20, epochs // 10)

    print(f"  パラメータ: d_model={d_model}, lstm_hidden={lstm_hidden}, "
          f"nhead={nhead}, lstm_layers={lstm_layers}")
    print(f"  データ: 学習={n_train}, 評価={n_test}, 系列長={seq_len}, "
          f"エポック={epochs}")
    print(f"  S_max={s_max}, σ={noise_scale}")

    # ── データ生成 ──
    print("  データ生成中...")
    X_train, Y_train, _, _, _ = generate_seed_dataset(
        n_train, lfsr_len, seq_len, noise_scale, s_max)
    X_test, Y_test, test_seeds, test_input_data, test_raw_obs = \
        generate_seed_dataset(n_test, lfsr_len, seq_len, noise_scale, s_max)

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

    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    best_state = None
    train_losses = []
    val_losses = []

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

    return {
        'ber': ber, 'exact_match': exact_match_rate,
        'hamming_dists': hamming_dists,
        'lfsr_len': lfsr_len, 's_max': s_max,
        'best_state': best_state,
        'model_config': {
            'd_model': d_model, 'lstm_hidden': lstm_hidden,
            'nhead': nhead, 'lstm_layers': lstm_layers,
            'out_dim': lfsr_len,
        },
    }


def plot_smax_results(results_matrix, lfsr_lengths, smax_values):
    """S_max 掃引の結果をグラフ化する（4パネル）。"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(lfsr_lengths)))

    # ── 左上: BER vs S_max ──
    ax = axes[0, 0]
    for i, l_len in enumerate(lfsr_lengths):
        bers = [results_matrix[l_len][s]['ber'] for s in smax_values]
        ax.plot(smax_values, bers, marker='o', linewidth=2, color=colors[i],
                label=f'{l_len}bit', markersize=6)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='ランダム (0.5)')
    ax.set_xlabel('S_max', fontsize=12)
    ax.set_ylabel('BER', fontsize=12)
    ax.set_title('シード推論BER vs S_max', fontsize=14)
    ax.set_ylim(-0.05, 0.55)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=10)

    # ── 右上: 完全一致率 vs S_max ──
    ax = axes[0, 1]
    for i, l_len in enumerate(lfsr_lengths):
        exact = [results_matrix[l_len][s]['exact_match'] for s in smax_values]
        ax.plot(smax_values, exact, marker='s', linewidth=2, color=colors[i],
                label=f'{l_len}bit', markersize=6)
    ax.set_xlabel('S_max', fontsize=12)
    ax.set_ylabel('Exact Match Rate', fontsize=12)
    ax.set_title('シード完全一致率 vs S_max', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=10)

    # ── 左下: 平均ハミング距離 vs S_max ──
    ax = axes[1, 0]
    for i, l_len in enumerate(lfsr_lengths):
        hd = [np.mean(results_matrix[l_len][s]['hamming_dists']) for s in smax_values]
        ax.plot(smax_values, hd, marker='^', linewidth=2, color=colors[i],
                label=f'{l_len}bit', markersize=6)
    ax.set_xlabel('S_max', fontsize=12)
    ax.set_ylabel('平均ハミング距離 (bits)', fontsize=12)
    ax.set_title('平均ハミング距離 vs S_max', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=10)

    # ── 右下: ヒートマップ（LFSR長 × S_max → BER）──
    ax = axes[1, 1]
    ber_matrix = np.array([
        [results_matrix[l][s]['ber'] for s in smax_values]
        for l in lfsr_lengths
    ])
    im = ax.imshow(ber_matrix, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=0.5)
    ax.set_xticks(range(len(smax_values)))
    ax.set_xticklabels([str(s) for s in smax_values])
    ax.set_yticks(range(len(lfsr_lengths)))
    ax.set_yticklabels([f'{l}bit' for l in lfsr_lengths])
    ax.set_xlabel('S_max', fontsize=12)
    ax.set_ylabel('LFSR Bit Length', fontsize=12)
    ax.set_title('BERヒートマップ（LFSR長 × S_max）', fontsize=14)
    for yi in range(len(lfsr_lengths)):
        for xi in range(len(smax_values)):
            val = ber_matrix[yi, xi]
            text_color = 'white' if val > 0.25 else 'black'
            ax.text(xi, yi, f'{val:.3f}', ha='center', va='center',
                    fontsize=8, color=text_color, fontweight='bold')
    fig.colorbar(im, ax=ax, label='BER')

    fig.suptitle(f'Y00 シード推論攻撃 — S_max 掃引実験\n'
                 f'（σ={QUANTUM_NOISE_SCALE}）',
                 fontsize=16, y=1.02)
    fig.tight_layout()
    out = 'seed_smax_sweep_results.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"\nグラフを {out} として保存しました。")


def plot_smax_ber_single(results_matrix, lfsr_lengths, smax_values):
    """S_max vs BER の1枚グラフを生成・保存する。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 0.4, len(lfsr_lengths)))
    markers = ['o', 's', '^', 'D']

    for i, l_len in enumerate(lfsr_lengths):
        bers = [results_matrix[l_len][s]['ber'] for s in smax_values]
        ax.plot(smax_values, bers, marker=markers[i], linewidth=2.5,
                color=colors[i], label=f'LFSR {l_len}bit', markersize=8)

    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='ランダム推測 (0.5)')
    ax.set_xlabel('S_max（信号レベル最大値）', fontsize=14)
    ax.set_ylabel('Seed BER（シード推論誤り率）', fontsize=14)
    ax.set_title(f'S_max によるシード推論誤り率の変化（σ={QUANTUM_NOISE_SCALE}）',
                 fontsize=15)
    ax.set_xticks(smax_values)
    ax.set_ylim(-0.02, 0.55)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=12, loc='upper right')
    fig.tight_layout()
    out = 'seed_smax_ber_single.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"BER単体グラフを {out} として保存しました。")


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


def main():
    lfsr_lengths = [4, 6, 8, 10]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print("  Y00 シード推論攻撃 — S_max 掃引学習実験")
    print("=" * 65)
    print(f"実行デバイス: {device}")
    print(f"量子ノイズスケール: {QUANTUM_NOISE_SCALE}")
    print(f"検証LFSR長: {lfsr_lengths}")
    print(f"S_max 掃引値: {SMAX_VALUES}")
    print()

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # results_matrix[lfsr_len][s_max] = result
    results_matrix = {l: {} for l in lfsr_lengths}

    for s_max in SMAX_VALUES:
        print(f"\n{'#' * 65}")
        print(f"  S_max = {s_max}")
        print(f"{'#' * 65}")

        for l_len in lfsr_lengths:
            print(f"\n  ─── LFSR {l_len}bit (周期: {(1 << l_len) - 1}) ───")
            t_start = time.time()
            result = train_and_evaluate(l_len, device, s_max=float(s_max))
            elapsed = time.time() - t_start
            print(f"  所要時間: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分)")

            results_matrix[l_len][s_max] = result

            # モデル保存
            if result['best_state'] is not None:
                model_path = os.path.join(
                    MODEL_SAVE_DIR, f'seed_predictor_{l_len}bit_smax{s_max}.pth')
                torch.save({
                    'model_state_dict': result['best_state'],
                    'model_config': result['model_config'],
                    'lfsr_length': l_len,
                    's_max': s_max,
                    'ber': result['ber'],
                    'exact_match': result['exact_match'],
                    'quantum_noise_scale': QUANTUM_NOISE_SCALE,
                }, model_path)
                print(f"  モデルを {model_path} に保存しました。")

    # ── 全結果サマリー（BER）──
    print("\n" + "=" * 80)
    print("  全結果サマリー（BER）")
    print("=" * 80)
    header = f"{'LFSR':>6} |"
    for s in SMAX_VALUES:
        header += f" S={s:>2} |"
    print(header)
    print("-" * len(header))
    for l_len in lfsr_lengths:
        row = f"{l_len:>4}bit |"
        for s in SMAX_VALUES:
            ber = results_matrix[l_len][s]['ber']
            row += f" {ber:.3f} |"
        print(row)

    # ── 全結果サマリー（完全一致率）──
    print(f"\n{'=' * 80}")
    print("  全結果サマリー（完全一致率）")
    print("=" * 80)
    header = f"{'LFSR':>6} |"
    for s in SMAX_VALUES:
        header += f" S={s:>2} |"
    print(header)
    print("-" * len(header))
    for l_len in lfsr_lengths:
        row = f"{l_len:>4}bit |"
        for s in SMAX_VALUES:
            em = results_matrix[l_len][s]['exact_match']
            row += f" {em:.3f} |"
        print(row)

    # ── グラフ描画 ──
    print("\nグラフを描画中...")
    plot_smax_results(results_matrix, lfsr_lengths, SMAX_VALUES)
    plot_smax_ber_single(results_matrix, lfsr_lengths, SMAX_VALUES)


if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'seed_claude_more_smax_log_{timestamp}.txt'
    tee = TeeLogger(log_filename)
    sys.stdout = tee

    print(f"ログファイル: {log_filename}")
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print(f"\n総実行時間: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分)")

    tee.close()
    print(f"ログを {log_filename} に保存しました。")
