# -*- coding: utf-8 -*-
"""
seed_inference_sigma.py

目的:
  ノイズあり学習済みモデル（saved_models_more/）を使用し、
  信号の量子ノイズスケール σ を 0.0〜0.9 まで 0.1 刻みで変化させた
  ときのシード推論性能を測定する。
  量子ノイズ耐性の閾値を定量的に評価するための実験。
"""

import os
import sys
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time

# ══════════════════════════════════════════════
# 定数・設定
# ══════════════════════════════════════════════

LFSR_TAPS = {
    4:  [3, 2],
    6:  [5, 4],
    8:  [7, 5, 4, 3],
    10: [9, 6],
    12: [11, 10, 9, 3],
    14: [13, 12, 11, 1],
    16: [15, 13, 12, 10],
}

SEQ_LEN_MAP = {4: 64, 6: 80, 8: 128, 10: 256}
N_INFERENCE = 5_000

# ノイズありモデルのディレクトリ
MODEL_DIR = 'saved_models_more'

# 信号ノイズスケールの範囲
SIGMA_VALUES = [round(s * 0.1, 1) for s in range(10)]  # 0.0, 0.1, ..., 0.9

LFSR_LENGTHS = [4, 6, 8, 10]


# ══════════════════════════════════════════════
# モデル定義（seed_claude_more.py と同一構造）
# ══════════════════════════════════════════════

class SeedPredictorEnhanced(nn.Module):
    """推論専用の再定義。seed_claude_more.py と構造を完全一致させる。"""
    def __init__(self, out_dim, d_model=128, lstm_hidden=128, nhead=4,
                 lstm_layers=2, dropout=0.15):
        super().__init__()
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


# ══════════════════════════════════════════════
# データ生成
# ══════════════════════════════════════════════

def generate_inference_data(n_samples, lfsr_length, seq_len, noise_scale):
    """推論用テストデータを生成（再現性のためrngシード固定）。"""
    N = 12
    BNum = 2 ** N
    S_max = 10.0
    S_levels = np.linspace(0, S_max, BNum * 2)
    taps = LFSR_TAPS[lfsr_length]
    mask = (1 << lfsr_length) - 1
    rng = np.random.default_rng(seed=42)
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

    if noise_scale > 0:
        xs_all = rng.normal(loc=output_levels, scale=noise_scale).astype(np.float32)
    else:
        xs_all = output_levels.astype(np.float32)

    X = xs_all.reshape(n_samples, seq_len, 1)
    mean_x = np.mean(X)
    std_x = np.std(X)
    X = (X - mean_x) / (std_x + 1e-8)
    return X, Y, seeds


# ══════════════════════════════════════════════
# 推論実行
# ══════════════════════════════════════════════

def load_model(model_path, device):
    """モデルをロードして返す（1度だけロードして使い回す用）。"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['model_config']
    model = SeedPredictorEnhanced(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def infer_with_model(model, lfsr_len, device, noise_scale):
    """ロード済みモデルで指定ノイズ条件のデータを推論する。"""
    seq_len = SEQ_LEN_MAP[lfsr_len]
    X, Y, seeds = generate_inference_data(N_INFERENCE, lfsr_len, seq_len, noise_scale)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            preds = model(bx)
            all_preds.append((preds >= 0.5).int().cpu().numpy())
            all_targets.append(by.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    ber = 1.0 - np.mean(all_preds == all_targets)
    exact_matches = np.all(all_preds == all_targets, axis=1)
    hamming_dists = np.sum(all_preds != all_targets, axis=1)

    pred_seeds = np.zeros(len(all_preds), dtype=np.int64)
    for i in range(lfsr_len):
        pred_seeds += all_preds[:, lfsr_len - 1 - i].astype(np.int64) << i

    return {
        'ber': ber,
        'exact_match': np.mean(exact_matches),
        'hamming_dists': hamming_dists,
        'true_seeds': seeds,
        'pred_seeds': pred_seeds,
    }


# ══════════════════════════════════════════════
# グラフ描画
# ══════════════════════════════════════════════

def plot_sigma_sweep(results_matrix, lfsr_lengths, sigma_values):
    """σ掃引の結果をグラフ化する（4パネル）。"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(lfsr_lengths)))

    # ── 左上: BER vs σ（各LFSR長を線で表示）──
    ax = axes[0, 0]
    for i, l_len in enumerate(lfsr_lengths):
        bers = [results_matrix[l_len][s]['ber'] for s in sigma_values]
        ax.plot(sigma_values, bers, marker='o', linewidth=2, color=colors[i],
                label=f'{l_len}bit', markersize=6)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='ランダム (0.5)')
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='学習時σ (0.5)')
    ax.set_xlabel('信号ノイズスケール σ', fontsize=12)
    ax.set_ylabel('BER', fontsize=12)
    ax.set_title('シード推論BER vs ノイズスケール', fontsize=14)
    ax.set_ylim(-0.05, 0.55)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=10)

    # ── 右上: 完全一致率 vs σ ──
    ax = axes[0, 1]
    for i, l_len in enumerate(lfsr_lengths):
        exact = [results_matrix[l_len][s]['exact_match'] for s in sigma_values]
        ax.plot(sigma_values, exact, marker='s', linewidth=2, color=colors[i],
                label=f'{l_len}bit', markersize=6)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='学習時σ (0.5)')
    ax.set_xlabel('信号ノイズスケール σ', fontsize=12)
    ax.set_ylabel('Exact Match Rate', fontsize=12)
    ax.set_title('シード完全一致率 vs ノイズスケール', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=10)

    # ── 左下: 平均ハミング距離 vs σ ──
    ax = axes[1, 0]
    for i, l_len in enumerate(lfsr_lengths):
        hd = [np.mean(results_matrix[l_len][s]['hamming_dists']) for s in sigma_values]
        ax.plot(sigma_values, hd, marker='^', linewidth=2, color=colors[i],
                label=f'{l_len}bit', markersize=6)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='学習時σ (0.5)')
    ax.set_xlabel('信号ノイズスケール σ', fontsize=12)
    ax.set_ylabel('平均ハミング距離 (bits)', fontsize=12)
    ax.set_title('平均ハミング距離 vs ノイズスケール', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=10)

    # ── 右下: ヒートマップ（LFSR長 × σ → BER）──
    ax = axes[1, 1]
    ber_matrix = np.array([
        [results_matrix[l][s]['ber'] for s in sigma_values]
        for l in lfsr_lengths
    ])
    im = ax.imshow(ber_matrix, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=0.5)
    ax.set_xticks(range(len(sigma_values)))
    ax.set_xticklabels([f'{s:.1f}' for s in sigma_values])
    ax.set_yticks(range(len(lfsr_lengths)))
    ax.set_yticklabels([f'{l}bit' for l in lfsr_lengths])
    ax.set_xlabel('信号ノイズスケール σ', fontsize=12)
    ax.set_ylabel('LFSR Bit Length', fontsize=12)
    ax.set_title('BERヒートマップ（LFSR長 × σ）', fontsize=14)
    # セル内にBER値を表示
    for yi in range(len(lfsr_lengths)):
        for xi in range(len(sigma_values)):
            val = ber_matrix[yi, xi]
            text_color = 'white' if val > 0.25 else 'black'
            ax.text(xi, yi, f'{val:.3f}', ha='center', va='center',
                    fontsize=8, color=text_color, fontweight='bold')
    fig.colorbar(im, ax=ax, label='BER')

    fig.suptitle('Y00 シード推論攻撃 — ノイズスケール掃引実験\n'
                 '（ノイズありモデル × 各種σの信号）',
                 fontsize=16, y=1.02)
    fig.tight_layout()
    out = 'seed_inference_sigma_sweep.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"\nグラフを {out} として保存しました。")


# ══════════════════════════════════════════════
# ログ保存用 TeeLogger
# ══════════════════════════════════════════════

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


# ══════════════════════════════════════════════
# メイン処理
# ══════════════════════════════════════════════

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 65)
    print("  Y00 シード推論攻撃 — ノイズスケール掃引実験")
    print("=" * 65)
    print(f"実行デバイス: {device}")
    print(f"推論サンプル数: {N_INFERENCE:,}")
    print(f"検証LFSR長: {LFSR_LENGTHS}")
    print(f"モデル: {MODEL_DIR}/ （ノイズあり σ=0.5 で学習済み）")
    print(f"信号ノイズスケール: {SIGMA_VALUES}")
    print()

    # results_matrix[lfsr_len][sigma] = { ber, exact_match, ... }
    results_matrix = {l: {} for l in LFSR_LENGTHS}

    for l_len in LFSR_LENGTHS:
        model_path = os.path.join(MODEL_DIR, f'seed_predictor_{l_len}bit.pth')
        if not os.path.exists(model_path):
            print(f"⚠ {model_path} が見つかりません。スキップ。")
            continue

        print(f"{'=' * 65}")
        print(f"  LFSR {l_len}bit (周期: {(1 << l_len) - 1})")
        print(f"  モデル: {model_path}")
        print(f"{'=' * 65}")

        # モデルを1度だけロード
        model, checkpoint = load_model(model_path, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  パラメータ数: {n_params:,}")
        if checkpoint.get('ber') is not None:
            print(f"  (参考) 学習時BER: {checkpoint['ber']:.4f}")

        for sigma in SIGMA_VALUES:
            result = infer_with_model(model, l_len, device, sigma)
            results_matrix[l_len][sigma] = result

            print(f"\n  ── σ = {sigma:.1f} ──")
            print(f"  BER:           {result['ber']:.4f}")
            print(f"  完全一致率:    {result['exact_match']:.4f} "
                  f"({int(result['exact_match'] * N_INFERENCE)}/{N_INFERENCE})")
            print(f"  平均ハミング距離: {np.mean(result['hamming_dists']):.2f} / {l_len}")

            # サンプル推論結果5件
            print(f"  【サンプル5件】")
            for idx in range(min(5, N_INFERENCE)):
                ts = result['true_seeds'][idx]
                ps = result['pred_seeds'][idx]
                m = "✓" if ts == ps else "✗"
                print(f"    {idx+1}: 真={ts:>{l_len+2}} ({format(ts, f'0{l_len}b')}) "
                      f"→ 予={ps:>{l_len+2}} ({format(ps, f'0{l_len}b')}) {m}")

        print()

    # ── 全結果サマリー表 ──
    print("\n" + "=" * 80)
    print("  全結果サマリー（BER）")
    print("=" * 80)

    # ヘッダー行
    header = f"{'LFSR':>6} |"
    for s in SIGMA_VALUES:
        header += f" σ={s:.1f} |"
    print(header)
    print("-" * len(header))

    for l_len in LFSR_LENGTHS:
        row = f"{l_len:>4}bit |"
        for s in SIGMA_VALUES:
            if s in results_matrix[l_len]:
                ber = results_matrix[l_len][s]['ber']
                row += f" {ber:.4f} |"
            else:
                row += "    -   |"
        print(row)

    # ── 完全一致率サマリー ──
    print(f"\n{'=' * 80}")
    print("  全結果サマリー（完全一致率）")
    print("=" * 80)

    header = f"{'LFSR':>6} |"
    for s in SIGMA_VALUES:
        header += f" σ={s:.1f} |"
    print(header)
    print("-" * len(header))

    for l_len in LFSR_LENGTHS:
        row = f"{l_len:>4}bit |"
        for s in SIGMA_VALUES:
            if s in results_matrix[l_len]:
                em = results_matrix[l_len][s]['exact_match']
                row += f" {em:.4f} |"
            else:
                row += "    -   |"
        print(row)

    # ── ノイズ耐性分析 ──
    print(f"\n{'=' * 80}")
    print("  ノイズ耐性分析（BER < 0.1 を維持できる最大σ）")
    print("=" * 80)
    for l_len in LFSR_LENGTHS:
        max_sigma = None
        for s in SIGMA_VALUES:
            if s in results_matrix[l_len] and results_matrix[l_len][s]['ber'] < 0.1:
                max_sigma = s
        if max_sigma is not None:
            print(f"  LFSR {l_len:>2}bit: σ ≤ {max_sigma:.1f} まで BER < 0.1 を維持")
        else:
            print(f"  LFSR {l_len:>2}bit: σ=0.0 でも BER ≥ 0.1")

    # ── グラフ描画 ──
    print("\nグラフを描画中...")
    plot_sigma_sweep(results_matrix, LFSR_LENGTHS, SIGMA_VALUES)


if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'seed_inference_sigma_log_{timestamp}.txt'
    tee = TeeLogger(log_filename)
    sys.stdout = tee

    print(f"ログファイル: {log_filename}")
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print(f"\n総実行時間: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分)")

    tee.close()
    print(f"ログを {log_filename} に保存しました。")
