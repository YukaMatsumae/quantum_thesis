# -*- coding: utf-8 -*-
"""
seed_bob_smax.py

目的:
  Y00量子変調における正規受信者（Bob）の復号性能を検証する。
  Bobはシード値を知っているため、LFSR鍵系列を再生成し、
  2つの候補信号レベルの閾値判定で送信データを復号する。
  S_max を 1〜10 まで変化させ、信号レベル間隔が
  正規受信者のBERに与える影響を定量的に評価する。

  ※ seed_claude_more_smax.py（盗聴者Eve）との比較用。
"""

import os
import sys
from datetime import datetime
import numpy as np
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

# 量子ノイズのスケール（Eveの学習条件と統一）
QUANTUM_NOISE_SCALE = 0.5

# S_max 掃引値
SMAX_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 検証パラメータ
LFSR_LENGTHS = [4, 6, 8, 10]
N_SAMPLES = 10_000        # サンプル数
SEQ_LEN_MAP = {4: 64, 6: 80, 8: 128, 10: 256}


# ══════════════════════════════════════════════
# 信号生成 + 正規受信者による閾値復号
# ══════════════════════════════════════════════

def bob_decode(n_samples, lfsr_length, seq_len, noise_scale, s_max):
    """
    正規受信者（Bob）のシミュレーション。
    1. ランダムシードでLFSRを駆動し、送信信号を生成
    2. 量子ノイズを付加した受信信号を作成
    3. Bobはシードを知っているので、base_idsを再生成
    4. 各シンボルで2つの候補レベルを計算し、近い方を選択（閾値復号）
    返り値: データビットのBER, 正解数/全数
    """
    N = 12
    BNum = 2 ** N
    S_levels = np.linspace(0, s_max, BNum * 2)
    taps = LFSR_TAPS[lfsr_length]
    mask = (1 << lfsr_length) - 1
    rng = np.random.default_rng(seed=123)

    seeds = rng.integers(1, 2**lfsr_length, size=n_samples, dtype=np.int64)

    # LFSR駆動で鍵系列を生成
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

    # 送信データの生成と変調
    input_data = rng.integers(0, 2, size=(n_samples, seq_len), dtype=np.int64)
    mod_indices = (input_data + base_ids % 2) % 2
    output_levels = S_levels[base_ids + BNum * mod_indices]

    # 量子ノイズの付加
    if noise_scale > 0:
        received = rng.normal(loc=output_levels, scale=noise_scale).astype(np.float32)
    else:
        received = output_levels.astype(np.float32)

    # ── Bobによる閾値復号 ──
    # Bobはbase_idsを知っている → 候補信号レベルを計算
    # input_data=0 のときの信号レベル
    mod_for_0 = (0 + base_ids % 2) % 2
    level_for_0 = S_levels[base_ids + BNum * mod_for_0]
    # input_data=1 のときの信号レベル
    mod_for_1 = (1 + base_ids % 2) % 2
    level_for_1 = S_levels[base_ids + BNum * mod_for_1]

    # 受信信号と各候補レベルとの距離を比較
    dist_0 = np.abs(received - level_for_0)
    dist_1 = np.abs(received - level_for_1)
    decoded = (dist_1 < dist_0).astype(np.int64)

    # BER算出
    n_errors = np.sum(decoded != input_data)
    n_total = n_samples * seq_len
    ber = n_errors / n_total

    # シンボル単位の誤り分布
    errors_per_sample = np.sum(decoded != input_data, axis=1)

    return {
        'ber': ber,
        'n_errors': int(n_errors),
        'n_total': n_total,
        'errors_per_sample': errors_per_sample,
        'seq_len': seq_len,
    }


# ══════════════════════════════════════════════
# グラフ描画
# ══════════════════════════════════════════════

def plot_bob_results(results_matrix, lfsr_lengths, smax_values):
    """Bobの復号BER vs S_max グラフ（4パネル）。"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(lfsr_lengths)))

    # ── Panel 1: BER vs S_max ──
    ax = axes[0, 0]
    for i, l_len in enumerate(lfsr_lengths):
        bers = [results_matrix[l_len][s]['ber'] for s in smax_values]
        ax.plot(smax_values, bers, marker='o', linewidth=2, color=colors[i],
                label=f'LFSR {l_len}bit', markersize=6)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Guess (0.5)')
    ax.set_xlabel('S_max (Signal Level Range)', fontsize=12)
    ax.set_ylabel('Data BER', fontsize=12)
    ax.set_title('Legitimate Receiver (Bob) — Data BER vs S_max', fontsize=14)
    ax.set_ylim(-0.01, max(0.1, ax.get_ylim()[1]))
    ax.set_xticks(smax_values)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=10)

    # ── Panel 2: BER vs S_max (log scale) ──
    ax = axes[0, 1]
    for i, l_len in enumerate(lfsr_lengths):
        bers = [max(results_matrix[l_len][s]['ber'], 1e-6) for s in smax_values]
        ax.semilogy(smax_values, bers, marker='s', linewidth=2, color=colors[i],
                     label=f'LFSR {l_len}bit', markersize=6)
    ax.set_xlabel('S_max (Signal Level Range)', fontsize=12)
    ax.set_ylabel('Data BER (log scale)', fontsize=12)
    ax.set_title('Legitimate Receiver (Bob) — Data BER vs S_max (Log)', fontsize=14)
    ax.set_xticks(smax_values)
    ax.grid(True, linestyle=':', alpha=0.7, which='both')
    ax.legend(fontsize=10)

    # ── Panel 3: Error Distribution (box plot at worst S_max) ──
    ax = axes[1, 0]
    # S_max=1（最も厳しい条件）でのエラー分布
    worst_smax = smax_values[0]
    box_data = [results_matrix[l][worst_smax]['errors_per_sample'] for l in lfsr_lengths]
    bp = ax.boxplot(box_data, labels=[f'{l}bit' for l in lfsr_lengths], patch_artist=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_xlabel('LFSR Bit Length', fontsize=12)
    ax.set_ylabel(f'Errors per Sample (seq_len symbols)', fontsize=12)
    ax.set_title(f'Error Distribution at S_max = {worst_smax}', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.7, axis='y')

    # ── Panel 4: Heatmap (LFSR length x S_max → BER) ──
    ax = axes[1, 1]
    ber_matrix = np.array([
        [results_matrix[l][s]['ber'] for s in smax_values]
        for l in lfsr_lengths
    ])
    im = ax.imshow(ber_matrix, aspect='auto', cmap='RdYlGn_r',
                   vmin=0, vmax=max(0.05, ber_matrix.max()))
    ax.set_xticks(range(len(smax_values)))
    ax.set_xticklabels([str(s) for s in smax_values])
    ax.set_yticks(range(len(lfsr_lengths)))
    ax.set_yticklabels([f'{l}bit' for l in lfsr_lengths])
    ax.set_xlabel('S_max', fontsize=12)
    ax.set_ylabel('LFSR Bit Length', fontsize=12)
    ax.set_title('Data BER Heatmap (LFSR Length x S_max)', fontsize=14)
    for yi in range(len(lfsr_lengths)):
        for xi in range(len(smax_values)):
            val = ber_matrix[yi, xi]
            text_color = 'white' if val > ber_matrix.max() * 0.5 else 'black'
            ax.text(xi, yi, f'{val:.4f}', ha='center', va='center',
                    fontsize=8, color=text_color, fontweight='bold')
    fig.colorbar(im, ax=ax, label='BER')

    fig.suptitle(f'Y00 Legitimate Receiver (Bob) Decoding Performance\n'
                 f'Noise Scale = {QUANTUM_NOISE_SCALE}',
                 fontsize=16, y=1.02)
    fig.tight_layout()
    out = 'seed_bob_smax_results.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"\n4パネルグラフを {out} として保存しました。")


def plot_bob_ber_single(results_matrix, lfsr_lengths, smax_values):
    """BER vs S_max の1枚グラフ。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 0.4, len(lfsr_lengths)))
    markers = ['o', 's', '^', 'D']

    for i, l_len in enumerate(lfsr_lengths):
        bers = [results_matrix[l_len][s]['ber'] for s in smax_values]
        ax.plot(smax_values, bers, marker=markers[i], linewidth=2.5,
                color=colors[i], label=f'LFSR {l_len}bit', markersize=8)

    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Guess (0.5)')
    ax.set_xlabel('S_max (Signal Level Range)', fontsize=14)
    ax.set_ylabel('Data BER (Legitimate Receiver)', fontsize=14)
    ax.set_title(f'Bob Decoding BER vs S_max (Noise Scale = {QUANTUM_NOISE_SCALE})',
                 fontsize=15)
    ax.set_xticks(smax_values)
    ax.set_ylim(-0.005, max(0.05, ax.get_ylim()[1]))
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=12, loc='upper right')
    fig.tight_layout()
    out = 'seed_bob_smax_ber_single.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"BER単体グラフを {out} として保存しました。")


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
    print("=" * 65)
    print("  Y00 正規受信者（Bob）復号性能 — S_max 掃引実験")
    print("=" * 65)
    print(f"サンプル数: {N_SAMPLES:,}")
    print(f"量子ノイズスケール: {QUANTUM_NOISE_SCALE}")
    print(f"検証LFSR長: {LFSR_LENGTHS}")
    print(f"S_max 掃引値: {SMAX_VALUES}")
    print()
    print("※ 正規受信者はシード値を知っており、閾値復号で送信データを復号します。")
    print("※ NNは使用しません（解析的復号）。")
    print()

    results_matrix = {l: {} for l in LFSR_LENGTHS}

    for s_max in SMAX_VALUES:
        print(f"\n{'#' * 65}")
        print(f"  S_max = {s_max}")
        print(f"{'#' * 65}")

        # S_maxとσから信号レベル間隔を計算して表示
        N = 12
        BNum = 2 ** N
        level_spacing = s_max / (BNum * 2 - 1)
        snr_approx = level_spacing / QUANTUM_NOISE_SCALE if QUANTUM_NOISE_SCALE > 0 else float('inf')
        print(f"  信号レベル間隔: {level_spacing:.6f}")
        print(f"  信号レベル間隔 / σ (SNR的指標): {snr_approx:.4f}")

        for l_len in LFSR_LENGTHS:
            seq_len = SEQ_LEN_MAP[l_len]
            print(f"\n  ── LFSR {l_len}bit (周期: {(1 << l_len) - 1}, "
                  f"系列長: {seq_len}) ──")

            t_start = time.time()
            result = bob_decode(N_SAMPLES, l_len, seq_len,
                                QUANTUM_NOISE_SCALE, float(s_max))
            elapsed = time.time() - t_start

            results_matrix[l_len][s_max] = result

            print(f"  データBER:     {result['ber']:.6f}")
            print(f"  エラービット数: {result['n_errors']:,} / {result['n_total']:,}")
            print(f"  所要時間: {elapsed:.2f} 秒")

    # ── 全結果サマリー（BER）──
    print("\n" + "=" * 80)
    print("  全結果サマリー — 正規受信者（Bob）データBER")
    print("=" * 80)
    header = f"{'LFSR':>6} |"
    for s in SMAX_VALUES:
        header += f" S={s:>2}  |"
    print(header)
    print("-" * len(header))
    for l_len in LFSR_LENGTHS:
        row = f"{l_len:>4}bit |"
        for s in SMAX_VALUES:
            ber = results_matrix[l_len][s]['ber']
            row += f" {ber:.4f} |"
        print(row)

    # ── S_max と BER の関係分析 ──
    print(f"\n{'=' * 80}")
    print("  分析: BER < 1e-3（実用的な通信品質）を達成する最小 S_max")
    print("=" * 80)
    for l_len in LFSR_LENGTHS:
        min_smax = None
        for s in SMAX_VALUES:
            if results_matrix[l_len][s]['ber'] < 1e-3:
                min_smax = s
                break
        if min_smax is not None:
            print(f"  LFSR {l_len:>2}bit: S_max >= {min_smax} で BER < 0.001 達成")
        else:
            print(f"  LFSR {l_len:>2}bit: S_max=10 でも BER >= 0.001")

    # ── グラフ描画 ──
    print("\nグラフを描画中...")
    plot_bob_results(results_matrix, LFSR_LENGTHS, SMAX_VALUES)
    plot_bob_ber_single(results_matrix, LFSR_LENGTHS, SMAX_VALUES)


if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'seed_bob_smax_log_{timestamp}.txt'
    tee = TeeLogger(log_filename)
    sys.stdout = tee

    print(f"ログファイル: {log_filename}")
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print(f"\n総実行時間: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分)")

    tee.close()
    print(f"ログを {log_filename} に保存しました。")
