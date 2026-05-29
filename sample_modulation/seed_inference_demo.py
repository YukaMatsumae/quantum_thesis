# -*- coding: utf-8 -*-
"""
seed_inference_demo.py

目的:
  学習済みモデル（.pthファイル）を読み込んで推論を行うデモンストレーション。
  ノイズあり（saved_models_more/）とノイズなし（saved_models_more_no_noise/）の
  両条件を1つのプログラムで比較解析し、以下を出力する:
    - 各条件・各LFSR長でのBER、完全一致率、平均ハミング距離
    - サンプル推論結果（真のシード vs 予測シード）
    - ノイズあり/なし比較グラフ（4パネル）
    - 全出力をtxtファイルにも保存
"""

import os
import sys
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

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

# 推論に使用するデータ数と系列長（CONFIGから系列長を参照）
SEQ_LEN_MAP = {4: 64, 6: 80, 8: 128, 10: 256}
N_INFERENCE = 5_000  # 推論に使うサンプル数

# モデル保存ディレクトリ
MODEL_DIRS = {
    'ノイズあり (σ=0.5)': 'saved_models_more',
    'ノイズなし (σ=0.0)': 'saved_models_more_no_noise',
}

# 推論時に使用するノイズスケール
NOISE_SCALES = {
    'ノイズあり (σ=0.2)': 0.5,
    'ノイズなし (σ=0.0)': 0.0,
}


# ══════════════════════════════════════════════
# モデル定義（seed_claude_more.py と同一構造）
# ══════════════════════════════════════════════

class SeedPredictorEnhanced(nn.Module):
    """
    強化版シード値推論モデル（推論専用として再定義）。
    学習済み重みをロードするため、構造は seed_claude_more.py と完全に一致させる。
    """
    def __init__(self, out_dim, d_model=128, lstm_hidden=128, nhead=4,
                 lstm_layers=2, dropout=0.15):
        super().__init__()
        self.d_model = d_model
        ch1 = d_model // 4
        ch2 = d_model // 2

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

        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden * 2
        self.attn1 = nn.MultiheadAttention(
            embed_dim=lstm_out_dim, num_heads=nhead,
            dropout=dropout, batch_first=True)
        self.attn_norm1 = nn.LayerNorm(lstm_out_dim)
        self.attn2 = nn.MultiheadAttention(
            embed_dim=lstm_out_dim, num_heads=nhead,
            dropout=dropout, batch_first=True)
        self.attn_norm2 = nn.LayerNorm(lstm_out_dim)

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
        a1, _ = self.attn1(h, h, h)
        h = self.attn_norm1(h + a1)
        a2, _ = self.attn2(h, h, h)
        h = self.attn_norm2(h + a2)
        return self.fc_out(h[:, -1, :])


# ══════════════════════════════════════════════
# データ生成（推論用）
# ══════════════════════════════════════════════

def generate_inference_data(n_samples, lfsr_length, seq_len, noise_scale):
    """推論用のテストデータを生成する。"""
    N = 12
    BNum = 2 ** N
    S_max = 10.0
    S_levels = np.linspace(0, S_max, BNum * 2)
    taps = LFSR_TAPS[lfsr_length]
    mask = (1 << lfsr_length) - 1
    rng = np.random.default_rng(seed=42)  # 再現性のためシード固定
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
# モデルロード & 推論
# ══════════════════════════════════════════════

def load_and_infer(model_path, lfsr_len, device, noise_scale):
    """
    .pthファイルからモデルをロードし、新規テストデータで推論を行う。
    返り値: BER, 完全一致率, ハミング距離配列, 真シード配列, 予測シード配列
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['model_config']

    # モデルの再構築と重みロード
    model = SeedPredictorEnhanced(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # テストデータの生成
    seq_len = SEQ_LEN_MAP[lfsr_len]
    X, Y, seeds = generate_inference_data(N_INFERENCE, lfsr_len, seq_len, noise_scale)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    # 推論実行
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            preds = model(bx)
            all_preds.append((preds >= 0.5).int().cpu().numpy())
            all_targets.append(by.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 指標計算
    ber = 1.0 - np.mean(all_preds == all_targets)
    exact_matches = np.all(all_preds == all_targets, axis=1)
    exact_match_rate = np.mean(exact_matches)
    hamming_dists = np.sum(all_preds != all_targets, axis=1)

    # 予測シード値を整数に変換
    pred_seeds = np.zeros(len(all_preds), dtype=np.int64)
    for i in range(lfsr_len):
        pred_seeds += all_preds[:, lfsr_len - 1 - i].astype(np.int64) << i

    return {
        'ber': ber,
        'exact_match': exact_match_rate,
        'hamming_dists': hamming_dists,
        'true_seeds': seeds,
        'pred_seeds': pred_seeds,
        'train_ber': checkpoint.get('ber', None),
        'train_exact': checkpoint.get('exact_match', None),
    }


# ══════════════════════════════════════════════
# 比較グラフ描画
# ══════════════════════════════════════════════

def plot_comparison(all_results, lfsr_lengths):
    """ノイズあり/なしの比較グラフ（4パネル）を生成・保存する。"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    conditions = list(all_results.keys())
    colors = {'ノイズあり (σ=0.5)': 'tab:blue', 'ノイズなし (σ=0.0)': 'tab:orange'}
    markers = {'ノイズあり (σ=0.5)': 'o', 'ノイズなし (σ=0.0)': 's'}

    # ── 左上: BER比較 ──
    ax = axes[0, 0]
    for cond in conditions:
        results = all_results[cond]
        lens = [r['lfsr_len'] for r in results]
        bers = [r['ber'] for r in results]
        ax.plot(lens, bers, marker=markers[cond], linewidth=2,
                color=colors[cond], label=cond, markersize=8)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='ランダム推測 (0.5)')
    ax.set_xlabel('LFSR Bit Length', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title('シード推論BER比較', fontsize=14)
    ax.set_xticks(lfsr_lengths)
    ax.set_ylim(-0.05, 0.55)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=10)

    # ── 右上: 完全一致率比較 ──
    ax = axes[0, 1]
    for cond in conditions:
        results = all_results[cond]
        lens = [r['lfsr_len'] for r in results]
        exact = [r['exact_match'] for r in results]
        ax.plot(lens, exact, marker=markers[cond], linewidth=2,
                color=colors[cond], label=cond, markersize=8)
    ax.set_xlabel('LFSR Bit Length', fontsize=12)
    ax.set_ylabel('Exact Match Rate', fontsize=12)
    ax.set_title('シード完全一致率比較', fontsize=14)
    ax.set_xticks(lfsr_lengths)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=10)

    # ── 左下: ハミング距離の箱ひげ図（ノイズあり vs なし 並列）──
    ax = axes[1, 0]
    positions = []
    box_data = []
    tick_labels = []
    for i, l_len in enumerate(lfsr_lengths):
        for j, cond in enumerate(conditions):
            results = all_results[cond]
            matching = [r for r in results if r['lfsr_len'] == l_len]
            if matching:
                box_data.append(matching[0]['hamming_dists'])
                positions.append(i * 3 + j)
                tick_labels.append(f"{l_len}bit\n{cond.split('(')[1].rstrip(')')}")

    bp = ax.boxplot(box_data, positions=positions, widths=0.8, patch_artist=True)
    color_cycle = [colors[c] for l in lfsr_lengths for c in conditions
                   if any(r['lfsr_len'] == l for r in all_results[c])]
    for patch, c in zip(bp['boxes'], color_cycle):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_xticks([i * 3 + 0.5 for i in range(len(lfsr_lengths))])
    ax.set_xticklabels([f'{l}bit' for l in lfsr_lengths])
    ax.set_ylabel('Hamming Distance (bits)', fontsize=12)
    ax.set_title('ハミング距離分布（ノイズあり vs なし）', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.7, axis='y')

    # ── 右下: BERの差分（ノイズの影響度）──
    ax = axes[1, 1]
    if len(conditions) == 2:
        noisy_results = all_results[conditions[0]]
        clean_results = all_results[conditions[1]]
        lens = [r['lfsr_len'] for r in noisy_results]
        ber_diff = []
        for nr, cr in zip(noisy_results, clean_results):
            ber_diff.append(nr['ber'] - cr['ber'])
        ax.bar(lens, ber_diff, width=1.2, color='tab:purple', alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('LFSR Bit Length', fontsize=12)
        ax.set_ylabel('ΔBER (ノイズあり − なし)', fontsize=12)
        ax.set_title('量子ノイズによるBER増加量', fontsize=14)
        ax.set_xticks(lfsr_lengths)
        ax.grid(True, linestyle=':', alpha=0.7, axis='y')

    fig.suptitle('Y00 シード推論攻撃 — 学習済みモデルによる推論デモ\n'
                 '（ノイズあり vs ノイズなし比較）',
                 fontsize=16, y=1.02)
    fig.tight_layout()
    output_file = 'seed_inference_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n比較グラフを {output_file} として保存しました。")


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
    lfsr_lengths = [4, 6, 8, 10]

    print("=" * 60)
    print("  Y00 シード推論攻撃 — 学習済みモデル推論デモ")
    print("=" * 60)
    print(f"実行デバイス: {device}")
    print(f"推論サンプル数: {N_INFERENCE:,}")
    print(f"検証LFSR長: {lfsr_lengths}")
    print()

    all_results = {}

    for cond_name, model_dir in MODEL_DIRS.items():
        noise_scale = NOISE_SCALES[cond_name]
        print("=" * 60)
        print(f"  条件: {cond_name}")
        print(f"  モデルディレクトリ: {model_dir}/")
        print("=" * 60)

        if not os.path.isdir(model_dir):
            print(f"  ⚠ ディレクトリが見つかりません。スキップします。\n")
            continue

        condition_results = []

        for l_len in lfsr_lengths:
            model_path = os.path.join(model_dir, f'seed_predictor_{l_len}bit.pth')
            if not os.path.exists(model_path):
                print(f"\n  LFSR {l_len}bit: モデルファイルが見つかりません。スキップ。")
                continue

            print(f"\n  ─── LFSR {l_len}bit (周期: {(1 << l_len) - 1}) ───")
            print(f"  モデル: {model_path}")

            result = load_and_infer(model_path, l_len, device, noise_scale)
            result['lfsr_len'] = l_len

            print(f"  推論BER:       {result['ber']:.4f}")
            print(f"  完全一致率:    {result['exact_match']:.4f} "
                  f"({int(result['exact_match'] * N_INFERENCE)}/{N_INFERENCE})")
            print(f"  平均ハミング距離: {np.mean(result['hamming_dists']):.2f} / {l_len}")

            if result['train_ber'] is not None:
                print(f"  (参考) 学習時BER: {result['train_ber']:.4f}, "
                      f"学習時完全一致率: {result['train_exact']:.4f}")

            # サンプル推論結果を10件表示
            print(f"\n  【サンプル推論結果（先頭10件）】")
            print(f"  {'No':>4} | {'真のシード':>12} | {'予測シード':>12} | "
                  f"{'一致':>4} | {'ハミング距離':>10}")
            print(f"  " + "-" * 56)
            for idx in range(min(10, N_INFERENCE)):
                true_s = result['true_seeds'][idx]
                pred_s = result['pred_seeds'][idx]
                match = "✓" if true_s == pred_s else "✗"
                hdist = result['hamming_dists'][idx]
                true_bin = format(true_s, f'0{l_len}b')
                pred_bin = format(pred_s, f'0{l_len}b')
                print(f"  {idx+1:>4} | {true_s:>5} ({true_bin}) | "
                      f"{pred_s:>5} ({pred_bin}) | {match:>4} | {hdist:>10}")

            condition_results.append(result)

        all_results[cond_name] = condition_results
        print()

    # ── 全条件の比較サマリー ──
    print("\n" + "=" * 70)
    print("  全条件 比較サマリー")
    print("=" * 70)
    print(f"{'条件':^20} | {'LFSR長':>6} | {'BER':>8} | "
          f"{'完全一致率':>10} | {'平均HD':>8}")
    print("-" * 70)
    for cond_name, results in all_results.items():
        for r in results:
            cond_short = "ノイズあり" if "0.5" in cond_name else "ノイズなし"
            print(f"{cond_short:^20} | {r['lfsr_len']:>4}bit | "
                  f"{r['ber']:>8.4f} | {r['exact_match']:>10.4f} | "
                  f"{np.mean(r['hamming_dists']):>8.2f}")

    # ── ノイズの影響度分析 ──
    if len(all_results) == 2:
        cond_keys = list(all_results.keys())
        noisy = all_results[cond_keys[0]]
        clean = all_results[cond_keys[1]]
        print(f"\n{'─' * 70}")
        print("  量子ノイズの影響度分析")
        print(f"{'─' * 70}")
        print(f"{'LFSR長':>8} | {'BER(ノイズあり)':>14} | {'BER(ノイズなし)':>14} | "
              f"{'ΔBER':>8} | {'ノイズ影響':>10}")
        print("-" * 70)
        for nr, cr in zip(noisy, clean):
            delta = nr['ber'] - cr['ber']
            impact = "大" if delta > 0.1 else "中" if delta > 0.02 else "小"
            print(f"{nr['lfsr_len']:>6}bit | {nr['ber']:>14.4f} | "
                  f"{cr['ber']:>14.4f} | {delta:>+8.4f} | {impact:>10}")

    # ── グラフ描画 ──
    valid_results = {k: v for k, v in all_results.items() if len(v) > 0}
    if valid_results:
        print("\n比較グラフを描画中...")
        plot_comparison(valid_results, lfsr_lengths)


if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'seed_inference_demo_log_{timestamp}.txt'
    tee = TeeLogger(log_filename)
    sys.stdout = tee

    print(f"ログファイル: {log_filename}")
    import time
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print(f"\n総実行時間: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分)")

    tee.close()
    print(f"ログを {log_filename} に保存しました。")
