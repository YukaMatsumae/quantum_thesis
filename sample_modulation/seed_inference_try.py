# -*- coding: utf-8 -*-
"""
seed_inference_try.py

目的:
  学習済みモデルと信号条件を「クロス（交差）」させた推論実験。
  - ノイズありモデル × ノイズなし信号
  - ノイズなしモデル × ノイズあり信号
  通常の組み合わせ（モデルと信号が一致）との比較により、
  モデルの汎化性能と量子ノイズの影響を定量的に分析する。
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

# 4つの実験条件: (条件名, モデルディレクトリ, 信号ノイズスケール)
EXPERIMENTS = [
    # 通常（一致）条件
    ('モデル:ノイズあり → 信号:ノイズあり', 'saved_models_more',          0.5),
    ('モデル:ノイズなし → 信号:ノイズなし', 'saved_models_more_no_noise', 0.0),
    # クロス（不一致）条件
    ('モデル:ノイズあり → 信号:ノイズなし', 'saved_models_more',          0.0),
    ('モデル:ノイズなし → 信号:ノイズあり', 'saved_models_more_no_noise', 0.5),
]


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

def load_and_infer(model_path, lfsr_len, device, noise_scale):
    """モデルをロードし、指定ノイズ条件のデータで推論を実行。"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['model_config']
    model = SeedPredictorEnhanced(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

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

def plot_cross_comparison(all_results, lfsr_lengths):
    """4条件のクロス比較グラフを描画する。"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # 条件ごとの色・マーカー設定
    styles = {
        'モデル:ノイズあり → 信号:ノイズあり': {'color': 'tab:blue',   'marker': 'o', 'ls': '-'},
        'モデル:ノイズなし → 信号:ノイズなし': {'color': 'tab:orange', 'marker': 's', 'ls': '-'},
        'モデル:ノイズあり → 信号:ノイズなし': {'color': 'tab:green',  'marker': '^', 'ls': '--'},
        'モデル:ノイズなし → 信号:ノイズあり': {'color': 'tab:red',    'marker': 'D', 'ls': '--'},
    }

    # ── 左上: BER比較 ──
    ax = axes[0, 0]
    for cond, results in all_results.items():
        s = styles[cond]
        lens = [r['lfsr_len'] for r in results]
        bers = [r['ber'] for r in results]
        # 凡例を短縮表示
        short = cond.replace('モデル:', 'M:').replace('信号:', 'S:').replace('ノイズ', 'N')
        ax.plot(lens, bers, marker=s['marker'], linestyle=s['ls'], linewidth=2,
                color=s['color'], label=short, markersize=8)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.6, label='ランダム (0.5)')
    ax.set_xlabel('LFSR Bit Length', fontsize=12)
    ax.set_ylabel('BER', fontsize=12)
    ax.set_title('シード推論BER — クロス条件比較', fontsize=14)
    ax.set_xticks(lfsr_lengths)
    ax.set_ylim(-0.05, 0.55)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=9)

    # ── 右上: 完全一致率比較 ──
    ax = axes[0, 1]
    for cond, results in all_results.items():
        s = styles[cond]
        lens = [r['lfsr_len'] for r in results]
        exact = [r['exact_match'] for r in results]
        short = cond.replace('モデル:', 'M:').replace('信号:', 'S:').replace('ノイズ', 'N')
        ax.plot(lens, exact, marker=s['marker'], linestyle=s['ls'], linewidth=2,
                color=s['color'], label=short, markersize=8)
    ax.set_xlabel('LFSR Bit Length', fontsize=12)
    ax.set_ylabel('Exact Match Rate', fontsize=12)
    ax.set_title('シード完全一致率 — クロス条件比較', fontsize=14)
    ax.set_xticks(lfsr_lengths)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=9)

    # ── 左下: 条件別BERの棒グラフ（LFSR長ごとにグループ化）──
    ax = axes[1, 0]
    n_cond = len(all_results)
    bar_width = 0.8 / n_cond
    cond_list = list(all_results.keys())
    for j, cond in enumerate(cond_list):
        results = all_results[cond]
        s = styles[cond]
        x_pos = np.arange(len(lfsr_lengths)) + j * bar_width
        bers = [r['ber'] for r in results]
        short = cond.replace('モデル:', 'M:').replace('信号:', 'S:').replace('ノイズ', 'N')
        ax.bar(x_pos, bers, width=bar_width, color=s['color'], alpha=0.75,
               label=short, edgecolor='black', linewidth=0.5)
    ax.set_xticks(np.arange(len(lfsr_lengths)) + bar_width * (n_cond - 1) / 2)
    ax.set_xticklabels([f'{l}bit' for l in lfsr_lengths])
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.6)
    ax.set_ylabel('BER', fontsize=12)
    ax.set_title('条件別BER（棒グラフ）', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.7, axis='y')
    ax.legend(fontsize=8)

    # ── 右下: クロス条件でのBER劣化量 ──
    ax = axes[1, 1]
    # 通常条件をベースラインとしたクロス条件の劣化量
    matched_noisy = all_results.get('モデル:ノイズあり → 信号:ノイズあり', [])
    cross_noisy_to_clean = all_results.get('モデル:ノイズあり → 信号:ノイズなし', [])
    matched_clean = all_results.get('モデル:ノイズなし → 信号:ノイズなし', [])
    cross_clean_to_noisy = all_results.get('モデル:ノイズなし → 信号:ノイズあり', [])

    x = np.arange(len(lfsr_lengths))
    w = 0.35
    if matched_noisy and cross_noisy_to_clean:
        delta1 = [c['ber'] - m['ber'] for m, c in zip(matched_noisy, cross_noisy_to_clean)]
        ax.bar(x - w/2, delta1, w, color='tab:green', alpha=0.75,
               label='Nありモデル: Nあり→Nなし信号', edgecolor='black', linewidth=0.5)
    if matched_clean and cross_clean_to_noisy:
        delta2 = [c['ber'] - m['ber'] for m, c in zip(matched_clean, cross_clean_to_noisy)]
        ax.bar(x + w/2, delta2, w, color='tab:red', alpha=0.75,
               label='Nなしモデル: Nなし→Nあり信号', edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}bit' for l in lfsr_lengths])
    ax.set_ylabel('ΔBER（クロス − 通常）', fontsize=12)
    ax.set_title('条件不一致によるBER劣化量', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.7, axis='y')
    ax.legend(fontsize=9)

    fig.suptitle('Y00 シード推論攻撃 — クロス条件推論実験\n'
                 '（モデル学習条件 ≠ 推論信号条件）',
                 fontsize=16, y=1.02)
    fig.tight_layout()
    out = 'seed_inference_cross_comparison.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"\n比較グラフを {out} として保存しました。")


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

    print("=" * 65)
    print("  Y00 シード推論攻撃 — クロス条件推論実験")
    print("=" * 65)
    print(f"実行デバイス: {device}")
    print(f"推論サンプル数: {N_INFERENCE:,}")
    print(f"検証LFSR長: {lfsr_lengths}")
    print()
    print("実験条件:")
    for i, (name, mdir, ns) in enumerate(EXPERIMENTS, 1):
        tag = "【通常】" if i <= 2 else "【クロス】"
        print(f"  {i}. {tag} {name}")
        print(f"     モデル: {mdir}/  |  信号ノイズ: σ={ns}")
    print()

    all_results = {}

    for exp_name, model_dir, noise_scale in EXPERIMENTS:
        print("─" * 65)
        print(f"  {exp_name}")
        print(f"  モデル: {model_dir}/  |  信号ノイズ: σ={noise_scale}")
        print("─" * 65)

        if not os.path.isdir(model_dir):
            print(f"  ⚠ ディレクトリが見つかりません。スキップ。\n")
            continue

        condition_results = []

        for l_len in lfsr_lengths:
            model_path = os.path.join(model_dir, f'seed_predictor_{l_len}bit.pth')
            if not os.path.exists(model_path):
                print(f"\n  LFSR {l_len}bit: モデルなし。スキップ。")
                continue

            print(f"\n  ── LFSR {l_len}bit (周期: {(1 << l_len) - 1}) ──")
            result = load_and_infer(model_path, l_len, device, noise_scale)
            result['lfsr_len'] = l_len

            print(f"  BER:           {result['ber']:.4f}")
            print(f"  完全一致率:    {result['exact_match']:.4f} "
                  f"({int(result['exact_match'] * N_INFERENCE)}/{N_INFERENCE})")
            print(f"  平均ハミング距離: {np.mean(result['hamming_dists']):.2f} / {l_len}")

            # サンプル5件表示
            print(f"  【サンプル5件】")
            for idx in range(min(5, N_INFERENCE)):
                ts = result['true_seeds'][idx]
                ps = result['pred_seeds'][idx]
                m = "✓" if ts == ps else "✗"
                print(f"    {idx+1}: 真={ts:>{l_len+2}} ({format(ts, f'0{l_len}b')}) "
                      f"→ 予={ps:>{l_len+2}} ({format(ps, f'0{l_len}b')}) {m}")

            condition_results.append(result)

        all_results[exp_name] = condition_results
        print()

    # ── 全条件 比較サマリー ──
    print("\n" + "=" * 75)
    print("  全条件 比較サマリー")
    print("=" * 75)
    print(f"{'条件':^40} | {'LFSR':>5} | {'BER':>7} | {'一致率':>7} | {'平均HD':>7}")
    print("-" * 75)
    for cond, results in all_results.items():
        short = cond.replace('モデル:', 'M:').replace('信号:', 'S:').replace('ノイズ', 'N')
        for r in results:
            print(f"{short:^40} | {r['lfsr_len']:>3}bit | "
                  f"{r['ber']:>7.4f} | {r['exact_match']:>7.4f} | "
                  f"{np.mean(r['hamming_dists']):>7.2f}")

    # ── クロス条件の影響度分析 ──
    print(f"\n{'=' * 75}")
    print("  クロス条件の影響度分析")
    print(f"{'=' * 75}")

    pairs = [
        ('モデル:ノイズあり → 信号:ノイズあり', 'モデル:ノイズあり → 信号:ノイズなし',
         'ノイズありモデルにノイズなし信号を入力'),
        ('モデル:ノイズなし → 信号:ノイズなし', 'モデル:ノイズなし → 信号:ノイズあり',
         'ノイズなしモデルにノイズあり信号を入力'),
    ]

    for base_key, cross_key, desc in pairs:
        if base_key not in all_results or cross_key not in all_results:
            continue
        base = all_results[base_key]
        cross = all_results[cross_key]
        print(f"\n  【{desc}】")
        print(f"  {'LFSR':>5} | {'BER(通常)':>10} | {'BER(クロス)':>11} | "
              f"{'ΔBER':>8} | {'影響度':>6}")
        print(f"  " + "-" * 55)
        for br, cr in zip(base, cross):
            delta = cr['ber'] - br['ber']
            if delta > 0.1:
                impact = "大↑"
            elif delta > 0.02:
                impact = "中↑"
            elif delta < -0.1:
                impact = "大↓"
            elif delta < -0.02:
                impact = "中↓"
            else:
                impact = "小"
            print(f"  {br['lfsr_len']:>3}bit | {br['ber']:>10.4f} | "
                  f"{cr['ber']:>11.4f} | {delta:>+8.4f} | {impact:>6}")

    # ── グラフ描画 ──
    valid = {k: v for k, v in all_results.items() if len(v) > 0}
    if valid:
        print("\n比較グラフを描画中...")
        plot_cross_comparison(valid, lfsr_lengths)


if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'seed_inference_try_log_{timestamp}.txt'
    tee = TeeLogger(log_filename)
    sys.stdout = tee

    print(f"ログファイル: {log_filename}")
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print(f"\n総実行時間: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分)")

    tee.close()
    print(f"ログを {log_filename} に保存しました。")
