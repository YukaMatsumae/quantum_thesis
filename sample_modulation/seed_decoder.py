# -*- coding: utf-8 -*-
"""
seed_decoder.py

目的:
  seed_claude.py で保存された実験データ（YAML + npz）を読み込み、
  推定シード値を使って通信データを「解読」し、通信データの解読BERを計算する。

  検証項目:
  1. 正解seedで解読した場合のBER（量子ノイズの影響のみ）
  2. 推定seedで解読した場合のBER
  3. seed完全一致/不一致サンプル別の解読BER
"""

import numpy as np
import yaml
import matplotlib.pyplot as plt
import os
import japanize_matplotlib

# ══════════════════════════════════════════════
# LFSR駆動 & 解読ロジック
# ══════════════════════════════════════════════

def run_lfsr(seed, lfsr_length, taps, n_bits):
    """
    指定されたseed値からLFSRをn_bits回駆動し、出力ビット列を返す。

    引数:
        seed: LFSR初期状態（整数）
        lfsr_length: LFSRのビット長
        taps: フィードバックタップ位置のリスト
        n_bits: 出力するビット数

    戻り値:
        bits: 出力ビット列 (numpy配列, shape=(n_bits,))
    """
    mask = (1 << lfsr_length) - 1
    reg = int(seed)
    bits = np.empty(n_bits, dtype=np.uint8)
    for k in range(n_bits):
        # フィードバック計算（タップ位置のXOR）
        sr = 0
        for t in taps:
            sr ^= (reg >> t)
        sr &= 1
        # MSBを出力
        bits[k] = (reg >> (lfsr_length - 1)) & 1
        # レジスタを左シフト＋フィードバック挿入
        reg = ((reg << 1) & mask) | sr
    return bits


def decode_with_seed(seed, lfsr_length, taps, observations, N, S_levels, BNum):
    """
    指定されたseed値を使ってLFSRを回し、観測系列から送信ビットを解読する。

    手順:
      1. seedからLFSRを駆動して鍵ストリームを生成
      2. Nビットずつまとめて基底IDを算出
      3. 各シンボルについて、bit=0 / bit=1 に対応する2つの信号レベルを計算
      4. 観測値に近い方の信号レベルを選び、送信ビットを判定（最近傍判定）

    引数:
        seed: 使用するseed値（整数）
        lfsr_length: LFSRのビット長
        taps: タップ位置
        observations: 観測系列（標準化前の生データ, shape=(seq_len,)）
        N: 基底決定ビット数
        S_levels: 信号レベル配列
        BNum: 基底数 (2^N)

    戻り値:
        decoded_bits: 解読結果のビット列 (shape=(seq_len,))
    """
    seq_len = len(observations)
    total_bits = seq_len * N

    # LFSRを回してビット列を生成
    lfsr_bits = run_lfsr(seed, lfsr_length, taps, total_bits)

    # Nビットずつまとめて基底IDを算出
    weights = (1 << np.arange(N)).astype(np.int64)
    bits_reshaped = lfsr_bits.reshape(seq_len, N)
    base_ids = bits_reshaped.dot(weights)

    # 各シンボルについて、bit=0 と bit=1 の信号レベルを一括計算
    mod_idx_0 = (0 + base_ids % 2) % 2  # bit=0の場合の変調インデックス
    mod_idx_1 = (1 + base_ids % 2) % 2  # bit=1の場合の変調インデックス
    level_0 = S_levels[base_ids + BNum * mod_idx_0]  # bit=0の信号レベル
    level_1 = S_levels[base_ids + BNum * mod_idx_1]  # bit=1の信号レベル

    # 最近傍判定: 観測値に近い方のビットを選択
    dist_0 = np.abs(observations - level_0)
    dist_1 = np.abs(observations - level_1)
    decoded_bits = (dist_1 < dist_0).astype(np.int64)

    return decoded_bits


# ══════════════════════════════════════════════
# メイン処理
# ══════════════════════════════════════════════

def main():
    """
    experiment_params.yaml と各npzファイルを読み込み、
    推定seedによる通信データ解読BERを検証する。
    """
    yaml_path = 'experiment_params.yaml'
    if not os.path.exists(yaml_path):
        print(f"エラー: {yaml_path} が見つかりません。")
        print("先に seed_claude.py を実行してデータを生成してください。")
        return

    # YAML読み込み
    with open(yaml_path, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)

    noise_scale = params['quantum_noise_scale']
    N = params['N']
    S_max = params['S_max']
    BNum = params['BNum']
    S_levels = np.linspace(0, S_max, BNum * 2)

    print("=" * 60)
    print(" Y00通信データ解読検証（seed_decoder.py）")
    print("=" * 60)
    print(f"  量子ノイズスケール: {noise_scale}")
    print(f"  基底決定ビット数 N: {N}")
    print(f"  信号レベル数: {BNum * 2}")
    print()

    all_results = []

    for lfsr_len_str, exp_config in sorted(params['experiments'].items(),
                                            key=lambda x: int(x[0])):
        lfsr_len = int(lfsr_len_str)
        taps = exp_config['taps']
        seq_len = exp_config['seq_len']
        n_verify = exp_config['n_verify']
        true_seeds = exp_config['true_seeds']
        pred_seeds = exp_config['predicted_seeds']
        data_file = exp_config['data_file']

        print(f"{'─' * 50}")
        print(f" LFSR長: {lfsr_len} bit | 周期: {(1 << lfsr_len) - 1} | "
              f"検証サンプル数: {n_verify}")
        print(f"{'─' * 50}")

        # npzファイル読み込み
        if not os.path.exists(data_file):
            print(f"  警告: {data_file} が見つかりません。スキップします。")
            continue

        data = np.load(data_file)
        input_data = data['input_data']    # 正解の送信ビット列
        observations = data['observations']  # 標準化前の観測系列

        # 各サンプルについて解読BERを計算
        ber_true_list = []    # 正解seedで解読した場合のBER
        ber_pred_list = []    # 推定seedで解読した場合のBER
        seed_match_list = []  # seedが完全一致かどうか

        for i in range(n_verify):
            true_seed = true_seeds[i]
            pred_seed = pred_seeds[i]
            obs = observations[i]
            true_bits = input_data[i]

            # 正解seedで解読（量子ノイズのみの影響を確認）
            decoded_true = decode_with_seed(
                true_seed, lfsr_len, taps, obs, N, S_levels, BNum)
            ber_true = np.mean(decoded_true != true_bits)
            ber_true_list.append(ber_true)

            # 推定seedで解読
            decoded_pred = decode_with_seed(
                pred_seed, lfsr_len, taps, obs, N, S_levels, BNum)
            ber_pred = np.mean(decoded_pred != true_bits)
            ber_pred_list.append(ber_pred)

            seed_match_list.append(true_seed == pred_seed)

        ber_true_arr = np.array(ber_true_list)
        ber_pred_arr = np.array(ber_pred_list)
        seed_match_arr = np.array(seed_match_list)

        n_match = int(np.sum(seed_match_arr))
        n_mismatch = n_verify - n_match

        print(f"  シード完全一致: {n_match}/{n_verify} "
              f"({n_match / n_verify * 100:.1f}%)")
        print(f"  正解seedでの平均解読BER: {np.mean(ber_true_arr):.4f}")
        print(f"  推定seedでの平均解読BER: {np.mean(ber_pred_arr):.4f}")
        if n_match > 0:
            print(f"  [seed一致]   解読BER: "
                  f"{np.mean(ber_pred_arr[seed_match_arr]):.4f}")
        if n_mismatch > 0:
            print(f"  [seed不一致] 解読BER: "
                  f"{np.mean(ber_pred_arr[~seed_match_arr]):.4f}")
        print()

        all_results.append({
            'lfsr_len': lfsr_len,
            'ber_true': ber_true_arr,
            'ber_pred': ber_pred_arr,
            'seed_match': seed_match_arr,
            'n_match': n_match,
            'n_verify': n_verify,
        })

    # 全結果サマリー
    if all_results:
        print("=" * 60)
        print(" 全結果サマリー")
        print("=" * 60)
        print(f"{'LFSR長':>8} | {'正解seed BER':>12} | "
              f"{'推定seed BER':>12} | {'seed一致率':>10}")
        print("-" * 60)
        for r in all_results:
            print(f"{r['lfsr_len']:>6} bit | "
                  f"{np.mean(r['ber_true']):>12.4f} | "
                  f"{np.mean(r['ber_pred']):>12.4f} | "
                  f"{r['n_match'] / r['n_verify']:>10.4f}")

        # グラフ描画
        print("\nグラフを描画・保存します...")
        plot_decode_results(all_results, noise_scale)


# ══════════════════════════════════════════════
# 可視化
# ══════════════════════════════════════════════

def plot_decode_results(all_results, noise_scale):
    """
    解読BERの結果を3つのサブプロットで可視化する。
    1. 正解seed vs 推定seed の解読BER比較
    2. seed一致/不一致別の解読BER
    3. シード完全一致率
    """
    lfsr_lengths = [r['lfsr_len'] for r in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x = np.arange(len(lfsr_lengths))
    width = 0.35

    # ── 左: 解読BER比較（正解seed vs 推定seed）──
    ax1 = axes[0]
    avg_ber_true = [np.mean(r['ber_true']) for r in all_results]
    avg_ber_pred = [np.mean(r['ber_pred']) for r in all_results]
    ax1.bar(x - width / 2, avg_ber_true, width,
            label='正解seedで解読', color='tab:green', alpha=0.8)
    ax1.bar(x + width / 2, avg_ber_pred, width,
            label='推定seedで解読', color='tab:red', alpha=0.8)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7,
                label='ランダム推測 (0.5)')
    ax1.set_xlabel('LFSR Bit Length', fontsize=12)
    ax1.set_ylabel('通信データ解読BER', fontsize=12)
    ax1.set_title('解読BER: 正解seed vs 推定seed', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(l) for l in lfsr_lengths])
    ax1.set_ylim(0, 0.6)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)

    # ── 中: seed一致/不一致別の解読BER ──
    ax2 = axes[1]
    ber_match = []
    ber_mismatch = []
    for r in all_results:
        if np.sum(r['seed_match']) > 0:
            ber_match.append(np.mean(r['ber_pred'][r['seed_match']]))
        else:
            ber_match.append(np.nan)
        if np.sum(~r['seed_match']) > 0:
            ber_mismatch.append(np.mean(r['ber_pred'][~r['seed_match']]))
        else:
            ber_mismatch.append(np.nan)
    ax2.bar(x - width / 2, ber_match, width,
            label='seed一致', color='tab:blue', alpha=0.8)
    ax2.bar(x + width / 2, ber_mismatch, width,
            label='seed不一致', color='tab:orange', alpha=0.8)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7,
                label='ランダム推測 (0.5)')
    ax2.set_xlabel('LFSR Bit Length', fontsize=12)
    ax2.set_ylabel('通信データ解読BER', fontsize=12)
    ax2.set_title('seed一致/不一致別の解読BER', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(l) for l in lfsr_lengths])
    ax2.set_ylim(0, 0.6)
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.7)

    # ── 右: シード完全一致率 ──
    ax3 = axes[2]
    match_rates = [r['n_match'] / r['n_verify'] for r in all_results]
    colors = plt.cm.RdYlGn(match_rates)  # 一致率が高いほど緑
    ax3.bar(x, match_rates, color=colors, alpha=0.85, edgecolor='gray')
    ax3.set_xlabel('LFSR Bit Length', fontsize=12)
    ax3.set_ylabel('シード完全一致率', fontsize=12)
    ax3.set_title('シード完全一致率 vs LFSR長', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(l) for l in lfsr_lengths])
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, linestyle=':', alpha=0.7)

    fig.suptitle(
        f'Y00通信データ解読検証 — 推定seedによる復号性能\n'
        f'(量子ノイズスケール = {noise_scale})',
        fontsize=16, y=1.02)
    fig.tight_layout()
    output_filename = 'decode_verification_results.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"グラフを {output_filename} として保存しました。")


if __name__ == "__main__":
    main()
