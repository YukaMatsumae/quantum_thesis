# -*- coding: utf-8 -*-
"""
seed_decoder_no_noise.py

目的:
  seed_claude_no_noise.py で保存された実験データ（YAML + npz）を読み込み、
  推定シード値を使って通信データを「解読」し、通信データの解読BERを計算する。

  ※ 本スクリプトは量子ノイズなしバージョンのデータを対象とする。

  検証項目:
  1. 正解seedで解読した場合のBER（ノイズなしの場合は理想的に0）
  2. 推定seedで解読した場合のBER
  3. seed完全一致/不一致サンプル別の解読BER
"""

import numpy as np
import yaml
import matplotlib.pyplot as plt
import os

# ══════════════════════════════════════════════
# LFSR drive & decoding logic
# ══════════════════════════════════════════════

def run_lfsr(seed, lfsr_length, taps, n_bits):
    """
    Drive the LFSR n_bits times from the specified seed value
    and return the output bit sequence.

    Args:
        seed: LFSR initial state (integer)
        lfsr_length: Bit length of the LFSR
        taps: List of feedback tap positions
        n_bits: Number of bits to output

    Returns:
        bits: Output bit sequence (numpy array, shape=(n_bits,))
    """
    mask = (1 << lfsr_length) - 1
    reg = int(seed)
    bits = np.empty(n_bits, dtype=np.uint8)
    for k in range(n_bits):
        # Feedback calculation (XOR of tap positions)
        sr = 0
        for t in taps:
            sr ^= (reg >> t)
        sr &= 1
        # Output MSB
        bits[k] = (reg >> (lfsr_length - 1)) & 1
        # Shift register left + insert feedback
        reg = ((reg << 1) & mask) | sr
    return bits


def decode_with_seed(seed, lfsr_length, taps, observations, N, S_levels, BNum):
    """
    Drive the LFSR with the specified seed value and decode
    the transmitted bits from the observation sequence.

    Steps:
      1. Generate key stream by driving LFSR from seed
      2. Group N bits to calculate basis ID
      3. Compute two signal levels for bit=0 / bit=1 for each symbol
      4. Select the signal level closest to the observation
         and determine the transmitted bit (nearest-neighbor decision)

    Args:
        seed: Seed value to use (integer)
        lfsr_length: Bit length of the LFSR
        taps: Tap positions
        observations: Observation sequence (raw data before normalization,
                      shape=(seq_len,))
        N: Number of bits for basis determination
        S_levels: Signal level array
        BNum: Number of bases (2^N)

    Returns:
        decoded_bits: Decoded bit sequence (shape=(seq_len,))
    """
    seq_len = len(observations)
    total_bits = seq_len * N

    # Drive LFSR to generate bit sequence
    lfsr_bits = run_lfsr(seed, lfsr_length, taps, total_bits)

    # Group N bits to calculate basis ID
    weights = (1 << np.arange(N)).astype(np.int64)
    bits_reshaped = lfsr_bits.reshape(seq_len, N)
    base_ids = bits_reshaped.dot(weights)

    # Compute signal levels for bit=0 and bit=1 for each symbol
    mod_idx_0 = (0 + base_ids % 2) % 2  # modulation index for bit=0
    mod_idx_1 = (1 + base_ids % 2) % 2  # modulation index for bit=1
    level_0 = S_levels[base_ids + BNum * mod_idx_0]  # signal level for bit=0
    level_1 = S_levels[base_ids + BNum * mod_idx_1]  # signal level for bit=1

    # Nearest-neighbor decision: select the bit closer to the observation
    dist_0 = np.abs(observations - level_0)
    dist_1 = np.abs(observations - level_1)
    decoded_bits = (dist_1 < dist_0).astype(np.int64)

    return decoded_bits


# ══════════════════════════════════════════════
# Main processing
# ══════════════════════════════════════════════

def main():
    """
    Load experiment_params.yaml and each npz file,
    and verify communication data decoding BER using estimated seeds.
    """
    yaml_path = 'experiment_params_no_noise.yaml'
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found.")
        print("先に seed_claude_no_noise.py を実行してデータを生成してください。")
        return

    # Load YAML
    with open(yaml_path, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)

    noise_scale = params['quantum_noise_scale']
    N = params['N']
    S_max = params['S_max']
    BNum = params['BNum']
    S_levels = np.linspace(0, S_max, BNum * 2)

    print("=" * 60)
    print(" Y00通信データ解読検証（seed_decoder_no_noise.py）")
    print("=" * 60)
    print(f"  Quantum noise scale: {noise_scale}")
    print(f"  Basis determination bits N: {N}")
    print(f"  Number of signal levels: {BNum * 2}")
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
        print(f" LFSR length: {lfsr_len} bit | Period: {(1 << lfsr_len) - 1} | "
              f"Verification samples: {n_verify}")
        print(f"{'─' * 50}")

        # Load npz file
        if not os.path.exists(data_file):
            print(f"  Warning: {data_file} not found. Skipping.")
            continue

        data = np.load(data_file)
        input_data = data['input_data']    # Ground-truth transmitted bit sequence
        observations = data['observations']  # Observation sequence (before normalization)

        # Calculate decoding BER for each sample
        ber_true_list = []    # BER when decoding with correct seed
        ber_pred_list = []    # BER when decoding with estimated seed
        seed_match_list = []  # Whether seed is a perfect match

        for i in range(n_verify):
            true_seed = true_seeds[i]
            pred_seed = pred_seeds[i]
            obs = observations[i]
            true_bits = input_data[i]

            # Decode with correct seed (verify quantum noise impact only)
            decoded_true = decode_with_seed(
                true_seed, lfsr_len, taps, obs, N, S_levels, BNum)
            ber_true = np.mean(decoded_true != true_bits)
            ber_true_list.append(ber_true)

            # Decode with estimated seed
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

        print(f"  Seed exact match: {n_match}/{n_verify} "
              f"({n_match / n_verify * 100:.1f}%)")
        print(f"  Average decoding BER (correct seed): {np.mean(ber_true_arr):.4f}")
        print(f"  Average decoding BER (estimated seed): {np.mean(ber_pred_arr):.4f}")
        if n_match > 0:
            print(f"  [Seed match]    Decoding BER: "
                  f"{np.mean(ber_pred_arr[seed_match_arr]):.4f}")
        if n_mismatch > 0:
            print(f"  [Seed mismatch] Decoding BER: "
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

    # Summary of all results
    if all_results:
        print("=" * 60)
        print(" Summary of All Results")
        print("=" * 60)
        print(f"{'LFSR len':>8} | {'Correct seed BER':>16} | "
              f"{'Estimated seed BER':>18} | {'Seed match rate':>15}")
        print("-" * 65)
        for r in all_results:
            print(f"{r['lfsr_len']:>6} bit | "
                  f"{np.mean(r['ber_true']):>16.4f} | "
                  f"{np.mean(r['ber_pred']):>18.4f} | "
                  f"{r['n_match'] / r['n_verify']:>15.4f}")

        # Plot graphs
        print("\nGenerating and saving graphs...")
        plot_decode_results(all_results, noise_scale)


# ══════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════

def plot_decode_results(all_results, noise_scale):
    """
    Visualize decoding BER results in three subplots.
    1. Decoding BER comparison: correct seed vs estimated seed
    2. Decoding BER by seed match / mismatch
    3. Seed exact match rate
    """
    lfsr_lengths = [r['lfsr_len'] for r in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x = np.arange(len(lfsr_lengths))
    width = 0.35

    # ── Left: Decoding BER comparison (correct seed vs estimated seed) ──
    ax1 = axes[0]
    avg_ber_true = [np.mean(r['ber_true']) for r in all_results]
    avg_ber_pred = [np.mean(r['ber_pred']) for r in all_results]
    ax1.bar(x - width / 2, avg_ber_true, width,
            label='Correct seed', color='tab:green', alpha=0.8)
    ax1.bar(x + width / 2, avg_ber_pred, width,
            label='Estimated seed', color='tab:red', alpha=0.8)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7,
                label='Random guess (0.5)')
    ax1.set_xlabel('LFSR Bit Length', fontsize=12)
    ax1.set_ylabel('Decoding BER', fontsize=12)
    ax1.set_title('Decoding BER: Correct Seed vs Estimated Seed', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(l) for l in lfsr_lengths])
    ax1.set_ylim(0, 0.6)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)

    # ── Center: Decoding BER by seed match / mismatch ──
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
            label='Seed match', color='tab:blue', alpha=0.8)
    ax2.bar(x + width / 2, ber_mismatch, width,
            label='Seed mismatch', color='tab:orange', alpha=0.8)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7,
                label='Random guess (0.5)')
    ax2.set_xlabel('LFSR Bit Length', fontsize=12)
    ax2.set_ylabel('Decoding BER', fontsize=12)
    ax2.set_title('Decoding BER by Seed Match / Mismatch', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(l) for l in lfsr_lengths])
    ax2.set_ylim(0, 0.6)
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.7)

    # ── Right: Seed exact match rate ──
    ax3 = axes[2]
    match_rates = [r['n_match'] / r['n_verify'] for r in all_results]
    colors = plt.cm.RdYlGn(match_rates)  # Green = higher match rate
    ax3.bar(x, match_rates, color=colors, alpha=0.85, edgecolor='gray')
    ax3.set_xlabel('LFSR Bit Length', fontsize=12)
    ax3.set_ylabel('Seed Exact Match Rate', fontsize=12)
    ax3.set_title('Seed Exact Match Rate vs LFSR Length', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(l) for l in lfsr_lengths])
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, linestyle=':', alpha=0.7)

    fig.suptitle(
        f'Y00通信データ解読検証 — 推定seedによる復号性能\n'
        f'(量子ノイズなし)',
        fontsize=16, y=1.02)
    fig.tight_layout()
    output_filename = 'decode_verification_results_no_noise.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Graph saved as {output_filename}.")


if __name__ == "__main__":
    main()
