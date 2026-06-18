# -*- coding: utf-8 -*-
"""
seed_claude_train_compare.py

目的:
  ノイズあり(sigma=0.5)で学習したモデルと、ノイズなし(sigma=0.0)で学習したモデルを、
  同じノイズあり(sigma=0.5)のテストデータで評価し、その性能（Seed BER, Data BER）を直接比較する。
  S_max = 10.0 固定。
"""

import os
import sys
import math
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ══════════════════════════════════════════════
# 定数・設定
# ══════════════════════════════════════════════

LFSR_TAPS = {
    4:  [3, 2],
    6:  [5, 4],
    8:  [7, 5, 4, 3],
    10: [9, 6],
}

# 物理定数
TEST_NOISE_SCALE = 0.5      # テスト時に加える量子ノイズ（シグマ）
SMAX_VALUE = 10.0           # 信号レベル最大値（S_max）

# 保存先
MODEL_SAVE_DIR = './saved_models_train_compare'

# ハイパーパラメータ設定 (LFSR長ごと)
# (d_model, lstm_hidden, nhead, lstm_layers, n_train, n_test, seq_len, epochs, batch_size, accum_steps)
CONFIG = {
    4:  (64,  64,  4, 2,  30_000, 5000, 64,  80, 256, 1),
    6:  (64,  64,  4, 2,  40_000, 5000, 80,  100, 256, 1),
    8:  (128, 128, 4, 2,  60_000, 5000, 128, 120, 256, 1),
    10: (256, 256, 4, 2, 100_000, 5000, 256, 150, 256, 2),
}


# ══════════════════════════════════════════════
# ニューラルネットワークモデル定義
# ══════════════════════════════════════════════

class SeedPredictorEnhanced(nn.Module):
    """Conv1D + BiLSTM + self-Attention x2 アーキテクチャ"""
    def __init__(self, out_dim, d_model=128, lstm_hidden=128, nhead=4, lstm_layers=2, dropout=0.1):
        super(SeedPredictorEnhanced, self).__init__()
        ch1 = 32
        ch2 = 64
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
# データ生成及び復号処理
# ══════════════════════════════════════════════

def generate_seed_dataset(n_samples, lfsr_length, seq_len, noise_scale, s_max=10.0, dataset_seed=None):
    """ランダムシードからLFSRを駆動し、量子ノイズ付き観測系列とシードビット列を生成。"""
    N = 12
    BNum = 2 ** N
    S_levels = np.linspace(0, s_max, BNum * 2)
    taps = LFSR_TAPS[lfsr_length]
    mask = (1 << lfsr_length) - 1
    rng = np.random.default_rng(dataset_seed)
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
    return X, Y, seeds, input_data, xs_all


def eve_decode_data(pred_seeds, lfsr_len, seq_len, s_max, test_raw_obs, test_input_data):
    """予測シードからデータビットを復号し、BERを算出する。"""
    n_samples = len(pred_seeds)
    N = 12
    BNum = 2 ** N
    S_levels = np.linspace(0, s_max, BNum * 2)
    taps = LFSR_TAPS[lfsr_len]
    mask = (1 << lfsr_len) - 1

    # LFSRを再駆動して各シンボルのbase_idを復元
    regs = pred_seeds.copy()
    total_bits = seq_len * N
    bits_all = np.empty((n_samples, total_bits), dtype=np.uint8)
    for k in range(total_bits):
        SR = np.zeros(n_samples, dtype=np.int64)
        for t in taps:
            SR ^= (regs >> t)
        SR &= 1
        bits_all[:, k] = (regs >> (lfsr_len - 1)) & 1
        regs = ((regs << 1) & mask) | SR

    weights = (1 << np.arange(N)).astype(np.int64)
    bits_reshaped = bits_all.reshape(n_samples, seq_len, N)
    base_ids = bits_reshaped.dot(weights)

    # 閾値判定によるデータ復号
    mod_for_0 = (0 + base_ids % 2) % 2
    level_for_0 = S_levels[base_ids + BNum * mod_for_0]
    mod_for_1 = (1 + base_ids % 2) % 2
    level_for_1 = S_levels[base_ids + BNum * mod_for_1]

    dist_0 = np.abs(test_raw_obs - level_for_0)
    dist_1 = np.abs(test_raw_obs - level_for_1)
    decoded_data = (dist_1 < dist_0).astype(np.int64)

    errors = (decoded_data != test_input_data)
    total_errors = np.sum(errors)
    total_elements = n_samples * seq_len
    ber = total_errors / total_elements
    errors_per_sample = np.sum(errors, axis=1).tolist()

    return ber, errors_per_sample


# ══════════════════════════════════════════════
# 学習ユーティリティ
# ══════════════════════════════════════════════

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_and_evaluate_scenario(lfsr_len, device, train_noise_scale, scenario_name):
    """特定の学習ノイズスケール（0.0 または 0.5）で学習し、sigma=0.5 のテストデータで評価する。"""
    d_model, lstm_hidden, nhead, lstm_layers, n_train, n_test, \
        seq_len, epochs, batch_size, accum_steps = CONFIG[lfsr_len]

    print(f"\n  [Scenario: {scenario_name}] Training with noise={train_noise_scale}...")

    # ── データ生成 ──
    print("    Generating training dataset (dataset_seed=12345)...")
    X_train, Y_train, _, _, _ = generate_seed_dataset(
        n_train, lfsr_len, seq_len, noise_scale=train_noise_scale, s_max=SMAX_VALUE, dataset_seed=12345)
    
    print(f"    Generating test dataset (sigma={TEST_NOISE_SCALE}, dataset_seed=54321)...")
    X_test, Y_test, test_seeds, test_input_data, test_raw_obs = \
        generate_seed_dataset(n_test, lfsr_len, seq_len, noise_scale=TEST_NOISE_SCALE, s_max=SMAX_VALUE, dataset_seed=54321)

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
    scheduler = get_warmup_cosine_scheduler(optimizer, min(20, epochs // 10), epochs)

    best_val_loss = float('inf')
    best_state = None
    patience = 50
    patience_counter = 0

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
            
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item() * accum_steps

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # 評価
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                preds = model(bx)
                val_loss += criterion(preds, by).item()
        avg_val_loss = val_loss / len(test_loader)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"      Epoch {epoch:>3}/{epochs} | Train Loss: {avg_loss:.5f} | Test Loss: {avg_val_loss:.5f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break

    # ── 最終評価 ──
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

    # 予測シードの復元
    pred_seeds = np.zeros(len(all_preds), dtype=np.int64)
    for i in range(lfsr_len):
        pred_seeds += all_preds[:, lfsr_len - 1 - i].astype(np.int64) << i

    # データ復号の評価
    data_ber, data_errors = eve_decode_data(
        pred_seeds, lfsr_len, seq_len, SMAX_VALUE, test_raw_obs, test_input_data)

    print(f"    => Seed BER: {ber:.4f}")
    print(f"    => Exact Match Rate: {exact_match_rate:.4f} ({np.sum(exact_matches)}/{n_test})")
    print(f"    => Data BER: {data_ber:.6f}")

    return {
        'ber': ber,
        'exact_match': exact_match_rate,
        'data_ber': data_ber,
        'best_state': best_state,
        'model_config': {
            'd_model': d_model,
            'lstm_hidden': lstm_hidden,
            'nhead': nhead,
            'lstm_layers': lstm_layers,
            'out_dim': lfsr_len,
        }
    }


# ══════════════════════════════════════════════
# 結果の可視化
# ══════════════════════════════════════════════

def plot_comparison_results(results, lfsr_lengths):
    """Noisy Training と Noiseless Training の性能を比較する英語の棒グラフを生成。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(lfsr_lengths))
    width = 0.35

    # ── Left: Seed BER Comparison ──
    noisy_seed_bers = [results[l]['noisy']['ber'] for l in lfsr_lengths]
    noiseless_seed_bers = [results[l]['noiseless']['ber'] for l in lfsr_lengths]

    ax1.bar(x - width/2, noisy_seed_bers, width, label='Noisy Train (sigma=0.5)', color='navy', alpha=0.8)
    ax1.bar(x + width/2, noiseless_seed_bers, width, label='Noiseless Train (sigma=0.0)', color='teal', alpha=0.7)
    ax1.set_ylabel('Seed BER', fontsize=12)
    ax1.set_xlabel('LFSR Bit Length', fontsize=12)
    ax1.set_title('Seed BER Comparison (Lower is Better)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{l}bit' for l in lfsr_lengths])
    ax1.set_ylim(-0.02, 0.55)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Guess (0.5)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=10)

    # ── Right: Data BER Comparison ──
    noisy_data_bers = [results[l]['noisy']['data_ber'] for l in lfsr_lengths]
    noiseless_data_bers = [results[l]['noiseless']['data_ber'] for l in lfsr_lengths]

    ax2.bar(x - width/2, noisy_data_bers, width, label='Noisy Train (sigma=0.5)', color='crimson', alpha=0.8)
    ax2.bar(x + width/2, noiseless_data_bers, width, label='Noiseless Train (sigma=0.0)', color='orange', alpha=0.7)
    ax2.set_ylabel('Data BER', fontsize=12)
    ax2.set_xlabel('LFSR Bit Length', fontsize=12)
    ax2.set_title('Decoded Data BER Comparison (Lower is Better)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{l}bit' for l in lfsr_lengths])
    ax2.set_ylim(-0.02, 0.55)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Guess (0.5)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=10)

    fig.suptitle(f'Generalization Performance: Noisy Train vs Noiseless Train (Test Noise sigma={TEST_NOISE_SCALE})', 
                 fontsize=15, y=1.02)
    fig.tight_layout()
    out = 'train_compare_results.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"\n[Plot] Comparison graph saved to: {out}")


# ══════════════════════════════════════════════
# メイン実行部
# ══════════════════════════════════════════════

class TeeLogger:
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
    print("=" * 70)
    print("  Y00 Generalization Analysis: Noisy Train vs Noiseless Train")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"S_max: {SMAX_VALUE}")
    print(f"Evaluation Test Noise Scale (sigma): {TEST_NOISE_SCALE}")
    print(f"LFSR Lengths: {lfsr_lengths}")
    print()

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    results = {}

    for l_len in lfsr_lengths:
        print(f"\n{'#' * 70}")
        print(f"  Target: LFSR {l_len}bit")
        print(f"{'#' * 70}")
        
        # 1. Noisy Training
        noisy_res = train_and_evaluate_scenario(l_len, device, train_noise_scale=0.5, scenario_name='Noisy Train')
        
        # 2. Noiseless Training
        noiseless_res = train_and_evaluate_scenario(l_len, device, train_noise_scale=0.0, scenario_name='Noiseless Train')

        results[l_len] = {
            'noisy': noisy_res,
            'noiseless': noiseless_res
        }

        # 各モデルの保存
        for key, res in [('noisy', noisy_res), ('noiseless', noiseless_res)]:
            if res['best_state'] is not None:
                model_path = os.path.join(MODEL_SAVE_DIR, f'seed_predictor_{l_len}bit_{key}.pth')
                torch.save({
                    'model_state_dict': res['best_state'],
                    'model_config': res['model_config'],
                    'lfsr_length': l_len,
                    's_max': SMAX_VALUE,
                    'ber': res['ber'],
                    'data_ber': res['data_ber'],
                }, model_path)
                print(f"  Model ({key}) saved to {model_path}")

    # 結果をYAMLにエクスポート
    yaml_data = {}
    for l_len in lfsr_lengths:
        yaml_data[int(l_len)] = {
            'noisy': {
                'seed_ber': float(results[l_len]['noisy']['ber']),
                'exact_match': float(results[l_len]['noisy']['exact_match']),
                'data_ber': float(results[l_len]['noisy']['data_ber']),
            },
            'noiseless': {
                'seed_ber': float(results[l_len]['noiseless']['ber']),
                'exact_match': float(results[l_len]['noiseless']['exact_match']),
                'data_ber': float(results[l_len]['noiseless']['data_ber']),
            }
        }
    yaml_path = 'train_compare_results.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    print(f"\nSaved all results to {yaml_path}")

    # サマリー印刷
    print("\n" + "=" * 80)
    print("  Final Comparison Summary (Evaluated on Test Noise sigma=0.5)")
    print("=" * 80)
    print(f"{'LFSR':>6} | {'Noisy Train (Seed BER)':>22} | {'Noiseless Train (Seed BER)':>26} | {'Noisy (Data BER)':>16} | {'Noiseless (Data BER)':>20}")
    print("-" * 105)
    for l_len in lfsr_lengths:
        n_seed = results[l_len]['noisy']['ber']
        nl_seed = results[l_len]['noiseless']['ber']
        n_data = results[l_len]['noisy']['data_ber']
        nl_data = results[l_len]['noiseless']['data_ber']
        print(f"{l_len:>4}bit | {n_seed:>22.4f} | {nl_seed:>26.4f} | {n_data:>16.6f} | {nl_data:>20.6f}")
    print("=" * 80)

    # グラフのプロット
    plot_comparison_results(results, lfsr_lengths)


if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'seed_claude_train_compare_log_{timestamp}.txt'
    tee = TeeLogger(log_filename)
    sys.stdout = tee

    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print(f"\nTotal Run Time: {elapsed:.1f} s ({elapsed / 60:.1f} min)")

    tee.close()
    print(f"Log saved to {log_filename}")
