# -*- coding: utf-8 -*-
"""
experiment_lfsr_comparison.py

LFSRのビット長（4, 6, 8, 10, 12, 14, 16）ごとに、
「量子ノイズあり」と「量子ノイズなし」の双方の条件下で
Transformerによる解読精度（BER）を検証し、比較グラフをプロットするスクリプト。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time
import math

# ── 1. Transformer の実装 ──
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.input_conv = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3, padding=1)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_conv(x)
        x = x.transpose(1, 2) 
        
        x = x * math.sqrt(self.d_model) 
        x = self.pos_encoder(x)
        
        memory = self.transformer_encoder(x)
        last_hidden = memory[:, -1, :] 
        prob = self.fc_out(last_hidden)
        return prob.squeeze(dim=-1)

# ── 2. データ準備 ──
def create_sequences(xs, ys, seq_len):
    X, Y = [], []
    for i in range(len(xs) - seq_len):
        X.append(xs[i : i + seq_len])
        Y.append(ys[i + seq_len - 1])  
    
    X_arr = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    Y_arr = np.array(Y, dtype=np.float32)
    return X_arr, Y_arr

# ── 3. 可変LFSR信号生成 ──
LFSR_TAPS = {
    4:  [3, 2],
    6:  [5, 4],
    8:  [7, 5, 4, 3],
    10: [9, 6],
    12: [11, 10, 9, 3],
    14: [13, 12, 11, 1],
    16: [15, 13, 12, 10]
}

def generate_signals_variable_lfsr(n_symbols: int, lfsr_length: int, add_noise: bool):
    N = 12
    BNum = 2 ** N
    S_max = 10.0
    S_levels = np.linspace(0, S_max, BNum * 2)

    reg = (1 << lfsr_length) - 1
    taps = LFSR_TAPS[lfsr_length]
    
    total_bits = n_symbols * N
    bits = np.empty(total_bits, dtype=np.uint8)
    mask = (1 << lfsr_length) - 1
    
    for k in range(total_bits):
        SR = 0
        for t in taps:
            SR ^= (reg >> t)
        SR &= 1
        
        bits[k] = (reg >> (lfsr_length - 1)) & 1
        reg = ((reg << 1) & mask) | SR
        
    weights = (1 << np.arange(N)).astype(np.int32)
    base_ids = bits.reshape(n_symbols, N).dot(weights)
    
    rng = np.random.default_rng(seed=42)
    input_data = rng.integers(0, 2, size=n_symbols, dtype=np.int32)
    
    mod_indices = (input_data + base_ids % 2) % 2
    output_levels = S_levels[base_ids + BNum * mod_indices]
    
    if add_noise:
        rng_meas = np.random.default_rng()
        xs = rng_meas.normal(loc=output_levels, scale=0.5).astype(np.float32)
    else:
        xs = output_levels.astype(np.float32)
        
    return xs, input_data

def train_and_evaluate(l_len, add_noise, n_train, n_test, seq_len, epochs, batch_size, device):
    total_symbols = n_train + n_test + seq_len
    xs, true_bits = generate_signals_variable_lfsr(total_symbols, l_len, add_noise)
    
    mean_x, std_x = np.mean(xs[:n_train]), np.std(xs[:n_train])
    xs_scaled = (xs - mean_x) / (std_x + 1e-8)
    
    X_train_np, Y_train_np = create_sequences(xs_scaled[:n_train+seq_len], true_bits[:n_train+seq_len], seq_len)
    X_test_np, Y_test_np   = create_sequences(xs_scaled[n_train:], true_bits[n_train:], seq_len)
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_np), torch.from_numpy(Y_train_np)), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_test_np), torch.from_numpy(Y_test_np)), batch_size=batch_size, shuffle=False)
    
    model = TransformerDecoder(d_model=64, nhead=4, num_layers=2, dim_feedforward=128).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            preds = model(bx)
            preds_bin = (preds >= 0.5).int().cpu().numpy()
            all_preds.extend(preds_bin)
            all_targets.extend(by.numpy())
            
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    return 1.0 - acc

# ── 4. メインループ ──
def main():
    lfsr_lengths = [4, 6, 8, 10, 12, 14, 16]
    ber_no_noise = []
    ber_with_noise = []
    
    n_train = 30_000
    n_test  = 10_000
    seq_len = 64
    epochs  = 100
    batch_size = 512
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"実行デバイス: {device}")
    print(f"エポック数: {epochs}")
    
    for l_len in lfsr_lengths:
        print(f"\n=========================================")
        print(f" LFSR長: {l_len} bit (周期: {(1<<l_len)-1}) の検証開始")
        print(f"=========================================")
        
        # ノイズなし
        print(" [1] 量子ノイズなし の学習中...")
        ber1 = train_and_evaluate(l_len, False, n_train, n_test, seq_len, epochs, batch_size, device)
        ber_no_noise.append(ber1)
        print(f"  => ノイズなし BER: {ber1:.4f}")
        
        # ノイズあり
        print(" [2] 量子ノイズあり の学習中...")
        ber2 = train_and_evaluate(l_len, True, n_train, n_test, seq_len, epochs, batch_size, device)
        ber_with_noise.append(ber2)
        print(f"  => ノイズあり BER: {ber2:.4f}")
        
    print("\nすべての検証が完了しました。グラフを描画・保存します...")
    plt.figure(figsize=(10, 6))
    
    plt.plot(lfsr_lengths, ber_no_noise, marker='o', linestyle='-', color='g', linewidth=2, markersize=8, label='No Quantum Noise')
    plt.plot(lfsr_lengths, ber_with_noise, marker='s', linestyle='-', color='b', linewidth=2, markersize=8, label='With Quantum Noise')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guessing (BER=0.5)')
    
    plt.title('Eve\'s Transformer BER vs. LFSR Bit Length', fontsize=16)
    plt.xlabel('LFSR Bit Length', fontsize=14)
    plt.ylabel('Bit Error Rate (BER)', fontsize=14)
    plt.ylim(-0.05, 0.55)
    plt.xticks(lfsr_lengths)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    
    output_filename = 'ber_vs_lfsr_length_comparison.png'
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"グラフを {output_filename} として保存しました。")
    
if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"総実行時間: {time.time() - t0:.1f} 秒")
