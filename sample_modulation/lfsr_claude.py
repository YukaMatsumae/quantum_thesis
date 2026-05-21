# -*- coding: utf-8 -*-
"""
lfsr_claude.py

4-bit LFSR (初期レジスタ4つ) の環境下で、
量子ノイズを含まない理想的な信号をTransformerが学習・解読できるかを検証するスクリプト。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import math

# ── 1. 4ビットLFSR & 信号生成 (量子ノイズなし) ──
N_group = 4 
BNum = 2 ** N_group  # 16
S_max = 10.0
S_levels = np.linspace(0, S_max, BNum * 2)

def generate_signals_no_noise(n_symbols: int):
    """
    4-bit LFSRで信号を生成。量子ノイズを含まない。
    """
    # 1. 4-bit LFSRで暗号用ベースIDを生成
    # 初期レジスタ: [1, 0, 1, 1] (最上位から1, 0, 1, 1 と想定)
    # 値としては 0b1011 = 11 または 0b1101 = 13。ここでは13を使用。
    reg = 13
    
    total_bits = n_symbols * N_group
    lfsr_bits = np.empty(total_bits, dtype=np.uint8)
    
    for k in range(total_bits):
        # 4-bit LFSR (タップ位置: bit3, bit2 -> 0-indexed)
        SR = ((reg >> 3) ^ (reg >> 2)) & 1
        lfsr_bits[k] = (reg >> 3) & 1
        reg = ((reg << 1) & 0xF) | SR
        
    # N_groupビットごとにまとめてbase_id化
    weights = (1 << np.arange(N_group)).astype(np.int32)
    base_ids = lfsr_bits.reshape(n_symbols, N_group).dot(weights)
    
    # 2. 送信ビットの生成 (ランダムな 0 or 1)
    rng = np.random.default_rng(seed=42)
    input_data = rng.integers(0, 2, size=n_symbols, dtype=np.int32)
    
    # 3. 信号レベル(振幅)の決定
    # Y00変調のロジック: base_id + BNum * mod_index
    mod_indices = (input_data + base_ids % 2) % 2
    output_levels = S_levels[base_ids + BNum * mod_indices]
    
    # 量子ノイズなしのため、ガウス乱数を使わず理想的な信号レベルをそのまま返す
    return output_levels.astype(np.float32), input_data

# ── 2. Transformer の実装 ──
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
    def __init__(self, d_model=32, nhead=2, num_layers=2, dim_feedforward=64, dropout=0.0):
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
            nn.Linear(d_model, 16),
            nn.GELU(),
            nn.Linear(16, 1),
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

# ── 3. データ準備 ──
def create_sequences(xs, ys, seq_len):
    X, Y = [], []
    for i in range(len(xs) - seq_len):
        X.append(xs[i : i + seq_len])
        Y.append(ys[i + seq_len - 1])
    
    X_arr = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    Y_arr = np.array(Y, dtype=np.float32)
    return X_arr, Y_arr

# ── 4. メイン処理 ──
def main():
    n_train = 5_000   # 周期15なので少ないデータでも十分学習可能
    n_test  = 1_000
    seq_len = 16      # 周期15より少し長めに設定してパターンを包含させる
    epochs  = 30
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"実行デバイス: {device}")
    
    total_symbols = n_train + n_test + seq_len
    print(f"信号生成中... (ノイズなし, 4-bit LFSR) 計 {total_symbols} サンプル")
    
    xs, true_bits = generate_signals_no_noise(total_symbols)
    
    # 標準化
    mean_x, std_x = np.mean(xs[:n_train]), np.std(xs[:n_train])
    xs_scaled = (xs - mean_x) / (std_x + 1e-8)
    
    print("データセット構築中...")
    X_train, Y_train = create_sequences(xs_scaled[:n_train+seq_len], true_bits[:n_train+seq_len], seq_len)
    X_test, Y_test   = create_sequences(xs_scaled[n_train:], true_bits[n_train:], seq_len)
    
    batch_size = 128
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test)), batch_size=batch_size, shuffle=False)
    
    model = TransformerDecoder().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    print("\n── 学習開始 ──")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)
            
        avg_loss = total_loss / len(X_train)
        print(f"Epoch [{epoch:02d}/{epochs}] Loss: {avg_loss:.4f}")
        
    # 評価
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            preds = model(bx)
            preds_bin = (preds >= 0.5).int().cpu().numpy()
            all_preds.extend(preds_bin)
            all_targets.extend(by.numpy())
            
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    ber = 1.0 - acc
    
    print(f"\n【検証結果】")
    print(f" - LFSR種別           : 4-bit (周期15)")
    print(f" - 量子ノイズ         : なし")
    print(f" - Attention参照長    : {seq_len}")
    print(f" - 盗聴者TransformerのBER : {ber:.4f} (ランダムなら約0.5)")
    
    if ber < 0.1:
        print("\n=> [結論] 量子ノイズがなく、かつ周期が短いため、Transformerは完全にLFSRの法則を学習・解読できました。")
    else:
        print("\n=> [結論] 解読できませんでした。(想定外)")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"処理時間: {time.time() - t0:.2f} 秒")
