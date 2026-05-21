# -*- coding: utf-8 -*-
"""
experiment_lfsr_no_noise.py (量子ノイズなし版)

このスクリプトの目的：
LFSR（擬似乱数生成器）の「ビット長（周期の長さ）」を変えながら、
「もし量子ノイズによる隠蔽効果が一切なかったら」という前提で、
Transformer（AI）が純粋にLFSRのアルゴリズム（周期パターン）を
どこまで解読できるかを検証します。
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
# （基本構造はノイズあり版と全く同じです。純粋なパターン推論能力を検証します）

class PositionalEncoding(nn.Module):
    """
    位置エンコーディング（Positional Encoding）
    入力データに「時間的な順序（順番）」の情報を与えるためのクラス。
    """
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
    """
    Eve（盗聴者）の頭脳となるTransformerモデル。
    """
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 1. 1次元畳み込み（Conv1d）
        self.input_conv = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3, padding=1)
        
        # 2. 位置情報の付加
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformerのコア部分（Self-Attention）
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 4. 最終予測層（0か1かの確率を出力）
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
    """
    時系列データの「窓（ウィンドウ）」を作ります。
    （ノイズあり版と全く同じ処理です）
    """
    X, Y = [], []
    for i in range(len(xs) - seq_len):
        X.append(xs[i : i + seq_len])
        Y.append(ys[i + seq_len - 1])  
    
    X_arr = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    Y_arr = np.array(Y, dtype=np.float32)
    return X_arr, Y_arr

# ── 3. 可変LFSR信号生成 (ここが一番の違い！) ──

LFSR_TAPS = {
    4:  [3, 2],
    6:  [5, 4],
    8:  [7, 5, 4, 3],
    10: [9, 6],
    12: [11, 10, 9, 3],
    14: [13, 12, 11, 1],
    16: [15, 13, 12, 10]
}

def generate_signals_variable_lfsr_no_noise(n_symbols: int, lfsr_length: int):
    """
    指定された長さのLFSRを使って暗号化を行いますが、
    【量子ノイズを含まない】理想的な計算値をそのまま出力します。
    """
    N = 12
    BNum = 2 ** N
    S_max = 10.0
    S_levels = np.linspace(0, S_max, BNum * 2)

    # 初期レジスタ
    reg = (1 << lfsr_length) - 1
    taps = LFSR_TAPS[lfsr_length]
    
    total_bits = n_symbols * N
    bits = np.empty(total_bits, dtype=np.uint8)
    mask = (1 << lfsr_length) - 1
    
    # LFSRによる擬似乱数生成
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
    
    # Y00変調 (データと乱数を混ぜて信号レベルを決定)
    mod_indices = (input_data + base_ids % 2) % 2
    output_levels = S_levels[base_ids + BNum * mod_indices]
    
    # ── 【重要】ここがノイズあり版との違い ──
    # ガウスノイズ（np.random.normal）を加えず、計算上の理想的な振幅値
    # （つまり数学的な正解そのもの）を、そのままEveが観測できたと仮定します。
    # ノイズがないため、EveはLFSRが算出したパターンを直接視認できてしまいます。
    xs = output_levels.astype(np.float32)
        
    return xs, input_data

# ── 4. メインループ ──
def main():
    lfsr_lengths = [4, 6, 8, 10, 12, 14, 16]
    ber_results = []
    
    n_train = 30_000
    n_test  = 10_000
    
    # Transformerが過去を振り返る長さ
    # ※ノイズがない場合、周期がこの長さ(64)以下のLFSR（4bitや6bit）は完全に解読されます。
    seq_len = 64
    epochs  = 100
    batch_size = 512
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"実行デバイス: {device}")
    print(f"学習サンプル: {n_train}, 評価サンプル: {n_test}, シーケンス長: {seq_len}, エポック: {epochs}\n")
    
    total_symbols = n_train + n_test + seq_len
    
    for l_len in lfsr_lengths:
        print(f"=========================================")
        print(f" LFSR長: {l_len} bit (周期: {(1<<l_len)-1}) の検証開始 (【量子ノイズなし】)")
        print(f"=========================================")
        
        # 【重要】ノイズなしの生成関数を呼び出しています
        xs, true_bits = generate_signals_variable_lfsr_no_noise(total_symbols, l_len)
        
        # 標準化
        mean_x, std_x = np.mean(xs[:n_train]), np.std(xs[:n_train])
        xs_scaled = (xs - mean_x) / (std_x + 1e-8)
        
        X_train_np, Y_train_np = create_sequences(xs_scaled[:n_train+seq_len], true_bits[:n_train+seq_len], seq_len)
        X_test_np, Y_test_np   = create_sequences(xs_scaled[n_train:], true_bits[n_train:], seq_len)
        
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_np), torch.from_numpy(Y_train_np)), batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_test_np), torch.from_numpy(Y_test_np)), batch_size=batch_size, shuffle=False)
        
        model = TransformerDecoder(d_model=64, nhead=4, num_layers=2, dim_feedforward=128).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # ── 学習ループ ──
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                preds = model(bx)
                loss = criterion(preds, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * len(by)
                
            avg_loss = total_loss / len(X_train_np)
            if epoch == 1 or epoch % 5 == 0:
                print(f"  Epoch [{epoch:03d}/{epochs}] Loss: {avg_loss:.4f}")
                
        # ── 評価ループ ──
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
        ber = 1.0 - acc
        
        print(f"  => 最終BER: {ber:.4f}\n")
        ber_results.append(ber)
        
    # ── グラフ描画 ──
    print("すべての検証が完了しました。グラフを描画・保存します...")
    plt.figure(figsize=(8, 5))
    # ノイズなしのグラフなので緑色で描画
    plt.plot(lfsr_lengths, ber_results, marker='o', linestyle='-', color='g', linewidth=2, markersize=8, label='No Quantum Noise')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guessing (BER=0.5)')
    
    plt.title('Eve\'s Transformer BER vs. LFSR Bit Length (No Quantum Noise)', fontsize=14)
    plt.xlabel('LFSR Bit Length (Longer = Better Security)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.ylim(-0.05, 0.55)
    plt.xticks(lfsr_lengths)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    output_filename = 'ber_vs_lfsr_length_no_noise.png'
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"グラフを {output_filename} として保存しました。")
    
if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"総実行時間: {time.time() - t0:.1f} 秒")
