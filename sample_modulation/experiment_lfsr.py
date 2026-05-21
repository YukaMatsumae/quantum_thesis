# -*- coding: utf-8 -*-
# 文字コードをUTF-8に指定

"""
experiment_lfsr.py (量子ノイズあり版)

目的:
LFSR（擬似乱数生成器）のビット長（周期）を変化させ、
量子ノイズ環境下においてTransformerモデルが
Y00変調の暗号パターンをどの程度学習・推測可能かを検証する。
"""

import numpy as np               # 数値計算・配列操作ライブラリ
import torch                     # 深層学習ライブラリPyTorch
import torch.nn as nn            # ニューラルネットワーク構築用モジュール
import torch.optim as optim      # 最適化アルゴリズム
from torch.utils.data import TensorDataset, DataLoader # データバッチ化・管理用
import matplotlib.pyplot as plt  # グラフ描画ライブラリ
import time                      # 実行時間計測用
import math                      # 数学関数用

# ── 1. Transformer の実装 ──
# 過去のホモダイン測定値（時系列データ）から、次のビットを予測するモデルを定義する。

class PositionalEncoding(nn.Module):
    """
    位置エンコーディング (Positional Encoding)
    Transformerは系列の順序を保持しないため、サイン波・コサイン波を用いて
    データに時間的な位置情報を埋め込む。
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # 時系列長に合わせた位置情報を縦ベクトルとして生成
        position = torch.arange(max_len).unsqueeze(1)
        # サイン波・コサイン波の周期調整項を計算
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # (1, max_len, d_model) のゼロテンソルを初期化
        pe = torch.zeros(1, max_len, d_model)
        # 偶数次元にサイン波の位置情報を代入
        pe[0, :, 0::2] = torch.sin(position * div_term)
        # 奇数次元にコサイン波の位置情報を代入
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # モデルのパラメータとして更新されないバッファとして登録
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 入力データに位置情報を加算
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerDecoder(nn.Module):
    """
    盗聴者(Eve)を模倣するTransformerモデル。
    観測データの時系列からLFSRのパターン（相関）を抽出する。
    """
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 1. 1次元畳み込み (Conv1d)
        # アナログ測定値を平滑化し、d_model次元の特徴量へ変換する
        self.input_conv = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3, padding=1)
        
        # 2. 位置情報の付加
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformerコア (Self-Attention)
        # 時系列データ間の長距離依存性（LFSRの相関など）を学習する
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,                 # 入出力次元数
            nhead=nhead,                     # マルチヘッドアテンションのヘッド数
            dim_feedforward=dim_feedforward, # 中間層の次元数
            dropout=dropout,                 # ドロップアウト率
            batch_first=True                 # (バッチサイズ, 系列長, 次元数) の入力形式
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 4. 最終予測層 (全結合層)
        # 抽出した特徴から0または1の確率を出力する
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32), # d_model次元から32次元への変換
            nn.GELU(),              # GELU活性化関数
            nn.Dropout(dropout),    # ドロップアウト
            nn.Linear(32, 1),       # 32次元から1次元（確率値）への変換
            nn.Sigmoid()            # シグモイド関数で0〜1に正規化
        )
        
    def forward(self, x):
        # Conv1dに合わせて次元を (Batch, Channels, SeqLen) に変換
        x = x.transpose(1, 2)
        x = self.input_conv(x)
        x = x.transpose(1, 2) 
        
        # Transformer入力前のスケール調整
        x = x * math.sqrt(self.d_model) 
        # 位置エンコーディングの加算
        x = self.pos_encoder(x)
        
        # Transformerによる系列特徴の抽出
        memory = self.transformer_encoder(x)
        
        # 最新（最後）の系列の特徴量を取得
        last_hidden = memory[:, -1, :] 
        # 次のビットが1である確率を予測
        prob = self.fc_out(last_hidden)
        
        # 次元を削減しスカラー値の確率を返す
        return prob.squeeze(dim=-1)

# ── 2. データ準備 ──
def create_sequences(xs, ys, seq_len):
    """
    時系列データの入力ウィンドウと対応する正解ラベルのペアを作成する。
    """
    X, Y = [], []
    for i in range(len(xs) - seq_len):
        # 過去 seq_len 分の観測値を入力データとする
        X.append(xs[i : i + seq_len])
        # 入力データの直後のビットを正解ラベルとする
        Y.append(ys[i + seq_len - 1])
    
    # 入力を (データ数, 系列長, 1) のNumPy配列に変換
    X_arr = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    # ラベルを1次元配列に変換
    Y_arr = np.array(Y, dtype=np.float32)
    return X_arr, Y_arr

# ── 3. 可変LFSR信号生成 ──

# LFSRの最長周期 (2^n - 1) を生成するためのタップ位置定義
LFSR_TAPS = {
    4:  [3, 2],
    6:  [5, 4],
    8:  [7, 5, 4, 3],
    10: [9, 6],
    12: [11, 10, 9, 3],
    14: [13, 12, 11, 1],
    16: [15, 13, 12, 10]
}

def generate_signals_variable_lfsr(n_symbols: int, lfsr_length: int):
    """
    指定長のLFSRで暗号化を実行し、量子ノイズを含むホモダイン測定値を生成する。
    """
    N = 12                     # 基底決定に用いるビット数
    BNum = 2 ** N              # 基底数 (4096)
    S_max = 10.0               # 最大信号レベル
    S_levels = np.linspace(0, S_max, BNum * 2) # 利用可能な信号レベル一覧を生成

    # シフトレジスタを全ビット1で初期化
    reg = (1 << lfsr_length) - 1
    taps = LFSR_TAPS[lfsr_length]
    
    total_bits = n_symbols * N
    bits = np.empty(total_bits, dtype=np.uint8)
    mask = (1 << lfsr_length) - 1
    
    # ── [A] LFSRによる擬似乱数生成 ──
    for k in range(total_bits):
        SR = 0
        for t in taps:
            # タップ位置のビットのXORを計算
            SR ^= (reg >> t) 
        SR &= 1
        
        # 最上位ビットを出力として保存
        bits[k] = (reg >> (lfsr_length - 1)) & 1 
        # レジスタを左シフトし、計算したビットを最下位に挿入
        reg = ((reg << 1) & mask) | SR           
        
    # Nビットをまとめ、0〜4095の基底ID (base_id) を算出
    weights = (1 << np.arange(N)).astype(np.int32)
    base_ids = bits.reshape(n_symbols, N).dot(weights)
    
    # 送信データをランダムに生成 (シード固定)
    rng = np.random.default_rng(seed=42)
    input_data = rng.integers(0, 2, size=n_symbols, dtype=np.int32)
    
    # ── [B] Y00変調 ──
    # 送信データと基底IDから変調インデックスを決定
    mod_indices = (input_data + base_ids % 2) % 2
    # 理想的な送信振幅レベルを取得
    output_levels = S_levels[base_ids + BNum * mod_indices] 
    
    # ── [C] 量子ノイズの付加 ──
    # 真空揺らぎに相当する標準偏差0.5のガウスノイズを加算
    rng_meas = np.random.default_rng()
    xs = rng_meas.normal(loc=output_levels, scale=0.5).astype(np.float32)
        
    return xs, input_data

# ── 4. メインループ ──
def main():
    # 評価対象のLFSRビット長
    lfsr_lengths = [4, 6, 8, 10, 12, 14, 16]
    ber_results = []
    
    # データ数の設定
    n_train = 30_000
    n_test  = 10_000
    
    seq_len = 64         # Transformerの入力時系列長　！！！！要調整
    epochs  = 100        # 学習エポック数
    batch_size = 512     # バッチサイズ
    
    # 計算デバイスの選定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"実行デバイス: {device}")
    print(f"学習サンプル: {n_train}, 評価サンプル: {n_test}, シーケンス長: {seq_len}, エポック: {epochs}\n")
    
    total_symbols = n_train + n_test + seq_len
    
    for l_len in lfsr_lengths:
        print(f"=========================================")
        print(f" LFSR長: {l_len} bit (周期: {(1<<l_len)-1}) の検証開始 (【量子ノイズあり】)")
        print(f"=========================================")
        
        # 信号および正解ラベルデータの生成
        xs, true_bits = generate_signals_variable_lfsr(total_symbols, l_len)
        
        # データの標準化 (平均0, 分散1)
        mean_x = np.mean(xs[:n_train])
        std_x = np.std(xs[:n_train])
        xs_scaled = (xs - mean_x) / (std_x + 1e-8)
        
        # スライディングウィンドウによる時系列シーケンスの作成
        X_train_np, Y_train_np = create_sequences(xs_scaled[:n_train+seq_len], true_bits[:n_train+seq_len], seq_len)
        X_test_np, Y_test_np   = create_sequences(xs_scaled[n_train:], true_bits[n_train:], seq_len)
        
        # データローダーの作成
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_np), torch.from_numpy(Y_train_np)), batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_test_np), torch.from_numpy(Y_test_np)), batch_size=batch_size, shuffle=False)
        
        # モデル、損失関数、最適化手法の定義
        model = TransformerDecoder(d_model=64, nhead=4, num_layers=2, dim_feedforward=128).to(device)
        criterion = nn.BCELoss() # 2値分類用損失関数
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # ── 学習ループ ──
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()       # 勾配の初期化
                preds = model(bx)           # 予測の実行
                loss = criterion(preds, by) # 損失の計算
                loss.backward()             # 誤差逆伝播
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 勾配クリッピング
                optimizer.step()            # パラメータ更新
                total_loss += loss.item() * len(by)
                
            avg_loss = total_loss / len(X_train_np)
            if epoch == 1 or epoch % 5 == 0:
                print(f"  Epoch [{epoch:03d}/{epochs}] Loss: {avg_loss:.4f}")
                
        # ── 評価ループ ──
        model.eval() 
        all_preds, all_targets = [], []
        with torch.no_grad(): # 勾配計算の無効化
            for bx, by in test_loader:
                bx = bx.to(device)
                preds = model(bx)
                # 確率0.5を閾値として0または1に2値化
                preds_bin = (preds >= 0.5).int().cpu().numpy() 
                all_preds.extend(preds_bin)
                all_targets.extend(by.numpy())
                
        # BER (Bit Error Rate) の計算
        acc = np.mean(np.array(all_preds) == np.array(all_targets)) # 正解率
        ber = 1.0 - acc # エラー率
        
        print(f"  => 最終BER: {ber:.4f}\n")
        ber_results.append(ber)
        
    # ── グラフ描画 ──
    print("すべての検証が完了しました。グラフを描画・保存します...")
    plt.figure(figsize=(8, 5))
    
    # BER推移のプロット
    plt.plot(lfsr_lengths, ber_results, marker='o', linestyle='-', color='b', linewidth=2, markersize=8, label='With Quantum Noise')
    # ランダム推測基準線 (BER=0.5) の描画
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guessing (BER=0.5)') 
    
    plt.title('Eve\'s Transformer BER vs. LFSR Bit Length (With Quantum Noise)', fontsize=14)
    plt.xlabel('LFSR Bit Length (Longer = Better Security)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.ylim(-0.05, 0.55)
    plt.xticks(lfsr_lengths)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    output_filename = 'ber_vs_lfsr_length.png'
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"グラフを {output_filename} として保存しました。")
    
if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"総実行時間: {time.time() - t0:.1f} 秒")
