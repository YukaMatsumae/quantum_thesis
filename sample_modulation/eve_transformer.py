import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import publisher_claude as pub
import time
import math

# ──────────────────────────────────────────────
# 1. 位置エンコーディング (Transformer 必須)
# ──────────────────────────────────────────────
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
        """
        x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

# ──────────────────────────────────────────────
# 2. Sionna相当の Neural Receiver (Transformer)
# ──────────────────────────────────────────────
class TransformerDecoder(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 1D-CNN 特徴抽出器 (Sionna等でもよく使われる入力の平滑化・埋め込み)
        self.input_conv = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3, padding=1)
        
        # 位置エンコード
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 解読ヘッド (Attentionで集約した全体情報を使って予測)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, 1)
        # Conv1d は (batch_size, channels, seq_len) を要求するため転置する
        x = x.transpose(1, 2)
        x = self.input_conv(x)
        x = x.transpose(1, 2) # (batch_size, seq_len, d_model) に戻す
        
        x = x * math.sqrt(self.d_model) # スケーリング
        x = self.pos_encoder(x)
        
        # Attention によって時系列の長距離相関（LFSRの周期など）を捉える
        memory = self.transformer_encoder(x)
        
        # ターゲット位置（一番最後）の特徴量を取得して推測
        last_hidden = memory[:, -1, :] 
        prob = self.fc_out(last_hidden)
        
        return prob.squeeze(dim=-1)

# ──────────────────────────────────────────────
# 3. データ準備
# ──────────────────────────────────────────────
def create_sequences(xs, ys, seq_len):
    """
    時系列のホモダイン測定値のリストから学習用のスライディングウィンドウを作成。
    xs: 測定値 (N,)
    ys: 正解ビット (N,)
    """
    X, Y = [], []
    for i in range(len(xs) - seq_len):
        X.append(xs[i : i + seq_len])
        Y.append(ys[i + seq_len - 1])  # シーケンス終端のビットを解読ターゲットに
    
    X_arr = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    Y_arr = np.array(Y, dtype=np.float32)
    return X_arr, Y_arr

# ──────────────────────────────────────────────
# 4. メインシミュレーション
# ──────────────────────────────────────────────
def run_transformer_eavesdropper(n_train=100_000, n_test=20_000, seq_len=64, epochs=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"実行デバイス: {device}")
    
    pub.reset_lfsr() 
    
    total_symbols = n_train + n_test + seq_len
    print(f"\n信号生成中... ({total_symbols} サンプル)")
    
    rng = np.random.default_rng(seed=42)
    true_bits = rng.integers(0, 2, size=total_symbols, dtype=np.int32)
    
    qstates, output_bits, _ = pub.generate_signals_with_labels(true_bits)
    
    print("Eve: ホモダイン測定実行中...")
    xs = np.array([qs.homodyne_measurement() for qs in qstates], dtype=np.float32)
    
    # 標準化
    mean_x, std_x = np.mean(xs[:n_train]), np.std(xs[:n_train])
    xs_scaled = (xs - mean_x) / (std_x + 1e-8)
    
    print("データセットの構築中...")
    X_train_np, Y_train_np = create_sequences(xs_scaled[:n_train+seq_len], true_bits[:n_train+seq_len], seq_len)
    X_test_np, Y_test_np   = create_sequences(xs_scaled[n_train:], true_bits[n_train:], seq_len)
    
    train_dataset = TensorDataset(torch.from_numpy(X_train_np), torch.from_numpy(Y_train_np))
    test_dataset  = TensorDataset(torch.from_numpy(X_test_np), torch.from_numpy(Y_test_np))
    
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # ── 学習 ──
    # Transformer は学習が収束しにくいため、Warmup等を使わない場合はLRを慎重に
    model = TransformerDecoder(d_model=64, nhead=4, num_layers=2, dim_feedforward=128).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # Sionna等の標準的なOptimizer
    
    print(f"\n── Transformer (Sionna 相当) Neural Receiver の学習開始 ──")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            
            # Gradient clipping (安定化)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item() * len(by)
            
        avg_loss = total_loss / len(train_dataset)
        
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                test_bx, test_by = next(iter(train_loader))
                test_bx, test_by = test_bx.to(device), test_by.to(device)
                p = model(test_bx)
                acc = ((p >= 0.5).float() == test_by).sum().item() / len(test_by)
            print(f"Epoch [{epoch:02d}/{epochs}] Loss: {avg_loss:.4f} | Train Batch Acc: {acc:.4f}")
    
    # ── 評価 ──
    model.eval()
    all_preds = []
    all_targets = []
    
    print("\n── テストデータに対する解読フェーズ ──")
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            preds = model(bx)
            preds_bin = (preds >= 0.5).int().cpu().numpy()
            all_preds.extend(preds_bin)
            all_targets.extend(by.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = np.mean(all_preds == all_targets)
    ber = 1.0 - acc
    
    print(f"\n【Sionna(Transformer)アーキテクチャ 検証結果】")
    print(f" - 学習サンプル数           : {n_train}")
    print(f" - シーケンス長(Attention)  : {seq_len}")
    print(f" - 評価サンプル数           : {len(all_preds)}")
    print(f" - 盗聴者TransformerのBER : {ber:.4f}  (ランダムの場合 0.5000)")
    
    if ber < 0.48:
        print("\n=> [WARNING] TransformerがLFSRの長距離的な相関を捉え、パターン学習に成功しました。")
    else:
        print("\n=> [SAFE] 最強クラスのTransformerを用いても解読できませんでした。（量子ノイズによる隠蔽はAttentionをも突破します）")

if __name__ == "__main__":
    t0 = time.time()
    run_transformer_eavesdropper(n_train=100_000, n_test=20_000, seq_len=64, epochs=15)
    print(f"\n総実行時間: {time.time() - t0:.1f} s")
