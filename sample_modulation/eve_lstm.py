import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import publisher_claude as pub
import time

# ──────────────────────────────────────────────
# 1. ネットワーク定義
# ──────────────────────────────────────────────
class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2):
        super().__init__()
        # 入力次元は1（ホモダイン測定値のスカラー）
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, 
                            dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid() # 0か1の確率を出力
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, 1)
        lstm_out, _ = self.lstm(x)
        # 最後のタイムステップの出力を特徴量として使用する
        last_out = lstm_out[:, -1, :] 
        prob = self.fc(last_out)
        return prob.squeeze(dim=-1)

# ──────────────────────────────────────────────
# 2. データ準備
# ──────────────────────────────────────────────
def create_sequences(xs, ys, seq_len):
    """
    時系列のホモダイン測定値のリストから学習用のスライディングウィンドウを作成。
    xs: 測定値 (N,)
    ys: 正解ビット (N,)
    seq_len: ウィンドウサイズ
    """
    X, Y = [], []
    for i in range(len(xs) - seq_len):
        X.append(xs[i : i + seq_len])
        Y.append(ys[i + seq_len - 1])  # シーケンス末尾のビットを予測ターゲットとする
    
    # PyTorch用に shape を (Batch, seq_len, 1) に変形
    X_arr = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    Y_arr = np.array(Y, dtype=np.float32)
    return X_arr, Y_arr

# ──────────────────────────────────────────────
# 3. メインシミュレーション
# ──────────────────────────────────────────────
def run_lstm_eavesdropper(n_train=100_000, n_test=20_000, seq_len=40):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"実行デバイス: {device}")
    
    # ── [攻撃シナリオ] Eveによる通信の傍受（全体） ──
    # Eveは LFSR をリセットせずに傍受を継続する
    pub.reset_lfsr() 
    
    total_symbols = n_train + n_test + seq_len
    print(f"\n信号生成中... ({total_symbols} サンプル)")
    
    # ランダムな平文ビット
    rng = np.random.default_rng(seed=42)
    true_bits = rng.integers(0, 2, size=total_symbols, dtype=np.int32)
    
    # 生成（publisher）
    qstates, output_bits, _ = pub.generate_signals_with_labels(true_bits)
    
    # ホモダイン測定
    print("Eve: ホモダイン測定実行中...")
    xs = np.array([qs.homodyne_measurement() for qs in qstates], dtype=np.float32)
    
    # 標準化（Z-score）
    mean_x, std_x = np.mean(xs[:n_train]), np.std(xs[:n_train])
    xs_scaled = (xs - mean_x) / (std_x + 1e-8)
    
    # スライディングウィンドウで学習/テストセットを作成
    print("データセットの構築中...")
    X_train_np, Y_train_np = create_sequences(xs_scaled[:n_train+seq_len], true_bits[:n_train+seq_len], seq_len)
    X_test_np, Y_test_np   = create_sequences(xs_scaled[n_train:], true_bits[n_train:], seq_len)
    
    # PyTorchテンソルへの変換
    train_dataset = TensorDataset(torch.from_numpy(X_train_np), torch.from_numpy(Y_train_np))
    test_dataset  = TensorDataset(torch.from_numpy(X_test_np), torch.from_numpy(Y_test_np))
    
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # ── 学習 ──
    model = LSTMDecoder(hidden_size=128, num_layers=2).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 15
    print("\n── LSTMの学習開始 ──")
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
            
        avg_loss = total_loss / len(train_dataset)
        
        # 1エポックごとの検証精度(Train上)
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            correct = 0
            with torch.no_grad():
                # サンプリングによる評価(Train上)
                test_bx, test_by = next(iter(train_loader))
                test_bx, test_by = test_bx.to(device), test_by.to(device)
                p = model(test_bx)
                p_bin = (p >= 0.5).float()
                correct = (p_bin == test_by).sum().item()
                acc = correct / len(test_by)
            print(f"Epoch [{epoch}/{epochs}] Loss: {avg_loss:.4f} | Train Batch Acc: {acc:.4f}")
    
    # ── 評価（未知のテストデータに対する解読） ──
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
    
    print(f"結果の統計:")
    print(f" - 学習サンプル数     : {n_train}")
    print(f" - シーケンス長(Window) : {seq_len}")
    print(f" - 評価サンプル数     : {len(all_preds)}")
    print(f" - 盗聴者LSTMのBER  : {ber:.4f}  (ランダムの場合は 0.5000)")
    
    if ber < 0.48:
        print("\n=> [WARNING] LSTMが時間的相関を捉え、鍵系列のパターン学習に成功している可能性があります（チートなしでも解読兆候あり）！")
    else:
        print("\n=> [SAFE] LSTMを用いても解読できていません（LFSRの周期性や相関が隠蔽されています）。")

if __name__ == "__main__":
    t0 = time.time()
    run_lstm_eavesdropper(n_train=100_000, n_test=20_000, seq_len=30)
    print(f"\nTotal Time: {time.time() - t0:.1f} s")
