"""
シンプルなBiLSTM: bit列を学習する
タスク: 過去N bitから次の1 bitを予測 (next-bit prediction)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


# ────────────────────────────────────────────
# 1. データセット
# ────────────────────────────────────────────
class BitDataset(Dataset):
    """bit列をスライディングウィンドウで (X, y) に変換"""

    def __init__(self, bits: list[int], seq_len: int = 32):
        self.seq_len = seq_len
        self.X, self.y = [], []
        for i in range(len(bits) - seq_len):
            self.X.append(bits[i : i + seq_len])
            self.y.append(bits[i + seq_len])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(-1)  # (seq_len, 1)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


# ────────────────────────────────────────────
# 2. モデル
# ────────────────────────────────────────────
class BiLSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),  # *2 for bidirectional
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.bilstm(x)          # (batch, seq_len, hidden*2)
        last = out[:, -1, :]             # 最後のタイムステップ
        return self.fc(last).squeeze(-1) # (batch,)


# ────────────────────────────────────────────
# 3. 学習ループ
# ────────────────────────────────────────────
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


# ────────────────────────────────────────────
# 4. メイン
# ────────────────────────────────────────────
def main():
    # ── ハイパーパラメータ ──
    SEQ_LEN    = 32
    HIDDEN     = 64
    LAYERS     = 2
    BATCH_SIZE = 128
    EPOCHS     = 20
    LR         = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── サンプルデータ: 周期的なbit列 (LFSRや実データに差し替え可) ──
    rng = np.random.default_rng(42)

# LFSR (例: 8段, 帰還多項式 x^8 + x^6 + x^5 + x^4 + 1)
    def lfsr(seed: int, taps: list[int], n_bits: int) -> list[int]:
        state = seed
        nbits_reg = max(taps)
        out = []
        for _ in range(n_bits):
            bit = state & 1
            out.append(bit)
            feedback = 0
            for t in taps:
                feedback ^= (state >> (t - 1)) & 1
            state = ((state >> 1) | (feedback << (nbits_reg - 1))) & ((1 << nbits_reg) - 1)
        return out

    bits = lfsr(seed=0b10110011, taps=[8, 6, 5, 4], n_bits=10000)

    # train / val 分割
    split = int(len(bits) * 0.8)
    train_ds = BitDataset(bits[:split],  seq_len=SEQ_LEN)
    val_ds   = BitDataset(bits[split:],  seq_len=SEQ_LEN)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # ── モデル / 損失 / 最適化 ──
    model     = BiLSTMPredictor(hidden_size=HIDDEN, num_layers=LAYERS).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}\n")

    # ── 学習 ──
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train(model, train_dl, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_dl, criterion, device)
        scheduler.step()
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f} | "
            f"Val loss: {va_loss:.4f}  acc: {va_acc:.4f}"
        )

    # ── 推論サンプル ──
    model.eval()
    sample_x, sample_y = val_ds[0]
    with torch.no_grad():
        logit = model(sample_x.unsqueeze(0).to(device))
        pred  = int((torch.sigmoid(logit) >= 0.5).item())
    print(f"\n[Inference] Input bits (last 8): {sample_x[-8:, 0].int().tolist()}")
    print(f"            True next bit: {int(sample_y.item())}  Predicted: {pred}")


if __name__ == "__main__":
    main()
