import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import publisher_claude as pub

class EveMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.layers(x)


def collect_training_data(n_samples: int = 50_000):
    rng = np.random.default_rng(seed=0)
    random_bits = rng.integers(0, 2, size=n_samples, dtype=np.int32)
    qstates, bit_vals, _ = pub.generate_signals_with_labels(input_data=random_bits)
    xs = [qs.homodyne_measurement() for qs in qstates]
    X = torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(np.array(bit_vals), dtype=torch.long)
    return X, y


def train(model, X, y, epochs=30, batch_size=256, lr=1e-3):
    dataset = TensorDataset(X, y)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | loss: {total_loss / len(loader):.4f}")


def evaluate(model, n_test: int = 10_000):
    rng = np.random.default_rng(seed=1)
    test_bits = rng.integers(0, 2, size=n_test, dtype=np.int32)
    qstates, true_bits, _ = pub.generate_signals_with_labels(input_data=test_bits)
    xs = [qs.homodyne_measurement() for qs in qstates]
    X_test = torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(1)

    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1).numpy()

    ber = np.mean(preds != true_bits)
    print(f"\n─── PyTorch MLP 盗聴評価 ───")
    print(f"基底数(BNum)   : {pub.BNum}")
    print(f"評価シンボル数 : {n_test}")
    print(f"盗聴者BER      : {ber:.4f}")
    print(f"ランダム推測   : 0.5000")
    return ber


if __name__ == "__main__":
    print("学習データ生成中...")
    X, y = collect_training_data(50_000)

    model = EveMLP()
    print("学習中...")
    train(model, X, y)

    evaluate(model, n_test=10_000)
