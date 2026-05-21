# -*- coding: utf-8 -*-
"""
seed_fixed_attack.py

目的:
  1つの固定されたseed値から生成された通信を大量に傍受し、
  NNが送信ビット(0/1)を直接解読できるかを検証する。

  seed_claude.py（②汎用攻撃）との違い:
  - seed値は各LFSR長ごとに1つ固定
  - Eveはseed値を知らないが、同じseedの通信を大量に傍受する前提
  - NNの出力はseed値ではなく、各シンボルの送信ビット（系列対系列）

  モデル構成:
  Conv1D + BiLSTM + Multi-Head Attention のハイブリッドモデル
  （seed_claude.pyと同じ基本構造だが、出力が系列対系列に変更）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time

# ── 量子ノイズのスケール設定 ──
QUANTUM_NOISE_SCALE = 0.5

# ── LFSRタップ位置の定義 ──
LFSR_TAPS = {
    4:  [3, 2],
    6:  [5, 4],
    8:  [7, 5, 4, 3],
    12: [11, 10, 9, 3],
    16: [15, 13, 12, 10],
}

# ── LFSR長に応じたモデル・学習パラメータの設定 ──
# seed_claude.py と同じ基本構成
# (d_model, lstm_hidden, nhead, lstm_layers, n_train, n_test, seq_len, epochs)
CONFIG = {
    4:  (64,  64,  4, 1,  20_000,  5_000,  64, 80),
    6:  (64,  64,  4, 1,  30_000,  8_000,  80, 100),
    8:  (128, 128, 4, 2,  50_000, 10_000, 128, 120),
    12: (128, 128, 8, 2,  80_000, 15_000, 256, 150),
    16: (256, 256, 8, 3, 100_000, 20_000, 512, 200),
}

# ── 各LFSR長で使用する固定seed値 ──
# 0以外のランダムな値（再現性のため固定値で定義）
FIXED_SEEDS = {
    4:  11,       # 0b1011
    6:  43,       # 0b101011
    8:  173,      # 0b10101101
    12: 2731,     # 0b101010101011
    16: 43691,    # 0b1010101010101011
}


# ══════════════════════════════════════════════
# モデル定義: Conv1D + BiLSTM + Attention
# 系列対系列（各シンボルの送信ビットを推定）
# ══════════════════════════════════════════════

class BitDecoderHybrid(nn.Module):
    """
    固定seed攻撃用の送信ビット解読モデル。
    seed_claude.py の SeedPredictorHybrid と同じ基本構造だが、
    出力が「各時間ステップの送信ビット」に変更。

    入力: (batch, seq_len, 1)  — 量子ノイズ付き観測系列
    出力: (batch, seq_len)     — 各シンボルの送信ビット推定値
    """
    def __init__(self, d_model=128, lstm_hidden=128, nhead=4,
                 lstm_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 1. 多層Conv1Dで局所特徴抽出
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # 2. 双方向LSTM
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # 3. Multi-Head Attention（BiLSTMの出力次元 = lstm_hidden * 2）
        lstm_out_dim = lstm_hidden * 2
        self.attn = nn.MultiheadAttention(
            embed_dim=lstm_out_dim, num_heads=nhead,
            dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(lstm_out_dim)

        # 4. 各ステップごとに1ビットを推定するヘッド
        self.fc_out = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, 1)
        h = x.transpose(1, 2)           # (batch, 1, seq_len)
        h = self.conv_block(h)           # (batch, d_model, seq_len)
        h = h.transpose(1, 2)           # (batch, seq_len, d_model)

        # BiLSTM
        h, _ = self.lstm(h)             # (batch, seq_len, lstm_hidden*2)

        # Self-Attention + 残差接続
        attn_out, _ = self.attn(h, h, h)
        h = self.attn_norm(h + attn_out)

        # 各ステップから1ビットを推定
        out = self.fc_out(h)            # (batch, seq_len, 1)
        return out.squeeze(-1)          # (batch, seq_len)


# ══════════════════════════════════════════════
# データ生成（固定seed版）
# ══════════════════════════════════════════════

def generate_fixed_seed_dataset(n_samples, lfsr_length, seq_len,
                                 noise_scale, fixed_seed):
    """
    固定されたseed値からLFSRを駆動し、送信データをランダムに変えた
    n_samples件の通信を生成する。

    引数:
        n_samples: 生成するサンプル数
        lfsr_length: LFSRのビット長
        seq_len: 1通信あたりのシンボル数
        noise_scale: 量子ノイズの標準偏差
        fixed_seed: 固定するseed値（整数）

    戻り値:
        X: 標準化済み観測系列 (n_samples, seq_len, 1)
        Y: 送信ビット列 (n_samples, seq_len) — これが正解ラベル
    """
    N = 12  # 基底決定に用いるビット数
    BNum = 2 ** N
    S_max = 10.0
    S_levels = np.linspace(0, S_max, BNum * 2)

    taps = LFSR_TAPS[lfsr_length]
    mask = (1 << lfsr_length) - 1

    # ── LFSRを1回だけ回して基底IDを計算（全サンプル共通）──
    reg = fixed_seed
    total_bits = seq_len * N
    lfsr_bits = np.empty(total_bits, dtype=np.uint8)
    for k in range(total_bits):
        sr = 0
        for t in taps:
            sr ^= (reg >> t)
        sr &= 1
        lfsr_bits[k] = (reg >> (lfsr_length - 1)) & 1
        reg = ((reg << 1) & mask) | sr

    # Nビットずつまとめて基底IDを算出
    weights = (1 << np.arange(N)).astype(np.int64)
    bits_reshaped = lfsr_bits.reshape(seq_len, N)
    base_ids = bits_reshaped.dot(weights)  # (seq_len,) — 全サンプル共通

    rng = np.random.default_rng()

    # 送信データをサンプルごとにランダム生成（これが正解ラベル）
    input_data = rng.integers(0, 2, size=(n_samples, seq_len), dtype=np.int64)

    # Y00変調（base_idsは全サンプル共通、input_dataだけ異なる）
    base_ids_2d = np.tile(base_ids, (n_samples, 1))  # (n_samples, seq_len)
    mod_indices = (input_data + base_ids_2d % 2) % 2
    output_levels = S_levels[base_ids_2d + BNum * mod_indices]

    # 量子ノイズの付加
    xs_all = rng.normal(loc=output_levels, scale=noise_scale).astype(np.float32)

    # (バッチ, 系列長, 1) に変形 + 標準化
    X = xs_all.reshape(n_samples, seq_len, 1)
    mean_x = np.mean(X)
    std_x = np.std(X)
    X = (X - mean_x) / (std_x + 1e-8)

    Y = input_data.astype(np.float32)  # 送信ビット列が正解ラベル

    return X, Y


# ══════════════════════════════════════════════
# 学習・評価ループ
# ══════════════════════════════════════════════

def train_and_evaluate(lfsr_len, device):
    """
    指定LFSR長で固定seed攻撃の学習・評価を行い、結果を返す。
    """
    d_model, lstm_hidden, nhead, lstm_layers, n_train, n_test, seq_len, epochs = \
        CONFIG[lfsr_len]
    batch_size = 512
    noise_scale = QUANTUM_NOISE_SCALE
    fixed_seed = FIXED_SEEDS[lfsr_len]

    print(f"  固定seed値: {fixed_seed} (0b{fixed_seed:0{lfsr_len}b})")
    print(f"  パラメータ: d_model={d_model}, lstm_hidden={lstm_hidden}, "
          f"nhead={nhead}, lstm_layers={lstm_layers}")
    print(f"  データ: 学習={n_train}, 評価={n_test}, 系列長={seq_len}, "
          f"エポック={epochs}")
    print(f"  量子ノイズスケール: {noise_scale}")

    # ── データ生成（同じseedで送信データだけ異なる）──
    print("  データ生成中...")
    X_train, Y_train = generate_fixed_seed_dataset(
        n_train, lfsr_len, seq_len, noise_scale, fixed_seed)
    X_test, Y_test = generate_fixed_seed_dataset(
        n_test, lfsr_len, seq_len, noise_scale, fixed_seed)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # ── モデル初期化 ──
    model = BitDecoderHybrid(
        d_model=d_model, lstm_hidden=lstm_hidden,
        nhead=nhead, lstm_layers=lstm_layers
    ).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2)

    # ── Early Stopping 用変数 ──
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_state = None

    train_losses = []
    val_losses = []

    # ── 学習ループ ──
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(by)
        scheduler.step()

        avg_train_loss = total_loss / n_train
        train_losses.append(avg_train_loss)

        # 検証Loss
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                preds = model(bx)
                val_loss += criterion(preds, by).item() * len(by)
                val_count += len(by)
        avg_val_loss = val_loss / val_count
        val_losses.append(avg_val_loss)

        if epoch == 1 or epoch % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch [{epoch:03d}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | LR: {lr_now:.6f}")

        # Early Stopping 判定
        if avg_val_loss < best_val_loss - 1e-5:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early Stopping at epoch {epoch} "
                      f"(patience={patience})")
                break

    # ベストモデルをロード
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # ── 最終評価 ──
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device, non_blocking=True)
            preds = model(bx)
            preds_bin = (preds >= 0.5).int().cpu().numpy()
            all_preds.append(preds_bin)
            all_targets.append(by.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 通信データ解読BER
    ber = np.mean(all_preds != all_targets)

    # シンボルごとのBER（位置別の解読精度）
    per_position_ber = np.mean(all_preds != all_targets, axis=0)

    # 1通信あたりの完全解読率（全ビット正解）
    exact_decode = np.all(all_preds == all_targets, axis=1)
    exact_decode_rate = np.mean(exact_decode)

    print(f"  => 通信データ解読BER: {ber:.4f}")
    print(f"  => 完全解読率: {exact_decode_rate:.4f} "
          f"({np.sum(exact_decode)}/{n_test})")
    print(f"  => ランダム推測との差: {0.5 - ber:+.4f}")

    return {
        'ber': ber,
        'exact_decode_rate': exact_decode_rate,
        'per_position_ber': per_position_ber,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'lfsr_len': lfsr_len,
        'fixed_seed': fixed_seed,
    }


# ══════════════════════════════════════════════
# 可視化
# ══════════════════════════════════════════════

def plot_results(results_list, lfsr_lengths):
    """
    固定seed攻撃の結果をまとめた統合グラフを生成・保存する。
    """
    n = len(lfsr_lengths)
    fig = plt.figure(figsize=(18, 12))

    # ── 上段左: 解読BER vs LFSR長 ──
    ax1 = fig.add_subplot(2, 2, 1)
    bers = [r['ber'] for r in results_list]
    ax1.plot(lfsr_lengths, bers, marker='o', linewidth=2,
             color='tab:red', label='解読BER')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7,
                label='ランダム推測 (0.5)')
    ax1.axhline(y=0.0, color='green', linestyle=':', alpha=0.5,
                label='完全解読 (0.0)')
    ax1.set_xlabel('LFSR Bit Length', fontsize=12)
    ax1.set_ylabel('通信データ解読BER', fontsize=12)
    ax1.set_title('固定seed攻撃: 解読BER vs LFSR長', fontsize=14)
    ax1.set_xticks(lfsr_lengths)
    ax1.set_ylim(-0.05, 0.55)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend()

    # ── 上段右: 完全解読率 vs LFSR長 ──
    ax2 = fig.add_subplot(2, 2, 2)
    exact = [r['exact_decode_rate'] for r in results_list]
    ax2.plot(lfsr_lengths, exact, marker='s', linewidth=2,
             color='tab:blue', label='完全解読率')
    ax2.set_xlabel('LFSR Bit Length', fontsize=12)
    ax2.set_ylabel('完全解読率', fontsize=12)
    ax2.set_title('固定seed攻撃: 完全解読率 vs LFSR長', fontsize=14)
    ax2.set_xticks(lfsr_lengths)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()

    # ── 下段左: 学習曲線 ──
    ax3 = fig.add_subplot(2, 2, 3)
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    for i, r in enumerate(results_list):
        epochs_range = range(1, len(r['train_losses']) + 1)
        ax3.plot(epochs_range, r['train_losses'], color=colors[i],
                 alpha=0.7, label=f'{r["lfsr_len"]}bit Train')
        ax3.plot(epochs_range, r['val_losses'], color=colors[i],
                 linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss (BCE)', fontsize=12)
    ax3.set_title('学習曲線（Train / Val）', fontsize=14)
    ax3.grid(True, linestyle=':', alpha=0.7)
    ax3.legend(fontsize=8, ncol=2)

    # ── 下段右: 位置別BER（各LFSR長の先頭64シンボル）──
    ax4 = fig.add_subplot(2, 2, 4)
    for i, r in enumerate(results_list):
        # 先頭64シンボルの位置別BERを表示
        show_len = min(64, len(r['per_position_ber']))
        ax4.plot(range(show_len), r['per_position_ber'][:show_len],
                 color=colors[i], alpha=0.7, linewidth=0.8,
                 label=f'{r["lfsr_len"]}bit')
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    ax4.set_xlabel('シンボル位置', fontsize=12)
    ax4.set_ylabel('BER', fontsize=12)
    ax4.set_title('位置別の解読BER（先頭64シンボル）', fontsize=14)
    ax4.grid(True, linestyle=':', alpha=0.7)
    ax4.legend(fontsize=8)

    fig.suptitle(
        f'Y00 固定seed攻撃 — 送信ビット直接解読\n'
        f'(量子ノイズスケール = {QUANTUM_NOISE_SCALE})',
        fontsize=16, y=1.01)
    fig.tight_layout()
    output_filename = 'fixed_seed_attack_results.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"グラフを {output_filename} として保存しました。")


# ══════════════════════════════════════════════
# メイン処理
# ══════════════════════════════════════════════

def main():
    lfsr_lengths = [4, 6, 8, 12, 16]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"実行デバイス: {device}")
    print(f"量子ノイズスケール: {QUANTUM_NOISE_SCALE}")
    print(f"検証LFSR長: {lfsr_lengths}")
    print(f"攻撃モード: 固定seed（同一seedの通信を大量傍受）\n")

    results_list = []

    for l_len in lfsr_lengths:
        print(f"=========================================")
        print(f" LFSR長: {l_len} bit (周期: {(1 << l_len) - 1})")
        print(f" 固定seed攻撃を開始")
        print(f"=========================================")

        result = train_and_evaluate(l_len, device)
        results_list.append(result)
        print()

    # ── 結果サマリー ──
    print("=" * 60)
    print(" 固定seed攻撃 — 全結果サマリー")
    print("=" * 60)
    print(f"{'LFSR長':>8} | {'固定seed':>10} | {'解読BER':>8} | "
          f"{'完全解読率':>10} | {'判定':>6}")
    print("-" * 60)
    for r in results_list:
        # BER < 0.45 なら「解読に成功（部分的）」と判定
        status = "成功" if r['ber'] < 0.45 else "失敗"
        print(f"{r['lfsr_len']:>6} bit | {r['fixed_seed']:>10} | "
              f"{r['ber']:>8.4f} | {r['exact_decode_rate']:>10.4f} | "
              f"{status:>6}")

    # ── グラフ描画 ──
    print("\nグラフを描画・保存します...")
    plot_results(results_list, lfsr_lengths)


if __name__ == "__main__":
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print(f"総実行時間: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分)")
