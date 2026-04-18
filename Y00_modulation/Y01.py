import numpy as np
import matplotlib.pyplot as plt

# 1. 論文に基づくパラメータ設定
M = 2048           # 基底の数
r = 2.0            # 電力比 P_2M / P_1
P_avg_dbm = 6.0    # 平均電力 6dBm
num_samples = 1000 # シミュレーションサンプル数

# 電力の物理量計算 (mW)
P_avg = 10**(P_avg_dbm / 10)
P1 = (2 * P_avg) / (r + 1)
P2M = r * P1
delta_P = (P2M - P1) / (2 * M - 1)

# 全強度レベルの生成 (P1 ~ P_2M)
P = np.array([P1 + i * delta_P for i in range(2 * M)])

# 2. 送信データと実行鍵の生成
np.random.seed(42)
bits = np.random.randint(0, 2, num_samples)    # 平文 b_n
keys = np.random.randint(0, M, num_samples)    # 実行鍵 S_n (0 ~ M-1)

# 3. マッピング (OSK: Overlap Selection Keying) [cite: 206, 207, 208]
tx_signals = np.zeros(num_samples)
for n in range(num_samples):
    sn = keys[n]
    bn = bits[n]
    # 基底番号が奇数か偶数かでマッピングを反転させる (論文図6(d)参照)
    if (sn + 1) % 2 != 0: # 奇数基底
        tx_signals[n] = P[sn] if bn == 0 else P[M + sn]
    else: # 偶数基底
        tx_signals[n] = P[M + sn] if bn == 0 else P[sn]

# 4. 量子ゆらぎ（ショットノイズ）の追加 [cite: 235, 480]
# 論文の実測値に近いノイズ強度をシミュレート
sigma_shot = 0.015  # 視覚的に分かりやすくするため調整
noise = np.random.normal(0, sigma_shot, num_samples)
rx_signals = tx_signals + noise

# 5. 可視化
plt.figure(figsize=(14, 8))

# --- Plot 1: 盗聴者の視点 (全ての信号が重なって見える) ---
plt.subplot(2, 2, 1)
plt.scatter(range(100), rx_signals[:100], c='gray', alpha=0.6, s=10)
plt.title("Eavesdropper's View (All overlapping levels)")
plt.ylabel("Intensity (mW)")
plt.grid(True, alpha=0.3)

# --- Plot 2: 正当な受信者の視点 (特定の基底 Sn=100 に着目) ---
plt.subplot(2, 2, 2)
target_basis = 100
idx = np.where(keys == target_basis)[0]
# ビット0と1を色分け
plt.scatter(idx, rx_signals[idx], c=bits[idx], cmap='bwr', s=30, edgecolors='k')
# 判定閾値を描画 
threshold = (P[target_basis] + P[M + target_basis]) / 2
plt.axhline(threshold, color='green', linestyle='--', label='Threshold')
plt.title(f"Authorized User's View (Fixed Basis Sn={target_basis})")
plt.legend(['Threshold', 'Bit 0', 'Bit 1'])
plt.grid(True, alpha=0.3)

# --- Plot 3: 時間波形と動的閾値 (最初の20サンプル) ---
plt.subplot(2, 1, 2)
time_axis = np.arange(20)
plt.step(time_axis, rx_signals[:20], where='mid', label='Received Signal', color='black', alpha=0.8)
# 各ビットごとの動的な閾値をプロット
dynamic_thresholds = [(P[keys[n]] + P[M + keys[n]]) / 2 for n in range(20)]
plt.step(time_axis, dynamic_thresholds, where='mid', label='Key-based Threshold', color='red', linestyle='--')
plt.title("Time-domain Waveform with Dynamic Key-based Threshold")
plt.xlabel("Time Slot")
plt.ylabel("Intensity (mW)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
