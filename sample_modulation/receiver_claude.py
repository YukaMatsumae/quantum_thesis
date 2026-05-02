import numpy as np
import publisher_claude as pub
import time

def process_reception(qstates, base_ids):
    """
    受信者(Bob)の判定ロジック:
    送信側と共有している(または同期計算した) base_ids から
    一括でしきい値を計算し、ホモダイン検波を行う。
    """
    # しきい値の計算 (OSK-Not用)
    # thld = S_max / (2 * BNum - 1) * (BNum / 2 + base_id)
    thlds = pub.S_max / (2 * pub.BNum - 1) * (pub.BNum / 2 + base_ids)
    
    # 測定実行
    # 量子状態オブジェクトのメソッド呼び出しは逐次処理
    vals = np.array([qs.homodyne(th) for qs, th in zip(qstates, thlds)], dtype=np.int32)
    
    # OSK-Notの反転処理を戻す
    decoded_bits = (vals + base_ids % 2) % 2
    return decoded_bits

def run_simulation(n_test=100_000):
    print(f"--- 正規受信者(Bob)の通信シミュレーション ---")
    print(f"信号生成中... ({n_test} サンプル)")
    
    # 送信するランダムデータ
    rng = np.random.default_rng(seed=42)
    original_data = rng.integers(0, 2, size=n_test, dtype=np.int32)
    
    # 1. 送信機から量子信号と正解データを取得
    # （実際の受信者は qstates のみを受け取ってから自身のLFSRで鍵を計算・同期させますが、
    # 本シミュレーションでは計算済みの正規の鍵(base_ids)をそのまま利用することで高速化しています）
    t0 = time.perf_counter()
    qstates, true_bits, base_ids_bob = pub.generate_signals_with_labels(original_data)
    
    # 2. 受信機で復調処理
    t1 = time.perf_counter()
    decoded_bits = process_reception(qstates, base_ids_bob)
    t2 = time.perf_counter()
    
    # 3. 誤り率(BER)の計算
    error_rate = np.mean(decoded_bits != true_bits)
    
    print(f"信号生成時間       : {t1 - t0:.4f} 秒")
    print(f"復調にかかった時間 : {t2 - t1:.4f} 秒")
    print(f"基底数(BNum)       : {pub.BNum}")
    print(f"送受信シンボル数   : {n_test}")
    print(f"正規受信者のBER    : {error_rate:.6f}")
    
    print(f"\n───────────────────")
    print(f"検証:")
    print(f" 復調データ(先頭10): {decoded_bits[:10]}")
    print(f" 正解データ(先頭10): {true_bits[:10]}")
    
    return error_rate, decoded_bits

if __name__ == "__main__":
    # 十分なサンプル数で正規受信者の性能を検証
    run_simulation(n_test=100_000)
