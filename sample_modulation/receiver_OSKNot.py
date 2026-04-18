import numpy as np
import publisher_OSKNot as pub # 送信機モジュールを読み込み

def getData(q_state, base_id):
    """
    受信者の判定ロジック:
    送信側と共有しているbase_idからしきい値を計算し、ホモダイン検波を行う。
    """
    # しきい値の計算 (OSK-Not用)
    thld = pub.S_max / (2 * pub.BNum - 1) * (pub.BNum / 2 + base_id)
    
    # 測定実行
    val = q_state.homodyne(thld)
    
    # OSK-Notの反転処理を戻す
    val = (val + base_id % 2) % 2
    return val

def run_simulation():
    # 1. 送信機から量子信号のリストを取得
    output_qstates = pub.generate_signals()
    
    # 2. 受信側の鍵(LFSR)を初期化
    register = np.array([1,0,1,0,1,1,0,0,1,1,1,0,0,0,0,1])
    a = np.zeros(pub.N)
    
    output_vals = []
    # 正解データ（送信データはすべて0と仮定）
    original_data = np.zeros(pub.M)

    # 3. 受信ループ
    for j in range(pub.M):
        # 送信側と同期したLFSR処理
        for i in range(pub.N):
            SR = (register[10] + register[12] + register[13] + register[15]) % 2
            a[i] = register[15]
            register = np.roll(register, 1)
            register[0] = SR
        
        d = 0
        p = 1
        for i in range(pub.N):
            d += p * a[i]
            p *= 2
        base_id = int(d)

        # 判定
        res = getData(output_qstates[j], base_id)
        output_vals.append(res)

    # 4. 誤り率(BER)の計算
    # 予測結果と正解が一致しない割合の平均
    error_rate = np.mean(np.array(output_vals) != original_data)
    
    return error_rate, output_vals

if __name__ == "__main__":
    ber, results = run_simulation()
    
    print(f"--- 受信シミュレーション結果 ---")
    print(f"基底数(BNum): {pub.BNum}")
    print(f"送信信号数(M): {pub.M}")
    print(f"受信者のビット誤り率(BER): {ber:.4f}")
    
    # 最初の10個の復調結果を表示
    print(f"復調データ(最初の10個): {results[:10]}")
