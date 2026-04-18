import numpy as np
import publisher_OSKNot as pub # 送信機モジュールをインポート

def eve_decision(q_state):
    """
    盗聴者の意思決定: 
    鍵を知らないため、全信号レベル(2 * BNum)の中から
    ホモダイン測定値に最も近いものを選択する。
    """
    # 1. ホモダイン測定（Eveはしきい値を知らないため、通常は0などの固定値か、そのままの値を測定）
    # ここでは、実装されている homodyne_measurement() を呼び出す
    x = q_state.homodyne_measurement()
    
    # 2. 全信号レベルとの距離を計算 (ベクトル化して高速化)
    # pub.S_levels は送信側で定義された全ての信号強度の配列
    distances = np.abs(x - pub.S_levels)
    
    # 3. 最も距離が近いインデックスを返す
    return np.argmin(distances)

def run_eavesdropper():
    # 1. 送信機から量子信号を取得
    output_qstates = pub.generate_signals()
    
    # 送信された元のデータ（誤り率計算用）
    # publisher側で input_Data = np.zeros(M) となっていることを想定
    original_data = np.zeros(pub.M) 
    
    eve_output_vals = []

    # 2. 盗聴ループ
    for j in range(pub.M):
        # 信号から最も可能性の高いレベル(ret)を推測
        ret = eve_decision(output_qstates[j])
        
        # 推測したレベルから、base_id と bit_val を抽出
        # 信号配置の仕様: output_level = S_levels[base_id + BNum * index]
        base_id_guess = ret % pub.BNum
        index_guess = 1 if ret >= pub.BNum else 0
        
        # OSKNotの反転ロジックを逆算して、元のデータを推定
        # 送信側: index = (data + base_id % 2) % 2
        # 逆算: data = (index + base_id % 2) % 2
        bit_val = (index_guess + base_id_guess % 2) % 2
        
        eve_output_vals.append(bit_val)

    # 3. 誤り率（BER）の計算
    # 元データが0以外の場合でも対応できるように比較演算を使用
    error_rate = np.mean(np.array(eve_output_vals) != original_data)
    return error_rate

if __name__ == "__main__":
    ber = run_eavesdropper()
    print(f"--- 盗聴シミュレーション結果 ---")
    print(f"基底数(BNum): {pub.BNum}")
    print(f"盗聴者のビット誤り率(BER): {ber:.4f}")
