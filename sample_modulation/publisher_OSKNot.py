import numpy as np
import Qstate

# 共有パラメータ
N = 8
M = 1000
S_max = 199
BNum = 2**N
S_levels = np.linspace(0, S_max, BNum*2)

def generate_signals():
    """量子信号を生成してリストで返す関数"""
    register = np.array([1,0,1,0,1,1,0,0,1,1,1,0,0,0,0,1])
    input_Data = np.zeros(M)
    output_qstates = []
    a = np.zeros(N)

    for j in range(M):
        for i in range(N):
            SR = (register[10]+register[12]+register[13]+register[15])%2 
            a[i] = register[15] 
            register = np.roll(register, 1) 
            register[0] = SR
            
        d = 0
        p = 1
        for i in range(N):
            d += p * a[i]
            p *= 2 
        base_id = int(d)
        
        index = (int(input_Data[j]) + base_id % 2) % 2
        output_level = S_levels[base_id + BNum * index]
        q_state = Qstate.Qstate(output_level)
        output_qstates.append(q_state)
        
    return output_qstates

# 直接実行した時だけ動く（import時には動かない）
if __name__ == "__main__":
    print(f"{M}個の信号を生成しました。")
