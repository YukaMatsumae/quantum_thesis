import numpy as np
import random
import Qstate

# --- パラメータ設定 ---
N = 8
M = 1000
BNum = 2**N
S_max = 10000
S_levels = np.linspace(0, S_max, BNum*2)

# Mapperの生成 (送受信共通)
# 乱数シードを固定することで、importするたびに同じl2が生成されるようにします
random.seed(42) 
l2 = list(range(BNum))
random.shuffle(l2)

def OSK_function(iD, iO):
    return int((iD + iO) % 2)

def generate_signals():
    register = np.array([1,0,1,0,1,1,0,0,1,1,1,0,0,0,0,1])
    input_Data = np.zeros(M)
    output_qstates = []
    a = np.zeros(N)

    for j in range(M):
        # 1. Mapper用のLFSR処理 (N回)
        for i in range(N):
            SR = (register[10]+register[12]+register[13]+register[15]) % 2
            a[i] = register[15]
            register = np.roll(register, 1)
            register[0] = SR
        
        d = 0
        p = 1
        for i in range(N):
            d += p * a[i]
            p *= 2
        input_Mapper = int(d)
        output_Mapper = l2[input_Mapper]

        # 2. OSK用のLFSR処理 (さらに1回追加)
        SR = (register[10]+register[12]+register[13]+register[15]) % 2 # publisherのコードに準拠
        register = np.roll(register, 1)
        register[0] = SR
        input_OSK = int(register[0])

        # 3. 信号生成
        output_OSK = OSK_function(input_Data[j], input_OSK)
        output_level = S_levels[output_Mapper + BNum * output_OSK]
        
        q_state = Qstate.Qstate(output_level)
        output_qstates.append(q_state)
        
    return output_qstates
