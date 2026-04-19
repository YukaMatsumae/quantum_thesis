#正弦波の生成(sine_wave_demoに関する)
def sine_wave(f, overSampRate, phase, nCy1):
    import numpy as np
    fs = overSampRate * f #1秒間に何回サンプリングするか。
    t  = np.arange(0, nCy1 * 1/f - 1/fs, 1/fs )
    #1/fsは、サンプリングの間隔
    #1/fは、 1周期の時間
    #nCy1 * 1/fは、波がnCy1個あるときの、nCy1周期の時間 
    g  = np.sin(2 * np.pi * f * t + phase)

    return (t, g)

#正弦波の例、可視化
def sine_wave_demo():
    import numpy as np
    import matplotlib.pyplot as plt 
    from signalgen import sine_wave

    f = 10 #周波数
    overSampRate = 40 #プロットの滑らかさ
    phase = 1 / 3 * np.pi #位相のズレ
    nCy1 = 5 #周期の数
    (t, g) = sine_wave(f, overSampRate, phase, nCy1)

    plt.plot(t, g)
    plt.title("Sine wave f = "+ str(f) + "Hz")
    plt.xlabel("Time[s]")
    plt.ylabel("Amplitude")
    plt.show()
