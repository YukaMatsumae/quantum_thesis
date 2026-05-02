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

#方形波の生成
def square_wave(f, overSampRate, nCy1):
    import numpy as np

    fs = overSampRate * f
    t  = np.arange(0, nCy1 * 1/f - 1/fs, 1/fs)
    g  = np.sign(np.sin(2 * np.pi * f * t))

    return (t, g)

#方形波の例、可視化
def square_wave_demo():
    import matplotlib.pyplot as plt
    f = 10
    overSampRate = 100
    nCy1 = 5

    (t, g) = square_wave(f, overSampRate, nCy1)
    plt.figure(num = "square_wave_standard" )
    plt.plot(t, g)
    plt.title("Square wave f = "+ str(f) + "Hz")
    plt.xlabel("Time[s]")
    plt.ylabel("Amplitude")
    plt.show()

#矩形波の生成
def rect_pulse(A, fs, T):
    import numpy as np
    t = np.arange(-0.5, 0.5, 1/fs)

    rect = np.where(np.abs(t) < T/2, 1.0, 0.0)  #where(条件、True、False)
    rect[np.isclose(np.abs(t), T/2)] = 0.5 #isclose　数字を丸めてくれる。

    g = A*rect
    return (t, g)

#矩形波の例、可視化
def rect_pulse_demo():
    import matplotlib.pyplot as plt
    A = 1
    fs = 500 #サンプリング周波数
    T = 0.4

    (t, g) = rect_pulse(A, fs, T)
    plt.figure(num = "rect_pulse_standard" )
    plt.plot(t, g)
    plt.xlabel("Time[s]")
    plt.ylabel("Amplitude")
    plt.show()

#ガウシアンパルスの生成
def gaussian_pulse(fs, sigma):
    import numpy as np

    t = np.arange(-0.5, 0.5, 1/fs)
    g = 1 / (np.sqrt(2*np.pi)*sigma) * ( np.exp(-t**2 /(sigma**2)))

    return (t, g)

# ガウシアンパルスの例、可視化
def gaussian_pulse_demo():
    import matplotlib.pyplot as plt

    fs = 500
    sigma = 0.1

    (t, g) = gaussian_pulse(fs, sigma)
    plt.figure(num = "gaussian_pulse_standard" )
    plt.plot(t, g)
    plt.xlabel("Time[s]")
    plt.ylabel("Amplitude")
    plt.show()

#チャープ信号の例、可視化(生成はなし)
def chirp_demo():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import chirp

    fs = 500
    t = np.arange(start = 0, stop = 1, step = 1/fs)
    g = chirp(t, f0 = 1, t1 = 0.5, f1 = 20, phi = 0, method = "linear")
    plt.plot(t, g)
    plt.show()
