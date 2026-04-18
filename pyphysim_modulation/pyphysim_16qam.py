import numpy as np
import matplotlib.pyplot as plt

# PyPhysimのモジュールをインポート
from pyphysim.modulators.fundamental import QAM
from pyphysim.util.conversion import dB2Linear

def main():
    # =========================
    # パラメータ設定
    # =========================
    M = 16                              # 16-QAM
    k = int(np.log2(M))                 # bits per symbol
    Nsym = 2000                         # シンボル数
    EbN0_dB = 15                        # Eb/N0 [dB]

    # =========================
    # PyPhysim 変調器の初期化
    # =========================
    # pyphysimのQAMクラスのインスタンスを生成
    # 自動的に16-QAMの理想的なコンスタレーション（星座点）などが内部に計算されます
    qam16 = QAM(M)

    # =========================
    # 送信シンボル生成 (PyPhysimを利用)
    # =========================
    # pyphysimの変調器(modulateメソッド)は、0からM-1までの整数値を入力とし、
    # シンボルへマッピングします。
    # 例：16QAMの場合、0~15のランダムな整数をNsym個生成する
    data_symbols = np.random.randint(0, M, Nsym)

    # 変調（整数値 -> 複素数データ）
    tx_symbols = qam16.modulate(data_symbols)

    # =========================
    # AWGN（Eb/N0 → ノイズ分散）の付加
    # =========================
    # pyphysimのdB2Linear関数を使ってdBからリニア値(真値)へ変換
    EbN0 = dB2Linear(EbN0_dB)
    EsN0 = EbN0 * k
    noise_variance = 1 / EsN0
    
    # 複素ガウスノイズを生成して送信シンボルに付加
    noise = np.sqrt(noise_variance / 2) * (
        np.random.randn(Nsym) + 1j * np.random.randn(Nsym)
    )
    rx_symbols = tx_symbols + noise

    # =========================
    # 可視化
    # =========================
    plt.figure(figsize=(6, 6))

    # 自分で座標を計算するのではなく、オブジェクトから
    # 理想的なコンスタレーション座標(`qam16.symbols`)を直接引っ張ってきます。
    ideal_symbols = qam16.symbols

    plt.plot(
        ideal_symbols.real,
        ideal_symbols.imag,
        'rx',
        c="red",
        markersize=10,
        label='PyPhysim Ideal constellation'
    )

    plt.plot(
        rx_symbols.real,
        rx_symbols.imag,
        '.',
        c="blue",
        markersize=5,
        alpha=0.4,
        label='Received (AWGN)'
    )

    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('In-phase')
    plt.ylabel('Quadrature')
    plt.title(f'PyPhysim 16-QAM with AWGN (Eb/N0 = {EbN0_dB} dB)')
    plt.legend()

    # =========================
    # 保存
    # =========================
    plt.tight_layout()
    output_filename = "constellation_16qam_awgn_pyphysim.png"
    plt.savefig(output_filename, dpi=150)
    plt.close()

    print(f"✅ {output_filename} を保存しました")

if __name__ == "__main__":
    main()
