import numpy as np
from matplotlib import pyplot as plt

from pyphysim.modulators.fundamental import BPSK, QAM, QPSK

np.set_printoptions(precision=2, linewidth=120)


def main():
    # Modulators
    bpsk = BPSK()
    qpsk = QPSK()
    qam16 = QAM(16)
    qam64 = QAM(64)

    # Figure
    fig, [[ax11, ax12], [ax21, ax22]] = plt.subplots(
        figsize=(10, 10),
        nrows=2,
        ncols=2
    )

    # BPSK
    ax11.set_title("BPSK")
    ax11.plot(bpsk.symbols.real, bpsk.symbols.imag, "r*", label="BPSK")
    ax11.axis("equal")
    ax11.grid(True)

    # QPSK
    ax12.set_title("QPSK")
    ax12.plot(qpsk.symbols.real, qpsk.symbols.imag, "r*", label="QPSK")
    ax12.axis("equal")
    ax12.grid(True)

    # 16-QAM
    ax21.set_title("16-QAM")
    ax21.plot(qam16.symbols.real, qam16.symbols.imag, "r*", label="16-QAM")
    ax21.axis("equal")
    ax21.grid(True)

    # 64-QAM
    ax22.set_title("64-QAM")
    ax22.plot(qam64.symbols.real, qam64.symbols.imag, "r*", label="64-QAM")
    ax22.axis("equal")
    ax22.grid(True)

    plt.tight_layout()

    plt.savefig("plot_modulation.png", dpi=300)


if __name__ == "__main__":
    main()
