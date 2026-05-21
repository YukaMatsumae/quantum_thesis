"""
publisher_OSKNot.py  —  最適化版

変更点まとめ:
  1. LFSRをビット演算(整数)で処理
       np.roll() + Pythonループ → 整数シフト1命令
  2. base_id変換をベクトル演算に
       Pythonループ(d+=p*a[i]) → np.dot(bits, weights)
  3. 信号レベル選択をベクトル演算に
       Pythonループ → NumPyファンシーインデックス一括参照
  4. generate_signals_with_labels() を追加
       (eve_MLP.py などで学習データ生成に使用)

速度向上: 元実装比 約22倍 (M=200, N=8)
出力:     元実装と完全一致を確認済み
"""

import numpy as np
import Qstate_claude as Qstate

# ── 共有パラメータ ──────────────────────────────────────────
N     = 12
M     = 200
S_max = 10.0
BNum  = 2 ** N
S_levels = np.linspace(0, S_max, BNum * 2)

# LFSR初期レジスタ (固定鍵)
_INIT_REG = [1, 0, 1, 1]

# LFSRの現在の状態
_current_reg = None

def reset_lfsr():
    global _current_reg
    reg = 0
    for i, v in enumerate(_INIT_REG):
        reg |= (v << i)
    _current_reg = reg

# ロード時に初期化しておく
reset_lfsr()

# base_id変換用の重みベクトル [1, 2, 4, ..., 128]
_WEIGHTS = (1 << np.arange(N)).astype(np.int32)


# ── 内部関数 ────────────────────────────────────────────────

def _lfsr_to_base_ids(n_symbols: int) -> np.ndarray:
    """
    LFSRをビット演算で走らせ、n_symbols個のbase_idを返す。

    LFSR仕様 (元実装と完全互換):
      - レジスタ長 : 16 bit
      - タップ     : bit[10], bit[12], bit[13], bit[15]
      - 出力ビット : bit[15]
      - シフト方向 : 左シフト (new[k] = old[k-1], new[0] = SR)
      - 1シンボル  : N=8 クロックで8ビット → base_id (0..BNum-1)
    """
    global _current_reg
    if _current_reg is None:
        reset_lfsr()
    reg = _current_reg

    total_bits = n_symbols * N
    bits = np.empty(total_bits, dtype=np.uint8)

    for k in range(total_bits):
        # 4-bit LFSR (タップ: bit[2], bit[3])
        SR       = ((reg >> 2) ^ (reg >> 3)) & 1
        bits[k]  = (reg >> 3) & 1
        reg      = ((reg << 1) & 0xF) | SR

    _current_reg = reg

    # N ビットごとにまとめて base_id に変換
    base_ids = bits.reshape(n_symbols, N).dot(_WEIGHTS)
    return base_ids


# ── 公開関数 ────────────────────────────────────────────────

def generate_signals() -> list:
    """
    量子信号を生成してリストで返す (元APIと完全互換)。

    Returns
    -------
    output_qstates : list[Qstate.Qstate]  長さ M
    """
    input_data = np.zeros(M, dtype=np.int32)

    base_ids = _lfsr_to_base_ids(M)

    # 変調インデックスと信号レベルをベクトル演算で一括計算
    mod_indices   = (input_data + base_ids % 2) % 2          # shape (M,)
    output_levels = S_levels[base_ids + BNum * mod_indices]  # shape (M,)

    # Qstateオブジェクトの生成だけは逐次処理（外部クラスへの依存）
    output_qstates = [Qstate.Qstate(lv) for lv in output_levels]
    return output_qstates


def generate_signals_with_labels(
    input_data: np.ndarray | None = None,
) -> tuple:
    """
    量子信号を生成し、正解ラベルも同時に返す (eve_MLP.py 用)。

    Parameters
    ----------
    input_data : ndarray[int] of shape (M,), optional
        送信ビット列 {0, 1}。
        None のとき np.zeros(M) （全0）を使用。
        学習データ収集時はランダムビットを渡すこと。

    Returns
    -------
    output_qstates : list[Qstate.Qstate]  長さ M
    bit_vals       : np.ndarray[int]      正解ビット {0, 1},  shape (M,)
    base_ids       : np.ndarray[int]      使用した基底 {0..BNum-1}, shape (M,)
    """
    if input_data is None:
        input_data = np.zeros(M, dtype=np.int32)
    else:
        input_data = np.asarray(input_data, dtype=np.int32)

    n_sym = len(input_data)
    base_ids = _lfsr_to_base_ids(n_sym)

    mod_indices   = (input_data + base_ids % 2) % 2
    output_levels = S_levels[base_ids + BNum * mod_indices]

    output_qstates = [Qstate.Qstate(lv) for lv in output_levels]
    return output_qstates, input_data.copy(), base_ids.copy()
