"""
eve_MLP.py
Y00方式 — MLPベース盗聴者レシーバー

前提:
  publisher_OSKNot.py に以下を追加済みであること:
    generate_signals_with_labels() -> (qstates, bit_vals, base_ids)
  追加方法は末尾の「publisher側への追記例」を参照。

依存:
  numpy, scikit-learn (pip install scikit-learn)
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import publisher_claude as pub


# ──────────────────────────────────────────────
# 1. 学習データ生成
# ──────────────────────────────────────────────

def collect_training_data(n_samples: int = 50_000):
    """
    ランダムな送信ビット列を使って学習データを収集する。

    publisher側の input_Data が np.zeros(M) 固定だと
    ラベルが常に bit=0 になりMLPが学習できないため、
    こちら側でランダムビットを生成して pub に渡す。

    Returns
    -------
    X : ndarray, shape (n_samples, 1)
        ホモダイン測定値（スカラー）
    y : ndarray, shape (n_samples,)
        正解ビット値 {0, 1}
    """
    rng = np.random.default_rng(seed=0)
    
    # 修正: 一括で n_samples 分のランダムビット列を生成して pub に渡す
    # （細切れに取得せず、ひと繋がりの通信傍受をシミュレートするため）
    random_bits = rng.integers(0, 2, size=n_samples, dtype=np.int32)
    qstates, bit_vals, _ = pub.generate_signals_with_labels(input_data=random_bits)
    
    xs = [qs.homodyne_measurement() for qs in qstates]

    X = np.array(xs).reshape(-1, 1)
    y = np.array(bit_vals)

    # 両クラスが含まれているか確認
    unique = np.unique(y)
    if len(unique) < 2:
        raise RuntimeError(
            f"学習データに含まれるクラスが {unique} のみです。"
            "publisher側の generate_signals_with_labels() が "
            "input_data 引数を正しく反映しているか確認してください。"
        )
    return X, y


# ──────────────────────────────────────────────
# 2. MLP盗聴者クラス
# ──────────────────────────────────────────────

class MLPEavesdropper:
    """
    ホモダイン測定値 x を受け取り、bit_val を予測するMLPベースの盗聴者。

    アーキテクチャ:
      入力層 : 1 ノード（x のスカラー値）
      隠れ層 : 128 → 64 → 32  (ReLU)
      出力層 : 2 ノード（softmax 相当、{0,1} の2クラス分類）

    Note:
      Eveは鍵（乱数列 base_id）を持たないため、
      入力は x のみ。MLP が非線形境界を学習することで
      最尤的なビット判定を行う。
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            verbose=False,
        )
        self._trained = False

    # ------------------------------------------
    def train(self, X_raw: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """
        Parameters
        ----------
        X_raw : ndarray, shape (N, 1)   — 生のホモダイン測定値
        y     : ndarray, shape (N,)     — 正解ビット {0, 1}
        test_size : float               — 評価用に分割する割合
        """
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_raw, y, test_size=test_size, random_state=42
        )

        # スケーリング（x の値域は pub 側のパラメータ依存）
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s = self.scaler.transform(X_te)

        print("MLPを学習中...")
        self.model.fit(X_tr_s, y_tr)
        self._trained = True

        # 評価レポート
        y_pred = self.model.predict(X_te_s)
        print("\n─── テストセット評価 ───")
        print(classification_report(
            y_te, y_pred,
            labels=[0, 1],                   # クラスを明示してエラー回避
            target_names=["bit=0", "bit=1"],
        ))

        ber = np.mean(y_pred != y_te)
        return ber

    # ------------------------------------------
    def predict_bit(self, x: float) -> int:
        """
        単一のホモダイン測定値からビット値を予測。

        Parameters
        ----------
        x : float — ホモダイン測定値

        Returns
        -------
        int — 予測ビット {0, 1}
        """
        if not self._trained:
            raise RuntimeError("先に train() を呼び出してください。")
        x_s = self.scaler.transform([[x]])
        return int(self.model.predict(x_s)[0])


# ──────────────────────────────────────────────
# 3. 盗聴シミュレーション本体
# ──────────────────────────────────────────────

def run_mlp_eavesdropper(n_train: int = 50_000):
    """
    1. 学習データを生成
    2. MLP を学習
    3. 新たな信号列に対して盗聴し BER を報告

    Parameters
    ----------
    n_train : int — 学習に使うサンプル数
    """

    # ── 学習フェーズ ──
    print(f"学習データ生成中... ({n_train} サンプル)")
    X_train, y_train = collect_training_data(n_train)

    eve = MLPEavesdropper()
    train_ber = eve.train(X_train, y_train)
    print(f"訓練時テストBER : {train_ber:.4f}")

    # ── 盗聴フェーズ（新鮮な信号列に対して評価）──
    n_test = 10_000
    print(f"\n新規信号列に対して盗聴評価中... ({n_test} サンプル)")
    
    # 修正: オール0ではなく、完全にランダムなテストデータを生成
    rng_test = np.random.default_rng(seed=1)
    test_bits = rng_test.integers(0, 2, size=n_test, dtype=np.int32)
    qstates, true_bits, _ = pub.generate_signals_with_labels(input_data=test_bits)

    predicted_bits = []
    for qs in qstates:
        x = qs.homodyne_measurement()
        predicted_bits.append(eve.predict_bit(x))

    predicted_bits = np.array(predicted_bits)
    true_bits      = np.array(true_bits)

    ber = np.mean(predicted_bits != true_bits)

    print(f"\n─── MLP盗聴シミュレーション結果 ───")
    print(f"基底数(BNum)          : {pub.BNum}")
    print(f"学習サンプル数        : {n_train}")
    print(f"評価シンボル数        : {n_test}")
    print(f"盗聴者BER (MLP)       : {ber:.4f}")
    print(f"比較: 従来手法BER目安 : 0.5000 (ランダム推測)")

    return ber, eve


# ──────────────────────────────────────────────
# 4. エントリポイント
# ──────────────────────────────────────────────

if __name__ == "__main__":
    run_mlp_eavesdropper(n_train=50_000)


# ══════════════════════════════════════════════
# publisher_OSKNot.py への追記例
# ══════════════════════════════════════════════
#
# 以下の関数を publisher_OSKNot.py の末尾に追加してください。
# generate_signals() をベースに、正解ラベルを同時に返します。
#
# def generate_signals_with_labels():
#     """
#     generate_signals() と同様に量子信号列を生成し、
#     正解の bit_vals と base_ids も返す。
#
#     Returns
#     -------
#     output_qstates : list[QuantumState]
#     bit_vals       : list[int]   正解ビット {0, 1}
#     base_ids       : list[int]   使用した基底インデックス {0, ..., BNum-1}
#     """
#     key_sequence  = generate_key()   # 乱数鍵列（base_id列）
#     input_Data    = np.zeros(M, dtype=int)   # 送信データ（全0を想定）
#
#     output_qstates = []
#     bit_vals       = []
#     base_ids       = []
#
#     for j in range(M):
#         base_id = key_sequence[j]
#         data    = int(input_Data[j])
#
#         # OSKNot の変調ロジック（publisher側と同じ）
#         index = (data + base_id % 2) % 2
#         output_level = S_levels[base_id + BNum * index]
#
#         qs = QuantumState(output_level)   # 量子状態オブジェクトを生成
#         output_qstates.append(qs)
#         bit_vals.append(data)
#         base_ids.append(base_id)
#
#     return output_qstates, bit_vals, base_ids