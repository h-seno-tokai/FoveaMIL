"""交差検証の指標に対する区間推定と有意差検定

fold 間の平均に対する信頼区間（t 分布・ブートストラップ），2 手法の対比較
（Wilcoxon 符号順位）と，交差検証の fold 間相関を補正した対 t 検定
（Nadeau-Bengio）を提供する標本が少ない/差が全て 0 等の縮退時は ``nan`` を返し
例外を投げない
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from scipy import stats

# 既定の有意水準
DEFAULT_ALPHA = 0.05
# 既定のブートストラップ反復数
DEFAULT_N_BOOT = 10000
# 区間推定に最低限必要な標本数
MIN_SAMPLES = 2

_NAN = float("nan")


def mean_ci_t(
    values: Sequence[float], alpha: float = DEFAULT_ALPHA
) -> Tuple[float, float, float]:
    """平均と t 分布ベースの ``(1-alpha)`` 信頼区間を返す

    Args:
        values: 標本（fold ごとの指標）
        alpha: 有意水準

    Returns:
        ``(mean, ci_low, ci_high)``標本が 2 未満なら区間は ``nan``
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(np.mean(arr)) if n else _NAN
    if n < MIN_SAMPLES:
        return mean, _NAN, _NAN
    sem = float(stats.sem(arr))
    half = sem * float(stats.t.ppf(1.0 - alpha / 2.0, n - 1))
    return mean, mean - half, mean + half


def mean_ci_bootstrap(
    values: Sequence[float],
    alpha: float = DEFAULT_ALPHA,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """平均とパーセンタイル・ブートストラップ信頼区間を返す

    Args:
        values: 標本（fold ごとの指標）
        alpha: 有意水準
        n_boot: 再標本化の反復数
        seed: 乱数シード（再現性のため固定）

    Returns:
        ``(mean, ci_low, ci_high)``標本が 2 未満なら区間は ``nan``
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(np.mean(arr)) if n else _NAN
    if n < MIN_SAMPLES:
        return mean, _NAN, _NAN
    rng = np.random.default_rng(seed)
    resampled = arr[rng.integers(0, n, size=(n_boot, n))]
    boot_means = resampled.mean(axis=1)
    low = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return mean, low, high


def wilcoxon_signed_rank(
    a: Sequence[float], b: Sequence[float]
) -> Dict[str, Any]:
    """対応のある 2 標本に Wilcoxon 符号順位検定を行う

    差が全て 0 や標本不足の縮退時は ``nan`` を返す（例外を投げない）

    Args:
        a: 手法 A の fold ごとの指標
        b: 手法 B の fold ごとの指標（``a`` と同長対応）

    Returns:
        ``{"statistic", "pvalue", "n"}``
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    n = int(min(arr_a.size, arr_b.size))
    diffs = arr_a[:n] - arr_b[:n]
    if n < 1 or np.count_nonzero(diffs) < 1:
        return {"statistic": _NAN, "pvalue": _NAN, "n": n}
    try:
        statistic, pvalue = stats.wilcoxon(arr_a[:n], arr_b[:n])
    except Exception:  # noqa: BLE001 - 縮退ケースは nan
        return {"statistic": _NAN, "pvalue": _NAN, "n": n}
    return {"statistic": float(statistic), "pvalue": float(pvalue), "n": n}


def nadeau_bengio_corrected_t(
    diffs: Sequence[float],
    n_train: int,
    n_test: int,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """交差検証の fold 間相関を補正した対応 t 検定（Nadeau-Bengio）

    通常の対 t 検定は CV の訓練集合の重なりで分散を過小評価する補正分散
    ``var(diffs) * (1/k + n_test/n_train)`` を用いて検定統計量と区間を求める

    Args:
        diffs: fold ごとの指標差（手法 A - 手法 B）
        n_train: 1 fold の訓練サンプル数
        n_test: 1 fold の test サンプル数
        alpha: 有意水準

    Returns:
        ``{"t", "pvalue", "df", "mean_diff", "ci_low", "ci_high"}``
    """
    arr = np.asarray(diffs, dtype=float)
    k = arr.size
    mean_diff = float(np.mean(arr)) if k else _NAN
    if k < MIN_SAMPLES:
        return {
            "t": _NAN, "pvalue": _NAN, "df": k - 1,
            "mean_diff": mean_diff, "ci_low": _NAN, "ci_high": _NAN,
        }

    variance = float(np.var(arr, ddof=1))
    correction = (1.0 / k) + (float(n_test) / float(n_train))
    corrected_var = variance * correction
    df = k - 1

    if corrected_var <= 0.0:
        # 差が全て同値補正分散 0
        t_stat = 0.0 if mean_diff == 0.0 else math.inf * math.copysign(1.0, mean_diff)
        pvalue = 1.0 if mean_diff == 0.0 else 0.0
        return {
            "t": t_stat, "pvalue": pvalue, "df": df,
            "mean_diff": mean_diff, "ci_low": mean_diff, "ci_high": mean_diff,
        }

    se = math.sqrt(corrected_var)
    t_stat = mean_diff / se
    pvalue = float(2.0 * stats.t.sf(abs(t_stat), df))
    half = se * float(stats.t.ppf(1.0 - alpha / 2.0, df))
    return {
        "t": float(t_stat), "pvalue": pvalue, "df": df,
        "mean_diff": mean_diff, "ci_low": mean_diff - half, "ci_high": mean_diff + half,
    }


def repeated_cv_corrected_t(
    diffs: Sequence[float],
    n_train: int,
    n_test: int,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """反復交差検証の指標差に対する補正リサンプル t 検定（Bouckaert-Frank）

    複数 seed × fold の全リサンプル（``m = len(diffs)`` 個）の対応差を 1 標本として
    扱い，補正分散 ``var(diffs) * (1/m + n_test/n_train)`` ・自由度 ``m-1`` で検定する
    （corrected resampled t-test）``diffs`` の並び順は問わない（seed と fold を平坦化
    した全差を渡してよい）

    Nadeau-Bengio の単一 CV 版が分母に fold 数 ``k`` を置くのに対し，本式は総リサンプル
    数 ``m`` を置く反復で ``m`` が増えても訓練集合の重なり由来の項 ``n_test/n_train``
    は残るため，平坦な対応 t（補正なし ``var/m``）より分散を大きく見積もり保守的になる
    重なりを無視して標本数だけ増やすと第一種の過誤を過大に出すのを防ぐ

    限界: 本式は訓練集合の重なり由来の単一相関 ``n_test/n_train`` のみを補正し，
    seed×fold の二段（ブロック）相関は補正しない seed 効果が支配的な場合は
    anti-conservative になり（第一種の過誤が ``alpha`` を超え）得る 主張に用いる p は
    プール予測上の並べ替え/ブートストラップ か seed 単位に集計した二段検定で併せて確認する

    縮退（標本 2 未満・全差同値で補正分散 0）では ``nan`` を返し例外を投げない

    Args:
        diffs: 全リサンプルの指標差（手法 A - 手法 B）を平坦化した列
        n_train: 1 リサンプルの訓練サンプル数
        n_test: 1 リサンプルの test サンプル数
        alpha: 有意水準

    Returns:
        ``{"t", "pvalue", "df", "mean_diff", "ci_low", "ci_high", "m"}``
    """
    arr = np.asarray(diffs, dtype=float)
    m = arr.size
    mean_diff = float(np.mean(arr)) if m else _NAN
    if m < MIN_SAMPLES:
        return {
            "t": _NAN, "pvalue": _NAN, "df": m - 1,
            "mean_diff": mean_diff, "ci_low": _NAN, "ci_high": _NAN, "m": m,
        }

    variance = float(np.var(arr, ddof=1))
    correction = (1.0 / m) + (float(n_test) / float(n_train))
    corrected_var = variance * correction
    df = m - 1

    if corrected_var <= 0.0:
        return {
            "t": _NAN, "pvalue": _NAN, "df": df,
            "mean_diff": mean_diff, "ci_low": _NAN, "ci_high": _NAN, "m": m,
        }

    se = math.sqrt(corrected_var)
    t_stat = mean_diff / se
    pvalue = float(2.0 * stats.t.sf(abs(t_stat), df))
    half = se * float(stats.t.ppf(1.0 - alpha / 2.0, df))
    return {
        "t": float(t_stat), "pvalue": pvalue, "df": df,
        "mean_diff": mean_diff, "ci_low": mean_diff - half,
        "ci_high": mean_diff + half, "m": m,
    }


# 多重比較補正法
ADJUST_HOLM = "holm"
ADJUST_FDR_BH = "fdr_bh"
ADJUST_METHODS = (ADJUST_HOLM, ADJUST_FDR_BH)


def adjust_pvalues(
    pvalues: Sequence[float],
    method: str = ADJUST_HOLM,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """複数の p 値に多重比較補正をかける

    ``holm``（Holm-Bonferroni・FWER 制御）または ``fdr_bh``（Benjamini-Hochberg・
    FDR 制御）``nan`` の p 値は族から除外し ``nan`` のまま返す（縮退検定を族に入れない）
    補正後 p は入力順で返し ``reject`` は補正後 p ≤ alpha の真偽列

    Args:
        pvalues: 補正対象の p 値列（``nan`` 可）
        method: ``"holm"`` / ``"fdr_bh"``
        alpha: 有意水準

    Returns:
        ``{"adjusted", "reject", "method", "n"}``（n は有効族サイズ）
    """
    if method not in ADJUST_METHODS:
        raise ValueError(
            f"method must be one of {ADJUST_METHODS}, got '{method}'"
        )
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(p.shape, _NAN, dtype=float)
    idx = np.where(~np.isnan(p))[0]
    m = int(idx.size)
    if m == 0:
        return {
            "adjusted": adjusted.tolist(),
            "reject": [False] * int(p.size),
            "method": method,
            "n": 0,
        }
    pv = p[idx]
    order = np.argsort(pv, kind="stable")
    ps = pv[order]
    adj = np.empty(m, dtype=float)
    if method == ADJUST_HOLM:
        running = 0.0  # step-down: 昇順に (m-i)*p の累積最大
        for i in range(m):
            running = max(running, (m - i) * ps[i])
            adj[i] = min(running, 1.0)
    else:  # fdr_bh step-up: 降順に (m/(i+1))*p の累積最小
        running = 1.0
        for i in range(m - 1, -1, -1):
            running = min(running, (m / (i + 1)) * ps[i])
            adj[i] = min(running, 1.0)
    restored = np.empty(m, dtype=float)
    restored[order] = adj
    adjusted[idx] = restored
    reject = [
        bool((not np.isnan(adjusted[i])) and adjusted[i] <= alpha)
        for i in range(int(p.size))
    ]
    return {"adjusted": adjusted.tolist(), "reject": reject, "method": method, "n": m}
