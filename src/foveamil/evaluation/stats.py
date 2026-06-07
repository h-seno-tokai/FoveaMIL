"""交差検証の指標に対する区間推定と有意差検定

fold 間の平均に対する信頼区間（t 分布・ブートストラップ），2 手法の対比較
（Wilcoxon 符号順位）と，交差検証の fold 間相関を補正した対 t 検定
（Nadeau-Bengio）を提供する標本が少ない/差が全て 0 等の縮退時は ``nan`` を返し
例外を投げない
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

# 既定の有意水準
DEFAULT_ALPHA = 0.05
# 既定のブートストラップ反復数
DEFAULT_N_BOOT = 10000
# 区間推定に最低限必要な標本数
MIN_SAMPLES = 2
# 並べ替え検定の既定反復数
DEFAULT_N_PERM = 10000
# 検定の既定乱数シード（決定的）
DEFAULT_SEED = 0
# f1_score のゼロ割時の値
ZERO_DIVISION = 0

_NAN = float("nan")


def _group_f1(
    y_true: np.ndarray, y_pred: np.ndarray, class_indices: Sequence[int]
) -> float:
    """指定クラス集合の非加重平均 F1（OvR・ゼロ割は 0）を返す空標本/空集合は nan"""
    labels = list(class_indices)
    if not labels or y_true.size == 0:
        return _NAN
    per_class = f1_score(
        y_true=y_true, y_pred=y_pred, labels=labels,
        average=None, zero_division=ZERO_DIVISION,
    )
    return float(np.mean(per_class))


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


def paired_group_f1_permutation_test(
    y_true: Sequence[int],
    y_pred_a: Sequence[int],
    y_pred_b: Sequence[int],
    class_indices: Sequence[int],
    n_perm: int = DEFAULT_N_PERM,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """対応ありプール group-F1 差の並べ替え検定

    検定単位は症例（スライド）であり ``y_true`` / ``y_pred_a`` / ``y_pred_b`` は
    同一テスト症例集合に対する対応した列である観測統計量は手法 A と B の
    プール group-F1 の差 ``gF1(y_true, y_pred_a) - gF1(y_true, y_pred_b)``
    （症例をまたいでプールした上で算出する単一値）

    帰無仮説は「各症例において手法 A と B の予測は交換可能（同分布）」である
    対応ありのため真ラベルを混ぜるのではなく，症例ごとに A と B の予測の割り当てを
    ランダムに入れ替えて（symmetric な符号入替）帰無分布を構成するこれにより
    各症例の正解と 2 予測の対応を保ったまま「どちらが A か」のみを帰無の下で
    入れ替える両側 p は ``|perm 統計量| >= |観測統計量|`` の割合（観測自身を 1 件
    含める）として返す

    縮退（症例 0・空クラス集合・観測/全 perm が nan）では p を nan にし例外を投げない

    Args:
        y_true: 対応する正解クラス ``[N]``
        y_pred_a: 手法 A の予測クラス ``[N]``
        y_pred_b: 手法 B の予測クラス ``[N]``
        class_indices: group-F1 を構成するクラス index 集合
        n_perm: 並べ替え反復数
        seed: 乱数シード（再現性のため固定）

    Returns:
        ``{"statistic", "pvalue", "n", "n_perm"}``
        （``statistic`` は観測 group-F1 差，``n`` は症例数）
    """
    yt = np.asarray(y_true)
    ya = np.asarray(y_pred_a)
    yb = np.asarray(y_pred_b)
    n = int(min(yt.size, ya.size, yb.size))
    yt, ya, yb = yt[:n], ya[:n], yb[:n]

    observed = (
        _group_f1(yt, ya, class_indices) - _group_f1(yt, yb, class_indices)
    )
    if n < 1 or not list(class_indices) or np.isnan(observed):
        return {"statistic": observed, "pvalue": _NAN, "n": n, "n_perm": 0}

    rng = np.random.default_rng(seed)
    abs_obs = abs(observed)
    ge = 1  # 観測自身を帰無分布に含める
    total = 1
    for _ in range(n_perm):
        swap = rng.integers(0, 2, size=n).astype(bool)
        perm_a = np.where(swap, yb, ya)
        perm_b = np.where(swap, ya, yb)
        stat = (
            _group_f1(yt, perm_a, class_indices)
            - _group_f1(yt, perm_b, class_indices)
        )
        if np.isnan(stat):
            continue
        total += 1
        if abs(stat) >= abs_obs:
            ge += 1
    pvalue = float(ge / total)
    return {"statistic": float(observed), "pvalue": pvalue, "n": n, "n_perm": n_perm}


def stratified_bootstrap_group_f1_ci(
    y_true: Sequence[int],
    y_pred_a: Sequence[int],
    class_indices: Sequence[int],
    y_pred_b: Optional[Sequence[int]] = None,
    alpha: float = DEFAULT_ALPHA,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """クラス層化ブートストラップによる group-F1（または差）のパーセンタイル CI

    真ラベルでクラス層化して症例を復元抽出し，各 bootstrap 標本で group-F1 を
    再算出して経験分布の ``(alpha/2, 1-alpha/2)`` パーセンタイルを CI とする
    層化により各 bootstrap 標本でクラス頻度を元と概ね保つ``y_pred_b`` を渡すと
    手法 A と B の group-F1 差（対応あり・同一抽出症例で両手法を評価）の CI を返す

    縮退（症体 0・空クラス集合・点推定が nan）では CI を nan にし例外を投げない

    Args:
        y_true: 正解クラス ``[N]``
        y_pred_a: 手法 A の予測クラス ``[N]``
        class_indices: group-F1 を構成するクラス index 集合
        y_pred_b: 差の CI を取る場合の手法 B の予測クラス ``[N]``（None で単独）
        alpha: 有意水準
        n_boot: 再標本化の反復数
        seed: 乱数シード（再現性のため固定）

    Returns:
        ``{"estimate", "ci_low", "ci_high", "n", "n_boot"}``
        （``estimate`` は点推定値差の場合は A-B）
    """
    yt = np.asarray(y_true)
    ya = np.asarray(y_pred_a)
    arrays = [yt, ya]
    yb = None
    if y_pred_b is not None:
        yb = np.asarray(y_pred_b)
        arrays.append(yb)
    n = int(min(a.size for a in arrays))
    yt, ya = yt[:n], ya[:n]
    if yb is not None:
        yb = yb[:n]

    def _stat(idx: np.ndarray) -> float:
        a = _group_f1(yt[idx], ya[idx], class_indices)
        if yb is None:
            return a
        return a - _group_f1(yt[idx], yb[idx], class_indices)

    estimate = _stat(np.arange(n))
    if n < 1 or not list(class_indices) or np.isnan(estimate):
        return {
            "estimate": estimate, "ci_low": _NAN, "ci_high": _NAN,
            "n": n, "n_boot": 0,
        }

    # クラスごとの症例 index（層化抽出のため）
    strata = [np.where(yt == c)[0] for c in np.unique(yt)]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        picks = [s[rng.integers(0, s.size, size=s.size)] for s in strata]
        idx = np.concatenate(picks)
        stat = _stat(idx)
        if not np.isnan(stat):
            boot.append(stat)
    if not boot:
        return {
            "estimate": float(estimate), "ci_low": _NAN, "ci_high": _NAN,
            "n": n, "n_boot": n_boot,
        }
    low = float(np.percentile(boot, 100.0 * alpha / 2.0))
    high = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "estimate": float(estimate), "ci_low": low, "ci_high": high,
        "n": n, "n_boot": n_boot,
    }
