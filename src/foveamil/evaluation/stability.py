"""学習履歴から学習の安定性を診断する

各 fold の ``history.csv`` を読み，終盤エポックの検証指標の標準偏差（振動）・
``val_loss`` 最小後の上昇量（過学習）・best epoch を算出し fold 平均でまとめる
2 構成の比較として，任意指標の per-fold std の分散比を決定的シードのブートストラップで
区間推定する学習や再推論はせず保存済み履歴を読むだけ
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# fold ディレクトリ名の接頭辞
FOLD_DIR_PREFIX = "fold"
# 学習履歴ファイル名
HISTORY_CSV = "history.csv"
# 検証損失の列名
VAL_LOSS_COL = "val_loss"
# エポック列名
EPOCH_COL = "epoch"
# val 指標列の接頭辞（history は val_ 接頭辞付きで保存される）
VAL_PREFIX = "val_"
# 振動を測る終盤エポック数の既定
DEFAULT_TAIL = 5
# ブートストラップ反復数の既定
DEFAULT_N_BOOT = 10000
# 区間推定の既定有意水準
DEFAULT_ALPHA = 0.05

_NAN = float("nan")


def fold_history_paths(combo_dir: str) -> List[str]:
    """combo 配下の各 fold の ``history.csv`` のパスを fold 順に返す"""
    if not os.path.isdir(combo_dir):
        return []
    folds = sorted(
        name for name in os.listdir(combo_dir)
        if name.startswith(FOLD_DIR_PREFIX)
        and os.path.isfile(os.path.join(combo_dir, name, HISTORY_CSV))
    )
    return [os.path.join(combo_dir, name, HISTORY_CSV) for name in folds]


def load_history(path: str) -> Optional[pd.DataFrame]:
    """``history.csv`` を読む無ければ ``None``"""
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _metric_column(history: pd.DataFrame, metric: str) -> Optional[str]:
    """指標名に対応する history 列名を返す（``val_`` 接頭辞も探す）"""
    if metric in history.columns:
        return metric
    prefixed = f"{VAL_PREFIX}{metric}"
    if prefixed in history.columns:
        return prefixed
    return None


def tail_std(
    history: pd.DataFrame, metric: str, tail: int = DEFAULT_TAIL
) -> float:
    """終盤 ``tail`` エポックの検証指標の標準偏差（振動の大きさ）を返す

    エポック昇順の末尾 ``tail`` 行を使う（行数が ``tail`` 未満なら全行）有効値が
    2 未満なら ``nan``指標列が無ければ ``nan``

    Args:
        history: 1 fold の学習履歴
        metric: 振動を測る検証指標名（``val_`` 接頭辞は省略可）
        tail: 末尾エポック数

    Returns:
        終盤の標準偏差
    """
    col = _metric_column(history, metric)
    if col is None:
        return _NAN
    ordered = _by_epoch(history)
    values = ordered[col].to_numpy(dtype=float)[-tail:]
    values = values[~np.isnan(values)]
    if values.size < 2:
        return _NAN
    return float(np.std(values, ddof=1))


def post_min_rise(history: pd.DataFrame) -> float:
    """``val_loss`` 最小後の最大上昇量（過学習の度合い）を返す

    ``val_loss`` の最小値を取った後のエポックでの最大値と最小値の差を返す最小が
    末尾（以後の悪化なし）なら 0``val_loss`` 列が無ければ ``nan``

    Args:
        history: 1 fold の学習履歴

    Returns:
        最小後の上昇量（>= 0）
    """
    if VAL_LOSS_COL not in history.columns:
        return _NAN
    ordered = _by_epoch(history)
    loss = ordered[VAL_LOSS_COL].to_numpy(dtype=float)
    if loss.size == 0 or np.all(np.isnan(loss)):
        return _NAN
    min_idx = int(np.nanargmin(loss))
    after = loss[min_idx:]
    after = after[~np.isnan(after)]
    if after.size == 0:
        return _NAN
    return float(np.max(after) - loss[min_idx])


def best_epoch(history: pd.DataFrame) -> float:
    """``val_loss`` が最小となる best epoch を返す

    ``epoch`` 列があればその値，無ければ昇順での行 index を返す``val_loss`` 列が
    無ければ ``nan``

    Args:
        history: 1 fold の学習履歴

    Returns:
        best epoch
    """
    if VAL_LOSS_COL not in history.columns:
        return _NAN
    ordered = _by_epoch(history)
    loss = ordered[VAL_LOSS_COL].to_numpy(dtype=float)
    if loss.size == 0 or np.all(np.isnan(loss)):
        return _NAN
    min_idx = int(np.nanargmin(loss))
    if EPOCH_COL in ordered.columns:
        return float(ordered[EPOCH_COL].to_numpy()[min_idx])
    return float(min_idx)


def _by_epoch(history: pd.DataFrame) -> pd.DataFrame:
    """``epoch`` 列があれば昇順に整列して返す（無ければそのまま）"""
    if EPOCH_COL in history.columns:
        return history.sort_values(EPOCH_COL, kind="stable").reset_index(drop=True)
    return history


def fold_stability(
    history: pd.DataFrame, metric: str, tail: int = DEFAULT_TAIL
) -> Dict[str, float]:
    """1 fold の安定性量（振動 std・最小後上昇・best epoch）を返す

    Args:
        history: 1 fold の学習履歴
        metric: 振動を測る検証指標名
        tail: 振動を測る末尾エポック数

    Returns:
        ``{"tail_std", "post_min_rise", "best_epoch"}``
    """
    return {
        "tail_std": tail_std(history, metric, tail=tail),
        "post_min_rise": post_min_rise(history),
        "best_epoch": best_epoch(history),
    }


def combo_stability(
    combo_dir: str, metric: str, tail: int = DEFAULT_TAIL
) -> Dict[str, Any]:
    """combo の全 fold の安定性量を集め fold 平均でまとめる

    各 fold の振動 std・最小後上昇・best epoch を算出し，``nan`` を除いた fold 平均と
    std を返す履歴が 1 つも読めなければ各平均は ``nan``

    Args:
        combo_dir: fold ディレクトリ群を含む combo ディレクトリ
        metric: 振動を測る検証指標名
        tail: 振動を測る末尾エポック数

    Returns:
        ``{"metric", "tail", "n_folds", "per_fold", "mean", "std"}``
        （``mean``/``std`` は各量ごとの fold 集約）
    """
    per_fold: List[Dict[str, float]] = []
    for path in fold_history_paths(combo_dir):
        history = load_history(path)
        if history is None or history.empty:
            continue
        per_fold.append(fold_stability(history, metric, tail=tail))

    keys = ("tail_std", "post_min_rise", "best_epoch")
    mean: Dict[str, float] = {}
    std: Dict[str, float] = {}
    for key in keys:
        values = [
            row[key] for row in per_fold if not np.isnan(row[key])
        ]
        mean[key] = float(np.mean(values)) if values else _NAN
        std[key] = float(np.std(values)) if values else _NAN

    return {
        "metric": metric,
        "tail": tail,
        "n_folds": len(per_fold),
        "per_fold": per_fold,
        "mean": mean,
        "std": std,
    }


def per_fold_tail_std(
    combo_dir: str, metric: str, tail: int = DEFAULT_TAIL
) -> List[float]:
    """combo の各 fold の終盤 std を fold 順に返す（``nan`` 除外）"""
    values: List[float] = []
    for path in fold_history_paths(combo_dir):
        history = load_history(path)
        if history is None or history.empty:
            continue
        value = tail_std(history, metric, tail=tail)
        if not np.isnan(value):
            values.append(value)
    return values


def variance_ratio_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Dict[str, Any]:
    """2 構成の per-fold 値の分散比 ``var(a)/var(b)`` をブートストラップで区間推定する

    各構成を独立に復元抽出して分散比の標本分布を作り，点推定とパーセンタイル信頼区間を
    返す決定的シードで再現する標本不足（各 2 未満）や ``var(b)==0`` の縮退では
    ``nan`` を返し例外を投げない

    Args:
        a: 構成 A の per-fold 値（例 終盤 std）
        b: 構成 B の per-fold 値
        n_boot: 再標本化の反復数
        alpha: 有意水準
        seed: 乱数シード（再現性のため固定）

    Returns:
        ``{"ratio", "ci_low", "ci_high", "n_a", "n_b"}``
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    n_a, n_b = arr_a.size, arr_b.size
    degenerate = {
        "ratio": _NAN, "ci_low": _NAN, "ci_high": _NAN,
        "n_a": int(n_a), "n_b": int(n_b),
    }
    if n_a < 2 or n_b < 2:
        return degenerate
    var_b = float(np.var(arr_b, ddof=1))
    if var_b <= 0.0:
        return degenerate
    ratio = float(np.var(arr_a, ddof=1) / var_b)

    rng = np.random.default_rng(seed)
    boot_a = arr_a[rng.integers(0, n_a, size=(n_boot, n_a))]
    boot_b = arr_b[rng.integers(0, n_b, size=(n_boot, n_b))]
    var_a_boot = boot_a.var(axis=1, ddof=1)
    var_b_boot = boot_b.var(axis=1, ddof=1)
    valid = var_b_boot > 0.0
    ratios = var_a_boot[valid] / var_b_boot[valid]
    if ratios.size == 0:
        return {**degenerate, "ratio": ratio}
    low = float(np.percentile(ratios, 100.0 * alpha / 2.0))
    high = float(np.percentile(ratios, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "ratio": ratio, "ci_low": low, "ci_high": high,
        "n_a": int(n_a), "n_b": int(n_b),
    }
