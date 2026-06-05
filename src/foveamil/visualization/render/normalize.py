"""アテンションスカラを配色前に ``[0, 1]`` へ写す正規化（純関数）

順位パーセンタイル化（症例間・パッチ数の違いに依らずコントラストを揃える）と
min-max を持つ正規化の種別と基準値を :class:`NormResult` に保持し，共有カラーバー用に
複数系列の min/max を :func:`shared_scale` で集約する値の意味は変えない純変換で
matplotlib に依存しない
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import rankdata

# 正規化の種別
NORM_PERCENTILE = "percentile"
NORM_MINMAX = "minmax"
NORM_RAW = "raw"
# min-max のゼロ幅回避用の最小値
_EPS = 1e-12


@dataclass
class NormResult:
    """正規化結果

    Attributes:
        values01: ``[0, 1]`` に写したスカラ列
        kind: 正規化の種別（``percentile`` / ``minmax`` / ``raw``）
        vmin: 正規化前の最小（カラーバー基準）
        vmax: 正規化前の最大（カラーバー基準）
    """

    values01: np.ndarray
    kind: str
    vmin: float
    vmax: float


def to_percentile(values: Sequence[float]) -> np.ndarray:
    """順位を ``[0, 1]`` に写す（``rank / N``）同値は平均順位"""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    return rankdata(arr) / arr.size


def to_minmax(values: Sequence[float]) -> np.ndarray:
    """min-max で ``[0, 1]`` に写す（幅 0 は全て 0）"""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < _EPS:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def normalize(values: Sequence[float], kind: str = NORM_PERCENTILE) -> NormResult:
    """``kind`` に応じてスカラを ``[0, 1]`` へ正規化し :class:`NormResult` を返す"""
    arr = np.asarray(values, dtype=float)
    vmin = float(arr.min()) if arr.size else 0.0
    vmax = float(arr.max()) if arr.size else 0.0
    if kind == NORM_PERCENTILE:
        values01 = to_percentile(arr)
    elif kind == NORM_MINMAX:
        values01 = to_minmax(arr)
    elif kind == NORM_RAW:
        values01 = np.clip(arr, 0.0, 1.0)
    else:
        raise ValueError(f"unknown norm kind: {kind}")
    return NormResult(values01=values01, kind=kind, vmin=vmin, vmax=vmax)


def shared_scale(series: Sequence[Sequence[float]]) -> tuple:
    """複数系列をまたいだ ``(vmin, vmax)`` を返す（共有カラーバー用）"""
    arrays = [np.asarray(s, dtype=float) for s in series if len(s)]
    if not arrays:
        return 0.0, 0.0
    flat = np.concatenate(arrays)
    return float(flat.min()), float(flat.max())
