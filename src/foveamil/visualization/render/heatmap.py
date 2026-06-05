"""正規化スカラとパッチ矩形からヒートマップの材料を作る（配列演算）

サムネ画素のキャンバス上で，パッチが存在する位置だけにスカラ場を置き（背景白地は
着色しない），連続カラーマップで RGBA に写す画像の合成（原画像との blend）は別部品が
担い，ここは材料（スカラ場・存在マスク・RGBA）を返すのみカラーマップの LUT には
matplotlib の colormap を用いる
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

# RGBA のチャネル数
_RGBA = 4


def scalar_field(
    canvas_hw: Tuple[int, int],
    rects: Sequence[Tuple[int, int, int, int]],
    scores01: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """パッチ矩形にスコアを置いたスカラ場と存在マスクを返す

    Args:
        canvas_hw: キャンバスの ``(H, W)`` 画素
        rects: パッチ矩形 ``(x, y, w, h)`` の列（サムネ画素・整数）
        scores01: 各パッチの ``[0, 1]`` スコア

    Returns:
        ``(field[H,W] float, mask[H,W] bool)``重なりはスコア最大を採る
    """
    height, width = canvas_hw
    field = np.zeros((height, width), dtype=float)
    mask = np.zeros((height, width), dtype=bool)
    for (x, y, w, h), score in zip(rects, scores01):
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(width, int(x) + int(w)), min(height, int(y) + int(h))
        if x1 <= x0 or y1 <= y0:
            continue
        region = field[y0:y1, x0:x1]
        np.maximum(region, float(score), out=region)
        mask[y0:y1, x0:x1] = True
    return field, mask


def to_rgba(field: np.ndarray, mask: np.ndarray, cmap_name: str) -> np.ndarray:
    """スカラ場を連続カラーマップで RGBA に写す存在しない画素は alpha 0

    Args:
        field: スカラ場 ``[H, W]``（``[0, 1]`` 前提）
        mask: 存在マスク ``[H, W]``
        cmap_name: カラーマップ名

    Returns:
        ``rgba[H, W, 4] float``（``[0, 1]``）背景は alpha 0
    """
    import matplotlib

    cmap = matplotlib.colormaps[cmap_name]
    rgba = cmap(np.clip(field, 0.0, 1.0))
    rgba[..., 3] = mask.astype(float)
    return rgba


def patch_value_field(
    canvas_hw: Tuple[int, int],
    rects: Sequence[Tuple[int, int, int, int]],
    scores01: Sequence[float],
    cmap_name: str,
) -> np.ndarray:
    """パッチ矩形から RGBA ヒートマップを一括で作る薄いヘルパ"""
    field, mask = scalar_field(canvas_hw, rects, scores01)
    return to_rgba(field, mask, cmap_name)
