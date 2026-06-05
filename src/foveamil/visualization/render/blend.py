"""原画像と RGBA オーバーレイの alpha 合成（配列演算）

ヒートマップ等の RGBA を原画像 RGB の上へ不透明度 ``alpha`` で重ねる照明（明暗の
変調）とは別概念で，ここは半透明合成のみを担う
"""

from __future__ import annotations

import numpy as np

# 8bit 画素の最大値
_PIXEL_MAX = 255.0
# 既定の合成不透明度
DEFAULT_ALPHA = 0.4


def alpha_over(
    base_rgb: np.ndarray, overlay_rgba: np.ndarray, alpha: float = DEFAULT_ALPHA
) -> np.ndarray:
    """``base_rgb`` の上に ``overlay_rgba`` を不透明度 ``alpha`` で合成する

    Args:
        base_rgb: 原画像 ``[H, W, 3] uint8``
        overlay_rgba: オーバーレイ ``[H, W, 4] float``（``[0, 1]``）
        alpha: オーバーレイ全体の不透明度

    Returns:
        合成画像 ``[H, W, 3] uint8``
    """
    base = base_rgb.astype(float) / _PIXEL_MAX
    over_rgb = overlay_rgba[..., :3]
    over_a = overlay_rgba[..., 3:4] * float(alpha)
    out = base * (1.0 - over_a) + over_rgb * over_a
    return np.clip(out * _PIXEL_MAX, 0, _PIXEL_MAX).astype(np.uint8)


def draw_border(image: np.ndarray, rect, color_rgb, thickness: int = 2) -> None:
    """矩形 ``(x, y, w, h)`` の枠を ``thickness`` 画素で塗る（in-place・配列演算）"""
    x, y, w, h = (int(v) for v in rect)
    t = max(1, int(thickness))
    color = np.asarray(color_rgb, dtype=image.dtype)
    image[y:y + t, x:x + w] = color
    image[y + h - t:y + h, x:x + w] = color
    image[y:y + h, x:x + t] = color
    image[y:y + h, x + w - t:x + w] = color
