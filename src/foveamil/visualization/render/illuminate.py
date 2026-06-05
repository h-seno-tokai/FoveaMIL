"""階層ズーム照明（40倍可視化問題の中核）

親パッチを拡大した画像に対し，内部の r×r 子セルを「子のアテンション（既定は
高倍率 pooling 寄与の primary）」の連続明度で照らす低スコアのセルは ``dim_factor``
方向へ減光し，高スコアのセルは原輝度を保つ次段でさらにズーム選択された子は
マゼンタの離散枠で標示する（chain 読解の補助）``select_weight`` は推論時ほぼハードで
連続 saliency ではないため，明度は子スコアで決め選択は枠で示す
"""

from __future__ import annotations

from typing import Optional, Sequence, Set

import numpy as np

from foveamil.visualization.render.blend import draw_border
from foveamil.visualization.render.geometry import child_cell_in_view, child_slot
from foveamil.visualization.render.palette import DIM_FACTOR, SELECT_EDGE_RGB

# 枠の最小太さ（画素）
_MIN_EDGE_PX = 2
# 枠太さを zoom_px から決める分母
_EDGE_PX_DIVISOR = 256


def build_child_slot_set(
    selected_global: Optional[Sequence[int]], parent_global: int, cpp: int
) -> Set[int]:
    """選択された子 global index のうち親 ``parent_global`` の子 slot 集合を返す"""
    slots: Set[int] = set()
    if selected_global is None:
        return slots
    for child_global in selected_global:
        slot = child_slot(int(child_global), parent_global, cpp)
        if 0 <= slot < cpp:
            slots.add(slot)
    return slots


def illuminate_children(
    parent_img: np.ndarray,
    child_scores01: Sequence[float],
    ratio: int,
    zoom_px: int,
    selected_slots: Optional[Set[int]] = None,
    dim_factor: float = DIM_FACTOR,
) -> np.ndarray:
    """親拡大画像の r×r 子セルを子スコアの連続明度で照らす

    Args:
        parent_img: 親領域の拡大 RGB ``[zoom_px, zoom_px, 3] uint8``
        child_scores01: 子 slot 順（0..cpp-1）の ``[0, 1]`` スコア（既定 primary）
        ratio: 倍率比 r（子は r×r）
        zoom_px: 拡大ビューの一辺画素
        selected_slots: マゼンタ枠で標示する子 slot 集合（``None`` で枠なし）
        dim_factor: スコア 0 の子セルに掛ける減光係数

    Returns:
        照明後の RGB ``[zoom_px, zoom_px, 3] uint8``
    """
    out = parent_img.astype(float).copy()
    scores = np.asarray(child_scores01, dtype=float)
    for slot in range(ratio * ratio):
        x, y, w, h = child_cell_in_view(slot, ratio, zoom_px)
        score = float(scores[slot]) if slot < scores.size else 0.0
        factor = dim_factor + (1.0 - dim_factor) * score
        out[y:y + h, x:x + w] *= factor
    out = np.clip(out, 0, 255).astype(np.uint8)

    if selected_slots:
        thickness = max(_MIN_EDGE_PX, zoom_px // _EDGE_PX_DIVISOR)
        for slot in selected_slots:
            rect = child_cell_in_view(slot, ratio, zoom_px)
            draw_border(out, rect, SELECT_EDGE_RGB, thickness)
    return out


__all__ = ["build_child_slot_set", "illuminate_children"]
