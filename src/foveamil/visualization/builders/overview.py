"""View A: WSI 全体オーバーレイ格子（倍率 × {主primary, 補助aux}）の組立

各倍率について主アテンション（pooling 寄与・viridis）と補助アテンション（選択スコア・
cividis）を H&E サムネ上に半透明合成し，次倍率へ選ばれたパッチをマゼンタ枠で示す
素材層（geometry/normalize/heatmap/blend/panels）を順に呼ぶだけで，値はトレースから渡す
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from foveamil.visualization.core.extraction import AttentionTrace, LayerTrace
from foveamil.visualization.render import heatmap, panels
from foveamil.visualization.render.blend import alpha_over, draw_border
from foveamil.visualization.render.geometry import (
    DEFAULT_PATCH_SIZE,
    level0_to_mag_px,
)
from foveamil.visualization.render.normalize import NORM_PERCENTILE, normalize
from foveamil.visualization.render.palette import (
    AUX_CMAP,
    ATTRIBUTION_DISCLAIMER,
    OVERLAY_ALPHA,
    PRIMARY_CMAP,
    SELECT_EDGE_RGB,
)

# 選択枠の太さ（画素）
_SELECT_EDGE_PX = 2


def _patch_rects_thumb(
    layer: LayerTrace, thumb_mag: float, actual_max_mag: int, patch_size: int
) -> List[Tuple[int, int, int, int]]:
    """層の全パッチをサムネ画素の矩形 ``(x, y, w, h)`` 列にする"""
    side = max(1, int(round(patch_size * thumb_mag / layer.magnification)))
    rects = []
    for x0, y0 in layer.coords:
        tx, ty = level0_to_mag_px((int(x0), int(y0)), thumb_mag, actual_max_mag)
        rects.append((int(round(tx)), int(round(ty)), side, side))
    return rects


def render_overlay(
    thumbnail: np.ndarray,
    layer: LayerTrace,
    scores: np.ndarray,
    cmap_name: str,
    thumb_mag: float,
    actual_max_mag: int,
    patch_size: int = DEFAULT_PATCH_SIZE,
    norm_kind: str = NORM_PERCENTILE,
    draw_selected: bool = False,
) -> np.ndarray:
    """サムネに 1 層分のスコアを連続ヒートで合成した RGB を返す

    ``draw_selected`` が真なら ``selected_local`` のパッチをマゼンタ枠で標示する
    """
    canvas_hw = thumbnail.shape[:2]
    rects = _patch_rects_thumb(layer, thumb_mag, actual_max_mag, patch_size)
    norm = normalize(scores, norm_kind)
    rgba = heatmap.patch_value_field(canvas_hw, rects, norm.values01, cmap_name)
    composited = alpha_over(thumbnail, rgba, OVERLAY_ALPHA)
    if draw_selected and layer.selected_local is not None:
        for local in layer.selected_local:
            draw_border(composited, rects[int(local)], SELECT_EDGE_RGB, _SELECT_EDGE_PX)
    return composited


def _title(trace: AttentionTrace, classes: Optional[List[str]]) -> str:
    """図のタイトル（slide_id・GT/予測・確率）を組む"""
    pred = trace.y_hat
    prob = float(trace.y_prob[pred]) if trace.y_prob is not None else float("nan")
    name = (lambda i: classes[i] if classes and i < len(classes) else str(i))
    return f"{trace.slide_id}  pred:{name(pred)} (p={prob:.2f})"


def build_overview_figure(
    trace: AttentionTrace,
    thumbnail: np.ndarray,
    thumb_mag: float,
    actual_max_mag: int,
    patch_size: int = DEFAULT_PATCH_SIZE,
    norm_kind: str = NORM_PERCENTILE,
    classes: Optional[List[str]] = None,
):
    """1 症例の倍率 × {主, 補助} オーバーレイ格子の Figure を作る"""
    layers = trace.layers
    fig, axes = panels.make_grid(len(layers), 2)
    for i, layer in enumerate(layers):
        primary_img = render_overlay(
            thumbnail, layer, layer.primary, PRIMARY_CMAP, thumb_mag,
            actual_max_mag, patch_size, norm_kind, draw_selected=True,
        )
        panels.draw_image(axes[i][0], primary_img, f"{layer.magnification:g}x primary")
        if layer.aux is not None:
            aux_img = render_overlay(
                thumbnail, layer, layer.aux, AUX_CMAP, thumb_mag,
                actual_max_mag, patch_size, norm_kind, draw_selected=False,
            )
            panels.draw_image(axes[i][1], aux_img, f"{layer.magnification:g}x aux")
        else:
            panels.blank(axes[i][1])

    panels.add_shared_colorbar(fig, axes[:, 0].tolist(), PRIMARY_CMAP, 0.0, 1.0, "primary percentile")
    panels.add_shared_colorbar(fig, axes[:, 1].tolist(), AUX_CMAP, 0.0, 1.0, "aux percentile")
    fig.suptitle(_title(trace, classes), fontsize=10)
    panels.add_select_legend(fig)
    panels.add_caption(fig, f"norm={norm_kind} (rank)  {ATTRIBUTION_DISCLAIMER}")
    return fig
