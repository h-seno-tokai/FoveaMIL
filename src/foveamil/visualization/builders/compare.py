"""View C: 成功 vs 失敗 症例の対比格子（行=症例, 列=倍率）

同一 best config・同一カラースケール・同一レイアウトで成功例（``y_true==y_pred``）と
失敗例を並置し，誤りパターン（コンファウンダ/アーチファクトへの注目）を診断する
各セルは View A の primary オーバーレイの縮約版で，成功/失敗をタイトル枠色で区別する
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from foveamil.visualization.builders.overview import render_overlay
from foveamil.visualization.render import panels
from foveamil.visualization.render.geometry import DEFAULT_PATCH_SIZE
from foveamil.visualization.render.normalize import NORM_PERCENTILE
from foveamil.visualization.render.palette import (
    ATTRIBUTION_DISCLAIMER,
    FAILURE_EDGE_COLOR,
    PRIMARY_CMAP,
    SUCCESS_EDGE_COLOR,
)


def _class_name(classes: Optional[List[str]], idx: int) -> str:
    return classes[idx] if classes and idx < len(classes) else str(idx)


def build_compare_figure(
    items: List[Dict[str, Any]],
    thumb_mag: float,
    patch_size: int = DEFAULT_PATCH_SIZE,
    norm_kind: str = NORM_PERCENTILE,
    classes: Optional[List[str]] = None,
):
    """成功/失敗症例 × 倍率 の primary オーバーレイ対比格子を作る

    Args:
        items: ``{trace, thumbnail, actual_max_mag, correct, y_true}`` の列（成功→失敗の順を想定）
        thumb_mag: サムネ倍率
        patch_size: パッチ一辺
        norm_kind: 正規化種別
        classes: クラス名の並び

    Returns:
        Figure
    """
    if not items:
        raise ValueError("compare items が空")
    mags = items[0]["trace"].magnifications
    fig, axes = panels.make_grid(len(items), len(mags))

    for row, item in enumerate(items):
        trace = item["trace"]
        edge = SUCCESS_EDGE_COLOR if item["correct"] else FAILURE_EDGE_COLOR
        gt = _class_name(classes, int(item["y_true"]))
        pred = _class_name(classes, trace.y_hat)
        prob = float(trace.y_prob[trace.y_hat])
        for col, layer in enumerate(trace.layers):
            img = render_overlay(
                item["thumbnail"], layer, layer.primary, PRIMARY_CMAP, thumb_mag,
                item["actual_max_mag"], patch_size, norm_kind, draw_selected=True,
            )
            title = f"{layer.magnification:g}x" if row == 0 else None
            panels.draw_image(axes[row][col], img, title, edge_color=edge)
        axes[row][0].set_ylabel(
            f"{trace.slide_id}\nGT:{gt} / pred:{pred} (p={prob:.2f})", fontsize=7,
        )

    panels.add_shared_colorbar(fig, axes.ravel().tolist(), PRIMARY_CMAP, 0.0, 1.0, "primary percentile")
    fig.suptitle("success (green) vs failure (red): attention", fontsize=10)
    panels.add_caption(
        fig, f"norm={norm_kind} (rank, shared scale)  {ATTRIBUTION_DISCLAIMER}"
    )
    return fig
