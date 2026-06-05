"""図のレイアウト素材（matplotlib の薄いヘルパ）

格子・タイトル・共有カラーバー・スケールバー・凡例・dpi 等の図品質要件をここに集約し，
builder は本部品を呼ぶだけで出版品質を満たすカラーバーは 1 つの ScalarMappable を
共有して付け，ラスタは高 dpi で書き出す
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

from foveamil.visualization.render.palette import (
    DEFAULT_DPI,
    MIN_FONT_SIZE,
    SELECT_EDGE_COLOR,
)

# 1 セルの既定インチ
_CELL_INCH = 3.0


def make_grid(
    nrows: int, ncols: int, cell_inch: float = _CELL_INCH
) -> Tuple[plt.Figure, np.ndarray]:
    """``nrows × ncols`` の格子 Figure と 2 次元軸配列を返す（``squeeze=False``）"""
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * cell_inch, nrows * cell_inch),
        squeeze=False,
    )
    return fig, axes


def draw_image(
    ax, image, title: Optional[str] = None, edge_color: Optional[str] = None
) -> None:
    """軸に画像を描く（軸目盛なし任意でタイトル・外枠色）"""
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=MIN_FONT_SIZE)
    if edge_color:
        for spine in ax.spines.values():
            spine.set_edgecolor(edge_color)
            spine.set_linewidth(2.0)


def blank(ax) -> None:
    """軸を非表示にする（空セル用）"""
    ax.axis("off")


def add_shared_colorbar(
    fig: plt.Figure, axes_list, cmap_name: str, vmin: float, vmax: float, label: str
) -> None:
    """1 つの ScalarMappable を共有するカラーバーを付ける"""
    mappable = ScalarMappable(
        norm=colors.Normalize(vmin=vmin, vmax=vmax),
        cmap=matplotlib.colormaps[cmap_name],
    )
    cbar = fig.colorbar(mappable, ax=axes_list, fraction=0.025, pad=0.02)
    cbar.set_label(label, fontsize=MIN_FONT_SIZE)
    cbar.ax.tick_params(labelsize=MIN_FONT_SIZE - 1)


def add_select_legend(fig: plt.Figure) -> None:
    """塗り=分類寄与 / 枠=ズーム選択 の凡例を図下部に付ける"""
    handles = [
        Rectangle((0, 0), 1, 1, facecolor="0.5", edgecolor="none"),
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=SELECT_EDGE_COLOR, linewidth=1.5),
    ]
    fig.legend(
        handles, ["fill = class contribution (primary)", "edge = zoom selection (aux)"],
        loc="lower center", ncol=2, fontsize=MIN_FONT_SIZE, frameon=False,
    )


def add_caption(fig: plt.Figure, text: str) -> None:
    """図下部にキャプション（正規化基準・但し書き）を付ける"""
    fig.text(0.5, 0.005, text, ha="center", fontsize=MIN_FONT_SIZE - 1, wrap=True)


def save_figure(fig: plt.Figure, path: str, dpi: int = DEFAULT_DPI) -> None:
    """図を保存して閉じる"""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
