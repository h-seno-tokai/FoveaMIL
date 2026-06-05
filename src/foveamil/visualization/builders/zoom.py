"""View B: 階層ズーム照明（40倍可視化問題の解）

選択された親パッチを高解像で拡大し，次倍率が観察した r×r 子のうち各子を「子の primary
アテンション（高倍率 pooling 寄与）」の連続明度で照らす全体図では潰れて見えない高倍率の
選択を，画面いっぱいの拡大視野で確実に視認させる単段（親→子）に加え，最低→最高倍率の
top 選択経路を 1 行で辿る多段連鎖 :func:`build_zoom_chain`（中心窩経路図）を持つ
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from foveamil.visualization.core.extraction import AttentionTrace, LayerTrace
from foveamil.visualization.render import panels
from foveamil.visualization.render.geometry import (
    DEFAULT_PATCH_SIZE,
    patch_footprint_level0,
    ratio_and_children,
)
from foveamil.visualization.render.illuminate import (
    build_child_slot_set,
    illuminate_children,
)
from foveamil.visualization.render.normalize import to_minmax
from foveamil.visualization.render.palette import ATTRIBUTION_DISCLAIMER, DIM_FACTOR

# 既定の拡大ビュー画素
DEFAULT_ZOOM_PX = 896
# 親選択の種別
PICK_TOP_AUX = "top_aux"
PICK_TOP_PRIMARY = "top_primary"
PICK_INDEX = "index"


def _global_to_value(layer: LayerTrace, values: np.ndarray) -> Dict[int, float]:
    """層の global index → スコアの辞書を作る"""
    return {int(g): float(v) for g, v in zip(layer.global_indices, values)}


def _child_scores(
    child_value_map: Dict[int, float], parent_global: int, cpp: int
) -> Tuple[np.ndarray, int]:
    """親の cpp 子の primary を slot 順に集め，親内 min-max で ``[0,1]`` 化する

    子が 1 つも存在しない（高倍率へ非展開＝aux 非選択の親）場合は全 0 を返し全セルを
    減光する（「子データ無し」を「一様に明るい」と混同しない）子が存在し分散がほぼ無い
    （真の一様注目）場合は全 1（均一に明るく）にする

    Returns:
        ``(scores[cpp], n_present)``
    """
    keys = [parent_global * cpp + slot for slot in range(cpp)]
    n_present = sum(1 for k in keys if k in child_value_map)
    if n_present == 0:
        return np.zeros(cpp, dtype=float), 0
    raw = np.array([child_value_map.get(k, 0.0) for k in keys], dtype=float)
    scaled = to_minmax(raw)
    if not np.any(scaled > 0):
        scaled = np.ones(cpp, dtype=float)
    return scaled, n_present


def _pick_parent_locals(
    parent_layer: LayerTrace,
    parent_pick: str,
    n_parents: int,
    indices: Optional[List[int]] = None,
) -> List[int]:
    """親として拡大する patch の local index 列を返す"""
    if parent_pick == PICK_INDEX and indices is not None:
        return list(indices)[:n_parents]
    if parent_pick == PICK_TOP_PRIMARY:
        order = np.argsort(parent_layer.primary)[::-1]
        return [int(i) for i in order[:n_parents]]
    # top_aux: 実際にズーム選択された patch（足りなければ aux 上位で補う）
    selected = (
        [int(i) for i in parent_layer.selected_local]
        if parent_layer.selected_local is not None else []
    )
    if len(selected) >= n_parents:
        return selected[:n_parents]
    if parent_layer.aux is not None:
        order = [int(i) for i in np.argsort(parent_layer.aux)[::-1]]
        for i in order:
            if i not in selected:
                selected.append(i)
            if len(selected) >= n_parents:
                break
    return selected[:n_parents]


def _zoom_one(
    reader, parent_layer, child_layer, parent_local, parent_mag, child_mag,
    actual_max_mag, zoom_px, dim_factor, patch_size,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """1 親について (親素画像, 子照明画像, r, 存在子数) を返す"""
    r, cpp = ratio_and_children(parent_mag, child_mag)
    parent_global = int(parent_layer.global_indices[parent_local])
    x0, y0 = parent_layer.coords[parent_local]
    side0 = patch_footprint_level0(parent_mag, actual_max_mag, patch_size)
    # zoom_px は r で割り切れる値に丸めて端の取りこぼしを防ぐ
    view_px = (zoom_px // r) * r
    parent_img = reader.read_level0_rect(int(x0), int(y0), side0, view_px)

    value_map = _global_to_value(child_layer, child_layer.primary)
    scores, n_present = _child_scores(value_map, parent_global, cpp)
    selected_slots = build_child_slot_set(child_layer.selected_global, parent_global, cpp)
    illum = illuminate_children(parent_img, scores, r, view_px, selected_slots, dim_factor)
    return parent_img, illum, r, n_present


def build_zoom_figure(
    trace: AttentionTrace,
    reader,
    parent_mag: float,
    actual_max_mag: int,
    parent_pick: str = PICK_TOP_AUX,
    n_parents: int = 4,
    zoom_px: int = DEFAULT_ZOOM_PX,
    dim_factor: float = DIM_FACTOR,
    patch_size: int = DEFAULT_PATCH_SIZE,
    indices: Optional[List[int]] = None,
):
    """単段ズーム照明（親→子）の Figure を作る（行=親, 列=[親素, 子照明]）"""
    mags = trace.magnifications
    if len(mags) < 2:
        raise ValueError("単一倍率ではズームは適用外")
    pidx = next(i for i, m in enumerate(mags) if abs(m - parent_mag) < 1e-6)
    if pidx >= len(mags) - 1:
        raise ValueError(f"parent_mag {parent_mag} は最終倍率で子を持たない")
    parent_layer, child_layer = trace.layers[pidx], trace.layers[pidx + 1]
    child_mag = mags[pidx + 1]

    parents = _pick_parent_locals(parent_layer, parent_pick, n_parents, indices)
    fig, axes = panels.make_grid(max(1, len(parents)), 2)
    for row, local in enumerate(parents):
        parent_img, illum, r, n_present = _zoom_one(
            reader, parent_layer, child_layer, local, parent_mag, child_mag,
            actual_max_mag, zoom_px, dim_factor, patch_size,
        )
        pg = int(parent_layer.global_indices[local])
        panels.draw_image(axes[row][0], parent_img, f"parent {parent_mag:g}x (g={pg})")
        suffix = "" if n_present else " [no high-mag children]"
        panels.draw_image(
            axes[row][1], illum, f"child {child_mag:g}x illuminated ({r}x{r}){suffix}",
        )
    fig.suptitle(f"{trace.slide_id}  zoom {parent_mag:g}x->{child_mag:g}x", fontsize=10)
    panels.add_caption(
        fig, f"brightness = child primary (per-parent norm)  {ATTRIBUTION_DISCLAIMER}"
    )
    return fig


def build_zoom_chain(
    trace: AttentionTrace,
    reader,
    actual_max_mag: int,
    zoom_px: int = DEFAULT_ZOOM_PX,
    dim_factor: float = DIM_FACTOR,
    patch_size: int = DEFAULT_PATCH_SIZE,
):
    """多段ズーム連鎖（最低→最高倍率の top 選択経路）の Figure を作る（行=1経路, 列=段）"""
    mags = trace.magnifications
    n_stages = len(mags) - 1
    if n_stages < 1:
        raise ValueError("単一倍率では多段ズームは適用外")

    # 経路の起点: 最低倍率で最も選ばれた（aux 上位）親
    current = _pick_parent_locals(trace.layers[0], PICK_TOP_AUX, 1)[0]
    fig, axes = panels.make_grid(1, n_stages)
    for stage in range(n_stages):
        parent_layer, child_layer = trace.layers[stage], trace.layers[stage + 1]
        parent_mag, child_mag = mags[stage], mags[stage + 1]
        r, cpp = ratio_and_children(parent_mag, child_mag)
        pg = int(parent_layer.global_indices[current])
        _, illum, _, _ = _zoom_one(
            reader, parent_layer, child_layer, current, parent_mag, child_mag,
            actual_max_mag, zoom_px, dim_factor, patch_size,
        )
        panels.draw_image(
            axes[0][stage], illum, f"{parent_mag:g}->{child_mag:g}x ({r}x{r})",
        )
        # 次段の親 = この親の子のうち「次層で実際に選択された」子の aux 最大
        current = _next_parent_local(child_layer, pg, cpp)
        if current is None:
            break
    fig.suptitle(f"{trace.slide_id}  foveated zoom path", fontsize=10)
    panels.add_caption(
        fig, f"brightness = child primary (per-parent norm)  {ATTRIBUTION_DISCLAIMER}"
    )
    return fig


def _next_parent_local(child_layer: LayerTrace, parent_global: int, cpp: int) -> Optional[int]:
    """子層で親の子のうち「実際に次層へ選択された」子の aux 最大 local を返す

    経路は実際の選択を辿るべきなので，``selected_local``（次層へズーム選択された patch）に
    含まれる子だけを候補にする最終層や該当無しは ``None``（経路打ち切り）
    """
    if child_layer.aux is None or child_layer.selected_local is None:
        return None
    child_globals = {parent_global * cpp + slot for slot in range(cpp)}
    best_local, best_aux = None, -np.inf
    for local in child_layer.selected_local:
        g = int(child_layer.global_indices[int(local)])
        a = child_layer.aux[int(local)]
        if g in child_globals and a > best_aux:
            best_local, best_aux = int(local), a
    return best_local
