"""可視化のオーケストレータ（loader→cases→io→builders を束ねる薄い本体）

best combo の解決→モデル再構築→症例選択→アテンショントレース抽出→WSI 解決→描画→保存→
サイドカー の一方向フローを統括するロジックは各部品に閉じ，ここは順序の制御のみを担う
View A（overview）/ B（zoom）/ C（compare）の 3 経路を提供する
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from foveamil.visualization import cases as cases_mod
from foveamil.visualization import io as io_mod
from foveamil.visualization import loader as loader_mod
from foveamil.visualization.builders import compare as compare_mod
from foveamil.visualization.builders import overview as overview_mod
from foveamil.visualization.builders import zoom as zoom_mod
from foveamil.visualization.core.extraction import extract_attention_trace
from foveamil.visualization.render import panels
from foveamil.visualization.render.geometry import DEFAULT_PATCH_SIZE
from foveamil.visualization.render.palette import DIM_FACTOR
from foveamil.visualization.render.region_reader import RegionReader

logger = logging.getLogger(__name__)

# 出力ファイル名の view ラベル
VIEW_OVERVIEW = "overview"
VIEW_ZOOM = "zoom"
VIEW_COMPARE = "compare"


@dataclass
class VizSpec:
    """可視化 1 実行の解決済み設定"""

    sweep_root: Optional[str] = None
    select: str = loader_mod.SELECT_BEST_VAL
    combo_index: Optional[int] = None
    combo_dir: Optional[str] = None
    fold: str = "1"
    weights_root: Optional[str] = None
    split: str = "test"
    outcome: str = "both"
    slide_id: Optional[List[str]] = None
    n: Optional[int] = None
    per_class: Optional[int] = None
    target_class: Optional[int] = None
    feature_root: str = ""
    coords_root: Optional[str] = None
    wsi_base_path: Optional[str] = None
    wsi_overrides_csv: Optional[str] = None
    out_dir: str = "."
    device: str = "cpu"
    dpi: int = 300
    patch_size: int = DEFAULT_PATCH_SIZE
    thumb_mag: Optional[float] = None
    norm: str = "percentile"
    parent_mag: Optional[float] = None
    parent_pick: str = zoom_mod.PICK_TOP_AUX
    n_parents: int = 4
    zoom_px: int = zoom_mod.DEFAULT_ZOOM_PX
    dim_factor: float = DIM_FACTOR
    chain: bool = False


@dataclass
class _Context:
    """解決済みの combo/モデル/症例選定の文脈"""

    combo_dir: str
    fold_dir: str
    weights_dir: Optional[str]
    loaded: "loader_mod.LoadedModel"
    fold_names: List[str] = field(default_factory=list)


def _resolve_context(spec: VizSpec) -> _Context:
    """combo を解決し fold のモデルをロードする"""
    combo_dir = spec.combo_dir or loader_mod.resolve_best_combo(
        spec.sweep_root, spec.select, spec.combo_index
    )
    fold_name = f"{loader_mod.FOLD_DIR_PREFIX}{spec.fold}"
    fold_dir = os.path.join(combo_dir, fold_name)
    combo_name = os.path.basename(os.path.normpath(combo_dir))
    weights_dir = (
        os.path.join(spec.weights_root, combo_name, fold_name)
        if spec.weights_root else None
    )
    loaded = loader_mod.load_fold(fold_dir, weights_dir=weights_dir, device=spec.device)
    return _Context(combo_dir, fold_dir, weights_dir, loaded, [fold_name])


def _select_slides(spec: VizSpec, ctx: _Context):
    """--slide-id 明示か predictions からの成功/失敗抽出で症例列を返す"""
    if spec.slide_id:
        return [cases_mod.CaseRef(s, -1, -1, False, float("nan"), float("nan"))
                for s in spec.slide_id]
    df = cases_mod.load_cases_frame(ctx.combo_dir, spec.split, ctx.fold_names)
    if df is None:
        raise ValueError(f"predictions_{spec.split}.csv が見つからない: {ctx.combo_dir}")
    success, failure = cases_mod.split_success_failure(df)
    picked = []
    if spec.outcome in ("success", "both"):
        picked += cases_mod.pick_cases(success, spec.n, spec.per_class, spec.target_class)
    if spec.outcome in ("failure", "both"):
        picked += cases_mod.pick_cases(failure, spec.n, spec.per_class, spec.target_class)
    return picked


def _trace(spec: VizSpec, ctx: _Context, slide_id: str):
    """1 症例のアテンショントレースを抽出する"""
    return extract_attention_trace(
        ctx.loaded.model, spec.feature_root, ctx.loaded.encoder, slide_id,
        ctx.loaded.magnifications, ctx.loaded.feature_type,
        device=torch.device(spec.device),
    )


def _open_reader(spec: VizSpec, resolver, slide_id: str, base_mag: float):
    """WSI を解決して RegionReader と actual_max_mag を返す"""
    wsi_path = resolver.resolve(slide_id)
    actual_max_mag = io_mod.resolve_actual_max_mag(
        slide_id, base_mag, wsi_path, spec.coords_root
    )
    return RegionReader(wsi_path), actual_max_mag


def run_overview(spec: VizSpec) -> List[str]:
    """View A を症例ごとに描画して保存しパス列を返す"""
    ctx = _resolve_context(spec)
    resolver = io_mod.make_resolver(spec.wsi_base_path, spec.wsi_overrides_csv)
    thumb_mag = spec.thumb_mag or ctx.loaded.magnifications[0]
    saved = []
    for case in _select_slides(spec, ctx):
        trace = _trace(spec, ctx, case.slide_id)
        reader, max_mag = _open_reader(spec, resolver, case.slide_id, ctx.loaded.magnifications[0])
        try:
            thumb = reader.read_thumbnail(thumb_mag, max_mag)
        finally:
            reader.close()
        fig = overview_mod.build_overview_figure(
            trace, thumb, thumb_mag, max_mag, spec.patch_size, spec.norm, ctx.loaded.classes
        )
        saved.append(_save(spec, fig, case.slide_id, VIEW_OVERVIEW, ctx))
    return saved


def run_zoom(spec: VizSpec) -> List[str]:
    """View B を症例ごとに描画して保存しパス列を返す"""
    ctx = _resolve_context(spec)
    resolver = io_mod.make_resolver(spec.wsi_base_path, spec.wsi_overrides_csv)
    saved = []
    for case in _select_slides(spec, ctx):
        trace = _trace(spec, ctx, case.slide_id)
        reader, max_mag = _open_reader(spec, resolver, case.slide_id, ctx.loaded.magnifications[0])
        try:
            if spec.chain:
                fig = zoom_mod.build_zoom_chain(
                    trace, reader, max_mag, spec.zoom_px, spec.dim_factor, spec.patch_size
                )
            else:
                parent_mag = spec.parent_mag or ctx.loaded.magnifications[-2]
                fig = zoom_mod.build_zoom_figure(
                    trace, reader, parent_mag, max_mag, spec.parent_pick,
                    spec.n_parents, spec.zoom_px, spec.dim_factor, spec.patch_size,
                )
        finally:
            reader.close()
        suffix = "chain" if spec.chain else ""
        saved.append(_save(spec, fig, case.slide_id, VIEW_ZOOM, ctx, suffix))
    return saved


def run_compare(spec: VizSpec) -> List[str]:
    """View C（成功 vs 失敗）を 1 図にまとめて保存しパスを返す"""
    ctx = _resolve_context(spec)
    resolver = io_mod.make_resolver(spec.wsi_base_path, spec.wsi_overrides_csv)
    thumb_mag = spec.thumb_mag or ctx.loaded.magnifications[0]
    items = []
    for case in _select_slides(spec, ctx):
        trace = _trace(spec, ctx, case.slide_id)
        reader, max_mag = _open_reader(spec, resolver, case.slide_id, ctx.loaded.magnifications[0])
        try:
            thumb = reader.read_thumbnail(thumb_mag, max_mag)
        finally:
            reader.close()
        items.append({
            "trace": trace, "thumbnail": thumb, "actual_max_mag": max_mag,
            "correct": case.correct, "y_true": case.y_true,
        })
    fig = compare_mod.build_compare_figure(
        items, thumb_mag, spec.patch_size, spec.norm, ctx.loaded.classes
    )
    out = io_mod.output_path(spec.out_dir, "compare", VIEW_COMPARE)
    panels.save_figure(fig, out, spec.dpi)
    io_mod.write_sidecar(out, _sidecar(spec, ctx))
    return [out]


def _save(spec: VizSpec, fig, slide_id: str, view: str, ctx: _Context, suffix: str = "") -> str:
    """図とサイドカーを保存しパスを返す"""
    out = io_mod.output_path(spec.out_dir, slide_id, view, suffix)
    panels.save_figure(fig, out, spec.dpi)
    io_mod.write_sidecar(out, _sidecar(spec, ctx))
    logger.info("saved %s", out)
    return out


def _sidecar(spec: VizSpec, ctx: _Context) -> dict:
    """サイドカー JSON の中身（config 要約・正規化・但し書き）"""
    return {
        "combo_dir": ctx.combo_dir,
        "fold": spec.fold,
        "encoder": ctx.loaded.encoder,
        "feature_type": ctx.loaded.feature_type,
        "magnifications": ctx.loaded.magnifications,
        "n_cls": ctx.loaded.n_cls,
        "save_metric": ctx.loaded.save_metric,
        "norm": spec.norm,
        "disclaimer": "生アテンションは分類寄与の代理であり厳密な特徴帰属ではない",
    }
