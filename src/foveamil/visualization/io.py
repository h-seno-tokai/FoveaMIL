"""可視化の副作用（WSI 解決・actual_max_mag 取得・図/サイドカー保存）を集約する

slide_id→WSI パスは :class:`WSIResolver`（env ``WSI_BASE_PATH`` / overrides CSV）で解決する
``actual_max_mag`` は座標 H5 の attr を優先し，無ければ WSI から再導出する（特徴 H5 には
この attr が無いため）出力パス命名・図保存・サイドカー JSON（正規化基準・config 要約・
忠実度の但し書き）を担い，公開コードに内部パス/ホスト名を書かない（全て引数/env）
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import h5py
import openslide

from foveamil.wsi.resolver import WSIResolver
from foveamil.wsi.slide import get_actual_max_magnification

# 座標 H5 の actual_max_mag attr 名
_ACTUAL_MAX_MAG_ATTR = "actual_max_mag"
# 倍率ディレクトリ名のテンプレート
_MAG_DIR_TEMPLATE = "{mag}x"
# サイドカー JSON の接尾辞
SIDECAR_SUFFIX = ".sidecar.json"


def make_resolver(
    wsi_base_path: Optional[str] = None, overrides_csv: Optional[str] = None
) -> WSIResolver:
    """WSI 解決器を作る（overrides CSV があれば併用）"""
    if overrides_csv:
        return WSIResolver.from_overrides_csv(overrides_csv, base_path=wsi_base_path)
    return WSIResolver(base_path=wsi_base_path)


def resolve_actual_max_mag(
    slide_id: str,
    base_mag: float,
    wsi_path: str,
    coords_root: Optional[str] = None,
) -> int:
    """``actual_max_mag`` を座標 H5 attr 優先で取得し，無ければ WSI から再導出する"""
    if coords_root:
        path = os.path.join(
            coords_root, _MAG_DIR_TEMPLATE.format(mag=base_mag), f"{slide_id}.h5"
        )
        if os.path.exists(path):
            with h5py.File(path, "r") as handle:
                if _ACTUAL_MAX_MAG_ATTR in handle.attrs:
                    return int(handle.attrs[_ACTUAL_MAX_MAG_ATTR])
    wsi = openslide.OpenSlide(wsi_path)
    try:
        return get_actual_max_magnification(wsi)
    finally:
        wsi.close()


def output_path(out_dir: str, slide_id: str, view: str, suffix: str = "") -> str:
    """``{out_dir}/{slide_id}_{view}{suffix}.png`` を組む"""
    os.makedirs(out_dir, exist_ok=True)
    tail = f"_{suffix}" if suffix else ""
    return os.path.join(out_dir, f"{slide_id}_{view}{tail}.png")


def write_sidecar(figure_path: str, payload: Dict[str, Any]) -> str:
    """図に対応するサイドカー JSON（正規化基準・config・但し書き）を書く"""
    path = figure_path + SIDECAR_SUFFIX
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path
