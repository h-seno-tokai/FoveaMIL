"""複数倍率の組織パッチ座標を抽出して H5 に保存する

処理:

1. 組織マスクをベース（最低）倍率で 1 回計算する（:class:`SimpleTissueMask`）
2. ベース倍率はグリッド走査し ``tissue_fraction >= tissue_threshold`` のパッチを採用する
   グリッド寸法は画像本体を読まず ``level_dimensions`` から決める
3. 以降の倍率は各親パッチを 2x2=4 子に座標演算で細分化する（子は親の 4 倍）

座標は全倍率で level-0（最大倍率）ピクセル空間の ``(x, y)`` で保持する
（x=列方向, y=行方向）

出力（WSI ごと・倍率ごとに 1 ファイル）:
  ファイル名 ``{mag}x/{slide_id}.h5``
  dataset ``coords``: shape ``(N, 2)``, dtype ``int32``, level-0 座標 ``(x, y)``
  共通 attrs: ``patch_size``, ``magnification``, ``stride``, ``downsample_factor``,
  ``actual_max_mag``, ``wsi_path``, ``tissue_threshold``, ``is_hierarchical``(=True)
  ベース以外には ``parent_magnification`` を追加
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Sequence

import h5py
import numpy as np
import openslide

from foveamil.utils.memory import force_gc, log_memory
from foveamil.wsi.slide import (
    get_actual_max_magnification,
    get_level_and_size,
    grid_shape,
    read_image_at,
)
from foveamil.wsi.tissue import SimpleTissueMask, make_tissue_mask

logger = logging.getLogger(__name__)

# 隣接倍率の比はちょうど 2.0 でなければならない（2x2 細分化の前提）
REQUIRED_MAGNIFICATION_RATIO = 2.0
# 浮動小数比較の許容誤差（倍率比のチェック用）
MAGNIFICATION_RATIO_TOLERANCE = 1e-6
# 親パッチを 2x2 に割るときの 1 軸あたりの分割数（子オフセット = 親サイズ / これ）
SUBDIVISION_PER_AXIS = 2
# 組織判定でマスク値が「組織」を表す値（:class:`SimpleTissueMask` の出力規約）
TISSUE_MASK_FOREGROUND = 1
# coords dataset の dtype（出力仕様で int32 固定）
COORDS_DTYPE = np.int32


def validate_magnifications(magnifications: Sequence[float]) -> None:
    """倍率列が昇順かつ隣接比ちょうど 2.0 であることを検証する（最低 2 倍率）

    Args:
        magnifications: 検証する倍率の列（低 → 高の順を期待）

    Raises:
        ValueError: 倍率が 2 個未満，または隣接比が 2.0 でない場合
    """
    if len(magnifications) < 2:
        raise ValueError(
            f"need at least 2 magnifications for a hierarchy, got {list(magnifications)}"
        )
    for parent, child in zip(magnifications, magnifications[1:]):
        ratio = child / parent
        if abs(ratio - REQUIRED_MAGNIFICATION_RATIO) > MAGNIFICATION_RATIO_TOLERANCE:
            raise ValueError(
                "magnifications must be ascending with each adjacent ratio == 2.0; "
                f"got {parent} -> {child} (ratio {ratio:.4g})"
            )


def downsample_factor(actual_max_mag: int, magnification: float) -> int:
    """level-0 から見た指定倍率のダウンサンプル係数 ``round(max_mag / mag)``"""
    return int(round(actual_max_mag / magnification))


def extract_base_coordinates(
    wsi: openslide.OpenSlide,
    magnification: float,
    patch_size: int,
    stride: int,
    mask: np.ndarray,
    tissue_threshold: float,
    actual_max_mag: int,
) -> np.ndarray:
    """ベース（最低）倍率の有効パッチ座標を level-0 ピクセル空間で抽出する

    画像本体は読まず ``level_dimensions`` から得たグリッド寸法でマスクを走査し，
    ``tissue_fraction >= tissue_threshold`` のパッチだけを採用するマスクは
    ベース倍率の解像度で作られている前提（追加ダウンサンプル不要）

    Args:
        wsi: 開いた :class:`openslide.OpenSlide`
        magnification: ベース倍率
        patch_size: パッチの一辺（ベース倍率の画素）
        stride: パッチ間ストライド（ベース倍率の画素）
        mask: ベース倍率解像度の組織マスク（組織=1）
        tissue_threshold: パッチ採用に必要な組織画素の最小割合
        actual_max_mag: WSI の最大倍率

    Returns:
        ``(N, 2)`` の ``int32`` 配列各行が level-0 座標 ``(x, y)``
    """
    log_memory(f"base extraction start @ {magnification}x")

    # 画像は読まずサイズ（=グリッド寸法）だけを得る
    _, _, width, height = get_level_and_size(wsi, magnification, actual_max_mag)
    n_rows, n_cols = grid_shape(width, height, patch_size, stride)
    logger.debug("base grid %dx%d = %d candidate patches", n_rows, n_cols, n_rows * n_cols)

    # ベース倍率座標 → level-0 座標 への倍率（= ダウンサンプル係数）
    level0_scale = actual_max_mag / magnification

    coords: List[tuple] = []
    for row in range(n_rows):
        for col in range(n_cols):
            y0 = row * stride
            x0 = col * stride
            y1 = y0 + patch_size
            x1 = x0 + patch_size
            if y1 > mask.shape[0] or x1 > mask.shape[1]:
                continue

            patch = mask[y0:y1, x0:x1]
            tissue_fraction = (patch == TISSUE_MASK_FOREGROUND).sum() / patch.size
            if tissue_fraction >= tissue_threshold:
                coords.append((int(x0 * level0_scale), int(y0 * level0_scale)))

    coords_array = np.asarray(coords, dtype=COORDS_DTYPE).reshape(-1, 2)
    log_memory(f"base extraction done @ {magnification}x: {len(coords_array)} patches")
    return coords_array


def subdivide_coordinates(
    parent_coords: np.ndarray,
    parent_mag: float,
    child_mag: float,
    patch_size: int,
    actual_max_mag: int,
) -> np.ndarray:
    """各親パッチを 2x2=4 子パッチに細分化する（level-0 空間の座標演算のみ）

    親パッチの level-0 上の一辺は ``patch_size * (actual_max_mag / parent_mag)``
    子のオフセットはその半分各親 ``(x, y)`` に対し
    ``for dy in (0, 1): for dx in (0, 1)`` の順で
    ``(x + dx*offset, y + dy*offset)`` を生成する（出力の並び順を固定するため重要）

    Args:
        parent_coords: 親の ``(N, 2)`` level-0 座標
        parent_mag: 親倍率
        child_mag: 子倍率（``parent_mag`` の 2 倍であること）
        patch_size: パッチの一辺（画素）
        actual_max_mag: WSI の最大倍率

    Returns:
        ``(N*4, 2)`` の ``int32`` 配列（level-0 座標）

    Raises:
        ValueError: ``child_mag / parent_mag`` が 2.0 でない場合
    """
    ratio = child_mag / parent_mag
    if abs(ratio - REQUIRED_MAGNIFICATION_RATIO) > MAGNIFICATION_RATIO_TOLERANCE:
        raise ValueError(
            f"magnification ratio must be 2.0, got {ratio:.4g} "
            f"({parent_mag}x -> {child_mag}x)"
        )

    # 親パッチの level-0 上の一辺と，その半分（子オフセット）
    parent_level0_patch_size = int(patch_size * (actual_max_mag / parent_mag))
    child_offset = parent_level0_patch_size // SUBDIVISION_PER_AXIS

    log_memory(f"subdivide {parent_mag}x -> {child_mag}x (offset={child_offset})")

    children: List[tuple] = []
    for parent_x, parent_y in parent_coords:
        for dy in range(SUBDIVISION_PER_AXIS):
            for dx in range(SUBDIVISION_PER_AXIS):
                children.append(
                    (
                        int(parent_x) + dx * child_offset,
                        int(parent_y) + dy * child_offset,
                    )
                )

    child_array = np.asarray(children, dtype=COORDS_DTYPE).reshape(-1, 2)
    logger.debug(
        "subdivided %d parents -> %d children", len(parent_coords), len(child_array)
    )
    return child_array


def _write_coords_h5(
    h5_path: str,
    coords: np.ndarray,
    *,
    patch_size: int,
    magnification: float,
    stride: int,
    actual_max_mag: int,
    wsi_path: str,
    tissue_threshold: float,
    parent_magnification: float | None,
) -> None:
    """1 倍率分の座標と属性を H5 に書き出す（出力仕様に厳密準拠）"""
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("coords", data=coords, dtype=COORDS_DTYPE)
        f.attrs["patch_size"] = patch_size
        f.attrs["magnification"] = magnification
        f.attrs["stride"] = stride
        f.attrs["downsample_factor"] = downsample_factor(actual_max_mag, magnification)
        f.attrs["actual_max_mag"] = actual_max_mag
        f.attrs["wsi_path"] = wsi_path
        f.attrs["tissue_threshold"] = tissue_threshold
        f.attrs["is_hierarchical"] = True
        if parent_magnification is not None:
            f.attrs["parent_magnification"] = parent_magnification


def process_wsi(
    wsi_path: str,
    output_dir: str,
    magnifications: Sequence[float],
    patch_size: int,
    stride: int,
    tissue_threshold: float,
    mask_generator: SimpleTissueMask,
    slide_id: str | None = None,
) -> List[str]:
    """WSI 1 枚を処理し，全倍率の座標 H5 を出力する

    ベース倍率で 1 度だけマスクを作り座標を抽出し，以降の倍率は座標演算だけで
    階層的に細分化するWSI 処理後は :func:`force_gc` で明示的に解放する

    Args:
        wsi_path: WSI ファイルの絶対パス
        output_dir: H5 出力ディレクトリ
        magnifications: 低 → 高の順の倍率列（隣接比 2.0）
        patch_size: パッチの一辺（画素）
        stride: パッチ間ストライド（画素）
        tissue_threshold: パッチ採用に必要な組織画素の最小割合
        mask_generator: 設定済みの :class:`SimpleTissueMask`
        slide_id: 出力ファイル名に使う slide_id``None`` なら WSI ベース名を使う

    Returns:
        書き出した H5 ファイルパスのリスト（倍率順）
    """
    validate_magnifications(magnifications)
    if slide_id is None:
        slide_id = Path(wsi_path).stem

    logger.info("processing %s", slide_id)
    os.makedirs(output_dir, exist_ok=True)
    written: List[str] = []

    wsi = openslide.OpenSlide(wsi_path)
    try:
        actual_max_mag = get_actual_max_magnification(wsi)
        logger.debug("%s actual_max_mag=%dx", slide_id, actual_max_mag)

        # --- マスクは最低倍率で 1 回だけ計算 ---
        base_mag = magnifications[0]
        base_image = read_image_at(wsi, base_mag, actual_max_mag)
        mask = make_tissue_mask(base_image, mask_generator)
        del base_image
        force_gc()

        # --- ベース倍率の座標抽出 ---
        base_coords = extract_base_coordinates(
            wsi=wsi,
            magnification=base_mag,
            patch_size=patch_size,
            stride=stride,
            mask=mask,
            tissue_threshold=tissue_threshold,
            actual_max_mag=actual_max_mag,
        )
        del mask
        force_gc()

        base_dir = os.path.join(output_dir, f"{base_mag}x")
        os.makedirs(base_dir, exist_ok=True)
        base_path = os.path.join(base_dir, f"{slide_id}.h5")
        _write_coords_h5(
            base_path,
            base_coords,
            patch_size=patch_size,
            magnification=base_mag,
            stride=stride,
            actual_max_mag=actual_max_mag,
            wsi_path=wsi_path,
            tissue_threshold=tissue_threshold,
            parent_magnification=None,
        )
        written.append(base_path)
        logger.info("  %.4gx: %d patches", base_mag, len(base_coords))

        # --- 以降の倍率は座標演算のみで階層細分化 ---
        parent_coords = base_coords
        parent_mag = base_mag
        for mag in magnifications[1:]:
            coords = subdivide_coordinates(
                parent_coords=parent_coords,
                parent_mag=parent_mag,
                child_mag=mag,
                patch_size=patch_size,
                actual_max_mag=actual_max_mag,
            )
            mag_dir = os.path.join(output_dir, f"{mag}x")
            os.makedirs(mag_dir, exist_ok=True)
            out_path = os.path.join(mag_dir, f"{slide_id}.h5")
            _write_coords_h5(
                out_path,
                coords,
                patch_size=patch_size,
                magnification=mag,
                stride=stride,
                actual_max_mag=actual_max_mag,
                wsi_path=wsi_path,
                tissue_threshold=tissue_threshold,
                parent_magnification=parent_mag,
            )
            written.append(out_path)
            logger.info("  %.4gx: %d patches", mag, len(coords))

            parent_coords = coords
            parent_mag = mag
    finally:
        wsi.close()
        force_gc()

    logger.info("done %s (%d files)", slide_id, len(written))
    return written
