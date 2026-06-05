"""OpenSlide WSI の倍率／レベル解決・画像読み出し・グリッド寸法算出

* level-0 の実最大倍率（例 40x）をメタデータから推定する
* 目標倍率に最も近いレベルを選び，必要なら追加ダウンサンプル係数を求める
* 画像本体を読まずに ``level_dimensions`` から指定倍率での画像サイズを算出する
* 指定倍率の画像全体を ``numpy`` 配列として読み出す
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import openslide
from PIL import Image

logger = logging.getLogger(__name__)

# Aperio SVS が持つ「対物レンズ倍率」プロパティ名（例 "40"）
APERIO_APP_MAG_PROPERTY = "aperio.AppMag"
# 解像度(microns per pixel)プロパティ名倍率不明時に mpp から逆算する
MPP_X_PROPERTY = "openslide.mpp-x"
# 40x スキャナのおおよその解像度 [microns/pixel]mpp から倍率を逆算する基準
# （actual_max_mag ≈ round(REFERENCE_MPP_AT_40X / mpp)）
REFERENCE_MPP_AT_40X = 0.25
# メタデータから倍率を判定できない場合のフォールバック最大倍率
DEFAULT_MAX_MAGNIFICATION = 40


def get_actual_max_magnification(wsi: openslide.OpenSlide) -> int:
    """WSI の level-0（最大解像度）の対物倍率を推定する

    判定順:
      1. Aperio の ``aperio.AppMag`` があればそれを整数化して使う
      2. なければ ``openslide.mpp-x`` から ``round(0.25 / mpp)`` で逆算する
      3. どちらも無ければ既定の 40x を使う（警告ログ）

    Args:
        wsi: 開いた :class:`openslide.OpenSlide`

    Returns:
        推定された最大倍率（整数, 例 ``40``）
    """
    app_mag = wsi.properties.get(APERIO_APP_MAG_PROPERTY)
    if app_mag is not None:
        return int(float(app_mag))

    mpp_x = wsi.properties.get(MPP_X_PROPERTY)
    if mpp_x is not None:
        return int(np.round(REFERENCE_MPP_AT_40X / float(mpp_x)))

    logger.warning(
        "magnification not found in WSI metadata; falling back to %dx",
        DEFAULT_MAX_MAGNIFICATION,
    )
    return DEFAULT_MAX_MAGNIFICATION


def get_level_and_size(
    wsi: openslide.OpenSlide, magnification: float, actual_max_mag: int
) -> Tuple[int, int, int, int]:
    """指定倍率に最適な WSI レベルと，その倍率での画像サイズを返す（画像は読まない）

    各レベルの「ネイティブ倍率」は ``actual_max_mag / level_downsample`` で求まる
    目標倍率に一致するレベルがあればそれを使い，無ければ
    :meth:`openslide.OpenSlide.get_best_level_for_downsample` で最寄りレベルを選び，
    そのレベルが目標より高倍率なら追加ダウンサンプル係数を求める

    Args:
        wsi: 開いた :class:`openslide.OpenSlide`
        magnification: 目標倍率
        actual_max_mag: :func:`get_actual_max_magnification` で得た最大倍率

    Returns:
        ``(level, additional_downsample, width, height)``
        ``width``/``height`` は追加ダウンサンプル適用後の，その倍率での画像サイズ
    """
    # ``level_downsamples`` は実測値で僅かに非整数になる（例 16.0009）
    # ピラミッドの段は整数倍なので丸めてネイティブ倍率を求める
    downsamples = wsi.level_downsamples
    native_mags = [actual_max_mag / round(df) for df in downsamples]

    if magnification in native_mags:
        level = native_mags.index(magnification)
        additional_downsample = 1
    else:
        level = wsi.get_best_level_for_downsample(actual_max_mag / magnification)
        native_mag = native_mags[level]
        # 端数を四捨五入して目標倍率ちょうどに合わせる（切り捨てない）
        additional_downsample = (
            max(1, int(round(native_mag / magnification)))
            if native_mag > magnification
            else 1
        )

    width, height = wsi.level_dimensions[level]
    if additional_downsample > 1:
        width //= additional_downsample
        height //= additional_downsample

    logger.debug(
        "magnification %.4gx -> level=%d (native %.4gx), extra downsample=%d, size=%dx%d",
        magnification,
        level,
        native_mags[level],
        additional_downsample,
        width,
        height,
    )
    return level, additional_downsample, width, height


def read_image_at(
    wsi: openslide.OpenSlide, magnification: float, actual_max_mag: int
) -> np.ndarray:
    """指定倍率の WSI 全体を RGB の ``numpy`` 配列として読み出す

    最寄りレベルを ``read_region`` で読み，目標より高倍率なら BILINEAR で縮小する

    Args:
        wsi: 開いた :class:`openslide.OpenSlide`
        magnification: 目標倍率
        actual_max_mag: 最大倍率

    Returns:
        ``(H, W, 3)`` の RGB 画像（``uint8``）
    """
    level, additional_downsample, _, _ = get_level_and_size(
        wsi, magnification, actual_max_mag
    )
    region = wsi.read_region(
        location=(0, 0), level=level, size=wsi.level_dimensions[level]
    ).convert("RGB")

    if additional_downsample > 1:
        w, h = region.size
        region = region.resize(
            (w // additional_downsample, h // additional_downsample), Image.BILINEAR
        )

    return np.array(region)


def grid_shape(width: int, height: int, patch_size: int, stride: int) -> Tuple[int, int]:
    """指定サイズ・ストライドで取れるパッチグリッドの行数・列数を返す

    端数は切り捨て（はみ出すパッチは作らない）``(n_rows, n_cols)`` を返す

    Args:
        width: 画像幅（列方向の画素数）
        height: 画像高さ（行方向の画素数）
        patch_size: パッチの一辺（画素）
        stride: パッチ間ストライド（画素）

    Returns:
        ``(n_rows, n_cols)`` のタプル負になる場合は ``(0, 0)`` を返す
    """
    n_rows = (height - patch_size) // stride + 1
    n_cols = (width - patch_size) // stride + 1
    if n_rows < 0 or n_cols < 0:
        return 0, 0
    return n_rows, n_cols
