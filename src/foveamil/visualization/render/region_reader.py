"""WSI の level-0 矩形を目標画素サイズの RGB として読む唯一の公開 API

``openslide.read_region`` の location は常に level-0 座標という契約をそのまま使い，
``features`` の私的な領域読込（最寄りピラミッドレベル選択＋必要なら縮小）の intent を
クリーンルーム再実装する（コピペはしない）レベル選択 :func:`best_level_for_side` は
``wsi`` を引数に取らない純関数にして ``features._best_level_and_read_size`` と数値一致を
テストで担保する全レベル読込の :func:`slide.read_image_at` は低倍サムネ専用に留める
"""

from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np
import openslide

from foveamil.preprocessing.features import MIN_QUALITY_FACTOR


def best_level_for_side(
    level_downsamples: Sequence[float],
    side0: int,
    out_px: int,
    min_quality: float = MIN_QUALITY_FACTOR,
) -> Tuple[int, int]:
    """level-0 一辺 ``side0`` を ``out_px`` で出すためのベストレベルと読み出しサイズを返す

    最上位レベルから下り ``side0 / downsample >= out_px * min_quality`` を満たす最初の
    レベルを採る``features._best_level_and_read_size`` と同じ規則（``patch_size`` を
    ``out_px``，``level0_size`` を ``side0`` とした一般化）

    Returns:
        ``(best_level, read_size)``
    """
    min_acceptable = out_px * min_quality
    best_level = 0
    read_size = int(side0)
    for level in range(len(level_downsamples) - 1, -1, -1):
        level_size = side0 / level_downsamples[level]
        if level_size >= min_acceptable:
            best_level = level
            read_size = int(np.ceil(level_size))
            break
    return best_level, read_size


class RegionReader:
    """WSI ハンドルを保持し level-0 矩形を読む

    Args:
        wsi_path: WSI ファイルの絶対パス
    """

    def __init__(self, wsi_path: str) -> None:
        self.wsi = openslide.OpenSlide(wsi_path)

    def read_level0_rect(
        self, x: int, y: int, side0: int, out_px: int
    ) -> np.ndarray:
        """level-0 左上 ``(x, y)``・一辺 ``side0`` の正方領域を ``out_px`` の RGB で返す

        Args:
            x: level-0 左上 x
            y: level-0 左上 y
            side0: level-0 上の一辺画素
            out_px: 出力の一辺画素

        Returns:
            ``[out_px, out_px, 3] uint8`` の RGB
        """
        best_level, read_size = best_level_for_side(
            self.wsi.level_downsamples, side0, out_px
        )
        region = self.wsi.read_region(
            location=(int(x), int(y)), level=best_level, size=(read_size, read_size)
        )
        img = np.asarray(region.convert("RGB"))
        if read_size != out_px:
            img = cv2.resize(img, (out_px, out_px), interpolation=cv2.INTER_LINEAR)
        return img

    def read_thumbnail(self, magnification: float, actual_max_mag: int) -> np.ndarray:
        """低倍率サムネを RGB で読む（全体オーバーレイのキャンバス用）"""
        from foveamil.wsi.slide import read_image_at

        img = read_image_at(self.wsi, magnification, actual_max_mag)
        return img[..., :3] if img.ndim == 3 and img.shape[-1] == 4 else img

    def close(self) -> None:
        """WSI ハンドルを閉じる"""
        self.wsi.close()

    def __enter__(self) -> "RegionReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
