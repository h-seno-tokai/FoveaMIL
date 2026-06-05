"""RGB 画像から組織マスク（組織=1 / 背景=0）を生成する

手順:
  1. RGB を HSV に変換し彩度チャネルを取り出す
  2. Otsu の自動しきい値で彩度を二値化する
  3. Gaussian 平滑をかけ，再二値化して滑らかなマスクにする
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# Otsu 二値化のターゲット最大値組織=1 のバイナリマスクを得る
_OTSU_MAX_VALUE = 1
# Gaussian 平滑後に再二値化するしきい値（0..1 の比率）
_SMOOTH_BINARIZE_THRESHOLD = 0.5
# 既定の Gaussian 標準偏差大きいほどマスクが滑らかになる
DEFAULT_SIGMA = 20


class SimpleTissueMask:
    """彩度の Otsu 二値化と Gaussian 平滑による組織マスク生成器

    設定値（``sigma``）のみを保持し，``multiprocessing.Pool`` で pickle できる

    Args:
        sigma: Gaussian 平滑の標準偏差``0`` 以下なら平滑をスキップする
    """

    def __init__(self, sigma: int = DEFAULT_SIGMA) -> None:
        self.sigma = sigma

    def process(self, image: np.ndarray) -> np.ndarray:
        """RGB 画像から組織マスク（``uint8``，組織=1/背景=0）を生成する

        Args:
            image: ``(H, W, 3)`` の RGB 画像（``uint8``）

        Returns:
            ``(H, W)`` の ``uint8`` バイナリマスク（組織=1, 背景=0）
        """
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation = image_hsv[:, :, 1]

        # Otsu: しきい値そのものは不要（_ で捨てる），二値マスクだけ受け取る
        _, mask = cv2.threshold(
            saturation, 0, _OTSU_MAX_VALUE, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        if self.sigma > 0:
            smoothed = gaussian_filter(mask.astype(float), sigma=self.sigma)
            mask = (smoothed > _SMOOTH_BINARIZE_THRESHOLD).astype(np.uint8)

        return mask


def make_tissue_mask(image: np.ndarray, mask_generator: SimpleTissueMask) -> np.ndarray:
    """彩度の Otsu 二値化で得た背景を白(255)で塗ってから組織マスクを生成する

    背景を白で埋めた画像に :meth:`SimpleTissueMask.process` を適用する

    Args:
        image: ``(H, W, 3)`` の RGB 画像（``uint8``）
        mask_generator: 設定済みの :class:`SimpleTissueMask`

    Returns:
        ``(H, W)`` の ``uint8`` バイナリマスク（組織=1, 背景=0）
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = image_hsv[:, :, 1]
    _, raw_mask = cv2.threshold(
        saturation, 0, _OTSU_MAX_VALUE, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 明背景を白で塗ってから本処理へ元画像は壊さずコピーする
    img_masked = image.copy()
    img_masked[raw_mask == 0] = 255

    return mask_generator.process(image=img_masked)
