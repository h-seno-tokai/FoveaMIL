"""正準レイアウトの特徴 H5 への部分ロードアクセサ

正準レイアウト ``{feature_root}/{encoder}/{mag}x/{slide_id}.h5`` の特徴を倍率ごとに
読む全パッチを読む ``load_all`` と，指定 index のパッチのみを読む ``load_patches``
を提供する``load_patches`` は h5py の fancy index 制約（昇順かつ重複不可）に対応
するため ``np.unique`` でユニーク昇順読みしてから元順序へ展開する（重複 index 可）
``feature_type`` に応じて pooled 特徴（``patches``）/ cls 特徴（``patches_cls``）/
両者の特徴次元連結（``concat``）を読む倍率をまたいで何度も開くため，倍率ごとの
ファイルハンドルをキャッシュする
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict

import h5py
import numpy as np
import torch
from torch import Tensor

# pooled 特徴を読む feature_type
FEATURE_TYPE_MEAN = "mean"
# cls 特徴を読む feature_type
FEATURE_TYPE_CLS = "cls"
# pooled と cls を特徴次元で連結する feature_type
FEATURE_TYPE_CONCAT = "concat"
# 選べる feature_type の一覧
FEATURE_TYPES = (FEATURE_TYPE_MEAN, FEATURE_TYPE_CLS, FEATURE_TYPE_CONCAT)
# pooled 特徴の dataset 名
POOLED_DATASET = "patches"
# cls 特徴の dataset 名
CLS_DATASET = "patches_cls"
# 座標の dataset 名
COORDS_DATASET = "coords"
# 座標を載せる numpy dtype
COORDS_DTYPE = np.int64
# 特徴を載せる numpy dtype
FEATURE_DTYPE = np.float32
# concat 時の特徴次元連結軸
CONCAT_AXIS = 1

# h5 open retry settings for transient I/O errors (e.g. NAS under load)
H5_OPEN_RETRIES = 5
H5_OPEN_WAIT_SEC = 1.0

logger = logging.getLogger(__name__)


def _format_mag(magnification: float) -> str:
    """倍率を正準レイアウトのディレクトリ名（例 ``1.25`` → ``"1.25x"``）にする"""
    return f"{magnification}x"


def _open_retry(
    path: str,
    retries: int = H5_OPEN_RETRIES,
    wait: float = H5_OPEN_WAIT_SEC,
) -> h5py.File:
    """Open an h5 file with linear-backoff retry on transient ``OSError``.

    ``FileNotFoundError`` is raised immediately without retry so that callers
    can distinguish a genuinely missing file from a transient I/O failure.

    Args:
        path: h5 file path
        retries: maximum number of attempts (≥1)
        wait: base wait in seconds; actual sleep is ``wait * attempt``

    Returns:
        An open ``h5py.File`` handle in read mode.

    Raises:
        FileNotFoundError: if the file does not exist (no retry)
        OSError: if all retry attempts are exhausted
    """
    for attempt in range(1, retries + 1):
        try:
            return h5py.File(path, "r")
        except FileNotFoundError:
            raise
        except OSError:
            if attempt == retries:
                raise
            logger.warning(
                "h5 open failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt,
                retries,
                wait * attempt,
                path,
            )
            time.sleep(wait * attempt)
    # unreachable – kept for type-checker satisfaction
    raise OSError(f"failed to open {path} after {retries} attempts")


class FeatureAccessor:
    """正準レイアウトの特徴 H5 への部分ロードアクセサ

    Args:
        feature_root: 特徴ルートディレクトリ
        encoder: エンコーダ名（特徴ルート直下のディレクトリ名）
        slide_id: スライド識別子
        feature_type: ``"mean"`` / ``"cls"`` / ``"concat"`` のいずれか
    """

    def __init__(
        self,
        feature_root: str,
        encoder: str,
        slide_id: str,
        feature_type: str = FEATURE_TYPE_MEAN,
    ) -> None:
        if feature_type not in FEATURE_TYPES:
            raise ValueError(
                f"feature_type must be one of {FEATURE_TYPES}, got '{feature_type}'"
            )
        self.feature_root = feature_root
        self.encoder = encoder
        self.slide_id = slide_id
        self.feature_type = feature_type
        self._handles: Dict[float, h5py.File] = {}

    def _path(self, magnification: float) -> str:
        """特徴 H5 パス ``{feature_root}/{encoder}/{mag}x/{slide_id}.h5`` を返す"""
        return os.path.join(
            self.feature_root,
            self.encoder,
            _format_mag(magnification),
            f"{self.slide_id}.h5",
        )

    def _file(self, magnification: float) -> h5py.File:
        """倍率のファイルハンドルを返す（キャッシュし無ければ開く）"""
        handle = self._handles.get(magnification)
        if handle is None:
            handle = _open_retry(self._path(magnification))
            self._handles[magnification] = handle
        return handle

    def _read_all(self, magnification: float) -> np.ndarray:
        """``feature_type`` に応じた全特徴配列 ``(N, dim) float32`` を読む"""
        handle = self._file(magnification)
        if self.feature_type == FEATURE_TYPE_MEAN:
            return np.asarray(handle[POOLED_DATASET][:], dtype=FEATURE_DTYPE)
        if self.feature_type == FEATURE_TYPE_CLS:
            return np.asarray(handle[CLS_DATASET][:], dtype=FEATURE_DTYPE)
        pooled = np.asarray(handle[POOLED_DATASET][:], dtype=FEATURE_DTYPE)
        cls = np.asarray(handle[CLS_DATASET][:], dtype=FEATURE_DTYPE)
        return np.concatenate([pooled, cls], axis=CONCAT_AXIS)

    def _read_indexed(
        self, magnification: float, unique_indices: np.ndarray
    ) -> np.ndarray:
        """昇順ユニーク index で部分読みした特徴 ``(U, dim) float32`` を返す"""
        handle = self._file(magnification)
        if self.feature_type == FEATURE_TYPE_MEAN:
            return np.asarray(
                handle[POOLED_DATASET][unique_indices], dtype=FEATURE_DTYPE
            )
        if self.feature_type == FEATURE_TYPE_CLS:
            return np.asarray(
                handle[CLS_DATASET][unique_indices], dtype=FEATURE_DTYPE
            )
        pooled = np.asarray(
            handle[POOLED_DATASET][unique_indices], dtype=FEATURE_DTYPE
        )
        cls = np.asarray(
            handle[CLS_DATASET][unique_indices], dtype=FEATURE_DTYPE
        )
        return np.concatenate([pooled, cls], axis=CONCAT_AXIS)

    def num_patches(self, magnification: float) -> int:
        """指定倍率のパッチ数を返す（データを読まず shape のみ参照）

        座標データセットの行数をパッチ数とみなす特徴抽出が空（パッチ 0）の
        スライド検出に使う

        Args:
            magnification: 倍率

        Returns:
            パッチ数（座標が無ければ 0）
        """
        handle = self._file(magnification)
        if COORDS_DATASET in handle:
            return int(handle[COORDS_DATASET].shape[0])
        return 0

    def load_all(self, magnification: float) -> Tensor:
        """指定倍率の全特徴 ``(N, dim) float32`` テンソルを返す

        Args:
            magnification: 倍率

        Returns:
            全特徴テンソル ``[N, dim]``
        """
        return torch.from_numpy(self._read_all(magnification))

    def load_patches(self, magnification: float, indices: np.ndarray) -> Tensor:
        """指定倍率の指定 index の特徴 ``(len(indices), dim) float32`` を返す

        h5py の fancy index 制約に対応するため ``np.unique`` でユニーク昇順読みし，
        inverse map で要求順（重複可）へ展開する

        Args:
            magnification: 倍率
            indices: 取り出す index の配列（重複可）

        Returns:
            要求順に並んだ特徴テンソル ``[len(indices), dim]``
        """
        indices = np.asarray(indices, dtype=np.int64)
        unique_indices, inverse = np.unique(indices, return_inverse=True)
        unique_feats = self._read_indexed(magnification, unique_indices)
        feats = unique_feats[inverse]
        return torch.from_numpy(feats)

    def load_coords_all(self, magnification: float) -> np.ndarray:
        """指定倍率の全座標 ``(N, 2)`` を返す

        Args:
            magnification: 倍率

        Returns:
            座標配列 ``(N, 2)``
        """
        handle = self._file(magnification)
        return np.asarray(handle[COORDS_DATASET][:], dtype=COORDS_DTYPE)

    def load_coords_indexed(
        self, magnification: float, indices: np.ndarray
    ) -> np.ndarray:
        """指定倍率の指定 index の座標 ``(len(indices), 2)`` を返す

        h5py の fancy index 制約に対応するため ``np.unique`` でユニーク昇順読みし，
        inverse map で要求順（重複可）へ展開する

        Args:
            magnification: 倍率
            indices: 取り出す index の配列（重複可）

        Returns:
            要求順に並んだ座標配列 ``(len(indices), 2)``
        """
        indices = np.asarray(indices, dtype=np.int64)
        unique_indices, inverse = np.unique(indices, return_inverse=True)
        handle = self._file(magnification)
        unique_coords = np.asarray(
            handle[COORDS_DATASET][unique_indices], dtype=COORDS_DTYPE
        )
        return unique_coords[inverse]

    def feature_dim(self, magnification: float) -> int:
        """指定倍率の特徴次元を返す（``concat`` は両 dataset の次元和）"""
        handle = self._file(magnification)
        if self.feature_type == FEATURE_TYPE_CLS:
            return int(handle[CLS_DATASET].shape[CONCAT_AXIS])
        if self.feature_type == FEATURE_TYPE_CONCAT:
            return int(
                handle[POOLED_DATASET].shape[CONCAT_AXIS]
                + handle[CLS_DATASET].shape[CONCAT_AXIS]
            )
        return int(handle[POOLED_DATASET].shape[CONCAT_AXIS])

    def close(self) -> None:
        """キャッシュした全ファイルハンドルを閉じる"""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
