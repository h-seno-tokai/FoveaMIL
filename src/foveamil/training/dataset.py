"""最低倍率の特徴量バッグを返す学習用データセット（Lazy 方式）

正準レイアウト ``{feature_root}/{encoder}/{mag}x/{slide_id}.h5`` のうち，最低倍率
（``magnifications`` の先頭）の全パッチのみを ``__getitem__`` で返す高倍率は学習
ループ側が選択結果に応じて :class:`FeatureAccessor` で都度ロードするため，ここでは
読まない``feature_type`` に応じて pooled 特徴（``patches``）/ cls 特徴
（``patches_cls``）/ 両者の特徴次元連結（``concat``）を選ぶ
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from foveamil.training.accessor import (
    FEATURE_TYPE_MEAN,
    FEATURE_TYPES,
    FeatureAccessor,
)

logger = logging.getLogger(__name__)


def build_label_dict(
    labels_csv: str, classes: Optional[Sequence[str]] = None
) -> Dict[str, int]:
    """``slide_id,label`` の CSV からクラス名→整数の辞書を作る

    ``classes`` 未指定時は CSV のユニークな ``label`` をソートして ``0..K-1`` を割り当てる
    指定時はその順序で ``0..K-1`` を割り当てる

    Args:
        labels_csv: ``slide_id,label`` 列を持つ CSV のパス
        classes: クラス名の並び指定時はこの順序で整数を割り当てる

    Returns:
        ``{クラス名: int}`` の辞書
    """
    if classes is None:
        labels = pd.read_csv(labels_csv)["label"].astype(str)
        classes = sorted(set(labels.tolist()))
    return {name: idx for idx, name in enumerate(classes)}


def feature_bag_collate(batch: List[Tuple[Tensor, str, int]]) -> Tuple[Tensor, str, int]:
    """バッチサイズ 1 前提の collate

    1 件の ``(base_feats[N,d], slide_id, label_int)`` を
    ``(base_feats[1,N,d], slide_id, label[1])`` に整える

    Args:
        batch: 長さ 1 の ``(base_feats, slide_id, label_int)`` のリスト

    Returns:
        ``(base_feats[1,N,d], slide_id, label[1])``
    """
    base_feats, slide_id, label = batch[0]
    return base_feats.unsqueeze(0), slide_id, torch.tensor([label], dtype=torch.long)


class FeatureBagDataset(Dataset):
    """最低倍率の特徴量バッグを返すデータセット

    Args:
        feature_root: 特徴ルートディレクトリ（``{encoder}/{mag}x/{slide_id}.h5``）
        encoder: エンコーダ名（特徴ルート直下のディレクトリ名）
        magnifications: 倍率の列（低→高の順を保ち倍率レイヤ順とする）
        slide_ids: この分割に含める ``slide_id`` の集合/列
        labels_csv: ``slide_id,label`` 列を持つ CSV のパス
        label_dict: クラス名→整数の辞書``label_dict`` に無いラベルの行は除外する
        feature_type: ``"mean"`` / ``"cls"`` / ``"concat"`` のいずれか

    Attributes:
        feature_root: 特徴ルートディレクトリ
        encoder: エンコーダ名
        magnifications: 倍率の列（低→高）
        feature_type: feature_type
        base_mag: 最低倍率（``magnifications`` の先頭）
        samples: ``(slide_id, label_int)`` の列
        n_cls: クラス数
    """

    def __init__(
        self,
        feature_root: str,
        encoder: str,
        magnifications: Sequence[float],
        slide_ids: Sequence[str],
        labels_csv: str,
        label_dict: Dict[str, int],
        feature_type: str = FEATURE_TYPE_MEAN,
    ) -> None:
        if feature_type not in FEATURE_TYPES:
            raise ValueError(
                f"feature_type must be one of {FEATURE_TYPES}, got '{feature_type}'"
            )
        if not magnifications:
            raise ValueError("magnifications must be non-empty")

        self.feature_root = feature_root
        self.encoder = encoder
        self.magnifications = list(magnifications)
        self.base_mag = self.magnifications[0]
        self.label_dict = dict(label_dict)
        self.feature_type = feature_type

        self.samples = self._build_samples(labels_csv, slide_ids)
        self.n_cls = len(set(self.label_dict.values()))

    def _build_samples(
        self, labels_csv: str, slide_ids: Sequence[str]
    ) -> List[Tuple[str, int]]:
        """``slide_ids`` で絞り ``(slide_id, label_int)`` の列を作る

        ``label_dict`` に無いラベルの行は除外する

        Args:
            labels_csv: ``slide_id,label`` 列を持つ CSV のパス
            slide_ids: 残す ``slide_id`` の集合/列

        Returns:
            ``(slide_id, label_int)`` の列（CSV の行順を保つ）
        """
        wanted = set(str(s) for s in slide_ids)
        df = pd.read_csv(labels_csv)
        df["slide_id"] = df["slide_id"].astype(str)
        df["label"] = df["label"].astype(str)

        samples: List[Tuple[str, int]] = []
        skipped = 0
        for slide_id, label in zip(df["slide_id"], df["label"]):
            if slide_id not in wanted:
                continue
            if label not in self.label_dict:
                skipped += 1
                continue
            samples.append((slide_id, self.label_dict[label]))

        if skipped:
            logger.info("skipped %d slides with labels outside label_dict", skipped)
        logger.info("FeatureBagDataset: %d slides selected", len(samples))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def get_label(self, idx: int) -> int:
        """サンプル ``idx`` のラベル整数を返す（サンプラ重み計算用）"""
        return self.samples[idx][1]

    def __getitem__(self, idx: int) -> Tuple[Tensor, str, int]:
        """最低倍率の全特徴とスライド識別子とラベル整数を返す

        Args:
            idx: サンプル添字

        Returns:
            ``(base_feats[N, dim], slide_id, label_int)``base_feats は ``float32``
        """
        slide_id, label = self.samples[idx]
        accessor = FeatureAccessor(
            self.feature_root, self.encoder, slide_id, self.feature_type
        )
        base_feats = accessor.load_all(self.base_mag).float()
        accessor.close()
        return base_feats, slide_id, label

    def class_counts(self) -> Dict[int, int]:
        """クラス整数ごとのサンプル件数を返す"""
        return dict(Counter(label for _, label in self.samples))
