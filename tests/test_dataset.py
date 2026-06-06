"""FeatureBagDataset の空バッグ除外のユニット"""

import os

import h5py
import numpy as np
import pandas as pd

from foveamil.training.accessor import CLS_DATASET, COORDS_DATASET
from foveamil.training.dataset import FeatureBagDataset

_ENC = "Virchow2"
_MAG = 20.0
_DIM = 8


def _write_h5(path, n):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            CLS_DATASET, data=np.random.rand(n, _DIM).astype(np.float16)
        )
        handle.create_dataset(
            COORDS_DATASET, data=np.zeros((n, 2), np.int64)
        )


def test_empty_bag_slides_are_excluded(tmp_path):
    # 空抽出スライド（パッチ 0）はサンプルから除外され非空のみ残る
    root = str(tmp_path / "feat")
    _write_h5(os.path.join(root, _ENC, f"{_MAG}x", "s_ok.h5"), 5)
    _write_h5(os.path.join(root, _ENC, f"{_MAG}x", "s_empty.h5"), 0)
    labels = str(tmp_path / "labels.csv")
    pd.DataFrame(
        {"slide_id": ["s_ok", "s_empty"], "label": ["A", "A"]}
    ).to_csv(labels, index=False)

    ds = FeatureBagDataset(
        root, _ENC, [_MAG], ["s_ok", "s_empty"], labels, {"A": 0},
        feature_type="cls",
    )
    ids = [s for s, _ in ds.samples]
    assert ids == ["s_ok"]
    assert len(ds) == 1


def test_missing_feature_file_treated_as_empty(tmp_path):
    # 特徴ファイルが無いスライドも学習に使えないため除外する
    root = str(tmp_path / "feat")
    _write_h5(os.path.join(root, _ENC, f"{_MAG}x", "s_ok.h5"), 3)
    labels = str(tmp_path / "labels.csv")
    pd.DataFrame(
        {"slide_id": ["s_ok", "s_absent"], "label": ["A", "A"]}
    ).to_csv(labels, index=False)

    ds = FeatureBagDataset(
        root, _ENC, [_MAG], ["s_ok", "s_absent"], labels, {"A": 0},
        feature_type="cls",
    )
    assert [s for s, _ in ds.samples] == ["s_ok"]
