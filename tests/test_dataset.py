"""FeatureBagDataset の空バッグ除外・h5 open リトライのユニット"""

import os
from unittest.mock import patch, MagicMock

import h5py
import numpy as np
import pandas as pd

from foveamil.training.accessor import (
    CLS_DATASET,
    COORDS_DATASET,
    FeatureAccessor,
)
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


def test_num_patches_uses_coords_independent_of_feature_type(tmp_path):
    # 座標件数で数える契約を固定する（feature_type 依存版へ退行すると落ちる）
    root = str(tmp_path / "feat")
    path = os.path.join(root, _ENC, f"{_MAG}x", "s.h5")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as handle:  # 特徴 dataset は置かず座標のみ
        handle.create_dataset(COORDS_DATASET, data=np.zeros((4, 2), np.int64))
    for ft in ("cls", "mean", "concat"):
        acc = FeatureAccessor(root, _ENC, "s", feature_type=ft)
        assert acc.num_patches(_MAG) == 4
        acc.close()


def test_num_patches_zero_when_coords_absent(tmp_path):
    # 座標が無いファイルは 0（空扱い）を返す
    root = str(tmp_path / "feat")
    path = os.path.join(root, _ENC, f"{_MAG}x", "s.h5")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(CLS_DATASET, data=np.zeros((3, _DIM), np.float16))
    acc = FeatureAccessor(root, _ENC, "s", feature_type="cls")
    assert acc.num_patches(_MAG) == 0
    acc.close()


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


# -- _open_retry tests ------------------------------------------------


def test_open_retry_succeeds_after_transient_errors(tmp_path):
    """Transient OSError is retried; after 2 failures the 3rd attempt succeeds."""
    from foveamil.training.accessor import _open_retry

    mock_handle = MagicMock()
    side = [OSError("busy"), OSError("busy"), mock_handle]
    with patch("foveamil.training.accessor.h5py.File", side_effect=side) as mock_file, \
         patch("foveamil.training.accessor.time.sleep") as mock_sleep:
        result = _open_retry("dummy.h5", retries=3, wait=0.0)
    assert result is mock_handle
    assert mock_file.call_count == 3
    # sleep called twice (after attempt 1 and 2, not after success)
    assert mock_sleep.call_count == 2


def test_open_retry_raises_filenotfounderror_immediately():
    """FileNotFoundError is never retried — raised on the first attempt."""
    from foveamil.training.accessor import _open_retry

    with patch("foveamil.training.accessor.h5py.File",
               side_effect=FileNotFoundError("no such file")) as mock_file, \
         patch("foveamil.training.accessor.time.sleep") as mock_sleep:
        try:
            _open_retry("missing.h5", retries=5, wait=0.0)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("FileNotFoundError was not raised")
    assert mock_file.call_count == 1
    mock_sleep.assert_not_called()


def test_open_retry_exhausts_retries():
    """If every attempt raises OSError, _open_retry re-raises after all retries."""
    import pytest
    from foveamil.training.accessor import _open_retry

    with patch("foveamil.training.accessor.h5py.File",
               side_effect=OSError("fail")), \
         patch("foveamil.training.accessor.time.sleep"):
        with pytest.raises(OSError, match="fail"):
            _open_retry("broken.h5", retries=3, wait=0.0)

