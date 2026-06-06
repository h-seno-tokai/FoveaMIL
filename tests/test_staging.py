"""特徴ステージング（cls-only 縮小コピー・容量判定・全体複製）のユニット"""

import os

import h5py
import numpy as np

from foveamil.training.accessor import (
    CLS_DATASET,
    COORDS_DATASET,
    POOLED_DATASET,
    FeatureAccessor,
)
from foveamil.training.staging import FeatureStager, _needed_datasets

_ENCODER = "Virchow2"
_MAG = 40.0
_DIM = 8
_N = 50


def _write_feature_h5(path, n=_N, dim=_DIM):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(POOLED_DATASET, data=np.random.rand(n, dim).astype(np.float32))
        handle.create_dataset(CLS_DATASET, data=np.random.rand(n, dim).astype(np.float32))
        handle.create_dataset(COORDS_DATASET, data=np.arange(n * 2).reshape(n, 2).astype(np.int64))
        handle.attrs["actual_max_mag"] = 40.0


def _make_root(tmp_path, slides=("s0", "s1")):
    root = str(tmp_path / "feat")
    for s in slides:
        _write_feature_h5(os.path.join(root, _ENCODER, f"{_MAG}x", f"{s}.h5"))
    return root, list(slides)


def test_needed_datasets_mapping():
    assert _needed_datasets("cls") == {CLS_DATASET, COORDS_DATASET}
    assert _needed_datasets("mean") == {POOLED_DATASET, COORDS_DATASET}
    assert _needed_datasets("concat") is None
    assert _needed_datasets(None) is None


def test_stage_cls_only_drops_pooled_and_is_readable(tmp_path):
    root, slides = _make_root(tmp_path)
    cache = str(tmp_path / "cache")
    stager = FeatureStager(cache_dir=cache)
    staged = stager.stage_set(root, _ENCODER, [_MAG], slides, feature_type="cls")
    assert staged == cache

    staged_h5 = os.path.join(cache, _ENCODER, f"{_MAG}x", "s0.h5")
    with h5py.File(staged_h5, "r") as handle:
        assert CLS_DATASET in handle and COORDS_DATASET in handle
        assert POOLED_DATASET not in handle  # pooled は削られる
        assert handle.attrs["actual_max_mag"] == 40.0

    # 縮小 h5 は元より小さい
    assert os.path.getsize(staged_h5) < os.path.getsize(
        os.path.join(root, _ENCODER, f"{_MAG}x", "s0.h5")
    )

    # FeatureAccessor が cls 特徴を正しく読める
    acc = FeatureAccessor(cache, _ENCODER, "s0", feature_type="cls")
    feats = acc.load_all(_MAG)
    assert feats.shape == (_N, _DIM)
    acc.close()


def test_stage_full_keeps_all_datasets(tmp_path):
    root, slides = _make_root(tmp_path)
    cache = str(tmp_path / "cache")
    stager = FeatureStager(cache_dir=cache)
    staged = stager.stage_set(root, _ENCODER, [_MAG], slides, feature_type=None)
    staged_h5 = os.path.join(staged, _ENCODER, f"{_MAG}x", "s0.h5")
    with h5py.File(staged_h5, "r") as handle:
        assert POOLED_DATASET in handle and CLS_DATASET in handle and COORDS_DATASET in handle


def test_required_bytes_cls_is_about_half_of_full(tmp_path):
    root, slides = _make_root(tmp_path)
    stager = FeatureStager(cache_dir=str(tmp_path / "cache"))
    rels = stager._target_files(root, _ENCODER, [_MAG], slides)
    full = stager._required_bytes(root, rels, None)
    cls_only = stager._required_bytes(root, rels, {CLS_DATASET, COORDS_DATASET})
    # cls + coords は patches + cls + coords より小さい（pooled の分だけ減る）
    assert 0 < cls_only < full


def test_stage_mean_only_drops_cls(tmp_path):
    root, slides = _make_root(tmp_path)
    cache = str(tmp_path / "cache")
    stager = FeatureStager(cache_dir=cache)
    stager.stage_set(root, _ENCODER, [_MAG], slides, feature_type="mean")
    with h5py.File(os.path.join(cache, _ENCODER, f"{_MAG}x", "s0.h5"), "r") as handle:
        assert POOLED_DATASET in handle and COORDS_DATASET in handle
        assert CLS_DATASET not in handle


def test_stage_subset_tolerates_absent_dataset(tmp_path):
    # cls 特徴の無い h5（pooled + coords のみ）に対し cls ステージしても壊れない
    root = str(tmp_path / "feat")
    path = os.path.join(root, _ENCODER, f"{_MAG}x", "s0.h5")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(POOLED_DATASET, data=np.zeros((4, _DIM), np.float32))
        handle.create_dataset(COORDS_DATASET, data=np.zeros((4, 2), np.int64))
    cache = str(tmp_path / "cache")
    stager = FeatureStager(cache_dir=cache)
    stager.stage_set(root, _ENCODER, [_MAG], ["s0"], feature_type="cls")
    with h5py.File(os.path.join(cache, _ENCODER, f"{_MAG}x", "s0.h5"), "r") as handle:
        assert COORDS_DATASET in handle and CLS_DATASET not in handle


def test_stage_falls_back_when_too_large(tmp_path):
    root, slides = _make_root(tmp_path)
    # 空きのほぼ全量をマージンに取れば必ず収まらない -> NAS 直読フォールバック
    stager = FeatureStager(cache_dir=str(tmp_path / "cache"), free_space_margin=1.0)
    staged = stager.stage_set(root, _ENCODER, [_MAG], slides, feature_type="cls")
    assert staged == root
