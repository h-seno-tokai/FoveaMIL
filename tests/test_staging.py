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


def test_resolve_copy_workers_precedence(monkeypatch):
    from foveamil.training.staging import (
        DEFAULT_COPY_WORKERS,
        STAGE_WORKERS_ENV,
        _resolve_copy_workers,
    )

    monkeypatch.delenv(STAGE_WORKERS_ENV, raising=False)
    assert _resolve_copy_workers(None) == DEFAULT_COPY_WORKERS  # 既定
    assert _resolve_copy_workers(4) == 4                        # 引数優先
    assert _resolve_copy_workers(0) == 1                        # 下限 1
    monkeypatch.setenv(STAGE_WORKERS_ENV, "3")
    assert _resolve_copy_workers(None) == 3                     # 環境変数
    assert _resolve_copy_workers(5) == 5                        # 引数 > 環境変数


def test_stage_parallel_copies_all_slides(tmp_path):
    # プロセス並列パス（copy_workers>1）で全スライドが正しく縮小コピーされる
    slides = tuple(f"s{i}" for i in range(5))
    root, slide_ids = _make_root(tmp_path, slides=slides)
    cache = str(tmp_path / "cache")
    stager = FeatureStager(cache_dir=cache, copy_workers=2)
    staged = stager.stage_set(root, _ENCODER, [_MAG], slide_ids, feature_type="cls")
    assert staged == cache
    for s in slides:
        path = os.path.join(cache, _ENCODER, f"{_MAG}x", f"{s}.h5")
        with h5py.File(path, "r") as handle:
            assert CLS_DATASET in handle and COORDS_DATASET in handle
            assert POOLED_DATASET not in handle


def test_stage_fp16_stores_half_precision_and_loads_float32(tmp_path):
    # fp16 ステージ: 特徴は float16 で保存 座標は元 dtype accessor は float32 で読む
    root, slides = _make_root(tmp_path)
    cache = str(tmp_path / "cache16")
    FeatureStager(cache_dir=cache, store_fp16=True).stage_set(
        root, _ENCODER, [_MAG], slides, feature_type="cls"
    )
    staged_h5 = os.path.join(cache, _ENCODER, f"{_MAG}x", "s0.h5")
    with h5py.File(staged_h5, "r") as handle:
        assert handle[CLS_DATASET].dtype == np.float16   # 特徴は fp16
        assert handle[COORDS_DATASET].dtype == np.int64  # 座標は元 dtype

    cache32 = str(tmp_path / "cache32")
    FeatureStager(cache_dir=cache32).stage_set(
        root, _ENCODER, [_MAG], slides, feature_type="cls"
    )
    fp32_h5 = os.path.join(cache32, _ENCODER, f"{_MAG}x", "s0.h5")
    assert os.path.getsize(staged_h5) < os.path.getsize(fp32_h5)

    acc = FeatureAccessor(cache, _ENCODER, "s0", feature_type="cls")
    feats = acc.load_all(_MAG)
    acc.close()
    assert feats.numpy().dtype == np.float32             # load 時に fp32 へ復元
    with h5py.File(os.path.join(root, _ENCODER, f"{_MAG}x", "s0.h5"), "r") as h:
        orig = h[CLS_DATASET][()]
    assert np.allclose(feats.numpy(), orig, atol=1e-2)   # fp16 丸め以内


def test_required_bytes_fp16_smaller_than_fp32(tmp_path):
    root, slides = _make_root(tmp_path)
    keep = {CLS_DATASET, COORDS_DATASET}
    s32 = FeatureStager(cache_dir=str(tmp_path / "c32"))
    s16 = FeatureStager(cache_dir=str(tmp_path / "c16"), store_fp16=True)
    rels = s32._target_files(root, _ENCODER, [_MAG], slides)
    b32 = s32._required_bytes(root, rels, keep)
    b16 = s16._required_bytes(root, rels, keep)
    assert 0 < b16 < b32   # cls 分は半分 coords は不変なので半分よりは大きい


def test_required_bytes_sampled_estimate_close_to_exact(tmp_path):
    # サイズの異なる多数スライドで サンプル推定が全件合計に近いことを確認
    root = str(tmp_path / "feat")
    slides = [f"s{i}" for i in range(80)]
    for i, s in enumerate(slides):
        _write_feature_h5(
            os.path.join(root, _ENCODER, f"{_MAG}x", f"{s}.h5"), n=_N + i
        )
    stager = FeatureStager(cache_dir=str(tmp_path / "cache"))
    rels = stager._target_files(root, _ENCODER, [_MAG], slides)
    keep = {CLS_DATASET, COORDS_DATASET}
    estimate = stager._required_bytes(root, rels, keep)

    exact = 0
    for rel in rels:
        with h5py.File(os.path.join(root, rel), "r") as handle:
            for name in keep:
                dataset = handle[name]
                exact += dataset.dtype.itemsize * dataset.size
    assert 0.8 * exact <= estimate <= 1.2 * exact
