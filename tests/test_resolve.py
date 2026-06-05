"""resolve モジュールのユニット"""

import os

import pytest

from foveamil.training.resolve import (
    RECOMMENDED_FOLDS,
    SUPPORTED_FOLDS,
    ResolvedPaths,
    normalize_mags,
    resolve_in_feat_dim,
    resolve_paths,
    resolve_split_files,
)


def test_normalize_mags_accepts_suffix_and_float():
    assert normalize_mags(["1.25x", "2.5x"]) == [1.25, 2.5]
    assert normalize_mags([1.25, 2.5]) == [1.25, 2.5]
    assert normalize_mags(["1.25", "2.5"]) == [1.25, 2.5]
    assert normalize_mags([1, 2]) == [1.0, 2.0]


@pytest.mark.parametrize(
    "encoder,feature_type,expected",
    [
        ("ResNet50", "mean", 1024),
        ("UNI2-h", "mean", 1536),
        ("UNI2-h", "cls", 1536),
        ("UNI2-h", "concat", 3072),
        ("Virchow2", "concat", 2560),
        ("Virchow2-mini-dinov2", "cls", 384),
    ],
)
def test_resolve_in_feat_dim(encoder, feature_type, expected):
    assert resolve_in_feat_dim(encoder, feature_type) == expected


def test_resolve_in_feat_dim_unknown_encoder():
    with pytest.raises(KeyError):
        resolve_in_feat_dim("NoSuchEncoder", "mean")


def test_resolve_paths_rejects_unsupported_folds():
    assert 7 not in SUPPORTED_FOLDS
    with pytest.raises(ValueError):
        resolve_paths(3, 7, "cohort", "/tmp/feat")


def test_resolve_paths_rejects_unresolved_feature_root():
    with pytest.raises(ValueError):
        resolve_paths(3, RECOMMENDED_FOLDS, "cohort", "${UNSET_FEATURE_ROOT_VAR}")


def _make_cohort(tmp_path, n_cls, folds):
    labels = tmp_path / "labels" / f"labels_{n_cls}class.csv"
    labels.parent.mkdir(parents=True)
    labels.write_text("slide_id,label\nS1,A\nS2,B\n", encoding="utf-8")
    splits = tmp_path / "splits" / f"{n_cls}class" / f"cv{folds}"
    splits.mkdir(parents=True)
    for i in range(1, folds + 1):
        (splits / f"split_fold{i}.csv").write_text(
            "train,val,test\nS1,S2,S1\n", encoding="utf-8"
        )
    return str(splits)


def test_resolve_paths_success(tmp_path):
    splits_dir = _make_cohort(tmp_path, 3, 5)
    resolved = resolve_paths(3, 5, str(tmp_path), "/tmp/feat")
    assert isinstance(resolved, ResolvedPaths)
    assert resolved.splits_dir == splits_dir
    assert resolved.labels_csv.endswith("labels_3class.csv")
    assert resolved.feature_root_base == "/tmp/feat"


def test_resolve_paths_missing_splits_hints_generation(tmp_path):
    labels = tmp_path / "labels" / "labels_3class.csv"
    labels.parent.mkdir(parents=True)
    labels.write_text("slide_id,label\nS1,A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="foveamil-cohort splits"):
        resolve_paths(3, 10, str(tmp_path), "/tmp/feat")


def test_resolve_split_files_counts_must_match(tmp_path):
    splits_dir = _make_cohort(tmp_path, 11, 5)
    files = resolve_split_files(splits_dir, 5)
    assert len(files) == 5
    assert [os.path.basename(p) for p in files] == [
        f"split_fold{i}.csv" for i in range(1, 6)
    ]
    with pytest.raises(ValueError):
        resolve_split_files(splits_dir, 10)
