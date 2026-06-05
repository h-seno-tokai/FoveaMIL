"""utils.provenance のユニット"""

from foveamil.utils.provenance import (
    META_SCHEMA_VERSION,
    collect_env,
    collect_run_meta,
    file_sha256,
    git_dirty,
    git_sha,
)


def test_git_helpers_graceful_outside_repo(tmp_path):
    # git 管理外/未初期化でも例外を投げず None を返す
    assert git_sha(str(tmp_path)) is None
    assert git_dirty(str(tmp_path)) is None


def test_collect_env_has_keys():
    env = collect_env()
    for key in ("python", "platform", "hostname", "torch", "cuda", "gpu_name"):
        assert key in env
    assert env["python"]  # python 版は必ず取れる


def test_file_sha256(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"hello")
    # echo -n hello | sha256sum
    assert file_sha256(str(path)) == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )
    assert file_sha256(None) is None
    assert file_sha256(str(tmp_path / "nope.txt")) is None


def test_collect_run_meta_structure(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("slide_id,label\nS1,A\n", encoding="utf-8")
    meta = collect_run_meta(
        config={"encoder": "ResNet50", "feature_type": "mean",
                "magnifications": [1.25, 2.5], "n_cls": 3, "classes": ["A", "B", "C"]},
        selection={"save_metric": "f1", "best_epoch": 7},
        timing={"start": "t0", "end": "t1", "duration_sec": 12.3},
        labels_csv=str(labels),
        split_csv=None,
        class_breakdown={"train": {"A": 10}, "val": {"A": 2}, "test": {"A": 2}},
        version="0.1.0",
    )
    assert meta["schema_version"] == META_SCHEMA_VERSION
    # git 管理下なら SHA 文字列, 管理外なら None（環境に依存しない検証）
    assert meta["code"]["git_sha"] is None or isinstance(meta["code"]["git_sha"], str)
    assert meta["code"]["foveamil_version"] == "0.1.0"
    assert meta["data"]["labels_csv_sha256"] is not None
    assert meta["data"]["split_csv_sha256"] is None
    assert meta["data"]["encoder"] == "ResNet50"
    assert meta["data"]["class_breakdown"]["train"] == {"A": 10}
    assert meta["selection"]["best_epoch"] == 7
    assert meta["duration_sec"] == 12.3
