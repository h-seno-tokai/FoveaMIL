"""特徴抽出のアトミック書き込み・デバイス解決・再開フィルタの単体検証"""

import types

import h5py
import numpy as np
import pytest

from foveamil.preprocessing import features


class _FakeEncoder:
    """``_write_features_h5`` が参照する属性だけを持つスタブ"""

    def __init__(self, name="ResNet50", feature_dim=4, has_cls=False):
        self.name = name
        self.feature_dim = feature_dim
        self.has_cls = has_cls


def _write_sample(out_path, encoder, n=3):
    coords = np.arange(n * 2, dtype=np.int32).reshape(n, 2)
    pooled = np.ones((n, encoder.feature_dim), dtype=np.float32)
    features._write_features_h5(
        out_path, coords, pooled, None,
        slide_id="S1", encoder=encoder, magnification=1.25,
    )
    return coords, pooled


def test_write_features_h5_writes_valid_layout(tmp_path):
    out_path = str(tmp_path / "enc" / "1.25x" / "S1.h5")
    encoder = _FakeEncoder()
    coords, pooled = _write_sample(out_path, encoder)
    with h5py.File(out_path, "r") as f:
        assert np.array_equal(f["coords"][:], coords)
        assert np.allclose(f["patches"][:], pooled)
        assert "patches_cls" not in f
        assert f.attrs["case_id"] == "S1"
        assert f.attrs["n_patches"] == 3
        assert f.attrs["magnification"] == "1.25x"


def test_write_features_h5_leaves_no_tmp_on_failure(tmp_path, monkeypatch):
    out_dir = tmp_path / "enc" / "1.25x"
    out_dir.mkdir(parents=True)
    out_path = str(out_dir / "S1.h5")

    class _RaisingFile:
        def __init__(self, path, mode):
            open(path, "wb").write(b"partial")  # tmp 作成後の書き込み中失敗を模す
            raise RuntimeError("boom")

    monkeypatch.setattr(features, "h5py", types.SimpleNamespace(File=_RaisingFile))
    with pytest.raises(RuntimeError):
        _write_sample(out_path, _FakeEncoder())

    assert not __import__("os").path.exists(out_path)
    leftovers = [p for p in out_dir.iterdir() if features.TMP_SUFFIX in p.name]
    assert leftovers == []


def test_write_features_h5_failure_keeps_existing_file(tmp_path, monkeypatch):
    out_path = str(tmp_path / "enc" / "1.25x" / "S1.h5")
    encoder = _FakeEncoder()
    _write_sample(out_path, encoder, n=2)  # 既存の完成ファイル

    class _RaisingFile:
        def __init__(self, path, mode):
            open(path, "wb").write(b"partial")
            raise RuntimeError("boom")

    monkeypatch.setattr(features, "h5py", types.SimpleNamespace(File=_RaisingFile))
    with pytest.raises(RuntimeError):
        _write_sample(out_path, encoder, n=5)

    # os.replace に到達しないので既存ファイルは無傷
    with h5py.File(out_path, "r") as f:
        assert f.attrs["n_patches"] == 2


def test_resolve_devices_from_visible_env(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    assert features.resolve_worker_devices(None) == [0, 1, 2]
    assert features.resolve_worker_devices([2, 1]) == [2, 1]


def test_resolve_devices_rejects_non_subset(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    with pytest.raises(ValueError):
        features.resolve_worker_devices([5])


def test_resolve_devices_empty_env_is_cpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert features.resolve_worker_devices(None) == [features.CPU_DEVICE_SENTINEL]


def test_resolve_devices_unset_env_uses_device_count(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(features.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(features.torch.cuda, "device_count", lambda: 2)
    assert features.resolve_worker_devices(None) == [0, 1]


def test_resolve_devices_no_cuda_is_cpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(features.torch.cuda, "is_available", lambda: False)
    assert features.resolve_worker_devices(None) == [features.CPU_DEVICE_SENTINEL]


def test_slides_needing_work_skips_complete(tmp_path):
    out_root = str(tmp_path)
    mags = [1.25, 2.5]
    enc = "ResNet50"
    slides = [("done", "/wsi/done.svs"), ("partial", "/wsi/partial.svs"), ("none", "/wsi/none.svs")]
    # done は全倍率，partial は片方だけ存在させる
    for mag in mags:
        p = features._output_h5_path(out_root, enc, mag, "done")
        __import__("os").makedirs(__import__("os").path.dirname(p), exist_ok=True)
        open(p, "w").close()
    p = features._output_h5_path(out_root, enc, mags[0], "partial")
    open(p, "w").close()

    need = features._slides_needing_work(slides, out_root, enc, mags)
    assert sorted(sid for sid, _ in need) == ["none", "partial"]
