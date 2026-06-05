"""visualization.loader のユニット（合成 sweep 出力でモデル復元を検証）"""

import json
import os

import torch

from foveamil.visualization.loader import (
    build_model,
    fold_dirs,
    load_fold,
    resolve_best_combo,
)

CONFIG = {
    "encoder": "ResNet50",
    "feature_type": "mean",
    "magnifications": [1.25, 2.5],
    "n_cls": 3,
    "in_feat_dim": 1024,
    "hidden_feat_dim": 256,
    "out_feat_dim": 512,
    "drop_out": None,
    "k_sample": 4,
    "k_sigma": 0.002,
    "topk_method": "perturbed",
    "fusion": "sum",
    "save_metric": "loss",
}


def _make_sweep(tmp_path, save_metric="loss"):
    combo = tmp_path / "combo_000__ResNet50_mean_m2"
    fold = combo / "fold1"
    fold.mkdir(parents=True)
    # 実モデルの state_dict を best 重みとして保存
    model = build_model(CONFIG)
    torch.save(model.state_dict(), fold / f"model_best_{save_metric}.pt")
    meta = {
        "config": CONFIG,
        "selection": {"save_metric": save_metric},
        "data": {"classes": ["DLBCL", "FL", "Reactive"]},
    }
    (fold / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    summary = {
        "best_by_val": {"index": 0, "name": combo.name, "out_dir": str(combo)},
        "combos": [{"index": 0, "name": combo.name, "out_dir": str(combo)}],
    }
    (tmp_path / "sweep_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return str(tmp_path), str(combo), str(fold)


def test_resolve_best_combo_and_folds(tmp_path):
    sweep_root, combo_dir, _ = _make_sweep(tmp_path)
    assert resolve_best_combo(sweep_root, "best_by_val") == combo_dir
    assert resolve_best_combo(sweep_root, "index", combo_index=0) == combo_dir
    assert [os.path.basename(d) for d in fold_dirs(combo_dir, "all")] == ["fold1"]


def test_load_fold_rebuilds_and_loads_weights(tmp_path):
    _, _, fold_dir = _make_sweep(tmp_path)
    loaded = load_fold(fold_dir, device="cpu")
    # _topk_kwargs 共有で state_dict 形状一致・load_state_dict が通る
    assert loaded.magnifications == [1.25, 2.5]
    assert loaded.encoder == "ResNet50"
    assert loaded.n_cls == 3
    assert loaded.classes == ["DLBCL", "FL", "Reactive"]
    assert loaded.save_metric == "loss"
    # eval 化されている
    assert not loaded.model.training


def test_build_model_topk_kwargs_roundtrip():
    # 同 config で 2 回構築し state_dict を相互ロード（topk 構築の一致を担保）
    a = build_model(CONFIG)
    b = build_model(CONFIG)
    b.load_state_dict(a.state_dict())  # 例外なし＝形状一致
