"""ModelSaver の best 保存フォールバックのユニット"""

import os

import torch.nn as nn

from foveamil.training.saver import ModelSaver


def test_best_saved_even_when_f1_never_improves(tmp_path):
    # val_weighted_f1 が全エポック 0.0 でも best が必ず保存される（last 黙評価を防ぐ）
    saver = ModelSaver(str(tmp_path), save_metric="f1")
    model = nn.Linear(2, 2)
    for ep in range(3):
        saver(model, {"val_loss": 1.0, "val_weighted_f1": 0.0}, epoch=ep)
    assert saver.best_epoch == 0                       # 初回が暫定 best
    assert saver.load_best_path() is not None
    assert os.path.exists(os.path.join(str(tmp_path), "model_best_f1.pt"))


def test_best_updates_on_improvement(tmp_path):
    # 改善時は best epoch が更新される
    saver = ModelSaver(str(tmp_path), save_metric="f1")
    model = nn.Linear(2, 2)
    saver(model, {"val_loss": 1.0, "val_weighted_f1": 0.2}, epoch=0)
    saver(model, {"val_loss": 1.0, "val_weighted_f1": 0.5}, epoch=1)
    saver(model, {"val_loss": 1.0, "val_weighted_f1": 0.3}, epoch=2)
    assert saver.best_epoch == 1
    assert abs(saver.best_value - 0.5) < 1e-9


def test_best_saved_first_epoch_for_loss_metric(tmp_path):
    # loss 指標でも初回は必ず保存される
    saver = ModelSaver(str(tmp_path), save_metric="loss")
    model = nn.Linear(2, 2)
    saver(model, {"val_loss": 2.0, "val_weighted_f1": 0.1}, epoch=0)
    assert saver.best_epoch == 0
    assert os.path.exists(os.path.join(str(tmp_path), "model_best_loss.pt"))
