"""機構M（差別化版 warm-start ＋背骨凍結）の基盤テスト

背骨 warm-start の流用契約（search net のみ欠損）・eval 凍結（requires_grad＋eval 両切替）・
凍結スケジュール・config 整合ガードを検証する既定 off で従来挙動にビット互換であることを確かめる
"""

import os

import pytest
import torch

from foveamil.models.mil import FoveaMIL
from foveamil.training.config import TrainConfig
from foveamil.training.trainer import Trainer
from foveamil.training.zoom_driver import build_zoom_driver

IN_DIM = 8
OUT_DIM = 12
N_CLS = 3
HIDDEN = 16
K = 3


def _model(num_layers=3):
    return FoveaMIL(
        in_feat_dim=IN_DIM, hidden_feat_dim=HIDDEN, out_feat_dim=OUT_DIM,
        k_sample=K, n_cls=N_CLS, num_layers=num_layers,
        topk_method="perturbed", fusion="sum",
    )


def _mcts_model(**overrides):
    model = _model()
    cfg = TrainConfig(
        in_feat_dim=IN_DIM, out_feat_dim=OUT_DIM, hidden_feat_dim=HIDDEN,
        k_sample=K, n_cls=N_CLS, drop_out=None,
        zoom_driver="mcts", mcts_simulations=8, mcts_max_considered=6,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    build_zoom_driver(cfg, model)  # search_policy/value を attach
    return model, cfg


class _Stub:
    """Trainer の M メソッドは self.{config,model,save_path,device} のみ参照＝stub で呼べる"""

    def __init__(self, model, config, save_path, device):
        self.model = model
        self.config = config
        self.save_path = save_path
        self.device = device

    def _backbone_modules(self):
        return Trainer._backbone_modules(self)


def _backbone_state(model):
    return {
        k: v for k, v in model.state_dict().items()
        if not k.startswith(("search_policy", "search_value"))
    }


def test_warm_start_loads_backbone_keeps_search_fresh(tmp_path):
    src_model, _ = _mcts_model()
    fold_dir = tmp_path / "fold1"
    fold_dir.mkdir()
    torch.save(_backbone_state(src_model), fold_dir / "model.pt")

    dst_model, _ = _mcts_model()
    cfg = TrainConfig(
        in_feat_dim=IN_DIM, n_cls=N_CLS,
        warm_start_checkpoint=str(tmp_path / "{fold}" / "model.pt"),
    )
    stub = _Stub(dst_model, cfg, str(tmp_path / "fold1"), torch.device("cpu"))
    Trainer._load_warm_start(stub)

    # 背骨は src と一致（流用された）
    src_bb, dst_bb = _backbone_state(src_model), _backbone_state(dst_model)
    for k in src_bb:
        assert torch.equal(dst_bb[k], src_bb[k]), f"backbone {k} がロードされていない"
    # search net は流用されず dst の初期値のまま（src と異なる）
    src_sd, dst_sd = src_model.state_dict(), dst_model.state_dict()
    search_keys = [k for k in dst_sd if k.startswith("search_")]
    assert search_keys, "search net が attach されていない"
    assert any(not torch.equal(dst_sd[k], src_sd[k]) for k in search_keys)


def test_warm_start_none_is_noop(tmp_path):
    model, _ = _mcts_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    cfg = TrainConfig(in_feat_dim=IN_DIM, n_cls=N_CLS, warm_start_checkpoint=None)
    stub = _Stub(model, cfg, str(tmp_path / "fold1"), torch.device("cpu"))
    Trainer._load_warm_start(stub)  # 何もしない
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])


def test_warm_start_contract_violation_fails_fast(tmp_path):
    model, _ = _mcts_model()
    fold_dir = tmp_path / "fold1"
    fold_dir.mkdir()
    bad = _backbone_state(model)
    bad["nonexistent.weight"] = torch.zeros(3)  # 余剰キー＝契約違反
    torch.save(bad, fold_dir / "model.pt")
    cfg = TrainConfig(
        in_feat_dim=IN_DIM, n_cls=N_CLS,
        warm_start_checkpoint=str(fold_dir / "model.pt"),
    )
    stub = _Stub(model, cfg, str(fold_dir), torch.device("cpu"))
    with pytest.raises(RuntimeError, match="流用契約違反"):
        Trainer._load_warm_start(stub)


def test_set_backbone_frozen_toggles_requires_grad_and_mode():
    model, _ = _mcts_model()
    stub = _Stub(model, TrainConfig(in_feat_dim=IN_DIM, n_cls=N_CLS), "x", torch.device("cpu"))
    Trainer._set_backbone_frozen(stub, True)
    # 背骨は requires_grad=False＋eval，search は影響なし
    assert all(not p.requires_grad for p in model.projections.parameters())
    assert all(not p.requires_grad for p in model.head.parameters())
    assert not model.head.training
    assert all(p.requires_grad for p in model.search_policy.parameters())
    Trainer._set_backbone_frozen(stub, False)
    assert all(p.requires_grad for p in model.projections.parameters())
    assert model.head.training


@pytest.mark.parametrize(
    "fb,unfreeze,epoch,expected",
    [
        (0, 0.0, 0, False),    # M off
        (30, 0.0, 0, True),    # freeze 相
        (30, 0.0, 29, True),
        (30, 0.0, 30, True),   # 恒久凍結(unfreeze=0)
        (30, 0.0, 49, True),
        (30, 0.1, 29, True),   # co-adapt: freeze 相は凍結
        (30, 0.1, 30, False),  # co-adapt: 相転移で解凍
        (30, 0.1, 49, False),
    ],
)
def test_is_backbone_frozen_schedule(fb, unfreeze, epoch, expected):
    cfg = TrainConfig(
        in_feat_dim=IN_DIM, n_cls=N_CLS,
        freeze_backbone_epochs=fb, unfreeze_lr_scale=unfreeze,
    )
    stub = _Stub(None, cfg, "x", torch.device("cpu"))
    assert Trainer._is_backbone_frozen(stub, epoch) is expected


def test_l_and_m_mutually_exclusive(tmp_path):
    model, _ = _mcts_model()
    cfg = TrainConfig(
        in_feat_dim=IN_DIM, n_cls=N_CLS,
        warm_start_checkpoint=str(tmp_path / "x.pt"),
        curriculum_warmup_epochs=10,  # L と併用＝排他違反
    )
    stub = _Stub(model, cfg, str(tmp_path / "fold1"), torch.device("cpu"))
    with pytest.raises(ValueError, match="排他"):
        Trainer._load_warm_start(stub)


class _MStub:
    """co-adapt 相転移テスト用 stub（optimizer/_base_lr/group index を持つ）"""

    def __init__(self, model, config):
        import torch.optim as optim

        self.model = model
        self.config = config
        self._base_lr = config.lr
        self._unfrozen = False
        self._backbone_group_idx = 0
        backbone, search = Trainer._split_backbone_search_params(self)
        self.optimizer = optim.Adam(
            [
                {"params": backbone, "lr": config.lr * config.unfreeze_lr_scale},
                {"params": search, "lr": config.lr},
            ],
            lr=config.lr,
        )
        self.scheduler = None

    def _backbone_modules(self):
        return Trainer._backbone_modules(self)

    def _set_backbone_frozen(self, frozen):
        return Trainer._set_backbone_frozen(self, frozen)

    def _split_backbone_search_params(self):
        return Trainer._split_backbone_search_params(self)

    def _build_scheduler(self):
        return Trainer._build_scheduler(self)


def test_maybe_unfreeze_transition_unfreezes_and_restores_lr():
    model, _ = _mcts_model()
    cfg = TrainConfig(
        in_feat_dim=IN_DIM, n_cls=N_CLS, lr=1e-4,
        freeze_backbone_epochs=30, unfreeze_lr_scale=0.1,
        scheduler_decay_rate=0.8, scheduler_patience=10,
    )
    stub = _MStub(model, cfg)
    Trainer._set_backbone_frozen(stub, True)  # freeze 相
    assert all(not p.requires_grad for p in model.head.parameters())
    # 相転移前は no-op
    Trainer._maybe_unfreeze_transition(stub, 29)
    assert stub._unfrozen is False
    # epoch==freeze で解凍＋背骨LRを base×unfreeze へ復元＋scheduler 作り直し
    Trainer._maybe_unfreeze_transition(stub, 30)
    assert stub._unfrozen is True
    assert all(p.requires_grad for p in model.head.parameters())
    assert stub.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4 * 0.1)
    assert stub.scheduler is not None
    # 二度目は no-op（_unfrozen 済）
    stub.optimizer.param_groups[0]["lr"] = 999.0
    Trainer._maybe_unfreeze_transition(stub, 31)
    assert stub.optimizer.param_groups[0]["lr"] == 999.0


def test_permanent_freeze_no_transition():
    # 恒久凍結(unfreeze=0)では相転移しない
    model, _ = _mcts_model()
    cfg = TrainConfig(
        in_feat_dim=IN_DIM, n_cls=N_CLS, lr=1e-4,
        freeze_backbone_epochs=30, unfreeze_lr_scale=0.0,
    )
    stub = _MStub(model, cfg)
    Trainer._maybe_unfreeze_transition(stub, 30)
    assert stub._unfrozen is False
