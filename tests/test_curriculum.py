"""機構L カリキュラム/安定化のテスト

RL 損失重みの epoch 依存 ramp（value→policy→actor-critic の順）と
探索ネット専用 param group（``search_lr_scale``）を検証する既定 off で従来挙動
（重み定数・単一 param group）にビット一致で畳まれることを確かめる
"""

import pytest

from foveamil.models.mil import FoveaMIL
from foveamil.training.config import TrainConfig
from foveamil.training.trainer import Trainer
from foveamil.training.zoom_driver import DifferentiableZoomDriver, build_zoom_driver

IN_DIM = 8
OUT_DIM = 12
N_CLS = 3
HIDDEN = 16
K = 3


def _model(num_layers=3):
    return FoveaMIL(
        in_feat_dim=IN_DIM,
        hidden_feat_dim=HIDDEN,
        out_feat_dim=OUT_DIM,
        k_sample=K,
        n_cls=N_CLS,
        num_layers=num_layers,
        topk_method="perturbed",
        fusion="sum",
    )


def _mcts_config(**overrides):
    cfg = TrainConfig(
        in_feat_dim=IN_DIM,
        out_feat_dim=OUT_DIM,
        hidden_feat_dim=HIDDEN,
        k_sample=K,
        n_cls=N_CLS,
        drop_out=None,
        zoom_driver="mcts",
        mcts_simulations=8,
        mcts_max_considered=6,
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        mcts_actor_critic_weight=0.5,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _mcts_driver(**overrides):
    return build_zoom_driver(_mcts_config(**overrides), _model())


class _StubTrainer:
    """``_optimizer_param_groups`` は ``self.model`` のみ参照するので最小 stub で呼べる"""

    def __init__(self, model):
        self.model = model


def test_curriculum_off_keeps_weights_constant():
    driver = _mcts_driver(curriculum_warmup_epochs=0)
    for epoch in (0, 1, 5, 99):
        driver.set_curriculum(epoch)
        assert driver.value_weight == 1.0
        assert driver.policy_weight == 1.0
        assert driver.actor_critic_weight == 0.5


def test_curriculum_epoch0_starts_all_at_zero():
    driver = _mcts_driver(curriculum_warmup_epochs=10, curriculum_value_lead_frac=0.5)
    driver.set_curriculum(0)
    assert driver.value_weight == 0.0
    assert driver.policy_weight == 0.0
    assert driver.actor_critic_weight == 0.0


def test_curriculum_value_leads_policy_leads_actor_critic():
    driver = _mcts_driver(curriculum_warmup_epochs=10, curriculum_value_lead_frac=0.5)
    # warmup 内では value_scale >= policy_scale >= ac_scale（critic が actor に先行）
    base_value, base_policy, base_ac = 1.0, 1.0, 0.5
    for epoch in range(1, 10):
        driver.set_curriculum(epoch)
        value_scale = driver.value_weight / base_value
        policy_scale = driver.policy_weight / base_policy
        ac_scale = driver.actor_critic_weight / base_ac
        assert value_scale >= policy_scale >= ac_scale


def test_curriculum_actor_critic_delayed_until_value_full():
    # lead=0.5, warmup=10 -> value full at epoch5, actor-critic 0 まで vfull
    driver = _mcts_driver(curriculum_warmup_epochs=10, curriculum_value_lead_frac=0.5)
    for epoch in range(0, 5):
        driver.set_curriculum(epoch)
        assert driver.actor_critic_weight == 0.0
    driver.set_curriculum(5)
    assert driver.value_weight == 1.0  # value は vfull で full


def test_curriculum_reaches_base_at_and_after_warmup():
    driver = _mcts_driver(curriculum_warmup_epochs=10, curriculum_value_lead_frac=0.5)
    for epoch in (10, 11, 50):
        driver.set_curriculum(epoch)
        assert driver.value_weight == 1.0
        assert driver.policy_weight == 1.0
        assert driver.actor_critic_weight == 0.5


def test_differentiable_driver_set_curriculum_is_noop():
    cfg = TrainConfig(
        in_feat_dim=IN_DIM, out_feat_dim=OUT_DIM, hidden_feat_dim=HIDDEN,
        k_sample=K, n_cls=N_CLS, zoom_driver="differentiable",
    )
    driver = build_zoom_driver(cfg, _model())
    assert isinstance(driver, DifferentiableZoomDriver)
    driver.set_curriculum(0)  # 例外を出さず何もしない
    driver.set_curriculum(99)


def test_param_groups_off_is_single_group():
    model = _model()
    build_zoom_driver(_mcts_config(), model)  # search ネットを attach
    groups = Trainer._optimizer_param_groups(_StubTrainer(model), lr=1e-4, search_lr_scale=1.0)
    # 1.0 では model.parameters() をそのまま返す（単一グループ＝従来挙動）
    params = list(groups)
    assert all(hasattr(p, "shape") for p in params)
    assert len(params) == len(list(model.parameters()))


def test_param_groups_split_isolates_search_nets():
    model = _model()
    build_zoom_driver(_mcts_config(), model)
    groups = Trainer._optimizer_param_groups(_StubTrainer(model), lr=1e-4, search_lr_scale=0.1)
    assert isinstance(groups, list) and len(groups) == 2
    other_group, search_group = groups
    # 探索グループは search_policy + search_value のパラメータと一致し LR は scale 倍
    expected_search = {
        id(p)
        for module in (model.search_policy, model.search_value)
        for p in module.parameters()
    }
    got_search = {id(p) for p in search_group["params"]}
    assert got_search == expected_search
    assert search_group["lr"] == pytest.approx(1e-4 * 0.1)
    # 2 グループで全パラメータを過不足なく分割（重複なし）
    other_ids = {id(p) for p in other_group["params"]}
    assert other_ids.isdisjoint(got_search)
    assert len(other_ids) + len(got_search) == len(list(model.parameters()))


@pytest.mark.parametrize("warmup,lead", [(10, 0.5), (4, 1.0), (1, 0.5), (3, 0.9)])
def test_curriculum_all_weights_reach_base_at_warmup(warmup, lead):
    # 縮退（vfull==warmup）でも value/policy/actor-critic とも epoch==warmup で base に達する
    driver = _mcts_driver(
        curriculum_warmup_epochs=warmup, curriculum_value_lead_frac=lead
    )
    driver.set_curriculum(warmup)
    assert driver.value_weight == pytest.approx(1.0)
    assert driver.policy_weight == pytest.approx(1.0)
    assert driver.actor_critic_weight == pytest.approx(0.5)


def test_param_groups_adam_defaults_propagate_to_both_groups():
    import torch.optim as optim

    model = _model()
    build_zoom_driver(_mcts_config(), model)
    groups = Trainer._optimizer_param_groups(
        _StubTrainer(model), lr=1e-4, search_lr_scale=0.1
    )
    opt = optim.Adam(groups, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    assert len(opt.param_groups) == 2
    # optimizer レベルの既定（betas/eps/weight_decay）は両グループへ伝播する
    for group in opt.param_groups:
        assert group["betas"] == (0.9, 0.999)
        assert group["eps"] == 1e-8
        assert group["weight_decay"] == 1e-4
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-4)  # head は base LR
    assert opt.param_groups[1]["lr"] == pytest.approx(1e-4 * 0.1)  # 探索は scale 倍


def test_param_groups_scale_without_search_nets_falls_back():
    # differentiable（探索ネット無し）では scale!=1 でも単一グループに畳む
    model = _model()
    cfg = TrainConfig(
        in_feat_dim=IN_DIM, out_feat_dim=OUT_DIM, hidden_feat_dim=HIDDEN,
        k_sample=K, n_cls=N_CLS, zoom_driver="differentiable",
    )
    build_zoom_driver(cfg, model)
    groups = Trainer._optimizer_param_groups(_StubTrainer(model), lr=1e-4, search_lr_scale=0.1)
    params = list(groups)
    assert len(params) == len(list(model.parameters()))
