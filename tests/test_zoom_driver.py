"""ズーム駆動シーム（既定の微分可能駆動・探索駆動）のユニット

既定駆動が従来ループの数値を再現する回帰ガードと，探索駆動が妥当な融合表現・有限な
合成損失を作り，方策・価値・共有ヘッドへ勾配を流し，シードで決定的であることを確認する
"""

import pytest
import torch
import torch.nn.functional as F

from foveamil.models.mil import FoveaMIL
from foveamil.training.config import TrainConfig
from foveamil.training.hierarchy import children_per_parent, compute_child_indices
from foveamil.training.zoom_driver import (
    DifferentiableZoomDriver,
    build_zoom_driver,
)

IN_DIM = 8
OUT_DIM = 12
N_CLS = 3
HIDDEN = 16
K = 3
MAGS_3 = [1.25, 2.5, 5.0]
MAGS_2 = [1.25, 2.5]


def _model(num_layers, k=K):
    return FoveaMIL(
        in_feat_dim=IN_DIM,
        hidden_feat_dim=HIDDEN,
        out_feat_dim=OUT_DIM,
        k_sample=k,
        n_cls=N_CLS,
        num_layers=num_layers,
        topk_method="perturbed",
        fusion="sum",
    )


def _seeded_child_loader():
    """合成子ローダ：``(mag, indices)`` で決定的な ``[1, Nc, IN_DIM]`` を返す"""
    store = {}

    def loader(next_mag, child_idx):
        n = len(child_idx)
        key = (round(float(next_mag), 4), tuple(int(i) for i in child_idx))
        if key not in store:
            seed = (int(next_mag * 1000) * 131 + sum(int(i) for i in child_idx)) % (2**31)
            g = torch.Generator().manual_seed(seed)
            store[key] = torch.randn(1, n, IN_DIM, generator=g)
        return store[key].clone()

    return loader


# --- 既定駆動の回帰ガード（従来ループと同一数値） ---


def _replay_prior_loop(model, base, mags, child_loader, device):
    """従来 ``Trainer._forward`` のループを逐語再生する（回帰ガードの参照）"""
    M_list = []
    selections = []
    x = base.to(device)
    global_idx = None
    num_layers = model.num_layers
    for layer_idx in range(num_layers):
        M, si, sw, _ = model.forward_layer(x, layer_idx)
        M_list.append(M)
        if layer_idx >= num_layers - 1:
            selections.append(None)
            continue
        selections.append({"select_indices": si, "select_weight": sw})
        cur, nxt = mags[layer_idx], mags[layer_idx + 1]
        cpp = children_per_parent(cur, nxt)
        local = si[0].detach().cpu().numpy()
        child = compute_child_indices(local, global_idx, children=cpp)
        x_next = child_loader(nxt, child).to(device)
        w_child = sw.repeat_interleave(cpp, dim=1)
        x_next = x_next * w_child.unsqueeze(-1)
        x = x_next
        global_idx = child
    return model.forward_final(M_list)


@pytest.mark.parametrize("num_layers,mags", [(2, MAGS_2), (3, MAGS_3)])
def test_differentiable_driver_matches_prior_loop(num_layers, mags):
    torch.manual_seed(11)
    model = _model(num_layers)
    model.eval()
    device = torch.device("cpu")
    base = torch.randn(1, 9, IN_DIM)
    loader = _seeded_child_loader()

    driver = DifferentiableZoomDriver(model, num_layers)
    d_logits, d_yhat, d_yprob, ctx = driver.run(base, mags, loader, device)
    r_logits, r_yhat, r_yprob = _replay_prior_loop(model, base, mags, loader, device)

    assert torch.equal(d_logits, r_logits)
    assert torch.equal(d_yhat, r_yhat)
    assert torch.equal(d_yprob, r_yprob)
    assert len(ctx.m_list) == num_layers


def test_differentiable_driver_train_propagates_gradient():
    torch.manual_seed(3)
    model = _model(num_layers=2)
    model.train()
    device = torch.device("cpu")
    base = torch.randn(1, 8, IN_DIM)
    driver = DifferentiableZoomDriver(model, 2)
    logits, _, _, _ = driver.run(base, MAGS_2, _seeded_child_loader(), device)
    F.cross_entropy(logits, torch.tensor([1])).backward()
    # 補助アテンション（選択経路）へ勾配が流れる
    aux_grad = model.aux_attentions[0].attention_c.weight.grad
    assert aux_grad is not None and aux_grad.abs().sum().item() > 0


# --- ファクトリ ---


def test_build_zoom_driver_default_is_differentiable():
    model = _model(num_layers=2)
    cfg = TrainConfig(
        in_feat_dim=IN_DIM, out_feat_dim=OUT_DIM, hidden_feat_dim=HIDDEN,
        k_sample=K, n_cls=N_CLS,
    )
    driver = build_zoom_driver(cfg, model)
    assert isinstance(driver, DifferentiableZoomDriver)


def test_build_zoom_driver_unknown_raises():
    model = _model(num_layers=2)
    cfg = TrainConfig(
        in_feat_dim=IN_DIM, out_feat_dim=OUT_DIM, hidden_feat_dim=HIDDEN,
        k_sample=K, n_cls=N_CLS, zoom_driver="does_not_exist",
    )
    with pytest.raises(KeyError):
        build_zoom_driver(cfg, model)


def _mcts_config(**overrides):
    cfg = TrainConfig(
        in_feat_dim=IN_DIM, out_feat_dim=OUT_DIM, hidden_feat_dim=HIDDEN,
        k_sample=K, n_cls=N_CLS, drop_out=None,
        zoom_driver="mcts", mcts_simulations=8, mcts_max_considered=6,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_build_zoom_driver_mcts_registers_search_modules():
    from foveamil.models.search import MCTSZoomDriver

    model = _model(num_layers=3)
    before = {n for n, _ in model.named_parameters()}
    driver = build_zoom_driver(_mcts_config(), model)
    after = {n for n, _ in model.named_parameters()}
    assert isinstance(driver, MCTSZoomDriver)
    # 方策・価値ネットが model のパラメータに加わる（共有 optimizer で最適化される）
    added = after - before
    assert any("search_policy" in n for n in added)
    assert any("search_value" in n for n in added)


# --- 探索駆動の forward・合成損失・勾配 ---


@pytest.mark.parametrize("num_layers,mags", [(2, MAGS_2), (3, MAGS_3)])
def test_mcts_driver_forward_and_composite_loss(num_layers, mags):
    torch.manual_seed(0)
    model = _model(num_layers)
    driver = build_zoom_driver(_mcts_config(), model)
    model.train()
    device = torch.device("cpu")
    base = torch.randn(1, 10, IN_DIM)
    label = torch.tensor([1])

    logits, Y_hat, Y_prob, ctx = driver.run(
        base, mags, _seeded_child_loader(), device, label=label
    )
    assert logits.shape == (1, N_CLS)
    assert Y_hat.shape == (1, 1)
    assert torch.allclose(Y_prob.sum(dim=-1), torch.ones(1), atol=1e-6)
    # context.m_list は駆動に依らず埋まる（A ブランチとの両立）
    assert len(ctx.m_list) == num_layers
    assert all(m is not None for m in ctx.m_list)

    # CE + 方策蒸留 + 価値回帰 が有限
    assert "mcts_policy" in ctx.extra_losses
    assert "mcts_value" in ctx.extra_losses
    composite = F.cross_entropy(logits, label) + sum(ctx.extra_losses.values())
    assert torch.isfinite(composite)


def test_mcts_driver_backward_populates_policy_value_and_head():
    torch.manual_seed(0)
    model = _model(num_layers=3)
    driver = build_zoom_driver(_mcts_config(), model)
    model.train()
    device = torch.device("cpu")
    base = torch.randn(1, 10, IN_DIM)
    label = torch.tensor([2])

    logits, _, _, ctx = driver.run(
        base, MAGS_3, _seeded_child_loader(), device, label=label
    )
    composite = F.cross_entropy(logits, label) + sum(ctx.extra_losses.values())
    composite.backward()

    def has_grad(module):
        return any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in module.parameters()
        )

    assert has_grad(model.search_policy)
    assert has_grad(model.search_value)
    # 共有識別器ヘッド・射影へも勾配が流れる
    assert model.head.fc.weight.grad is not None
    assert model.head.fc.weight.grad.abs().sum().item() > 0
    assert has_grad(model.projections[0])


def test_mcts_driver_inference_has_no_extra_losses():
    torch.manual_seed(0)
    model = _model(num_layers=3)
    driver = build_zoom_driver(_mcts_config(), model)
    model.eval()
    device = torch.device("cpu")
    base = torch.randn(1, 9, IN_DIM)
    with torch.no_grad():
        logits, _, _, ctx = driver.run(
            base, MAGS_3, _seeded_child_loader(), device, label=None
        )
    assert logits.shape == (1, N_CLS)
    assert ctx.extra_losses == {}


def test_mcts_driver_deterministic_under_seed():
    device = torch.device("cpu")
    base = torch.randn(1, 10, IN_DIM)

    def run_once():
        torch.manual_seed(5)
        model = _model(num_layers=3)
        driver = build_zoom_driver(_mcts_config(), model)
        model.eval()
        with torch.no_grad():
            logits, _, _, ctx = driver.run(
                base, MAGS_3, _seeded_child_loader(), device, label=None
            )
        sel = [
            None if s is None else s["select_indices"].cpu().numpy().tolist()
            for s in ctx.selections
        ]
        return logits, sel

    a_logits, a_sel = run_once()
    b_logits, b_sel = run_once()
    assert torch.allclose(a_logits, b_logits)
    assert a_sel == b_sel


def _zoom_search_problem(model, value_net, rollout_depth=1, stochastic=False):
    """``_ZoomSearchProblem`` の最小インスタンスを作る（``evaluate`` の単体検証用）

    同一の ``model`` ``value_net`` を共有し，木全体で共有する ``_RolloutContext`` を
    1 個張る単一倍率比の最小木で ``rollout_depth=1`` 既定では値の差は評価モードのみに由来する
    """
    import numpy as np

    from foveamil.models.search.driver import _RolloutContext, _ZoomSearchProblem

    ctx = _RolloutContext(
        model=model,
        value_net=value_net,
        child_loader=_seeded_child_loader(),
        magnifications=MAGS_2,
        num_layers=2,
        planner_name="gumbel",
        rollout_simulations=4,
        rollout_considered=4,
        stochastic=stochastic,
        device=torch.device("cpu"),
    )
    return _ZoomSearchProblem(
        prior_np=np.full(3, 1.0 / 3),
        x_fc=torch.zeros(1, 3, OUT_DIM),
        layer_idx=0,
        next_mag=MAGS_2[1],
        cpp=children_per_parent(MAGS_2[0], MAGS_2[1]),
        global_idx=None,
        rollout_depth=rollout_depth,
        seed=0,
        ctx=ctx,
    )


def test_evaluate_independent_of_value_net_train_eval_mode():
    from foveamil.models.search.value import ValueNetwork

    torch.manual_seed(0)
    model = _model(num_layers=2)
    value_net = ValueNetwork(OUT_DIM, HIDDEN, dropout=0.5)

    value_net.train()
    torch.manual_seed(7)
    reward_train = _zoom_search_problem(model, value_net).evaluate(0)

    value_net.eval()
    torch.manual_seed(7)
    reward_eval = _zoom_search_problem(model, value_net).evaluate(0)

    # dropout 0.5 でも train/eval どちらのモードでも葉評価は一致する（eval を強制）
    assert reward_train == pytest.approx(reward_eval)


def test_evaluate_restores_value_net_training_mode():
    from foveamil.models.search.value import ValueNetwork

    model = _model(num_layers=2)
    value_net = ValueNetwork(OUT_DIM, HIDDEN, dropout=0.5)
    value_net.train()
    _zoom_search_problem(model, value_net).evaluate(0)
    # 前向き後に元の train モードへ戻る
    assert value_net.training


def test_mcts_driver_entropy_term_when_enabled():
    torch.manual_seed(0)
    model = _model(num_layers=2)
    driver = build_zoom_driver(_mcts_config(policy_entropy_weight=0.1), model)
    model.train()
    device = torch.device("cpu")
    base = torch.randn(1, 9, IN_DIM)
    _, _, _, ctx = driver.run(
        base, MAGS_2, _seeded_child_loader(), device, label=torch.tensor([0])
    )
    assert "mcts_entropy" in ctx.extra_losses
    assert torch.isfinite(ctx.extra_losses["mcts_entropy"])


# --- 価値ターゲット軸 mcts_value_target ---


def test_mcts_value_target_default_is_realised():
    model = _model(num_layers=3)
    driver = build_zoom_driver(_mcts_config(), model)
    # 既定は従来挙動（最終 CE を全状態へ broadcast）
    assert driver.value_target == "realised"


def test_mcts_value_target_realised_uses_single_broadcast_target():
    """realised の価値回帰は最終 CE の単一スカラを全状態へ broadcast した目標を使う

    leaf_ce が状態依存目標を使うのと対照に，realised は同一スカラを全状態へ広げる
    両モードで価値損失が一致しないことで broadcast 経路が leaf 経路と別物だと確認する
    """
    base = torch.randn(1, 10, IN_DIM)
    label = torch.tensor([1])

    def value_loss(value_target):
        torch.manual_seed(0)
        model = _model(num_layers=3)
        driver = build_zoom_driver(
            _mcts_config(mcts_value_target=value_target), model
        )
        model.train()
        _, _, _, ctx = driver.run(
            base, MAGS_3, _seeded_child_loader(), torch.device("cpu"), label=label
        )
        return float(ctx.extra_losses["mcts_value"].detach())

    realised_loss = value_loss("realised")
    leaf_loss = value_loss("leaf_ce")
    assert realised_loss >= 0.0
    # 状態依存目標とスカラ broadcast 目標で価値損失が異なる
    assert realised_loss != leaf_loss


def test_mcts_value_target_default_bit_compat_is_deterministic():
    """既定 realised は state_dict 固定で再現可能（数値が run 間で一致）"""
    base = torch.randn(1, 10, IN_DIM)
    label = torch.tensor([2])

    def run_once():
        torch.manual_seed(3)
        model = _model(num_layers=3)
        driver = build_zoom_driver(_mcts_config(), model)
        model.train()
        logits, _, _, ctx = driver.run(
            base, MAGS_3, _seeded_child_loader(), torch.device("cpu"), label=label
        )
        return (
            float(logits.detach().sum()),
            {k: float(v.detach()) for k, v in ctx.extra_losses.items()},
        )

    a = run_once()
    b = run_once()
    assert a == b


def test_mcts_value_target_leaf_ce_is_state_dependent():
    """leaf_ce では部分選択状態ごとに価値ターゲット（leaf 報酬）が異なる"""
    torch.manual_seed(0)
    model = _model(num_layers=3)
    driver = build_zoom_driver(_mcts_config(mcts_value_target="leaf_ce"), model)
    model.train()
    device = torch.device("cpu")
    base = torch.randn(1, 10, IN_DIM)
    label = torch.tensor([1])

    _, _, _, ctx = driver.run(
        base, MAGS_3, _seeded_child_loader(), device, label=label
    )
    # 各探索層（num_layers-1 個）の状態で leaf 報酬を再計算する
    rewards = driver._leaf_rewards(
        [m.detach() for m in ctx.m_list], label, num_states=len(MAGS_3) - 1
    )
    assert rewards.shape[0] == len(MAGS_3) - 1
    # 前置融合が状態で異なるため目標は状態を弁別する
    assert not torch.allclose(rewards[0], rewards[1])


def _policy_grad_from_extra_losses(value_target, label, base):
    """``extra_losses`` のみで backward し方策ネット勾配の絶対値和を返す

    主 CE を加えないため方策ネットへ流れる勾配は方策蒸留・actor-critic 項のみに由来する
    （共有ヘッドの自明な主 CE 勾配を検証から排除する）
    """
    torch.manual_seed(0)
    model = _model(num_layers=3)
    driver = build_zoom_driver(_mcts_config(mcts_value_target=value_target), model)
    model.train()
    _, _, _, ctx = driver.run(
        base, MAGS_3, _seeded_child_loader(), torch.device("cpu"), label=label
    )
    model.zero_grad()
    sum(ctx.extra_losses.values()).backward()
    return sum(
        p.grad.abs().sum().item()
        for p in model.search_policy.parameters()
        if p.grad is not None
    )


def test_mcts_value_target_leaf_ce_actor_critic_drives_policy_gradient():
    """leaf_ce の actor-critic 項が方策勾配へ寄与し realised と方策勾配が異なる

    主 CE を除いた extra_losses のみで backward し方策ネット勾配を測るleaf_ce は方策
    蒸留に actor-critic 項を上乗せするため，蒸留のみの realised と勾配の大きさが異なる
    （head 勾配は主 CE から自明に来るので検証には使わない）
    """
    label = torch.tensor([2])
    torch.manual_seed(13)
    base = torch.randn(1, 10, IN_DIM)
    leaf_grad = _policy_grad_from_extra_losses("leaf_ce", label, base)
    realised_grad = _policy_grad_from_extra_losses("realised", label, base)
    assert leaf_grad > 0
    assert realised_grad > 0
    # actor-critic 項の上乗せにより方策勾配が realised と一致しない
    assert leaf_grad != pytest.approx(realised_grad)


def test_mcts_value_target_leaf_ce_advantage_reflects_selection_effect():
    """選択 j の advantage が選択 j の結果状態 M_{j+1} の良し悪しを反映する

    M_0 を固定し選択 0 の結果状態 M_1 を「良い（負 CE が高い）」「悪い」で差し替え，
    共通の価値ベースラインの下で advantage（leaf 報酬 − 価値推定）が良い選択で大きくなる
    ことを確認する選択 j のリターンが M_{j+1} に依存する（off-by-one でない）ことの検証
    """
    torch.manual_seed(0)
    model = _model(num_layers=3)
    driver = build_zoom_driver(_mcts_config(mcts_value_target="leaf_ce"), model)
    model.eval()
    label = torch.tensor([1])

    torch.manual_seed(1)
    cands = [torch.randn(1, 1, OUT_DIM) for _ in range(60)]

    def ce(prefix):
        return float(F.cross_entropy(model.classify(model.fuse_repr(prefix))[0], label))

    m0, m2 = cands[0], cands[1]
    good_m1 = min(cands, key=lambda c: ce([m0, c]))
    bad_m1 = max(cands, key=lambda c: ce([m0, c]))

    # 選択 0 の報酬は m_list[:2]＝fuse(M_0, M_1) の負 CE（M_1 が選択 0 の結果状態）
    reward_good = driver._leaf_rewards(
        [m0, good_m1, m2], label, num_states=len(MAGS_3) - 1
    )[0]
    reward_bad = driver._leaf_rewards(
        [m0, bad_m1, m2], label, num_states=len(MAGS_3) - 1
    )[0]

    # 共通の価値ベースライン v の下で advantage = reward - v良い選択で advantage が大きい
    baseline = (reward_good.detach() + reward_bad.detach()) / 2
    adv_good = reward_good.detach() - baseline
    adv_bad = reward_bad.detach() - baseline
    assert float(reward_good) > float(reward_bad)
    assert float(adv_good) > 0 > float(adv_bad)


def test_mcts_value_target_leaf_ce_loss_is_finite():
    torch.manual_seed(0)
    model = _model(num_layers=3)
    driver = build_zoom_driver(_mcts_config(mcts_value_target="leaf_ce"), model)
    model.train()
    device = torch.device("cpu")
    base = torch.randn(1, 9, IN_DIM)
    _, _, _, ctx = driver.run(
        base, MAGS_3, _seeded_child_loader(), device, label=torch.tensor([0])
    )
    assert torch.isfinite(ctx.extra_losses["mcts_value"])
    assert torch.isfinite(ctx.extra_losses["mcts_policy"])


# --- actor-critic 安定化軸 mcts_actor_critic_weight ---


def test_mcts_actor_critic_weight_default_is_one():
    model = _model(num_layers=3)
    driver = build_zoom_driver(_mcts_config(), model)
    # 既定は等倍（正規化 advantage を 1.0 で方策蒸留へ上乗せ）
    assert driver.actor_critic_weight == 1.0


def test_normalize_advantage_zero_mean_unit_var():
    """advantage 正規化は選択状態軸でゼロ平均・単位分散にする（eps 付き）"""
    from foveamil.models.search import MCTSZoomDriver

    adv = torch.tensor([3.0, -1.0, 0.5, 7.0, -4.0])
    out = MCTSZoomDriver._normalize_advantage(adv)
    assert float(out.mean().abs()) < 1e-5
    # 分散 1（不偏でない母分散）に近い eps による僅かな縮みは許容する
    assert float(out.var(unbiased=False)) == pytest.approx(1.0, abs=1e-3)


def test_normalize_advantage_preserves_sign():
    """正規化は線形変換のため要素の符号と大小順を保つ（advantage>0 で chosen 確率↑）"""
    from foveamil.models.search import MCTSZoomDriver

    adv = torch.tensor([2.0, -3.0, 5.0, -0.5])
    out = MCTSZoomDriver._normalize_advantage(adv)
    # 平均を引くため符号自体は不変でないが大小順（argsort）は保たれる
    assert torch.equal(adv.argsort(), out.argsort())
    # 元の最大が正規化後も最大最小が最小（順序保存＝符号保存の弁別シグナル）
    assert out.argmax() == adv.argmax()
    assert out.argmin() == adv.argmin()


def test_normalize_advantage_single_state_no_div():
    """状態が 1 個なら平均引きのみで分散正規化を省く（ゼロ割を回避し有限）"""
    from foveamil.models.search import MCTSZoomDriver

    out = MCTSZoomDriver._normalize_advantage(torch.tensor([4.0]))
    assert torch.isfinite(out).all()
    assert float(out.abs().max()) == pytest.approx(0.0)


def _leaf_ce_extra_losses(actor_critic_weight, label, base):
    """``leaf_ce`` で driver.run を回し ``extra_losses`` を返す（actor-critic 重み可変）"""
    torch.manual_seed(0)
    model = _model(num_layers=3)
    driver = build_zoom_driver(
        _mcts_config(
            mcts_value_target="leaf_ce", mcts_actor_critic_weight=actor_critic_weight
        ),
        model,
    )
    model.train()
    _, _, _, ctx = driver.run(
        base, MAGS_3, _seeded_child_loader(), torch.device("cpu"), label=label
    )
    return {k: float(v.detach()) for k, v in ctx.extra_losses.items()}


def test_mcts_actor_critic_weight_zero_removes_actor_critic_term():
    """``mcts_actor_critic_weight=0`` で leaf_ce の方策損失が蒸留のみ（actor-critic 消失）

    重み 0 では actor-critic 項が落ち方策損失が realised と同一になる価値回帰は leaf_ce の
    状態依存目標を保つため value 損失は realised と異なる（value だけ残り planner を導く）
    """
    label = torch.tensor([2])
    torch.manual_seed(13)
    base = torch.randn(1, 10, IN_DIM)

    leaf_zero = _leaf_ce_extra_losses(0.0, label, base)

    torch.manual_seed(0)
    model = _model(num_layers=3)
    driver = build_zoom_driver(_mcts_config(mcts_value_target="realised"), model)
    model.train()
    _, _, _, ctx = driver.run(
        base, MAGS_3, _seeded_child_loader(), torch.device("cpu"), label=label
    )
    realised = {k: float(v.detach()) for k, v in ctx.extra_losses.items()}

    # actor-critic 無効で方策損失は蒸留のみ＝realised と一致する
    assert leaf_zero["mcts_policy"] == pytest.approx(realised["mcts_policy"])
    # value は状態依存 leaf 目標で realised の broadcast 目標と異なる
    assert leaf_zero["mcts_value"] != pytest.approx(realised["mcts_value"])


def test_mcts_actor_critic_weight_scales_policy_loss():
    """actor-critic 重みが方策損失を線形にスケールする（重みで項が増減する）"""
    label = torch.tensor([2])
    torch.manual_seed(13)
    base = torch.randn(1, 10, IN_DIM)

    zero = _leaf_ce_extra_losses(0.0, label, base)["mcts_policy"]
    half = _leaf_ce_extra_losses(0.5, label, base)["mcts_policy"]
    full = _leaf_ce_extra_losses(1.0, label, base)["mcts_policy"]

    # policy_loss = distill - w * (正規化 advantage * log_select) の平均
    # 重み 0/0.5/1.0 で actor-critic 寄与が線形に効き half は zero と full の中点
    assert half == pytest.approx((zero + full) / 2)
    assert full != pytest.approx(zero)


def test_mcts_leaf_ce_advantage_sign_preserved_in_policy_term():
    """正規化後も advantage>0 の選択で chosen log 確率の係数が正（chosen 確率↑方向）

    actor-critic 項は ``- w * adv * log_select`` で policy_terms へ加わるadv>0 のとき
    log_select の係数 ``w*adv`` が正＝最小化で log_select を増やす（chosen 確率↑）方向を保つ
    正規化（平均引き・分散割り）は線形変換のためこの符号関係を壊さないことを確認する
    """
    from foveamil.models.search import MCTSZoomDriver

    raw = torch.tensor([2.0, -3.0, 5.0, -1.0])
    norm = MCTSZoomDriver._normalize_advantage(raw)
    # 正規化後の符号と大小が選択の弁別を保つ（最良選択の係数が最大正）
    assert norm.argmax() == raw.argmax()
    # 平均引きで正負が分かれ良い選択は正悪い選択は負（chosen 確率↑/↓を弁別）
    assert float(norm[raw.argmax()]) > 0 > float(norm[raw.argmin()])


def _toy_train_loss_trace(value_target, steps=12, actor_critic_weight=1.0):
    """toy データで数ステップ学習し合成 train_loss の系列を返す（発散検証用）

    主 CE + extra_losses を合成損失とし optimizer で更新する単一 bag を繰り返し回し
    train_loss が有界（発散しない）かを realised/正規化 leaf_ce で比較する
    """
    torch.manual_seed(0)
    model = _model(num_layers=3)
    driver = build_zoom_driver(
        _mcts_config(
            mcts_value_target=value_target,
            mcts_actor_critic_weight=actor_critic_weight,
        ),
        model,
    )
    model.train()
    device = torch.device("cpu")
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    loader = _seeded_child_loader()
    base = torch.randn(1, 12, IN_DIM)
    label = torch.tensor([1])

    losses = []
    for _ in range(steps):
        opt.zero_grad()
        logits, _, _, ctx = driver.run(base, MAGS_3, loader, device, label=label)
        loss = F.cross_entropy(logits, label) + sum(ctx.extra_losses.values())
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def test_toy_leaf_ce_train_loss_bounded():
    """toy で正規化 leaf_ce の train_loss が有界（発散しない）realised と同程度に収まる

    正規化前の actor-critic 項はスケール暴走で train_loss が負へ発散しうる正規化で
    advantage を単位分散に抑えるため数ステップ学習しても有限な範囲に留まることを確認する
    """
    realised = _toy_train_loss_trace("realised")
    leaf_ce = _toy_train_loss_trace("leaf_ce")

    # 全ステップで有限
    assert all(torch.isfinite(torch.tensor(realised)))
    assert all(torch.isfinite(torch.tensor(leaf_ce)))
    # 正規化 leaf_ce の train_loss が realised から大きく乖離せず有界に留まる
    # （正規化 advantage は単位分散で actor-critic 寄与が ~O(1) に抑えられる）
    bound = abs(max(realised)) + 10.0
    assert max(abs(v) for v in leaf_ce) < bound


# --- 多段 rollout 軸 mcts_rollout_depth ---


def test_mcts_rollout_depth_default_is_one():
    model = _model(num_layers=3)
    driver = build_zoom_driver(_mcts_config(), model)
    # 既定は 1 段評価（従来挙動）で確率評価は無効
    assert driver.rollout_depth == 1
    assert driver.eval_stochastic is False


def test_mcts_rollout_depth_one_bit_compat_with_default():
    """``mcts_rollout_depth=1`` を明示しても既定（未指定）と数値が完全一致する

    既定が 1 なので明示は no-op であるべきで logits・全 extra_losses が一致する
    """
    base = torch.randn(1, 10, IN_DIM)
    label = torch.tensor([2])

    def run_once(**overrides):
        torch.manual_seed(3)
        model = _model(num_layers=3)
        driver = build_zoom_driver(_mcts_config(**overrides), model)
        model.train()
        logits, _, _, ctx = driver.run(
            base, MAGS_3, _seeded_child_loader(), torch.device("cpu"), label=label
        )
        return (
            float(logits.detach().sum()),
            {k: float(v.detach()) for k, v in ctx.extra_losses.items()},
        )

    assert run_once() == run_once(mcts_rollout_depth=1)


def _evaluate_with_depth(rollout_depth, action=0):
    """3 層モデルで深さ ``rollout_depth`` の最上層 evaluate を回し ``(ctx, reward)`` を返す

    共有 ``_RolloutContext`` を観測し，葉評価回数・子キャッシュの到達層を確認する
    """
    import numpy as np

    from foveamil.models.search.driver import _RolloutContext, _ZoomSearchProblem

    torch.manual_seed(0)
    model = _model(num_layers=3)
    build_zoom_driver(_mcts_config(), model)  # search_policy/value を登録する
    model.eval()
    x_fc = torch.randn(1, 6, OUT_DIM)
    ctx = _RolloutContext(
        model=model,
        value_net=model.search_value,
        child_loader=_seeded_child_loader(),
        magnifications=MAGS_3,
        num_layers=3,
        planner_name="gumbel",
        rollout_simulations=4,
        rollout_considered=4,
        stochastic=False,
        device=torch.device("cpu"),
    )
    problem = _ZoomSearchProblem(
        prior_np=np.full(6, 1.0 / 6),
        x_fc=x_fc,
        layer_idx=0,
        next_mag=MAGS_3[1],
        cpp=children_per_parent(MAGS_3[0], MAGS_3[1]),
        global_idx=None,
        rollout_depth=rollout_depth,
        seed=0,
        ctx=ctx,
    )
    reward = problem.evaluate(action)
    return ctx, reward


def test_mcts_rollout_depth_two_evaluates_deeper_state():
    """``rollout_depth=2`` は子を更に次倍率へ展開し最深層 (layer 1 の子) を葉評価する

    depth=1 では子キャッシュは layer 0 の鍵のみ（最深評価は projections[1] の状態）
    depth=2 では rollout が layer 1 へ降り layer 1 の子キャッシュ鍵が増える
    """
    ctx1, _ = _evaluate_with_depth(rollout_depth=1)
    ctx2, _ = _evaluate_with_depth(rollout_depth=2)

    layers1 = {layer for (layer, _) in ctx1.child_cache}
    layers2 = {layer for (layer, _) in ctx2.child_cache}
    # depth=1 は layer 0 の子ロードのみ depth=2 は layer 1 へ降り子ロードが増える
    assert layers1 == {0}
    assert 1 in layers2
    # depth=1 は 1 候補 1 葉 depth=2 は層 1 の sub-planner が複数葉を評価し増える
    assert ctx1.leaf_evals == 1
    assert ctx2.leaf_evals > ctx1.leaf_evals


def test_mcts_rollout_depth_two_runs_end_to_end():
    """``rollout_depth=2`` の driver.run が有限な合成損失を作り深い rollout が回る"""
    torch.manual_seed(0)
    model = _model(num_layers=3)
    driver = build_zoom_driver(_mcts_config(mcts_rollout_depth=2), model)
    model.train()
    device = torch.device("cpu")
    base = torch.randn(1, 12, IN_DIM)
    label = torch.tensor([1])
    logits, _, _, ctx = driver.run(
        base, MAGS_3, _seeded_child_loader(), device, label=label
    )
    composite = F.cross_entropy(logits, label) + sum(ctx.extra_losses.values())
    assert torch.isfinite(composite)


# --- 確率的葉評価 mcts_eval_stochastic ---


def test_mcts_eval_stochastic_varies_across_simulations():
    """確率評価では同一候補を 2 回評価しても MC dropout で値が異なる（memoize 撤廃）"""
    from foveamil.models.search.value import ValueNetwork

    torch.manual_seed(0)
    model = _model(num_layers=2)
    value_net = ValueNetwork(OUT_DIM, HIDDEN, dropout=0.5)

    problem = _zoom_search_problem(
        model, value_net, rollout_depth=1, stochastic=True
    )
    torch.manual_seed(7)
    first = problem.evaluate(0)
    second = problem.evaluate(0)
    # memoize されないため同一候補でも MC dropout で評価が異なる
    assert first != second


def test_mcts_eval_non_stochastic_memoizes_same_value():
    """非確率（既定）では同一候補の評価は memoize され同値（従来挙動）"""
    from foveamil.models.search.value import ValueNetwork

    torch.manual_seed(0)
    model = _model(num_layers=2)
    value_net = ValueNetwork(OUT_DIM, HIDDEN, dropout=0.5)
    problem = _zoom_search_problem(
        model, value_net, rollout_depth=1, stochastic=False
    )
    assert problem.evaluate(0) == problem.evaluate(0)


def test_mcts_stochastic_improved_policy_differs_from_deterministic():
    """確率評価の simulation 分散が選択へ波及する

    評価値が simulation 間で変動することは別テストが担保する 本テストはその分散が
    改良方策を介し選択へ波及することを確認する 単一シードでは argmax が偶然一致
    しうるため複数シードで集め少なくとも1つが決定版と異なることを要求する
    """
    base = torch.randn(1, 12, IN_DIM)
    label = torch.tensor([1])

    def first_layer_selection(stochastic, run_seed):
        torch.manual_seed(0)
        model = _model(num_layers=3)
        driver = build_zoom_driver(
            _mcts_config(
                mcts_eval_stochastic=stochastic,
                drop_out=0.5,
                mcts_simulations=24,
                mcts_max_considered=8,
            ),
            model,
        )
        model.train()
        torch.manual_seed(run_seed)
        _, _, _, ctx = driver.run(
            base, MAGS_3, _seeded_child_loader(), torch.device("cpu"), label=label
        )
        return tuple(
            None
            if s is None
            else tuple(s["select_indices"].cpu().numpy().ravel().tolist())
            for s in ctx.selections
        )

    sel_det = first_layer_selection(False, 0)
    sto_variants = {first_layer_selection(True, seed) for seed in range(6)}
    assert any(variant != sel_det for variant in sto_variants)


# --- 深い rollout でも policy/value へ勾配が流れる（lazy選択未学習バグの再来防止）---


def test_mcts_rollout_depth_two_propagates_policy_value_gradient():
    """``rollout_depth=2`` でも extra_losses が policy/value ネットへ勾配を流す

    主 CE を除き extra_losses のみで backward し，深い木でも方策・価値ネットへ
    勾配が流れる（>0）ことを確認する（探索が学習されないバグの再来防止）
    """
    torch.manual_seed(0)
    model = _model(num_layers=3)
    driver = build_zoom_driver(
        _mcts_config(mcts_rollout_depth=2, mcts_value_target="leaf_ce"), model
    )
    model.train()
    base = torch.randn(1, 12, IN_DIM)
    label = torch.tensor([2])
    _, _, _, ctx = driver.run(
        base, MAGS_3, _seeded_child_loader(), torch.device("cpu"), label=label
    )
    model.zero_grad()
    sum(ctx.extra_losses.values()).backward()

    def grad_sum(module):
        return sum(
            p.grad.abs().sum().item()
            for p in module.parameters()
            if p.grad is not None
        )

    assert grad_sum(model.search_policy) > 0
    assert grad_sum(model.search_value) > 0
