"""探索部品（方策・価値ネット，Gumbel-AlphaZero / PUCT プランナ）のユニット

方策の和 1・価値の形状・有限非ゼロ勾配，プランナの改良方策が妥当（非負・和 1）で，
仕込んだ高報酬アクションに事前より強く集中し，同一シードで決定的であることを確認する
"""

import numpy as np
import pytest
import torch

from foveamil.models.search import (
    PolicyNetwork,
    SearchProblem,
    ValueNetwork,
    build_planner,
)
from foveamil.models.search.mcts import (
    GumbelAlphaZeroPlanner,
    PuctPlanner,
    _completed_q,
    _sequential_halving_schedule,
)


# --- 方策ネット ---


@pytest.mark.parametrize("n", [1, 5, 20])
def test_policy_outputs_distribution(n):
    pol = PolicyNetwork(feat_dim=12, hidden_dim=8)
    pol.eval()
    p = pol(torch.randn(1, n, 12))
    assert p.shape == (1, n)
    assert torch.all(p >= 0)
    assert torch.allclose(p.sum(dim=-1), torch.ones(1), atol=1e-6)


def test_policy_logits_shape():
    pol = PolicyNetwork(feat_dim=8, hidden_dim=4)
    logits = pol.logits(torch.randn(2, 7, 8))
    assert logits.shape == (2, 7)


def test_policy_finite_nonzero_gradient():
    torch.manual_seed(0)
    pol = PolicyNetwork(feat_dim=8, hidden_dim=4)
    pol.train()
    x = torch.randn(1, 10, 8, requires_grad=True)
    # softmax 分布の和は定数なので重み付き和で勾配を評価する
    weight = torch.linspace(0.1, 1.0, 10)
    (pol(x) * weight).sum().backward()
    grads = [p.grad for p in pol.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0


# --- 価値ネット ---


@pytest.mark.parametrize("n", [1, 6, 15])
def test_value_outputs_scalar_per_batch(n):
    val = ValueNetwork(feat_dim=12, hidden_dim=8)
    val.eval()
    v = val(torch.randn(3, n, 12))
    assert v.shape == (3,)
    assert torch.isfinite(v).all()


def test_value_finite_nonzero_gradient():
    torch.manual_seed(0)
    val = ValueNetwork(feat_dim=8, hidden_dim=4)
    val.train()
    x = torch.randn(2, 9, 8, requires_grad=True)
    val(x).sum().backward()
    grads = [p.grad for p in val.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0


# --- プランナの toy 問題 ---


class _PlantedProblem(SearchProblem):
    """1 つの仕込みアクションのみ高報酬を返す toy 問題（純粋）"""

    def __init__(self, n, planted, prior=None):
        self.n = n
        self.planted = planted
        self._prior = (
            np.full(n, 1.0 / n) if prior is None else np.asarray(prior, dtype=float)
        )

    def num_actions(self):
        return self.n

    def prior(self):
        return self._prior

    def evaluate(self, action):
        return 1.0 if action == self.planted else 0.0


PLANNER_NAMES = ["gumbel", "puct"]


@pytest.mark.parametrize("name", PLANNER_NAMES)
def test_planner_improved_policy_is_valid_distribution(name):
    prob = _PlantedProblem(n=10, planted=4)
    planner = build_planner(name, simulations=32, max_considered=8, seed=0)
    result = planner.run(prob, num_select=3)
    pol = result.improved_policy
    assert pol.shape == (10,)
    assert np.all(pol >= 0)
    assert abs(float(pol.sum()) - 1.0) < 1e-9


@pytest.mark.parametrize("name", PLANNER_NAMES)
def test_planner_concentrates_on_planted_more_than_prior(name):
    n, planted = 12, 7
    prob = _PlantedProblem(n=n, planted=planted)
    planner = build_planner(name, simulations=48, max_considered=n, seed=1)
    result = planner.run(prob, num_select=2)
    raw_prior = result.prior[planted]
    improved = result.improved_policy[planted]
    # 仕込みアクションへ事前より強く集中する
    assert improved > raw_prior
    # 改良方策の argmax は仕込みアクション
    assert int(result.improved_policy.argmax()) == planted


@pytest.mark.parametrize("name", PLANNER_NAMES)
def test_planner_chosen_actions_are_valid(name):
    prob = _PlantedProblem(n=8, planted=3)
    planner = build_planner(name, simulations=24, max_considered=8, seed=0)
    result = planner.run(prob, num_select=4)
    chosen = result.chosen_actions
    assert len(chosen) == 4
    assert np.all((chosen >= 0) & (chosen < 8))
    # 昇順かつ重複なし
    assert list(chosen) == sorted(set(chosen.tolist()))
    # 仕込みアクションは選ばれる
    assert prob.planted in chosen.tolist()


@pytest.mark.parametrize("name", PLANNER_NAMES)
def test_planner_deterministic_under_seed(name):
    prob = _PlantedProblem(n=15, planted=9)
    a = build_planner(name, simulations=40, max_considered=10, seed=123).run(
        prob, num_select=3
    )
    b = build_planner(name, simulations=40, max_considered=10, seed=123).run(
        prob, num_select=3
    )
    assert np.allclose(a.improved_policy, b.improved_policy)
    assert np.array_equal(a.chosen_actions, b.chosen_actions)


def test_gumbel_seed_changes_explored_candidate_set():
    # Gumbel は探索する候補集合を変える（改良方策は完了 Q から決定的だが探索は seed 依存）
    # 一様事前・無報酬なら訪問された候補集合がシードで変わる
    prob_flat = _PlantedProblem(n=20, planted=-1)  # planted=-1 はどの index にも当たらない
    a = build_planner("gumbel", simulations=20, max_considered=4, seed=0).run(
        prob_flat, num_select=4
    )
    b = build_planner("gumbel", simulations=20, max_considered=4, seed=999).run(
        prob_flat, num_select=4
    )
    visited_a = set(np.flatnonzero(a.visit_counts).tolist())
    visited_b = set(np.flatnonzero(b.visit_counts).tolist())
    assert visited_a != visited_b


def test_build_planner_unknown_raises():
    with pytest.raises(KeyError):
        build_planner("does_not_exist", simulations=8, max_considered=4)


def test_build_planner_types():
    assert isinstance(
        build_planner("gumbel", simulations=8, max_considered=4),
        GumbelAlphaZeroPlanner,
    )
    assert isinstance(
        build_planner("puct", simulations=8, max_considered=4), PuctPlanner
    )


def test_prior_length_mismatch_raises():
    class Bad(_PlantedProblem):
        def prior(self):
            return np.ones(self.n + 1)

    planner = build_planner("gumbel", simulations=8, max_considered=4, seed=0)
    with pytest.raises(ValueError, match="prior length"):
        planner.run(Bad(n=5, planted=0), num_select=2)


# --- 内部ヘルパ ---


def test_sequential_halving_schedule_halves_to_one():
    assert _sequential_halving_schedule(8, 32) == [8, 4, 2, 1]
    assert _sequential_halving_schedule(1, 4) == [1]
    assert _sequential_halving_schedule(3, 8)[-1] == 1


def test_completed_q_fills_unvisited_with_prior_weighted_mix():
    prior = np.array([0.5, 0.3, 0.2])
    q_sum = np.array([2.0, 0.0, 0.0])
    visit = np.array([2, 0, 0])
    evaluated = {0: 1.0}
    q = _completed_q(prior, q_sum, visit, evaluated)
    # 訪問済みは実測平均
    assert q[0] == pytest.approx(1.0)
    # 未訪問は訪問済みの prior 重み付き平均（ここでは唯一の訪問 mean=1.0）
    assert q[1] == pytest.approx(1.0)
    assert q[2] == pytest.approx(1.0)


def test_completed_q_zero_when_no_visits():
    q = _completed_q(np.array([0.5, 0.5]), np.zeros(2), np.zeros(2, dtype=int), {})
    assert np.all(q == 0.0)
