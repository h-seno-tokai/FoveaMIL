"""非学習の選択コントローラ（random / uniform）のユニット

学習あり選択器と同じ出力形状・k で動き，random は同シードで再現し global RNG を汚さず，
uniform は決定的で，空バッグ・k>N の縮退で安全に動き，registry から解決できることを
回帰ガードする
"""

import torch

from foveamil.models import FoveaMIL, build_selection_controller
from foveamil.models.selection import available_selection_controllers
from foveamil.models.topk import build_topk

# 候補数
NUM_ELEMENTS = 10
# バッチ数
BATCH_SIZE = 2
# 特徴次元
FEAT_DIM = 8
# 選択数 k
K = 4


def _inputs(num_elements: int = NUM_ELEMENTS, batch_size: int = BATCH_SIZE):
    """正規化済みスコア ``[B, N]`` と射影特徴 ``[B, N, D]`` を作る"""
    scores = torch.softmax(torch.randn(batch_size, num_elements), dim=-1)
    features = torch.randn(batch_size, num_elements, FEAT_DIM)
    return scores, features


def test_baselines_registered():
    available = available_selection_controllers()
    assert "random" in available
    assert "uniform" in available


def test_shape_matches_learned_topk():
    scores, features = _inputs()
    ref = build_topk("perturbed", k=K)
    ref.eval()
    ref_out = ref(scores)
    for name in ("random", "uniform"):
        controller = build_selection_controller(name, k=K)
        controller.eval()
        out = controller.select(scores, features)
        assert out.shape == ref_out.shape
        assert out.dtype == ref_out.dtype


def test_rows_are_hard_one_hot_and_ascending():
    scores, features = _inputs()
    for name in ("random", "uniform"):
        controller = build_selection_controller(name, k=K)
        controller.eval()
        out = controller.select(scores, features)
        # 各行は和=1 の hard one-hot
        assert torch.allclose(out.sum(dim=-1), torch.ones(BATCH_SIZE, K))
        assert float(out.max(dim=-1).values.min()) == 1.0
        idx = out.argmax(dim=-1)
        # 行は index 昇順で k 個は相異なる
        assert (idx[:, 1:] > idx[:, :-1]).all()


def test_random_reproducible_with_same_seed():
    scores, features = _inputs()
    controller = build_selection_controller("random", k=K, seed=123)
    first = controller.select(scores, features)
    second = controller.select(scores, features)
    assert torch.equal(first, second)


def test_random_differs_across_seeds():
    scores, features = _inputs()
    a = build_selection_controller("random", k=K, seed=1).select(scores, features)
    b = build_selection_controller("random", k=K, seed=2).select(scores, features)
    assert not torch.equal(a, b)


def test_random_does_not_perturb_global_rng():
    scores, features = _inputs()
    controller = build_selection_controller("random", k=K, seed=7)
    torch.manual_seed(999)
    before = torch.rand(5)
    controller.select(scores, features)
    torch.manual_seed(999)
    after = torch.rand(5)
    assert torch.allclose(before, after)


def test_random_ignores_scores_and_features():
    _, features = _inputs()
    controller = build_selection_controller("random", k=K, seed=0)
    a = controller.select(torch.softmax(torch.randn(BATCH_SIZE, NUM_ELEMENTS), -1), features)
    controller2 = build_selection_controller("random", k=K, seed=0)
    b = controller2.select(torch.softmax(torch.randn(BATCH_SIZE, NUM_ELEMENTS), -1), features)
    # スコアが違っても同シードなら同じ選択
    assert torch.equal(a, b)


def test_uniform_deterministic_and_even():
    scores, features = _inputs(num_elements=10, batch_size=1)
    controller = build_selection_controller("uniform", k=K)
    controller.eval()
    first = controller.select(scores, features)
    second = controller.select(scores, features)
    assert torch.equal(first, second)
    # 0..N-1 を k 分割した等間隔 index
    assert controller.select(scores, features).argmax(dim=-1)[0].tolist() == [0, 3, 6, 9]


def test_uniform_independent_of_scores():
    _, features = _inputs(batch_size=1)
    controller = build_selection_controller("uniform", k=K)
    a = controller.select(torch.softmax(torch.randn(1, NUM_ELEMENTS), -1), features)
    b = controller.select(torch.softmax(torch.randn(1, NUM_ELEMENTS), -1), features)
    assert torch.equal(a, b)


def test_k_clamps_to_n():
    scores, features = _inputs(num_elements=5)
    for name in ("random", "uniform"):
        controller = build_selection_controller(name, k=100)
        controller.eval()
        out = controller.select(scores, features)
        assert out.shape == (BATCH_SIZE, 5, 5)


def test_empty_bag_is_safe():
    scores = torch.zeros(BATCH_SIZE, 0)
    features = torch.zeros(BATCH_SIZE, 0, FEAT_DIM)
    for name in ("random", "uniform"):
        controller = build_selection_controller(name, k=K)
        controller.eval()
        out = controller.select(scores, features)
        assert out.shape == (BATCH_SIZE, 0, 0)


def test_no_grad_required():
    scores, features = _inputs()
    for name in ("random", "uniform"):
        controller = build_selection_controller(name, k=K)
        controller.eval()
        out = controller.select(scores, features)
        # 非学習選択器は勾配を要求しない
        assert not out.requires_grad


def test_foveamil_integration_weights_are_one():
    torch.manual_seed(0)
    for name in ("random", "uniform"):
        model = FoveaMIL(
            in_feat_dim=FEAT_DIM,
            hidden_feat_dim=16,
            out_feat_dim=12,
            k_sample=K,
            n_cls=3,
            num_layers=2,
            selector=name,
        )
        model.eval()
        M, idx, weight, aux = model.forward_layer(
            torch.randn(1, 20, FEAT_DIM), layer_idx=0
        )
        assert M.shape == (1, 1, 12)
        assert idx.shape == (1, K) and weight.shape == (1, K)
        # 行が index 昇順なので下流の argmax→sort→gather が選択重み 1.0 を拾う
        assert (idx[:, 1:] > idx[:, :-1]).all()
        assert torch.allclose(weight, torch.ones_like(weight))
