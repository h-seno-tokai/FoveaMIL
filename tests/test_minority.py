"""少数クラス学習信号強化（mixup・サンプラ温度・ordinal 補助損失）のユニット

3 機構が既定値（``mixup_alpha=0`` / ``sampler_temp=1.0`` / ``ordinal_aux_weight=0``）で
現行と数値一致すること（bit 互換）各機構の数式・挙動・勾配・決定性を確かめる
"""

import torch
import torch.nn.functional as F

from foveamil.training.minority import (
    BagMixup,
    OrdinalAuxLoss,
    temper_sampler_weights,
)


# --- サンプラ温度 ---


def test_sampler_temp_identity_returns_input_unchanged():
    # temp=1.0 で重みが不変（bit 互換）
    w = torch.tensor([3.0, 1.0, 0.0, 5.0], dtype=torch.float64)
    out = temper_sampler_weights(w, 1.0)
    assert torch.equal(out, w)


def test_sampler_temp_below_one_relaxes_toward_uniform():
    # temp<1 で最大/最小の比が縮む（一様寄り）
    w = torch.tensor([8.0, 1.0], dtype=torch.float64)
    base_ratio = (w[0] / w[1]).item()
    relaxed = temper_sampler_weights(w, 0.5)
    assert (relaxed[0] / relaxed[1]).item() < base_ratio


def test_sampler_temp_above_one_emphasizes():
    # temp>1 で比が広がる（少数クラス強調）
    w = torch.tensor([8.0, 1.0], dtype=torch.float64)
    base_ratio = (w[0] / w[1]).item()
    emphasized = temper_sampler_weights(w, 2.0)
    assert (emphasized[0] / emphasized[1]).item() > base_ratio


def test_sampler_temp_keeps_zero_weight_zero():
    # 不在クラス（0 重み）は temp に依らず 0 のまま
    w = torch.tensor([4.0, 0.0, 2.0], dtype=torch.float64)
    out = temper_sampler_weights(w, 0.3)
    assert out[1].item() == 0.0


def test_sampler_temp_power_formula():
    w = torch.tensor([2.0, 4.0, 9.0], dtype=torch.float64)
    out = temper_sampler_weights(w, 1.5)
    assert torch.allclose(out, w.pow(1.5))


# --- mixup（既定 off で bit 互換） ---


def test_mixup_disabled_returns_repr_unchanged_and_plain_loss():
    mixup = BagMixup(alpha=0.0, n_cls=3)
    repr_bag = torch.randn(1, 5, requires_grad=True)
    target = torch.tensor([2])
    # 2 回呼んでも（buffer ができても）alpha=0 では混ぜない
    for _ in range(2):
        repr_out, soft, loss_fn = mixup.mix(repr_bag, target)
        assert torch.equal(repr_out, repr_bag)
        assert torch.equal(soft.argmax(dim=-1), target)
    logits = torch.randn(1, 3)
    crit = torch.nn.CrossEntropyLoss()
    assert torch.allclose(loss_fn(crit, logits), crit(logits, target))


def test_mixup_first_call_is_plain_then_interpolates():
    mixup = BagMixup(alpha=1.0, n_cls=3)
    crit = torch.nn.CrossEntropyLoss()
    repr_a = torch.randn(1, 4)
    target_a = torch.tensor([0])
    # 初回は buffer が無いため素のラベル損失
    _, _, loss_fn0 = mixup.mix(repr_a, target_a)
    logits = torch.randn(1, 3)
    assert torch.allclose(loss_fn0(crit, logits), crit(logits, target_a))


def test_mixup_label_interpolation_and_repr_interpolation():
    # 2 サンプル目で repr とラベルが λ で補間される
    mixup = BagMixup(alpha=1.0, n_cls=3)
    repr_a = torch.full((1, 4), 1.0)
    repr_b = torch.full((1, 4), 5.0)
    mixup.mix(repr_a, torch.tensor([0]))  # buffer に a を退避
    repr_mixed, soft_mixed, loss_fn = mixup.mix(repr_b, torch.tensor([1]))
    # 補間後の repr は a と b の間（全要素が 1 と 5 の凸結合）
    assert (repr_mixed >= 1.0).all() and (repr_mixed <= 5.0).all()
    # soft ラベルはクラス 0,1 のみに質量を持ち和 1
    assert torch.allclose(soft_mixed.sum(), torch.tensor(1.0))
    assert soft_mixed[0, 2].item() == 0.0
    # 混合損失は λ·CE(·,1) + (1-λ)·CE(·,0) の形（両クラスの凸結合）
    logits = torch.tensor([[2.0, 0.5, -1.0]])
    lam = soft_mixed[0, 1].item()
    crit = torch.nn.CrossEntropyLoss()
    expected = lam * crit(logits, torch.tensor([1])) + (1.0 - lam) * crit(
        logits, torch.tensor([0])
    )
    assert torch.allclose(loss_fn(crit, logits), expected)


def test_mixup_gradient_flows_to_current_repr_only():
    # 補間 repr の勾配は現サンプル（勾配付き）にのみ流れ buffer 側（detach）には流れない
    mixup = BagMixup(alpha=1.0, n_cls=3)
    repr_a = torch.randn(1, 4, requires_grad=True)
    mixup.mix(repr_a, torch.tensor([0]))
    repr_b = torch.randn(1, 4, requires_grad=True)
    repr_mixed, _, _ = mixup.mix(repr_b, torch.tensor([1]))
    repr_mixed.sum().backward()
    assert repr_b.grad is not None and repr_b.grad.abs().sum() > 0
    # a は detach されて buffer に入るため勾配が伝わらない
    assert repr_a.grad is None


def test_mixup_deterministic_with_seeded_generator():
    def run():
        gen = torch.Generator()
        gen.manual_seed(123)
        mixup = BagMixup(alpha=0.7, n_cls=3, generator=gen)
        mixup.mix(torch.zeros(1, 4), torch.tensor([0]))
        _, soft, _ = mixup.mix(torch.ones(1, 4), torch.tensor([1]))
        return soft.clone()

    assert torch.equal(run(), run())


def _lam_samples(alpha, n, seed=0):
    # generator 経路（本番経路）で λ を n 本引く
    gen = torch.Generator()
    gen.manual_seed(seed)
    mixup = BagMixup(alpha=alpha, n_cls=2, generator=gen)
    return torch.stack(
        [mixup._sample_lam(torch.device("cpu")) for _ in range(n)]
    )


def test_mixup_lam_variance_decreases_monotonically_in_alpha():
    # Beta(α,α) の分散 1/(4(2α+1)) は α 増で単調減少 generator 経路で再現すること
    n = 8000
    alphas = [0.2, 0.5, 1.0, 2.0, 5.0]
    variances = [_lam_samples(a, n).var().item() for a in alphas]
    for lo, hi in zip(variances, variances[1:]):
        assert hi < lo, f"var が単調減少でない: {variances}"


def test_mixup_lam_variance_close_to_beta_theory():
    # generator 経路の var が Beta(α,α) 理論値に近接（旧バグでは α 非依存の一様だった）
    n = 12000
    for alpha in (0.5, 1.0, 3.0):
        var = _lam_samples(alpha, n).var().item()
        theory = 1.0 / (4.0 * (2.0 * alpha + 1.0))
        assert abs(var - theory) < 0.015, (
            f"alpha={alpha}: var={var:.5f} theory={theory:.5f}"
        )


def test_mixup_lam_not_uniform_for_concentrated_alpha():
    # α 大は λ が 0.5 付近に集中（旧バグの一様 var=1/12 とは明確に異なる）
    # 旧バグでは α に依らず var≈1/12 だったため この差が回帰を捕える
    var = _lam_samples(5.0, 12000).var().item()
    assert var < 1.0 / 12.0 - 0.03


def test_mixup_lam_alpha_one_is_uniform():
    # α=1 で Beta(1,1)=Uniform var≈1/12
    var = _lam_samples(1.0, 12000).var().item()
    assert abs(var - 1.0 / 12.0) < 0.01


def test_mixup_lam_deterministic_same_seed():
    # 同一 seed で λ 列が完全一致（決定的・global RNG 非汚染）
    assert torch.equal(_lam_samples(0.3, 200, seed=7), _lam_samples(0.3, 200, seed=7))


def test_mixup_lam_does_not_pollute_global_rng():
    # generator 経路は global RNG を進めない（汚染しない）
    torch.manual_seed(0)
    before = torch.rand(3)
    torch.manual_seed(0)
    _lam_samples(0.4, 100)
    after = torch.rand(3)
    assert torch.equal(before, after)


def test_mixup_lam_finite_and_in_unit_interval():
    # α<1（境界）でも λ は有限かつ (0,1) に収まる
    lams = _lam_samples(0.1, 3000)
    assert torch.isfinite(lams).all()
    assert (lams >= 0.0).all() and (lams <= 1.0).all()


# --- ordinal 補助損失 ---


def test_ordinal_zero_when_prediction_matches_rank():
    # 期待ランクが正解ランクに一致すれば損失 0
    loss = OrdinalAuxLoss(n_cls=4)
    # クラス 2 にほぼ全質量
    logits = torch.tensor([[-10.0, -10.0, 10.0, -10.0]])
    out = loss(logits, torch.tensor([2]))
    assert out.item() < 1e-6


def test_ordinal_monotonic_in_rank_distance():
    # 同一予測に対し 正解ランクが遠いほど損失が大きい（順序の単調性）
    loss = OrdinalAuxLoss(n_cls=4)
    logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]])  # 期待ランク ~0
    near = loss(logits, torch.tensor([1])).item()
    far = loss(logits, torch.tensor([3])).item()
    assert far > near


def test_ordinal_matches_manual_formula():
    n_cls = 4
    loss = OrdinalAuxLoss(n_cls)
    logits = torch.tensor([[0.5, 1.0, -0.5, 0.2]])
    target = torch.tensor([1])
    probs = F.softmax(logits, dim=-1)
    ranks = torch.arange(n_cls, dtype=torch.float32)
    expected_rank = (probs * ranks).sum(dim=-1)
    diff = (expected_rank - ranks[target]) / (n_cls - 1)
    expected = (diff * diff).mean()
    assert torch.allclose(loss(logits, target), expected)


def test_ordinal_gradient_flows_to_logits():
    loss = OrdinalAuxLoss(n_cls=3)
    logits = torch.randn(4, 3, requires_grad=True)
    out = loss(logits, torch.tensor([0, 1, 2, 1]))
    out.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_ordinal_single_class_is_safe():
    # n_cls=1 では順序が無く scale が恒等（0 除算なし）損失は有限
    loss = OrdinalAuxLoss(n_cls=1)
    out = loss(torch.zeros(2, 1), torch.tensor([0, 0]))
    assert torch.isfinite(out).all()


# --- モデルの分割 forward が従来 forward_final と bit 互換 ---


def test_classify_fuse_repr_matches_forward_final():
    from foveamil.models import FoveaMIL

    torch.manual_seed(0)
    model = FoveaMIL(
        in_feat_dim=8, hidden_feat_dim=16, out_feat_dim=12, n_cls=4, num_layers=2
    )
    model.eval()
    m_list = [torch.randn(1, 1, 12), torch.randn(1, 1, 12)]
    logits_a, yhat_a, prob_a = model.forward_final(m_list)
    logits_b, yhat_b, prob_b = model.classify(model.fuse_repr(m_list))
    assert torch.equal(logits_a, logits_b)
    assert torch.equal(yhat_a, yhat_b)
    assert torch.equal(prob_a, prob_b)


def test_instance_repr_matches_instance_loss_logits():
    from foveamil.models import FoveaMIL

    torch.manual_seed(0)
    model = FoveaMIL(
        in_feat_dim=8,
        hidden_feat_dim=16,
        out_feat_dim=12,
        n_cls=3,
        num_layers=1,
        instance_loss=True,
    )
    model.eval()
    x = torch.randn(1, 10, 8)
    label = torch.tensor([1])
    logits, _, _, inst_loss = model.forward_with_instance_loss(x, label)
    fused, inst_loss2 = model.forward_with_instance_repr(x, label)
    logits2, _, _ = model.classify(fused)
    assert torch.equal(logits, logits2)
    assert torch.allclose(inst_loss, inst_loss2)
