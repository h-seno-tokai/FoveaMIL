"""機能を組み合わせたときの整合（合成損失の有限性・sweep 軸の同時展開と畳み込み）のユニット

スパース補助アテンション（B）・DPP 選択（D）・脱相関と多様性の正則化（A/D）を同時に有効化し，
合成損失が有限で勾配が流れること，および 4 機能を sweep 軸にしたときの展開と単一倍率での畳み込み
を確認する
"""

import torch

from foveamil.models import ForwardContext, iter_active_regularizers
from foveamil.training.config import TrainConfig
from foveamil.training.resolve import ResolvedPaths
from foveamil.training.sweep import expand_combos
from foveamil.training.trainer import build_foveamil_from_config, regularizer_loss

# 合成検証に使う倍率列・特徴次元
_MAGS = [1.25, 2.5, 5.0]
_IN_DIM = 8
_OUT_DIM = 12
_K = 3
_N_CLS = 3
# 子パッチ数（連続 2x）
_CHILDREN = 4


def _resolved():
    return ResolvedPaths(
        n_cls=3,
        folds=10,
        labels_csv="/c/labels.csv",
        splits_dir="/c/splits",
        feature_root_base="/feat",
    )


def _base_sweep(**overrides):
    sweep = {
        "encoder": ["ResNet50"],
        "feature_type": ["mean"],
        "magnifications": [[1.25, 2.5]],
    }
    sweep.update(overrides)
    return sweep


def test_sparse_norm_dpp_selector_and_regularizers_compose():
    torch.manual_seed(0)
    config = TrainConfig(
        magnifications=_MAGS,
        in_feat_dim=_IN_DIM,
        hidden_feat_dim=16,
        out_feat_dim=_OUT_DIM,
        k_sample=_K,
        n_cls=_N_CLS,
        aux_norm="entmax",
        aux_norm_alpha=1.5,
        selector="dpp",
        dpp_similarity="cosine",
        dpp_diversity_weight=0.1,
        decorrelation_weight=0.05,
        decorrelation_method="cosine",
    )
    model = build_foveamil_from_config(config, num_layers=len(_MAGS))
    regularizers = iter_active_regularizers(config)
    assert {r.name for r in regularizers} == {"decorrelation", "dpp_diversity"}

    model.train()
    m_list, layer_aux, dpp_log_dets = [], [], []
    x = torch.randn(1, 16, _IN_DIM)
    for layer_idx in range(len(_MAGS)):
        M, _, weight, aux = model.forward_layer(x, layer_idx)
        m_list.append(M)
        layer_aux.append(aux)
        log_det = model.selector.pop_log_det()
        if log_det is not None:
            dpp_log_dets.append(log_det)
        if layer_idx < len(_MAGS) - 1:
            x = torch.randn(1, weight.shape[1] * _CHILDREN, _IN_DIM)
    context = ForwardContext(
        m_list=m_list, layer_aux=layer_aux, dpp_log_dets=dpp_log_dets
    )
    logits, _, _ = model.forward_final(m_list)
    label = torch.tensor([1])
    loss = torch.nn.functional.cross_entropy(logits, label) + regularizer_loss(
        regularizers, context, label
    )
    assert torch.isfinite(loss)
    loss.backward()
    grad_sum = sum(
        float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None
    )
    assert grad_sum > 0


def test_all_feature_axes_expand_for_multi_mag_and_fold_for_single_mag():
    axes = dict(
        aux_norm=["softmax", "entmax"],
        selector=["topk", "dpp"],
        decorrelation_weight=[0.0, 0.05],
        zoom_driver=["differentiable", "mcts"],
    )
    multi = expand_combos(_base_sweep(**axes), {}, _resolved())
    assert len(multi) == 16  # 2 * 2 * 2 * 2
    single = expand_combos(
        _base_sweep(magnifications=[[40]], **axes), {}, _resolved()
    )
    assert len(single) == 1  # 単一倍率では全軸が畳まれる
