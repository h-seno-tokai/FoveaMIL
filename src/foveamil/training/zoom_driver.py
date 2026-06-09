"""倍率ごとのズーム駆動の差し替え可能シーム

各倍率（低→高）でどの親パッチを高解像度の子へ展開するかを決め，段階 forward を
回して ``(logits, Y_hat, Y_prob, ForwardContext)`` を返す共通インタフェースを定義する
子特徴のロードは callable ``child_loader(next_mag, child_global_indices) -> [1, Nc, D]``
で注入し，学習ループ側は :class:`FeatureAccessor` 由来の callable を，テストは合成子を
返す callable を与えられる

:class:`DifferentiableZoomDriver` は補助アテンションの top-k 選択を一括で確定する既定
駆動で，従来の学習ループと同一の数値を再現する探索ベースの駆動は
``foveamil.models.search`` の :class:`MCTSZoomDriver` が提供する駆動はレジストリ
``build_zoom_driver`` で構築する
"""

from __future__ import annotations

import abc
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from foveamil.models.regularizers import ForwardContext
from foveamil.training.hierarchy import children_per_parent, compute_child_indices

# 子特徴ローダの型（次倍率と子 global index から ``[1, Nc, D]`` を返す）
ChildLoader = Callable[[float, np.ndarray], Tensor]

# 既定駆動名
DRIVER_DIFFERENTIABLE = "differentiable"
# 探索駆動名
DRIVER_MCTS = "mcts"


class ZoomDriver(abc.ABC):
    """倍率ごとのズーム駆動の抽象基底

    Args:
        model: 段階 forward を持つ :class:`FoveaMIL`
        num_layers: 倍率数
    """

    def __init__(self, model, num_layers: int) -> None:
        self.model = model
        self.num_layers = num_layers

    def set_curriculum(self, epoch: int) -> None:
        """機構L カリキュラム hook（既定 no-op）

        :class:`MCTSZoomDriver` のみ override し epoch 依存で RL 損失重みを ramp する
        探索を持たない駆動では何もしない
        """

    def _pop_log_det(self):
        """選択コントローラが log-det を保持していれば取り出す（無ければ ``None``）"""
        pop = getattr(self.model.selector, "pop_log_det", None)
        if pop is None:
            return None
        return pop()

    @abc.abstractmethod
    def run(
        self,
        base_feats: Tensor,
        magnifications: List[float],
        child_loader: ChildLoader,
        device: torch.device,
        label: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, ForwardContext]:
        """段階 forward を回して予測と forward 文脈を返す

        Args:
            base_feats: 最低倍率の全特徴 ``[1, N, in_feat_dim]``
            magnifications: 低→高の倍率列
            child_loader: 子特徴ローダ ``(next_mag, child_global_indices) -> [1, Nc, D]``
            device: 計算デバイス
            label: 正解クラス ``[B]``（学習時の補助損失用無ければ推論）

        Returns:
            ``(logits, Y_hat, Y_prob, ForwardContext)``
        """


class DifferentiableZoomDriver(ZoomDriver):
    """補助アテンションの top-k 選択を一括確定する既定のズーム駆動

    各層の選択結果から子の global index を求め，次倍率の子を ``child_loader`` でロード
    し，選択重み（学習時 soft / 推論時 hard）を子特徴へ掛けて補助アテンションへ勾配を
    流す各倍率のプーリング表現と各層の選択を :class:`ForwardContext` に集める従来の
    学習ループと同一の数値を再現する
    """

    def run(
        self,
        base_feats: Tensor,
        magnifications: List[float],
        child_loader: ChildLoader,
        device: torch.device,
        label: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, ForwardContext]:
        M_list: List[Tensor] = []
        selections: List[Optional[dict]] = []
        layer_aux: List[Optional[Tensor]] = []
        dpp_log_dets: List[Tensor] = []
        x = base_feats.to(device)
        global_idx: Optional[np.ndarray] = None
        for layer_idx in range(self.num_layers):
            M, select_indices, select_weight, aux = self.model.forward_layer(
                x, layer_idx
            )
            M_list.append(M)
            layer_aux.append(aux)
            # DPP 選択コントローラなら選択部分カーネル log-det を排出して文脈へ積む
            log_det = self._pop_log_det()
            if log_det is not None:
                dpp_log_dets.append(log_det)
            if layer_idx >= self.num_layers - 1:
                selections.append(None)
                continue

            selections.append(
                {"select_indices": select_indices, "select_weight": select_weight}
            )
            cur_mag = magnifications[layer_idx]
            next_mag = magnifications[layer_idx + 1]
            cpp = children_per_parent(cur_mag, next_mag)

            local_idx = select_indices[0].detach().cpu().numpy()
            child = compute_child_indices(local_idx, global_idx, children=cpp)
            x_next = child_loader(next_mag, child).to(device)
            # 選択重みを子特徴へ掛け，補助アテンションへ勾配を流す
            # （各親の子が連続する r^2 個なので重みを cpp 回繰り返す）
            w_child = select_weight.repeat_interleave(cpp, dim=1)
            x_next = x_next * w_child.unsqueeze(-1)
            x = x_next
            global_idx = child
        logits, Y_hat, Y_prob = self.model.forward_final(M_list)
        context = ForwardContext(
            m_list=M_list,
            selections=selections,
            layer_aux=layer_aux,
            dpp_log_dets=dpp_log_dets,
        )
        return logits, Y_hat, Y_prob, context


def build_zoom_driver(config, model) -> ZoomDriver:
    """設定とモデルからズーム駆動を構築する

    ``config.zoom_driver`` の名前に応じて駆動を選ぶ既定 ``"differentiable"`` は
    従来挙動を再現し，``"mcts"`` は探索ベースの :class:`MCTSZoomDriver` を返す

    Args:
        config: ``TrainConfig``
        model: 段階 forward を持つ :class:`FoveaMIL`

    Returns:
        構築した :class:`ZoomDriver`

    Raises:
        KeyError: ``config.zoom_driver`` が未知の名前の場合
    """
    num_layers = model.num_layers
    name = getattr(config, "zoom_driver", DRIVER_DIFFERENTIABLE)
    if name == DRIVER_DIFFERENTIABLE:
        return DifferentiableZoomDriver(model, num_layers)
    if name == DRIVER_MCTS:
        from foveamil.models.search.driver import MCTSZoomDriver

        return MCTSZoomDriver.from_config(config, model, num_layers)
    raise KeyError(
        f"unknown zoom_driver '{name}'; "
        f"available: {(DRIVER_DIFFERENTIABLE, DRIVER_MCTS)}"
    )
