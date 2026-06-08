"""探索ベースのズーム駆動とその学習目的

各倍率で :class:`Planner` を回し，どの親を高解像度の子へ展開するかを look-ahead で
選ぶプランナの事前は方策ネット ``π``，葉評価は価値ネット ``v`` が与える選んだ親の
子を ``child_loader`` で materialise し，選択確率を子へ掛けて勾配を流し，融合表現を作る

学習損失は :class:`ForwardContext` の ``extra_losses`` に名前付きで積む分類損失（CE）
は学習ループが従来どおり加える本駆動は ``λ_π · 方策蒸留``（π から改良方策への交差
エントロピー）と ``λ_v · 価値回帰``を加える（任意でエントロピー項）識別器ヘッド・射影は
基線と共有し，並列分類器は持たない推論時はラベル無しで同じ探索を回しズーム先を選ぶ

価値回帰の目標は ``mcts_value_target`` で選ぶ``"realised"`` は探索表現の負分類損失
（最終 CE）を全状態へ broadcast する（detach）``"leaf_ce"`` は選択 j の結果状態を含む
融合を共有ヘッドへ通した状態依存 leaf 報酬（detach）を目標にし，選択確率には advantage
（leaf 報酬 − 価値推定）由来の actor-critic 項を加えることで，どの親を選ぶかが分類の良し
悪しを弁別する leaf 報酬は detach され共有ヘッド・融合へは勾配を流さない
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from foveamil.models.regularizers import ForwardContext
from foveamil.models.search.mcts import PlannerResult, SearchProblem, build_planner
from foveamil.models.search.policy import PolicyNetwork
from foveamil.models.search.value import ValueNetwork
from foveamil.training.hierarchy import children_per_parent, compute_child_indices
from foveamil.training.zoom_driver import ChildLoader, ZoomDriver

logger = logging.getLogger(__name__)

# 方策蒸留損失のキー
LOSS_POLICY = "mcts_policy"
# 価値回帰損失のキー
LOSS_VALUE = "mcts_value"
# 方策エントロピー損失のキー
LOSS_ENTROPY = "mcts_entropy"
# 価値ターゲット種別：最終 CE を全状態へ broadcast（従来挙動）
VALUE_TARGET_REALISED = "realised"
# 価値ターゲット種別：各部分選択状態の暫定融合を共有ヘッドへ通した状態依存 leaf 報酬
VALUE_TARGET_LEAF_CE = "leaf_ce"
# モデルへ登録する方策ネットの属性名
POLICY_ATTR = "search_policy"
# モデルへ登録する価値ネットの属性名
VALUE_ATTR = "search_value"
# 確率の数値下限（log 安定化）
_PROB_FLOOR = 1e-12
# プランナへ渡すシードの倍率ごとオフセット係数（層で別シードにする）
_LAYER_SEED_STRIDE = 1009


class _ZoomSearchProblem(SearchProblem):
    """1 倍率のズーム決定を解く探索問題

    候補は現倍率の親パッチ事前は方策ネット出力各候補の評価は，その親の子を
    ``child_loader`` で materialise し次倍率へ射影・プーリングした状態の価値ネット推定
    （look-ahead）同一候補の子ロードと評価をキャッシュし h5 読みの重複を避ける

    Args:
        prior_np: 事前方策 ``(N,)``（numpy detached）
        x_fc: 現倍率の射影特徴 ``[1, N, D]``
        layer_idx: 現倍率の添字
        next_mag: 次倍率
        cpp: 1 親あたりの子数（倍率比^2）
        global_idx: 現倍率パッチの global index（``None`` なら local=global）
        model: 段階 forward を持つ :class:`FoveaMIL`
        value_net: 価値ネット
        child_loader: 子特徴ローダ
        device: 計算デバイス
    """

    def __init__(
        self,
        prior_np: np.ndarray,
        x_fc: Tensor,
        layer_idx: int,
        next_mag: float,
        cpp: int,
        global_idx: Optional[np.ndarray],
        model,
        value_net: ValueNetwork,
        child_loader: ChildLoader,
        device: torch.device,
    ) -> None:
        self._prior = prior_np
        self.x_fc = x_fc
        self.layer_idx = layer_idx
        self.next_mag = next_mag
        self.cpp = cpp
        self.global_idx = global_idx
        self.model = model
        self.value_net = value_net
        self.child_loader = child_loader
        self.device = device
        self._reward_cache: Dict[int, float] = {}
        self._child_cache: Dict[int, Tensor] = {}

    def num_actions(self) -> int:
        return self._prior.shape[0]

    def prior(self) -> np.ndarray:
        return self._prior

    def _load_child(self, action: int) -> Tensor:
        """候補親 ``action`` の子特徴 ``[1, cpp, D_in]`` をロードしてキャッシュする"""
        cached = self._child_cache.get(action)
        if cached is not None:
            return cached
        child = compute_child_indices(
            np.asarray([action], dtype=np.int64), self.global_idx, children=self.cpp
        )
        feats = self.child_loader(self.next_mag, child).to(self.device)
        self._child_cache[action] = feats
        return feats

    def evaluate(self, action: int) -> float:
        """候補 ``action`` の子を次倍率へ射影・プーリングした状態の価値推定を返す

        価値ネットは探索シグナルなので eval モードで前向きし dropout 等の確率性を
        除く前向きの間だけ eval へ切り替え終了後に元のモード（train/eval）へ戻す
        """
        cached = self._reward_cache.get(action)
        if cached is not None:
            return cached
        was_training = self.value_net.training
        self.value_net.eval()
        try:
            with torch.no_grad():
                feats = self._load_child(action)
                x_next = self.model.projections[self.layer_idx + 1](feats)
                reward = float(self.value_net(x_next).item())
        finally:
            self.value_net.train(was_training)
        self._reward_cache[action] = reward
        return reward


class MCTSZoomDriver(ZoomDriver):
    """探索ベースのズーム駆動

    各倍率で方策ネットを事前，価値ネットを葉評価とし :class:`Planner` でどの親を展開
    するかを選ぶ選んだ親の子を materialise し，方策確率を子へ掛けて勾配を流し，
    各倍率のプーリング表現を融合して分類する方策蒸留・価値回帰（・任意でエントロピー）
    損失を :class:`ForwardContext` の ``extra_losses`` に積む方策・価値ネットは
    ``model`` の子モジュールとして登録し ``model.parameters()`` で最適化される
    （識別器ヘッド・射影は基線と共有する）

    価値ネットは価値回帰項からのみ勾配を受けるため ``value_weight > 0`` のときに限り
    学習される 0 なら価値ネットの重みは初期値のまま固定され葉評価は学習されない

    ``context.selections[i]['select_weight']`` は選んだ親の方策確率（``PolicyNetwork``
    出力）であり differentiable 駆動の補助アテンション重みとは別物 アテンション
    ベースの正則化器など下流の消費者はこの違いに留意する

    Args:
        model: 段階 forward を持つ :class:`FoveaMIL`
        num_layers: 倍率数
        feat_dim: 射影特徴次元 D（``model.fusion.out_dim`` と一致）
        hidden_dim: 方策・価値ネットの中間次元
        dropout: Dropout 率
        k_sample: 1 倍率で展開する親数 k
        planner_name: 探索プランナ名（``"gumbel"`` / ``"puct"``）
        simulations: 模擬予算
        max_considered: 検討最大候補数 m
        policy_weight: 方策蒸留損失の重み λ_π
        value_weight: 価値回帰損失の重み λ_v
        entropy_weight: 方策エントロピー損失の重み
        value_target: 価値回帰目標の作り方（``"realised"`` で従来挙動，``"leaf_ce"`` で状態依存 leaf 報酬）
        seed: 乱数シード
    """

    def __init__(
        self,
        model,
        num_layers: int,
        feat_dim: int,
        hidden_dim: int,
        dropout: Optional[float],
        k_sample: int,
        planner_name: str,
        simulations: int,
        max_considered: int,
        policy_weight: float,
        value_weight: float,
        entropy_weight: float,
        value_target: str = VALUE_TARGET_REALISED,
        seed: int = 0,
    ) -> None:
        super().__init__(model, num_layers)
        self.k_sample = k_sample
        self.planner_name = planner_name
        self.simulations = simulations
        self.max_considered = max_considered
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.value_target = value_target
        self.seed = seed

        device = next(model.parameters()).device
        self.policy = PolicyNetwork(feat_dim, hidden_dim, dropout).to(device)
        self.value = ValueNetwork(feat_dim, hidden_dim, dropout).to(device)
        # 方策・価値を model の子モジュールとして登録し model.parameters() に含める
        # （基線の識別器ヘッド・射影と同一 optimizer で最適化される）
        model.add_module(POLICY_ATTR, self.policy)
        model.add_module(VALUE_ATTR, self.value)

    @classmethod
    def from_config(cls, config, model, num_layers: int) -> "MCTSZoomDriver":
        """``TrainConfig`` とモデルから探索駆動を構築する

        Args:
            config: ``TrainConfig``
            model: 段階 forward を持つ :class:`FoveaMIL`
            num_layers: 倍率数

        Returns:
            構築した :class:`MCTSZoomDriver`
        """
        feat_dim = model.fusion.out_dim
        hidden_dim = config.mcts_hidden_dim or config.hidden_feat_dim
        if config.value_loss_weight == 0.0:
            logger.warning(
                "value_loss_weight=0: 価値ネットは学習されず葉評価は初期重みのまま固定される"
            )
        return cls(
            model=model,
            num_layers=num_layers,
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            dropout=config.drop_out,
            k_sample=config.k_sample,
            planner_name=config.mcts_planner,
            simulations=config.mcts_simulations,
            max_considered=config.mcts_max_considered,
            policy_weight=config.policy_loss_weight,
            value_weight=config.value_loss_weight,
            entropy_weight=config.policy_entropy_weight,
            value_target=config.mcts_value_target,
            seed=config.seed,
        )

    def _project_and_pool(self, x: Tensor, layer_idx: int) -> Tuple[Tensor, Tensor]:
        """射影と集約器でプーリング表現を作り ``(M, x_fc)`` を返す

        プーリングは基線と同じ集約器に委ね，集約器軸の差し替えへ追随する射影特徴
        ``x_fc`` は方策・価値ネットへ渡すため別途返す
        """
        x_fc = self.model.projections[layer_idx](x)
        M, _ = self.model.aggregators[layer_idx](x_fc)
        return M, x_fc

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
        priors: List[Tensor] = []
        improved: List[Tensor] = []
        value_preds: List[Tensor] = []

        x = base_feats.to(device)
        global_idx: Optional[np.ndarray] = None
        for layer_idx in range(self.num_layers):
            M, x_fc = self._project_and_pool(x, layer_idx)
            M_list.append(M)
            if layer_idx >= self.num_layers - 1:
                selections.append(None)
                continue

            cur_mag = magnifications[layer_idx]
            next_mag = magnifications[layer_idx + 1]
            cpp = children_per_parent(cur_mag, next_mag)

            prior = self.policy(x_fc).squeeze(0)
            value_preds.append(self.value(x_fc).squeeze(0))
            priors.append(prior)

            problem = _ZoomSearchProblem(
                prior_np=prior.detach().cpu().numpy(),
                x_fc=x_fc,
                layer_idx=layer_idx,
                next_mag=next_mag,
                cpp=cpp,
                global_idx=global_idx,
                model=self.model,
                value_net=self.value,
                child_loader=child_loader,
                device=device,
            )
            planner = build_planner(
                self.planner_name,
                simulations=self.simulations,
                max_considered=self.max_considered,
                seed=self.seed + layer_idx * _LAYER_SEED_STRIDE,
            )
            result: PlannerResult = planner.run(problem, num_select=self.k_sample)
            improved.append(
                torch.as_tensor(
                    result.improved_policy, dtype=prior.dtype, device=device
                )
            )

            chosen = np.sort(result.chosen_actions)
            select_weight = prior[torch.as_tensor(chosen, device=device)].unsqueeze(0)
            selections.append(
                {
                    "select_indices": torch.as_tensor(
                        chosen, device=device
                    ).unsqueeze(0),
                    "select_weight": select_weight,
                }
            )

            child = compute_child_indices(chosen, global_idx, children=cpp)
            x_next = child_loader(next_mag, child).to(device)
            w_child = select_weight.repeat_interleave(cpp, dim=1)
            x_next = x_next * w_child.unsqueeze(-1)
            x = x_next
            global_idx = child

        logits, Y_hat, Y_prob = self.model.forward_final(M_list)
        context = ForwardContext(m_list=M_list, selections=selections)
        self._attach_losses(
            context, logits, label, priors, improved, value_preds, M_list, selections
        )
        return logits, Y_hat, Y_prob, context

    def _leaf_rewards(
        self,
        m_list: List[Tensor],
        label: Tensor,
        num_states: int,
    ) -> Tensor:
        """選択 ``j`` の結果状態の leaf 報酬 ``[num_states]`` を返す

        run() の構造上選択 ``j`` の効果は次倍率のプーリング表現 ``M_{j+1}`` に現れる
        よって選択 ``j`` の報酬は ``m_list[:j+2]``（``M_0..M_{j+1}``）を共有融合・識別
        ヘッドへ通した負分類損失とする最深選択は ``m_list`` 全長の融合を含み報酬に現れる
        報酬は消費側で detach され勾配源にはしない（共有ヘッド・融合は主 CE で学習される）
        """
        rewards: List[Tensor] = []
        for j in range(num_states):
            prefix = m_list[: j + 2]
            prefix_logits = self.model.classify(self.model.fuse_repr(prefix))[0]
            rewards.append(-F.cross_entropy(prefix_logits, label))
        return torch.stack(rewards, dim=0)

    def _attach_losses(
        self,
        context: ForwardContext,
        logits: Tensor,
        label: Optional[Tensor],
        priors: List[Tensor],
        improved: List[Tensor],
        value_preds: List[Tensor],
        m_list: List[Tensor],
        selections: List[Optional[dict]],
    ) -> None:
        """方策蒸留・価値回帰（・エントロピー）損失を ``extra_losses`` に積む

        ラベルが無い（推論）または探索層が無い場合は何も積まない
        ``value_target="realised"`` では実現報酬（探索表現の負分類損失）を全状態へ
        broadcast して価値回帰の目標にする（detach）``value_target="leaf_ce"`` では選択 j の
        結果状態を含む融合の負分類損失を状態依存 leaf 報酬（detach）として価値回帰の目標にし，
        選択確率には advantage（leaf 報酬 − 価値推定）由来の actor-critic 項を加える
        leaf 報酬は detach されるため価値回帰・方策勾配のどちらも共有ヘッド・融合へは
        勾配を流さない（共有ヘッド・融合・各倍率表現は主 CE 経由でのみ学習される）
        """
        if label is None or not priors:
            return

        # 倍率ごとに候補数 N が異なる（子数 k·cpp が層で変わる）ため層別に集計する
        policy_terms: List[Tensor] = []
        entropy_terms: List[Tensor] = []
        for prior, target in zip(priors, improved):
            log_prior = torch.log(prior.clamp_min(_PROB_FLOOR))
            policy_terms.append(-(target.detach() * log_prior).sum())
            if self.entropy_weight != 0.0:
                entropy_terms.append(-(prior * log_prior).sum())

        value_stack = torch.stack(value_preds, dim=0)
        if self.value_target == VALUE_TARGET_LEAF_CE:
            # leaf[j] は選択 j の結果状態 M_{j+1} を含む融合の負分類損失で選択 j に依存する
            leaf = self._leaf_rewards(m_list, label, value_stack.shape[0])
            value_loss = F.mse_loss(value_stack, leaf.detach())
            # 価値推定をベースラインに選択 j のリターン残差で選択を弁別する
            # （良い選択ほど結果状態の leaf 報酬が高く advantage が正へ向く）
            advantage = leaf.detach() - value_stack.detach()
            for state_idx, selection in enumerate(
                s for s in selections if s is not None
            ):
                select_weight = selection["select_weight"]
                log_select = torch.log(select_weight.clamp_min(_PROB_FLOOR)).sum()
                policy_terms[state_idx] = (
                    policy_terms[state_idx] - advantage[state_idx] * log_select
                )
        else:
            realised = (-F.cross_entropy(logits, label)).detach()
            value_loss = F.mse_loss(value_stack, realised.expand_as(value_stack))

        policy_loss = torch.stack(policy_terms).mean()
        context.extra_losses[LOSS_POLICY] = self.policy_weight * policy_loss
        context.extra_losses[LOSS_VALUE] = self.value_weight * value_loss

        if entropy_terms:
            entropy = torch.stack(entropy_terms).mean()
            context.extra_losses[LOSS_ENTROPY] = -self.entropy_weight * entropy
