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

葉評価は ``mcts_rollout_depth`` と ``mcts_eval_stochastic`` でノブ化する``rollout_depth=1``
は選んだ親の子を 1 段だけ次倍率へ射影して価値ネットで評価する（従来挙動）``>1`` では
その子を更に次倍率へ再帰展開し（各段でプランナが 1 子を選ぶ）最深状態の価値を葉評価に
する子ロードは木全体で ``child_cache`` を共有し重複 h5 読みを抑える``eval_stochastic``
は葉評価を MC dropout で確率化し報酬 memoize を撤廃して simulation 間に分散を出す
（既定 ``False`` は eval モード葉評価＋memoize で従来挙動）
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

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
# advantage 正規化の分散下限（ゼロ割回避）
_ADV_EPS = 1e-8
# プランナへ渡すシードの倍率ごとオフセット係数（層で別シードにする）
_LAYER_SEED_STRIDE = 1009


class _RolloutContext:
    """rollout 木全体で共有する状態（子キャッシュ・葉評価カウンタ・倍率列）

    ``child_cache`` は ``(layer_idx, global_parent)`` を鍵に子特徴を保持し，深さ×分岐で
    増える h5 読みの重複を抑える木の全 :class:`_ZoomSearchProblem` がこの 1 個を共有する
    ``leaf_evals`` は価値ネットを葉として評価した回数（テストで深さ依存を確認する）

    Args:
        model: 段階 forward を持つ :class:`FoveaMIL`
        value_net: 価値ネット
        child_loader: 子特徴ローダ
        magnifications: 倍率列（再帰で次倍率比を引く）
        num_layers: 倍率数（これ以上展開できない最深層の判定に使う）
        planner_name: 子選択に用いるプランナ名
        rollout_simulations: 各 rollout 段のプランナ模擬予算
        rollout_considered: 各 rollout 段のプランナ検討候補数 m
        stochastic: 葉評価を MC dropout で確率化するか
        device: 計算デバイス
    """

    def __init__(
        self,
        model,
        value_net: ValueNetwork,
        child_loader: ChildLoader,
        magnifications: List[float],
        num_layers: int,
        planner_name: str,
        rollout_simulations: int,
        rollout_considered: int,
        stochastic: bool,
        device: torch.device,
    ) -> None:
        self.model = model
        self.value_net = value_net
        self.child_loader = child_loader
        self.magnifications = magnifications
        self.num_layers = num_layers
        self.planner_name = planner_name
        self.rollout_simulations = rollout_simulations
        self.rollout_considered = rollout_considered
        self.stochastic = stochastic
        self.device = device
        self.child_cache: Dict[Tuple[int, int], Tensor] = {}
        self.leaf_evals = 0

    def load_child(
        self, layer_idx: int, global_parent: int, next_mag: float, cpp: int
    ) -> Tensor:
        """親の global index ``global_parent`` の子特徴 ``[1, cpp, D_in]`` を返す

        木全体で鍵 ``(layer_idx, global_parent)`` のキャッシュを共有し重複読みを避ける
        """
        key = (layer_idx, int(global_parent))
        cached = self.child_cache.get(key)
        if cached is not None:
            return cached
        child = compute_child_indices(
            np.asarray([global_parent], dtype=np.int64), None, children=cpp
        )
        feats = self.child_loader(next_mag, child).to(self.device)
        self.child_cache[key] = feats
        return feats

    def value_leaf(self, x_next: Tensor) -> float:
        """射影済み状態 ``x_next`` を価値ネットで葉評価する（カウンタを進める）

        確率的でない（既定）ときは eval モードへ切り替え dropout 等の確率性を除く前向き
        の間だけモードを変え終了後に元へ戻す確率的なときは train モードのまま MC dropout
        で評価し simulation 間に分散を出す
        """
        self.leaf_evals += 1
        was_training = self.value_net.training
        self.value_net.train(self.stochastic)
        try:
            with torch.no_grad():
                return float(self.value_net(x_next).item())
        finally:
            self.value_net.train(was_training)

    def value_leaf_batch(self, x_states: List[Tensor]) -> List[float]:
        """射影済み状態列 ``[m × (1, cpp, D)]`` を価値ネットで葉評価し ``[m]`` を返す

        各候補を ``[1, cpp, D]`` 単体で前向きし（葉評価値を改修前とビット同一に保つ），
        出力テンソルを ``.item()`` せず保持して末尾で 1 回だけ ``.cpu().tolist()`` する
        per-leaf の GPU→CPU 同期（律速）を 1 回へ畳む単体前向きは非同期にキューされ
        最後の同期だけがブロックするため逐次同期待ちが消える``leaf_evals`` は件数で進める

        値の生成は per-leaf 評価と同一前向き・同一縮約順序のため決定論時の報酬列は
        改修前と完全一致する（バッチ GEMM の縮約順序差による ULP 揺れを持ち込まない）
        """
        self.leaf_evals += len(x_states)
        was_training = self.value_net.training
        self.value_net.train(self.stochastic)
        try:
            with torch.no_grad():
                outputs = [self.value_net(x).reshape(1) for x in x_states]
            return torch.cat(outputs, dim=0).cpu().tolist()
        finally:
            self.value_net.train(was_training)

    def policy_prior(self, x_next: Tensor) -> Tensor:
        """rollout 中段の方策事前 ``π(a|state)`` を勾配なしで返す（探索シグナル）

        確率的でない（既定）ときは eval モードで dropout を除き決定的にする確率的な
        ときは train モードのまま MC dropout で標本化する前向きの間だけモードを変える
        """
        policy_net = self.model.search_policy
        was_training = policy_net.training
        policy_net.train(self.stochastic)
        try:
            with torch.no_grad():
                return policy_net(x_next)
        finally:
            policy_net.train(was_training)


class _ZoomSearchProblem(SearchProblem):
    """1 倍率のズーム決定を解く探索問題

    候補は現倍率の親パッチ事前は方策ネット出力各候補の評価は，その親の子を
    ``child_loader`` で materialise し次倍率へ射影した状態の rollout 評価``rollout_depth``
    が 1 ならその状態を直接価値ネットで評価する（look-ahead）``>1`` なら子を更に次倍率へ
    再帰展開し（各段でプランナが 1 子を選ぶ）最深状態を価値ネットで評価する子ロードと
    （非確率時の）評価をキャッシュし h5 読みの重複を避ける

    Args:
        prior_np: 事前方策 ``(N,)``（numpy detached）
        x_fc: 現倍率の射影特徴 ``[1, N, D]``
        layer_idx: 現倍率の添字
        next_mag: 次倍率
        cpp: 1 親あたりの子数（倍率比^2）
        global_idx: 現倍率パッチの global index（``None`` なら local=global）
        rollout_depth: 展開する rollout 深さ（1 で 1 段評価し更に展開しない）
        seed: 子選択プランナのシード（決定性のため）
        ctx: 木全体で共有する :class:`_RolloutContext`
    """

    def __init__(
        self,
        prior_np: np.ndarray,
        x_fc: Tensor,
        layer_idx: int,
        next_mag: float,
        cpp: int,
        global_idx: Optional[np.ndarray],
        rollout_depth: int,
        seed: int,
        ctx: _RolloutContext,
    ) -> None:
        self._prior = prior_np
        self.x_fc = x_fc
        self.layer_idx = layer_idx
        self.next_mag = next_mag
        self.cpp = cpp
        self.global_idx = global_idx
        self.rollout_depth = int(rollout_depth)
        self.seed = int(seed)
        self.ctx = ctx
        self._reward_cache: Dict[int, float] = {}

    def num_actions(self) -> int:
        return self._prior.shape[0]

    def prior(self) -> np.ndarray:
        return self._prior

    def _global_parent(self, action: int) -> int:
        """候補 local index ``action`` を現倍率の global index へ写す"""
        if self.global_idx is None:
            return int(action)
        return int(self.global_idx[action])

    def _load_child(self, action: int) -> Tensor:
        """候補親 ``action`` の子特徴 ``[1, cpp, D_in]`` を共有キャッシュ経由でロードする"""
        return self.ctx.load_child(
            self.layer_idx, self._global_parent(action), self.next_mag, self.cpp
        )

    def _can_expand(self) -> bool:
        """次倍率の子を更に展開できるか（最深層手前まで）を返す"""
        return self.layer_idx + 1 < self.ctx.num_layers - 1

    def _rollout(self, x_next: Tensor, global_child: np.ndarray) -> float:
        """子状態 ``x_next`` を 1 段深く展開し最深状態の価値を返す（再帰）

        子状態の方策事前でプランナを 1 回回し，最良の孫を選んでさらに展開する残り深さが
        尽きるか最深層に達したら価値ネットで葉評価する子ロードは ``ctx`` の共有キャッシュ
        を使い，乱数は ``seed`` で固定して決定的に動く
        """
        next_layer = self.layer_idx + 1
        sub_mag = self.ctx.magnifications[next_layer]
        sub_next_mag = self.ctx.magnifications[next_layer + 1]
        sub_cpp = children_per_parent(sub_mag, sub_next_mag)
        sub_prior = self.ctx.policy_prior(x_next).squeeze(0)
        sub_problem = _ZoomSearchProblem(
            prior_np=sub_prior.detach().cpu().numpy(),
            x_fc=x_next,
            layer_idx=next_layer,
            next_mag=sub_next_mag,
            cpp=sub_cpp,
            global_idx=global_child,
            rollout_depth=self.rollout_depth - 1,
            seed=self.seed,
            ctx=self.ctx,
        )
        planner = build_planner(
            self.ctx.planner_name,
            simulations=self.ctx.rollout_simulations,
            max_considered=self.ctx.rollout_considered,
            seed=self.seed,
        )
        result: PlannerResult = planner.run(sub_problem, num_select=1)
        return sub_problem.evaluate(int(result.chosen_actions[0]))

    def evaluate(self, action: int) -> float:
        """候補 ``action`` の rollout 評価（最深状態の価値推定）を返す

        ``rollout_depth<=1`` または最深層手前なら子を次倍率へ射影した状態を価値ネットで
        直接評価する``>1`` なら子を更に再帰展開し最深状態を葉評価にする確率的でないとき
        は同一候補の評価を memoize し，確率的なときは memoize せず simulation 毎に再評価する
        """
        if not self.ctx.stochastic:
            cached = self._reward_cache.get(action)
            if cached is not None:
                return cached
        feats = self._load_child(action)
        with torch.no_grad():
            x_next = self.ctx.model.projections[self.layer_idx + 1](feats)
        if self.rollout_depth > 1 and self._can_expand():
            global_child = compute_child_indices(
                np.asarray([self._global_parent(action)], dtype=np.int64),
                None,
                children=self.cpp,
            )
            reward = self._rollout(x_next, global_child)
        else:
            reward = self.ctx.value_leaf(x_next)
        if not self.ctx.stochastic:
            self._reward_cache[action] = reward
        return reward

    def prefetch_batch(self, actions: Sequence[int]) -> None:
        """候補集合 ``actions`` の葉評価を先に計算し ``_reward_cache`` を充填する

        確率的でない（既定）ときに限り有効各候補の子を ``[m, cpp, D_in]`` へ stack し
        射影を 1 回（候補軸独立で batched GEMM は per-row とビット同一）で通したうえで，
        価値ネットは行ごとに前向きしつつ GPU→CPU 同期を末尾 1 回へ畳む以後の per-action
        :meth:`evaluate` は全てキャッシュヒットになり，探索算術へ渡る報酬列は per-leaf
        評価と完全一致する（葉評価値はビット同一・同期回数のみ m→1 へ減る）

        確率的なとき（``stochastic``）は memoize を撤廃し simulation 間に分散を出す
        設計のため何もしない（per-action 経路の MC dropout 標本化をそのまま使う）
        ``rollout_depth>1`` で更に展開できる候補は入れ子 planner.run が逐次依存のため
        バッチ化対象外として何もしない（葉到達時のバッチ化は入れ子側で効く）
        """
        if self.ctx.stochastic:
            return
        if self.rollout_depth > 1 and self._can_expand():
            return
        todo = [int(a) for a in actions if int(a) not in self._reward_cache]
        if not todo:
            return
        # 射影は候補軸独立で batched GEMM が per-row とビット同一のため 1 回で射影し，
        # value-net は行ごと前向き＋末尾 1 同期で per-leaf 評価とビット同一に保つ
        feats = torch.cat([self._load_child(a) for a in todo], dim=0)
        with torch.no_grad():
            x_next = self.ctx.model.projections[self.layer_idx + 1](feats)
        states = [x_next[i : i + 1] for i in range(x_next.shape[0])]
        values = self.ctx.value_leaf_batch(states)
        for action, value in zip(todo, values):
            self._reward_cache[action] = value


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
        rollout_depth: 葉評価の rollout 深さ（1 で従来＝1 段評価，>1 で再帰展開）
        rollout_simulations: rollout 各段の入れ子プランナ模擬予算（``None`` で ``simulations`` に一致＝従来挙動）
        eval_stochastic: 葉評価を MC dropout で確率化するか（False で従来＝eval＋memoize）
        actor_critic_weight: ``leaf_ce`` の actor-critic 項スケール（正規化 advantage × 選択 log 確率の上乗せ重み，0 で無効）
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
        rollout_depth: int = 1,
        rollout_simulations: Optional[int] = None,
        eval_stochastic: bool = False,
        actor_critic_weight: float = 1.0,
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
        self.rollout_depth = int(rollout_depth)
        # None なら simulations と同値＝従来挙動（rollout 段の予算を最上層と分離する任意ノブ）
        self.rollout_simulations = (
            int(rollout_simulations)
            if rollout_simulations is not None
            else int(simulations)
        )
        self.eval_stochastic = bool(eval_stochastic)
        self.actor_critic_weight = float(actor_critic_weight)
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
            rollout_depth=config.mcts_rollout_depth,
            rollout_simulations=config.mcts_rollout_simulations,
            eval_stochastic=config.mcts_eval_stochastic,
            actor_critic_weight=config.mcts_actor_critic_weight,
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

        # rollout 木全体で子キャッシュ・葉カウンタ・確率設定を共有する
        rollout_ctx = _RolloutContext(
            model=self.model,
            value_net=self.value,
            child_loader=child_loader,
            magnifications=magnifications,
            num_layers=self.num_layers,
            planner_name=self.planner_name,
            rollout_simulations=self.rollout_simulations,
            rollout_considered=self.max_considered,
            stochastic=self.eval_stochastic,
            device=device,
        )

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

            layer_seed = self.seed + layer_idx * _LAYER_SEED_STRIDE
            problem = _ZoomSearchProblem(
                prior_np=prior.detach().cpu().numpy(),
                x_fc=x_fc,
                layer_idx=layer_idx,
                next_mag=next_mag,
                cpp=cpp,
                global_idx=global_idx,
                rollout_depth=self.rollout_depth,
                seed=layer_seed,
                ctx=rollout_ctx,
            )
            planner = build_planner(
                self.planner_name,
                simulations=self.simulations,
                max_considered=self.max_considered,
                seed=layer_seed,
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

    @staticmethod
    def _normalize_advantage(advantage: Tensor) -> Tensor:
        """advantage ``[num_states]`` を選択状態軸でゼロ平均・単位分散へ正規化する

        平均を引き標準偏差（eps 付き）で割る標準的な policy-gradient 安定化で advantage の
        スケール暴走を抑える符号（advantage>0 で chosen 確率↑）は線形変換で保たれる状態が
        1 個で分散が定義できないときは平均引きのみ行い分散正規化は省く（ゼロ割回避）
        """
        if advantage.numel() <= 1:
            return advantage - advantage.mean()
        std = advantage.std(unbiased=False)
        return (advantage - advantage.mean()) / (std + _ADV_EPS)

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
        advantage は選択状態軸でゼロ平均・単位分散へ正規化し（eps 付き・符号は保つ）スケール
        暴走を抑える``actor_critic_weight`` で actor-critic 項を方策蒸留と分離して独立に
        スケールする（0 で actor-critic 無効＝状態依存 value は価値回帰のみ残る）
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
            advantage = self._normalize_advantage(
                leaf.detach() - value_stack.detach()
            )
            for state_idx, selection in enumerate(
                s for s in selections if s is not None
            ):
                select_weight = selection["select_weight"]
                log_select = torch.log(select_weight.clamp_min(_PROB_FLOOR)).sum()
                policy_terms[state_idx] = (
                    policy_terms[state_idx]
                    - self.actor_critic_weight * advantage[state_idx] * log_select
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
