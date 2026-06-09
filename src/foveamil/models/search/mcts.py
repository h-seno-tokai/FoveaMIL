"""ズーム木の探索プランナ（Gumbel-AlphaZero / PUCT）

候補親（=単一パッチ展開アクション）上で，事前方策を prior，葉評価を価値として
探索木を辿り，訪問・スコア由来の改良方策（非負・和 1）と選択アクションを返す

候補数が多く模擬予算が小さい設定に合わせ，既定では Gumbel-AlphaZero
（Danihelka et al., ICLR 2022）流のアクション選択を用いる Gumbel で上位 m 候補を
標本化し，sequential halving で予算を配分し，完了 Q 値から決定的な改良方策を作る
これにより PUCT 定数の手調整を避ける PUCT 版も選べる

報酬は :class:`SearchProblem` の callback が与える（純粋にでき，torch I/O を要しない）
全乱数はシードで固定し，同一シードでは決定的に動く
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

# Gumbel-AlphaZero の Q 値変換 σ(q) の visit 係数 c_visit
DEFAULT_C_VISIT = 50.0
# Gumbel-AlphaZero の Q 値変換 σ(q) のスケール係数 c_scale
DEFAULT_C_SCALE = 1.0
# PUCT 探索定数
DEFAULT_C_PUCT = 1.25
# 改良方策で 0 確率を避ける下限
_PROB_FLOOR = 1e-12
# 確率正規化の許容下限（総和がこれ未満なら一様分布へ）
_SUM_FLOOR = 1e-12


def _softmax(logits: np.ndarray) -> np.ndarray:
    """数値安定な softmax を返す（最終軸）"""
    if logits.size == 0:
        return logits
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    total = exp.sum()
    if total < _SUM_FLOOR:
        return np.full_like(exp, 1.0 / exp.size)
    return exp / total


def _sequential_halving_schedule(
    num_considered: int, simulations: int
) -> List[int]:
    """sequential halving の各ラウンドで残す候補数列を返す

    候補数を ``num_considered`` から半減させていき 1 まで縮める（各 >=1）模擬予算
    ``simulations`` が小さく 1 ラウンド分も賄えない場合でも最低 1 ラウンドは回す

    Args:
        num_considered: 最初に検討する候補数 m
        simulations: 模擬予算

    Returns:
        各ラウンドの候補数（降順，末尾は 1）
    """
    schedule: List[int] = []
    width = max(1, num_considered)
    while width > 1:
        schedule.append(width)
        width = max(1, width // 2)
    schedule.append(1)
    return schedule


def _sigma(q_values: np.ndarray, max_visit: int, c_scale: float) -> np.ndarray:
    """Gumbel-AlphaZero の単調変換 σ(q) を返す

    ``σ(q) = (c_visit + max_visit) * c_scale * q`` で visit 数に応じ Q の寄与を強める

    Args:
        q_values: 完了 Q 値
        max_visit: 任意候補の最大訪問数
        c_scale: スケール係数

    Returns:
        変換後の値（logits へ加える）
    """
    return (DEFAULT_C_VISIT + max_visit) * c_scale * q_values


class SearchProblem(abc.ABC):
    """探索プランナが解く問題のインタフェース

    候補アクション数と，事前方策・各候補の報酬評価を与える報酬評価は純粋でよく，
    プランナはこのインタフェースのみに依存する
    """

    @abc.abstractmethod
    def num_actions(self) -> int:
        """候補アクション数 N を返す"""

    @abc.abstractmethod
    def prior(self) -> np.ndarray:
        """候補上の事前方策 ``(N,)``（非負・和 1）を返す"""

    @abc.abstractmethod
    def evaluate(self, action: int) -> float:
        """候補 ``action`` を 1 段展開した状態の報酬推定（スカラ）を返す

        Args:
            action: 候補 index

        Returns:
            報酬推定（大きいほど良い）
        """

    def prefetch_batch(self, actions: Sequence[int]) -> None:
        """候補集合 ``actions`` の評価を 1 バッチで先に計算しキャッシュへ充填する

        基底は何もしない（後方互換）``evaluate`` をメモ化する実装が，この呼び出しで
        per-leaf 評価・同期をバッチ 1 回へ畳み，以後の per-action :meth:`evaluate` を
        キャッシュヒットへ退化させるために override する報酬列・評価順序は不変

        Args:
            actions: 先に評価する候補 index 列
        """

    def prefetch_round(self, action_counts: Dict[int, int]) -> None:
        """1 ラウンドで候補 ``a`` を ``action_counts[a]`` 回評価する分を先に一括計算する

        基底は何もしない（後方互換）確率的葉評価をする実装が，この呼び出しで候補 ``a``
        を ``action_counts[a]`` 回 repeat した状態を 1 テンソルへ並べ value-net を 1 回だけ
        前向き（train モード＝dropout 有効）し，得た独立標本を候補ごとの FIFO へ積む
        以後の per-action :meth:`evaluate` はこの FIFO から 1 標本ずつ取り出す各候補は
        依然 ``action_counts[a]`` 個の独立標本を持ち（標本数・独立性・分散構造は不変）
        forward と GPU→CPU 同期だけがラウンド単位 1 回へ畳まれる

        Args:
            action_counts: 候補 index → このラウンドでの評価回数
        """


@dataclass
class PlannerResult:
    """探索プランナの返り値

    Attributes:
        improved_policy: 候補上の改良方策 ``(N,)``（非負・和 1）
        chosen_actions: 選んだ候補 index の列（改良方策上位 k，昇順）
        q_values: 候補ごとの完了 Q 値 ``(N,)``（未評価候補は prior 由来で補完）
        visit_counts: 候補ごとの訪問数 ``(N,)``
        prior: 入力事前方策 ``(N,)``
    """

    improved_policy: np.ndarray
    chosen_actions: np.ndarray
    q_values: np.ndarray
    visit_counts: np.ndarray
    prior: np.ndarray


class Planner(abc.ABC):
    """探索プランナの基底

    Args:
        simulations: 模擬予算
        max_considered: 検討する最大候補数 m（Gumbel top-m）
        seed: 乱数シード（決定性のため）
    """

    def __init__(
        self,
        simulations: int,
        max_considered: int,
        seed: int = 0,
    ) -> None:
        self.simulations = int(simulations)
        self.max_considered = int(max_considered)
        self.seed = int(seed)

    @abc.abstractmethod
    def run(self, problem: SearchProblem, num_select: int) -> PlannerResult:
        """``problem`` を探索し改良方策と選択アクションを返す

        Args:
            problem: 探索対象
            num_select: 選ぶアクション数 k

        Returns:
            :class:`PlannerResult`
        """

    @staticmethod
    def _top_actions(policy: np.ndarray, num_select: int) -> np.ndarray:
        """方策上位 ``num_select`` の候補 index を昇順で返す（同点は index 昇順）"""
        n = policy.shape[0]
        k = max(1, min(num_select, n))
        order = np.argsort(-policy, kind="stable")[:k]
        return np.sort(order)


@dataclass
class _GumbelRunState:
    """``GumbelAlphaZeroPlanner.run`` の途中状態をラウンド単位で持ち越す

    スライド跨ぎバッチ化（lockstep）で複数プランナを同一ラウンドまで進めるため
    ``run`` を ``_prepare`` / ``_round_step`` / ``_finalize`` に分けて駆動する容器
    """

    n: int
    prior: np.ndarray
    logits: np.ndarray
    gumbel: np.ndarray
    candidates: np.ndarray
    schedule: List[int]
    sims_per_round: int
    visit_counts: np.ndarray
    q_sum: np.ndarray
    evaluated: Dict[int, float]
    active: List[int]


class GumbelAlphaZeroPlanner(Planner):
    """Gumbel-AlphaZero 流のアクション選択を行うプランナ

    Gumbel で上位 m 候補を標本化し，sequential halving で予算を配分して各候補を
    評価し，完了 Q 値から決定的な改良方策 ``softmax(logits + σ(completedQ))`` を作る
    PUCT 定数の手調整を要しない（Danihelka et al., ICLR 2022）

    Args:
        simulations: 模擬予算
        max_considered: 検討する最大候補数 m
        seed: 乱数シード
        c_scale: σ(q) のスケール係数
    """

    def __init__(
        self,
        simulations: int,
        max_considered: int,
        seed: int = 0,
        c_scale: float = DEFAULT_C_SCALE,
    ) -> None:
        super().__init__(simulations, max_considered, seed)
        self.c_scale = float(c_scale)

    def run(self, problem: SearchProblem, num_select: int) -> PlannerResult:
        # ラウンド単位の駆動に分割（挙動は分割前と完全一致・スライド跨ぎ lockstep の土台）
        state = self._prepare(problem)
        for round_width in state.schedule:
            self._round_step(problem, state, round_width)
        return self._finalize(state, num_select)

    def _prepare(self, problem: SearchProblem) -> _GumbelRunState:
        """候補集合・スケジュールを確定し全候補の葉評価を先に充填する

        候補集合は run 冒頭で確定し sequential halving は部分集合のみへ縮む（新規候補は
        出ない）ため全候補の葉評価を先に充填し per-leaf 同期を 1 回へ畳む以後の evaluate は
        キャッシュヒットになり報酬列・訪問・スコア順序は per-leaf 評価と一致する
        （メモ化実装のみ有効・確率時は no-op）
        """
        state = self._prepare_candidates(problem)
        problem.prefetch_batch([int(a) for a in state.candidates])
        return state

    def _prepare_candidates(self, problem: SearchProblem) -> _GumbelRunState:
        """葉充填を行わず候補集合・gumbel・スケジュールのみ確定して状態を返す

        Gumbel top-m で初期候補集合を決め sequential halving の各ラウンド幅を確定する
        葉評価の充填（prefetch）は呼ばず numpy 算術のみ行う（:meth:`_prepare` は本メソッド
        の後に prefetch_batch を 1 回呼ぶ＝挙動は分離前と完全一致）
        """
        n = problem.num_actions()
        prior = np.asarray(problem.prior(), dtype=np.float64).reshape(-1)
        if prior.shape[0] != n:
            raise ValueError(
                f"prior length {prior.shape[0]} does not match num_actions {n}"
            )
        logits = np.log(np.clip(prior, _PROB_FLOOR, None))

        rng = np.random.default_rng(self.seed)
        gumbel = rng.gumbel(size=n)

        considered = max(1, min(self.max_considered, n))
        # Gumbel top-m: logits + gumbel の上位 m を最初の候補集合にする
        candidates = np.argsort(-(logits + gumbel), kind="stable")[:considered]

        schedule = _sequential_halving_schedule(considered, self.simulations)
        sims_per_round = max(1, self.simulations // max(1, len(schedule)))
        return _GumbelRunState(
            n=n,
            prior=prior,
            logits=logits,
            gumbel=gumbel,
            candidates=candidates,
            schedule=schedule,
            sims_per_round=sims_per_round,
            visit_counts=np.zeros(n, dtype=np.int64),
            q_sum=np.zeros(n, dtype=np.float64),
            evaluated={},
            active=list(candidates),
        )

    def _round_step(
        self, problem: SearchProblem, state: _GumbelRunState, round_width: int
    ) -> None:
        """1 ラウンドぶん（上位 ``round_width`` 候補を per_action 回評価し再ソート）進める

        ラウンドの全評価（候補 a を per_action 回）を 1 バッチへ畳む確率的実装のみ有効
        （基底 no-op）標本数・独立性・期待値/分散構造は不変で forward/同期のみ畳まれる
        """
        self._round_prefetch(problem, state, round_width)
        self._round_consume(problem, state, round_width)

    def _round_prefetch(
        self, problem: SearchProblem, state: _GumbelRunState, round_width: int
    ) -> None:
        """ラウンドの全評価入力（active を per_action 回）を先取り充填する

        ``active`` と ``per_action`` は ``_round_consume`` と同一に確定する（その間 state.active
        は不変）複数 problem の prefetch を先に済ませ葉前向きを後段で連結 1 同期へ畳む分離点
        """
        active = state.active[:round_width]
        per_action = max(1, state.sims_per_round // max(1, len(active)))
        problem.prefetch_round({int(a): per_action for a in active})

    def _round_consume(
        self, problem: SearchProblem, state: _GumbelRunState, round_width: int
    ) -> None:
        """先取り済みの評価を消費し Q/訪問を更新し次ラウンドの active を再ソートする"""
        active = state.active[:round_width]
        per_action = max(1, state.sims_per_round // max(1, len(active)))
        for action in active:
            for _ in range(per_action):
                reward = float(problem.evaluate(int(action)))
                state.q_sum[action] += reward
                state.visit_counts[action] += 1
                state.evaluated[int(action)] = reward
        # 各候補の平均 Q で次ラウンドへ残す上位を選ぶ（Gumbel 補正込み）
        state.active = sorted(
            active,
            key=lambda a: -(
                state.logits[a]
                + state.gumbel[a]
                + _sigma(
                    np.asarray([_safe_mean(state.q_sum[a], state.visit_counts[a])]),
                    int(state.visit_counts.max()),
                    self.c_scale,
                )[0]
            ),
        )

    def _finalize(
        self, state: _GumbelRunState, num_select: int
    ) -> PlannerResult:
        """完了 Q から改良方策と選択アクションを作る"""
        q_values = _completed_q(
            state.prior, state.q_sum, state.visit_counts, state.evaluated
        )
        improved_logits = state.logits + _sigma(
            q_values, int(state.visit_counts.max(initial=0)), self.c_scale
        )
        improved_policy = _softmax(improved_logits)
        chosen = self._top_actions(improved_policy, num_select)
        return PlannerResult(
            improved_policy=improved_policy,
            chosen_actions=chosen,
            q_values=q_values,
            visit_counts=state.visit_counts,
            prior=state.prior,
        )


class PuctPlanner(Planner):
    """PUCT のアクション選択を行うプランナ

    各模擬で ``argmax_a Q(a) + c_puct * prior(a) * sqrt(ΣN) / (1 + N(a))`` を選び評価
    する訪問数で改良方策を作る基本的な AlphaZero 流の参照実装で，``c_puct`` を要する

    Args:
        simulations: 模擬予算
        max_considered: 検討する最大候補数 m
        seed: 乱数シード
        c_puct: PUCT 探索定数
    """

    def __init__(
        self,
        simulations: int,
        max_considered: int,
        seed: int = 0,
        c_puct: float = DEFAULT_C_PUCT,
    ) -> None:
        super().__init__(simulations, max_considered, seed)
        self.c_puct = float(c_puct)

    def run(self, problem: SearchProblem, num_select: int) -> PlannerResult:
        n = problem.num_actions()
        prior = np.asarray(problem.prior(), dtype=np.float64).reshape(-1)
        if prior.shape[0] != n:
            raise ValueError(
                f"prior length {prior.shape[0]} does not match num_actions {n}"
            )

        considered = max(1, min(self.max_considered, n))
        candidates = np.argsort(-prior, kind="stable")[:considered]

        visit_counts = np.zeros(n, dtype=np.int64)
        q_sum = np.zeros(n, dtype=np.float64)
        evaluated: Dict[int, float] = {}

        for _ in range(max(1, self.simulations)):
            total_visits = int(visit_counts.sum())
            best_action = None
            best_score = -math.inf
            for action in candidates:
                q = _safe_mean(q_sum[action], visit_counts[action])
                u = (
                    self.c_puct
                    * prior[action]
                    * math.sqrt(total_visits + 1.0)
                    / (1.0 + visit_counts[action])
                )
                score = q + u
                if score > best_score:
                    best_score = score
                    best_action = int(action)
            reward = float(problem.evaluate(best_action))
            q_sum[best_action] += reward
            visit_counts[best_action] += 1
            evaluated[best_action] = reward

        # 改良方策は訪問数比（探索した候補のみ非ゼロ）
        improved_policy = visit_counts.astype(np.float64)
        if improved_policy.sum() < _SUM_FLOOR:
            improved_policy = prior.copy()
        improved_policy = improved_policy / improved_policy.sum()
        q_values = _completed_q(prior, q_sum, visit_counts, evaluated)
        chosen = self._top_actions(improved_policy, num_select)
        return PlannerResult(
            improved_policy=improved_policy,
            chosen_actions=chosen,
            q_values=q_values,
            visit_counts=visit_counts,
            prior=prior,
        )


def _safe_mean(total: float, count: int) -> float:
    """訪問数で割った平均を返す（未訪問は 0）"""
    return total / count if count > 0 else 0.0


def _completed_q(
    prior: np.ndarray,
    q_sum: np.ndarray,
    visit_counts: np.ndarray,
    evaluated: Dict[int, float],
) -> np.ndarray:
    """完了 Q 値を返す（未評価候補を prior 重み付き平均で補完する）

    Gumbel-AlphaZero の completed-Q（評価済み候補は実測平均，未評価候補は訪問済み
    候補の prior 重み付き平均 v_mix で補完）に倣う訪問が皆無なら 0 で埋める

    Args:
        prior: 事前方策 ``(N,)``
        q_sum: 候補ごとの報酬総和 ``(N,)``
        visit_counts: 候補ごとの訪問数 ``(N,)``
        evaluated: 評価済み候補 index → 最終報酬

    Returns:
        完了 Q 値 ``(N,)``
    """
    n = prior.shape[0]
    q_values = np.zeros(n, dtype=np.float64)
    visited = visit_counts > 0
    if not np.any(visited):
        return q_values

    means = np.where(visited, q_sum / np.maximum(visit_counts, 1), 0.0)
    prior_visited = prior[visited]
    weight = prior_visited.sum()
    if weight < _SUM_FLOOR:
        v_mix = float(means[visited].mean())
    else:
        v_mix = float((prior_visited * means[visited]).sum() / weight)

    q_values[visited] = means[visited]
    q_values[~visited] = v_mix
    return q_values


# プランナ名 → クラス
PLANNERS = {
    "gumbel": GumbelAlphaZeroPlanner,
    "puct": PuctPlanner,
}


def build_planner(
    name: str,
    simulations: int,
    max_considered: int,
    seed: int = 0,
    **kwargs,
) -> Planner:
    """名前から探索プランナを構築する

    Args:
        name: ``PLANNERS`` に登録されたプランナ名（``"gumbel"`` / ``"puct"``）
        simulations: 模擬予算
        max_considered: 検討する最大候補数 m
        seed: 乱数シード
        **kwargs: 各プランナ固有の追加引数

    Returns:
        構築した :class:`Planner`

    Raises:
        KeyError: ``name`` が未登録の場合
    """
    if name not in PLANNERS:
        raise KeyError(
            f"unknown planner '{name}'; available: {sorted(PLANNERS)}"
        )
    return PLANNERS[name](
        simulations=simulations,
        max_considered=max_considered,
        seed=seed,
        **kwargs,
    )
