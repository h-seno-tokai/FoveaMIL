"""微分可能 k-DPP による選択コントローラ

品質と多様性を同時に測る決定点過程（DPP）のカーネル ``L_ij = q_i q_j k(z_i, z_j)``
を候補上に張る品質 ``q_i`` は正規化済み補助アテンション ``scores`` から，類似度 ``k``
は射影特徴 ``features`` から作るサイズ k の部分集合を貪欲 MAP（Chen et al. 2018）で
選び，各段の条件付き限界利得（残差ノルム）に対する argmax を温度付き soft argmax /
Gumbel-softmax で緩和して学習時に連続化する推論時は hard な貪欲 MAP で one-hot 行を
返すこれにより品質（アテンション）と多様性（特徴射影）の双方へ勾配が流れ，密な
高品質クラスタへ偏る top-k と違い各クラスタから 1 点ずつ拾う傾向を持つ
Kulesza & Taskar 2012Chen et al. NeurIPS 2018Jang et al. ICLR 2017 に基づく
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from foveamil.models.selection import register_selection_controller
from foveamil.models.selection.base import SelectionController

# 類似度名（cosine / rbf）
SIMILARITY_COSINE = "cosine"
SIMILARITY_RBF = "rbf"
SIMILARITIES = (SIMILARITY_COSINE, SIMILARITY_RBF)
# 既定の類似度名
DEFAULT_SIMILARITY = SIMILARITY_COSINE
# 既定の緩和温度（soft argmax / Gumbel-softmax の温度）
DEFAULT_TEMPERATURE = 1.0
# 品質 q_i = exp(beta * scores_i) の既定スケール
DEFAULT_QUALITY_BETA = 1.0
# RBF カーネルの既定帯域（gamma）
DEFAULT_RBF_GAMMA = 1.0
# Gumbel-softmax を使うか（既定は決定的な soft argmax）
DEFAULT_USE_GUMBEL = False
# 数値安定化の微小量
_EPS = 1e-10
# 特徴正規化の微小量
_NORM_EPS = 1e-8
# 既選択をマスクするための大きな負の値
_NEG_INF = -1e9


def cosine_similarity_matrix(features: Tensor) -> Tensor:
    """特徴 ``[B, N, D]`` の L2 正規化後の内積で類似度 ``[B, N, N]`` を作る"""
    normed = features / (features.norm(dim=-1, keepdim=True) + _NORM_EPS)
    return normed @ normed.transpose(-1, -2)


def rbf_similarity_matrix(features: Tensor, gamma: float) -> Tensor:
    """特徴 ``[B, N, D]`` の対距離から RBF 類似度 ``exp(-gamma d^2)`` を作る"""
    sq = (features * features).sum(dim=-1, keepdim=True)
    dist2 = sq + sq.transpose(-1, -2) - 2.0 * (features @ features.transpose(-1, -2))
    dist2 = dist2.clamp_min(0.0)
    return torch.exp(-gamma * dist2)


def build_dpp_kernel(
    scores: Tensor, similarity: Tensor, beta: float
) -> Tensor:
    """品質 ``q_i = exp(beta·scores_i)`` と類似度から DPP カーネル ``L`` を作る

    ``L_ij = q_i q_j k(z_i, z_j)`` を返す品質は正規化済みアテンションから作り，
    勾配がアテンションへ流れる類似度は特徴から作り，勾配が射影へ流れる

    Args:
        scores: 正規化済み補助アテンション ``[B, N]``
        similarity: 類似度行列 ``[B, N, N]``
        beta: 品質スケール

    Returns:
        DPP カーネル ``[B, N, N]``
    """
    quality = torch.exp(beta * scores)
    return quality.unsqueeze(-1) * similarity * quality.unsqueeze(-2)


def _soft_onehot(
    gains: Tensor,
    chosen_mask: Tensor,
    temperature: float,
    hard: bool,
    use_gumbel: bool,
) -> Tensor:
    """限界利得 ``gains [B, N]`` から 1 段ぶんの選択ベクトル ``[B, N]`` を作る

    既選択（``chosen_mask`` が真）を ``-inf`` でマスクし，温度付き softmax で緩和した
    one-hot 風ベクトルを返す``hard`` 時は argmax の hard one-hot を straight-through
    で返す``use_gumbel`` 時は Gumbel 雑音を加える（学習時の確率的探索）
    """
    logits = gains / temperature
    logits = logits.masked_fill(chosen_mask, _NEG_INF)
    if use_gumbel and not hard:
        uniform = torch.rand_like(logits).clamp_min(_EPS)
        gumbel = -torch.log((-torch.log(uniform)).clamp_min(_EPS))
        logits = logits + gumbel
    soft = F.softmax(logits, dim=-1)
    if not hard:
        return soft
    index = soft.argmax(dim=-1, keepdim=True)
    hard_vec = torch.zeros_like(soft).scatter_(-1, index, 1.0)
    return hard_vec + (soft - soft.detach())


@register_selection_controller("dpp")
class DPPSelectionController(SelectionController):
    """品質と多様性を測る微分可能 k-DPP の選択コントローラ

    品質 ``q_i`` を正規化済み補助アテンション，類似度を射影特徴から作り，DPP カーネル
    の貪欲 MAP（Chen et al. 2018）でサイズ k の部分集合を選ぶ各段の条件付き限界利得
    に対する argmax を温度付き soft argmax / Gumbel-softmax で緩和し，学習時は soft な
    選択行列，推論時は hard な one-hot 行を返す``k`` が ``N`` を超える場合は ``min(N, k)``
    に丸める

    Args:
        k: 選択する要素数
        similarity: 類似度名（``"cosine"`` / ``"rbf"``）
        temperature: 緩和温度（soft argmax / Gumbel-softmax）
        quality_beta: 品質 ``q_i = exp(beta·scores_i)`` のスケール
        rbf_gamma: RBF カーネルの帯域（``similarity="rbf"`` 時のみ）
        use_gumbel: 学習時に Gumbel 雑音で確率的に選ぶか
        seed: Gumbel 標本の決定性を保つための乱数シード（``None`` なら固定しない）

    Raises:
        ValueError: ``similarity`` が未知の名前の場合
    """

    def __init__(
        self,
        k: int,
        similarity: str = DEFAULT_SIMILARITY,
        temperature: float = DEFAULT_TEMPERATURE,
        quality_beta: float = DEFAULT_QUALITY_BETA,
        rbf_gamma: float = DEFAULT_RBF_GAMMA,
        use_gumbel: bool = DEFAULT_USE_GUMBEL,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(k)
        if similarity not in SIMILARITIES:
            raise ValueError(
                f"unknown similarity '{similarity}'; available: {list(SIMILARITIES)}"
            )
        self.similarity = similarity
        self.temperature = temperature
        self.quality_beta = quality_beta
        self.rbf_gamma = rbf_gamma
        self.use_gumbel = use_gumbel
        self.seed = seed
        # 直近 forward の選択部分カーネルの log-det（多様性正則化が排出する）
        self._last_log_det: Optional[Tensor] = None

    def _similarity_matrix(self, features: Tensor) -> Tensor:
        """設定の類似度名に応じて類似度行列 ``[B, N, N]`` を作る"""
        if self.similarity == SIMILARITY_RBF:
            return rbf_similarity_matrix(features, self.rbf_gamma)
        return cosine_similarity_matrix(features)

    def kernel(self, scores: Tensor, features: Tensor) -> Tensor:
        """品質と類似度から DPP カーネル ``L [B, N, N]`` を作る"""
        similarity = self._similarity_matrix(features)
        return build_dpp_kernel(scores, similarity, self.quality_beta)

    def selected_log_det(self, kernel: Tensor, selection: Tensor) -> Tensor:
        """選択された要素の部分カーネル ``L_S`` の log-det をバッチ平均で返す

        各行の argmax で選択 index 集合 ``S`` を決め（非微分），``L`` から ``L_S = L[S, S]``
        を取り出してその log-det を返すカーネル要素（品質・類似度）に対して微分可能で，
        多様な（互いに非類似な）選択ほど log-det が大きい対角へ微小量を足し数値安定化する

        Args:
            kernel: DPP カーネル ``[B, N, N]``
            selection: 選択行列 ``[B, k, N]``

        Returns:
            スカラ（バッチ平均 log-det）
        """
        with torch.no_grad():
            indices = selection.argmax(dim=-1)
        k = indices.shape[-1]
        num_elements = kernel.shape[-1]
        row_index = indices.unsqueeze(-1).expand(-1, -1, num_elements)
        rows = kernel.gather(1, row_index)
        col_index = indices.unsqueeze(-2).expand(-1, k, -1)
        sub = rows.gather(2, col_index)
        eye = torch.eye(k, device=sub.device, dtype=sub.dtype)
        sub = sub + _EPS * eye
        return torch.logdet(sub).mean()

    def _greedy_map(self, kernel: Tensor, k: int, hard: bool) -> Tensor:
        """貪欲 MAP（Chen et al. 2018）でサイズ k の選択行列 ``[B, k, N]`` を作る

        各段で条件付き限界利得（残差分散 ``d_i^2``）を計算し，その argmax を soft /
        hard に緩和した選択ベクトルを 1 行として積む選んだ要素の Cholesky 因子
        ``c`` を漸進更新し ``d_i^2 -= c_i^2`` で残差を縮める既選択はマスクする
        """
        batch_size, num_elements, _ = kernel.shape
        device = kernel.device
        diag = torch.diagonal(kernel, dim1=-2, dim2=-1)
        d2 = diag.clone()
        cis: list = []
        chosen_mask = torch.zeros(
            batch_size, num_elements, dtype=torch.bool, device=device
        )

        rows = []
        for _ in range(k):
            gains = torch.log(d2.clamp_min(_EPS))
            row = _soft_onehot(
                gains, chosen_mask, self.temperature, hard, self.use_gumbel
            )
            rows.append(row)
            # 選択ベクトルで該当列を集約し Cholesky 因子を漸進更新する
            l_col = torch.einsum("bn,bmn->bm", row, kernel)
            if cis:
                prev_factors = torch.stack(cis, dim=1)
                prev = torch.einsum("bsn,bn->bs", prev_factors, row)
                proj = torch.einsum("bsm,bs->bm", prev_factors, prev)
            else:
                proj = torch.zeros_like(l_col)
            denom = torch.sqrt(
                (d2 * row).sum(dim=-1, keepdim=True).clamp_min(_EPS)
            )
            ci = (l_col - proj) / denom
            cis.append(ci)
            d2 = (d2 - ci * ci).clamp_min(0.0)
            chosen_mask = chosen_mask | (row > 0.5)

        return torch.stack(rows, dim=1)

    def select(self, scores: Tensor, features: Tensor) -> Tensor:
        """品質と多様性から選択行列 ``[B, k, N]`` を返す（学習時 soft / 推論時 hard）

        Args:
            scores: 正規化済み補助アテンション ``[B, N]``
            features: 射影特徴 ``[B, N, D]``

        Returns:
            選択行列 ``[B, k, N]``
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
        num_elements = scores.shape[-1]
        k = min(self.k, num_elements)
        kernel = self.kernel(scores, features)
        selection = self._greedy_map(kernel, k, hard=not self.training)
        # 多様性正則化が後で排出できるよう選択部分カーネルの log-det を保持する
        self._last_log_det = self.selected_log_det(kernel, selection)
        return selection

    def pop_log_det(self) -> Optional[Tensor]:
        """直近 forward の選択部分カーネル log-det を取り出して消費する"""
        value = self._last_log_det
        self._last_log_det = None
        return value
