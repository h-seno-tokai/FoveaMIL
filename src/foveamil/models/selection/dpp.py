"""微分可能 k-DPP による選択コントローラ

品質と多様性を同時に測る決定点過程（DPP）のカーネル ``L_ij = q_i q_j k(z_i, z_j)``
を候補上に張る品質 ``q_i`` は正規化済み補助アテンション ``scores`` から，類似度 ``k``
は射影特徴 ``features`` から作るサイズ k の部分集合を貪欲 MAP で選び，各段で選んだ
index を温度付き soft one-hot で緩和して学習時に連続化する推論時は hard な one-hot 行を
返す各段の選択 index は no_grad で hard に決め，その index で残差（Cholesky 因子）を
更新する一方，返す行は同 index を中心とする soft one-hot とし品質・特徴双方へ勾配を流す
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
# log-det 安定化のため対角の大きさへ比例させるジッタ係数
_LOGDET_JITTER = 1e-4
# soft one-hot が hard one-hot とみなせる閾値（既選択判定）
_CHOSEN_THRESHOLD = 0.5


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


def _masked_logits(
    gains: Tensor,
    chosen_mask: Tensor,
    temperature: float,
    use_gumbel: bool,
    generator: Optional[torch.Generator],
) -> Tensor:
    """限界利得 ``gains [B, N]`` を温度で割り既選択をマスクしたロジット ``[B, N]`` を作る

    既選択（``chosen_mask`` が真）を ``-inf`` でマスクする``use_gumbel`` 時は Gumbel
    雑音を加える（``generator`` で標本の決定性を保つ）
    """
    logits = gains / temperature
    logits = logits.masked_fill(chosen_mask, _NEG_INF)
    if use_gumbel:
        uniform = torch.rand(
            logits.shape, generator=generator, device=logits.device, dtype=logits.dtype
        ).clamp_min(_EPS)
        gumbel = -torch.log((-torch.log(uniform)).clamp_min(_EPS))
        logits = logits + gumbel
    return logits


def _soft_onehot_at(logits: Tensor, index: Tensor) -> Tensor:
    """選択 index を中心とする 1 段ぶんの選択ベクトル ``[B, N]`` を作る

    前向きの値は ``index`` の hard one-hot に一致させ（straight-through），勾配は温度付き
    softmax を通して品質・特徴へ流す``logits`` は温度反映済みとし``[B, 1]`` の ``index``
    は呼び出し側が no_grad で hard に決め，残差更新と返す行の中心を一致させる
    """
    hard_vec = torch.zeros_like(logits).scatter_(-1, index, 1.0)
    soft = F.softmax(logits, dim=-1)
    return hard_vec + (soft - soft.detach())


@register_selection_controller("dpp")
class DPPSelectionController(SelectionController):
    """品質と多様性を測る微分可能 k-DPP の選択コントローラ

    品質 ``q_i`` を正規化済み補助アテンション，類似度を射影特徴から作り，DPP カーネル
    の貪欲 MAP でサイズ k の部分集合を選ぶ各段の条件付き限界利得に対する argmax を温度付き
    soft argmax / Gumbel-softmax で緩和し，学習時は soft な選択行列，推論時は hard な
    one-hot 行を返す``k`` が ``N`` を超える場合は ``min(N, k)`` に丸める

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
        # Gumbel 標本専用の乱数生成器（global RNG を汚さない select の入力デバイスで作る）
        self._gen: Optional[torch.Generator] = None
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
        多様な（互いに非類似な）選択ほど log-det が大きい対角へ大きさ比例のジッタを足し，
        符号付き ``slogdet`` で数値安定に評価する

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
        diag_scale = torch.diagonal(sub, dim1=-2, dim2=-1).abs().mean(
            dim=-1, keepdim=True
        ).clamp_min(_EPS)
        jitter = (_LOGDET_JITTER * diag_scale).unsqueeze(-1)
        sub = sub + jitter * eye
        sign, logabsdet = torch.linalg.slogdet(sub)
        # 正定値なら sign>0 でそのまま，非正の符号は数値由来として log-det を 0 へ寄せる
        logabsdet = torch.where(sign > 0, logabsdet, torch.zeros_like(logabsdet))
        return logabsdet.mean()

    def _greedy_map(self, kernel: Tensor, k: int, hard: bool) -> Tensor:
        """貪欲 MAP でサイズ k の選択行列 ``[B, k, N]`` を作る

        各段で条件付き限界利得（残差分散 ``d_i^2``）を計算し，その argmax の index を
        no_grad で hard に決める残差（Cholesky 因子 ``c``）はその hard index に対して
        厳密に更新し ``d_i^2 -= c_i^2`` で縮める既選択はマスクし k 個の hard index は
        相異なる返す行は同 index を中心とする温度付き soft one-hot（学習時）または hard
        one-hot（推論時）で，品質・特徴双方へ勾配を流す
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
            logits = _masked_logits(
                gains, chosen_mask, self.temperature, self.use_gumbel, self._gen
            )
            # 選択 index は no_grad で hard に決め，残差更新と返す行の中心へ共用する
            with torch.no_grad():
                index = logits.argmax(dim=-1, keepdim=True)
            sel = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            rows.append(sel if hard else _soft_onehot_at(logits, index))
            # hard index の列で Cholesky 因子を厳密更新する（soft 平均を使わない）
            l_col = torch.einsum("bn,bmn->bm", sel, kernel)
            if cis:
                prev_factors = torch.stack(cis, dim=1)
                prev = torch.einsum("bsn,bn->bs", prev_factors, sel)
                proj = torch.einsum("bsm,bs->bm", prev_factors, prev)
            else:
                proj = torch.zeros_like(l_col)
            denom = torch.sqrt(
                (d2 * sel).sum(dim=-1, keepdim=True).clamp_min(_EPS)
            )
            ci = (l_col - proj) / denom
            cis.append(ci)
            d2 = (d2 - ci * ci).clamp_min(0.0)
            chosen_mask = chosen_mask.scatter(-1, index, True)

        return torch.stack(rows, dim=1)

    def select(self, scores: Tensor, features: Tensor) -> Tensor:
        """品質と多様性から選択行列 ``[B, k, N]`` を返す（学習時 soft / 推論時 hard）

        Args:
            scores: 正規化済み補助アテンション ``[B, N]``
            features: 射影特徴 ``[B, N, D]``

        Returns:
            選択行列 ``[B, k, N]``
        """
        num_elements = scores.shape[-1]
        k = min(self.k, num_elements)
        kernel = self.kernel(scores, features)
        self._sync_generator_device(kernel.device)
        selection = self._greedy_map(kernel, k, hard=not self.training)
        # 多様性正則化が後で排出できるよう選択部分カーネルの log-det を保持する
        self._last_log_det = self.selected_log_det(kernel, selection)
        return selection

    def _sync_generator_device(self, device: torch.device) -> None:
        """Gumbel 用生成器を入力デバイスへ合わせ seed を巻き戻す（seed 指定時のみ）

        ``torch.Generator`` はデバイス固有のため，入力が別デバイスなら作り直す各 select で
        同 seed へ巻き戻すことで，連続呼び出しでも同一の Gumbel 標本を再現する
        """
        if self.seed is None:
            return
        if self._gen is None or self._gen.device != device:
            self._gen = torch.Generator(device=device)
        self._gen.manual_seed(self.seed)

    def pop_log_det(self) -> Optional[Tensor]:
        """直近 forward の選択部分カーネル log-det を取り出して消費する"""
        value = self._last_log_det
        self._last_log_det = None
        return value
