"""自己アテンションによる相関考慮の集約器（TransMIL 風 + Nyström 近似）

各要素を独立にスコアリングする ABMIL と異なり，まずパッチ列に多頭自己アテンションを
かけてパッチ間コンテキスト（近傍構成）を表現へ取り込み，その文脈付き表現をゲート付き
アテンションでプーリングする低倍率では数千パッチに達し full self-attention は N² で
重いため，landmark を用いた Nyström 近似で計算量を O(N·m) に抑える``N`` が landmark
数以下のときは近似誤差を避けて厳密な自己アテンションへ縮退する

出力契約は :class:`Aggregator` と同一（``M=[B,1,D]`` / ``A=[B,1,N]``）で，後段の
インスタンス補助損失・可視化はプーリング重み ``A`` を従来どおり参照できる
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from foveamil.models.aggregator.base import Aggregator
from foveamil.models.aggregator.registry import register_aggregator
from foveamil.models.attention import GatedAttention

# 既定の注意ヘッド数
DEFAULT_NUM_HEADS = 4
# 既定の landmark 数（Nyström 近似の解像度）
DEFAULT_NUM_LANDMARKS = 64
# プーリング用アテンションのクラス数（1 スコア/要素）
_ATTENTION_N_CLS = 1


@register_aggregator("self_attn")
class SelfAttentionAggregator(Aggregator):
    """Nyström 近似付き多頭自己アテンション集約器

    Args:
        dim: 入力特徴次元（出力 ``M`` の次元も同一）
        hidden_dim: ゲート付きプーリングの中間次元
        dropout: Dropout 率``None`` なら Dropout を挟まない
        num_heads: 注意ヘッド数（``dim`` を割り切ること）
        num_landmarks: Nyström の landmark 数

    Raises:
        ValueError: ``dim`` が ``num_heads`` で割り切れない場合
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: Optional[float] = None,
        num_heads: int = DEFAULT_NUM_HEADS,
        num_landmarks: int = DEFAULT_NUM_LANDMARKS,
    ) -> None:
        super().__init__(dim, hidden_dim, dropout)
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_landmarks = num_landmarks
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.attn_dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()
        self.pool = GatedAttention(dim, hidden_dim, dropout, n_cls=_ATTENTION_N_CLS)

    def _split_heads(self, t: Tensor) -> Tensor:
        """``[B, N, D]`` を ``[B, H, N, head_dim]`` に分割する"""
        b, n, _ = t.shape
        return t.view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _exact_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """厳密な softmax 自己アテンション（``N`` 小・landmark 以下のとき）"""
        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        return attn @ v

    def _nystrom_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """landmark を用いた Nyström 近似自己アテンション

        Q・K を ``num_landmarks`` 個のセグメント平均で landmark に圧縮し，
        ``softmax(Q·K̃)·pinv(softmax(Q̃·K̃))·softmax(Q̃·K)·V`` で N×N 行列を陽に
        作らず近似する中核は ``m×m``（小）なので擬似逆は ``torch.linalg.pinv`` で
        厳密に取り，計算量は N に対し線形に保つ``N`` が landmark の倍数でないときは
        ゼロ詰めの均等セグメントで平均を取る
        """
        m = self.num_landmarks
        q_land = self._segment_means(q, m)
        k_land = self._segment_means(k, m)

        kernel1 = F.softmax((q @ k_land.transpose(-2, -1)) * self.scale, dim=-1)
        kernel2 = F.softmax((q_land @ k_land.transpose(-2, -1)) * self.scale, dim=-1)
        kernel3 = F.softmax((q_land @ k.transpose(-2, -1)) * self.scale, dim=-1)
        return kernel1 @ torch.linalg.pinv(kernel2) @ (kernel3 @ v)

    def _segment_means(self, t: Tensor, m: int) -> Tensor:
        """``[B, H, N, d]`` を ``m`` 個のセグメント平均 ``[B, H, m, d]`` に圧縮する

        ``N`` を ``m`` 個へ均等分割し各セグメントの平均を landmark とする
        ``N`` が ``m`` で割り切れないときはゼロ詰めし，各セグメントの実要素数で
        割って平均を保つ
        """
        b, h, n, d = t.shape
        seg = -(-n // m)
        pad = seg * m - n
        if pad > 0:
            t = F.pad(t, (0, 0, 0, pad))
            mask = F.pad(t.new_ones(b, h, n, 1), (0, 0, 0, pad))
        else:
            mask = t.new_ones(b, h, n, 1)
        t = t.view(b, h, m, seg, d).sum(dim=-2)
        counts = mask.view(b, h, m, seg, 1).sum(dim=-2)
        return t / counts.clamp_min(1.0)

    def forward(self, x_fc: Tensor) -> Tuple[Tensor, Tensor]:
        """自己アテンションで文脈を取り込みゲート付きプーリングで ``(M, A)`` を返す"""
        residual = x_fc
        x = self.norm1(x_fc)
        qkv = self.qkv(x).chunk(3, dim=-1)
        q = self._split_heads(qkv[0])
        k = self._split_heads(qkv[1])
        v = self._split_heads(qkv[2])

        n = x_fc.size(1)
        if n <= self.num_landmarks:
            ctx = self._exact_attention(q, k, v)
        else:
            ctx = self._nystrom_attention(q, k, v)

        b = x_fc.size(0)
        ctx = ctx.permute(0, 2, 1, 3).reshape(b, n, self.dim)
        ctx = residual + self.proj(ctx)
        ctx = ctx + self.ffn(self.norm2(ctx))

        A, _ = self.pool(ctx)
        A = F.softmax(A.permute(0, 2, 1), dim=-1)
        M = A @ ctx
        return M, A
