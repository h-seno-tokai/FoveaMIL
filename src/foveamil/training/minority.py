"""少数クラスの学習信号を強化する機構（いずれも既定 off で従来挙動と一致）

3 機構を提供する いずれも既定値（``mixup_alpha=0`` / ``sampler_temp=1.0`` /
``ordinal_aux_weight=0``）で現行と数値一致する縮退安全な部品とする
- :class:`BagMixup` ＝ bag 表現レベル（融合後 ``[B, dim]``）の manifold-mixup
  2 サンプルの表現を線形補間しラベルも補間する バッチサイズ 1 前提のため直前
  サンプルの表現/ラベルを buffer に持ち現サンプルと混ぜる ``alpha=0`` で無効
- :func:`temper_sampler_weights` ＝ バランスサンプラ重みの温度付け 重みを ``temp``
  乗する ``temp=1.0`` で現行 ``<1`` で緩和（一様寄り）``>1`` で強調
- :class:`OrdinalAuxLoss` ＝ クラス順序を活かす ordinal 補助損失 softmax 確率の
  期待ランクと正解ランクの二乗距離 順序が遠い誤りほど大きく罰する ``weight=0`` で無効
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# mixup を無効化する alpha（この値で従来挙動と一致）
MIXUP_DISABLED_ALPHA = 0.0
# サンプラ温度の現行値（この値で重みが不変）
SAMPLER_TEMP_IDENTITY = 1.0
# ordinal 補助損失を無効化する重み（この値で寄与 0）
ORDINAL_DISABLED_WEIGHT = 0.0


def temper_sampler_weights(weights: Tensor, temp: float) -> Tensor:
    """バランスサンプラ重みへ温度を掛ける

    ``temp=1.0`` で恒等（現行と一致）``temp<1`` で重みを一様へ緩和し ``temp>1`` で
    少数クラスを強調する 0 重み（不在クラス）は ``temp`` に依らず 0 のままとする

    Args:
        weights: サンプルごとの非負重み（クラス頻度逆数など）
        temp: 温度

    Returns:
        ``weights ** temp``（``temp=1.0`` なら入力をそのまま返す）
    """
    if temp == SAMPLER_TEMP_IDENTITY:
        return weights
    return weights.pow(temp)


class BagMixup:
    """bag 表現レベルの manifold-mixup（バッチサイズ 1 対応）

    融合後の bag 表現 ``[B, dim]`` を直前サンプルの表現と線形補間し ラベルも対応する
    比で補間する バッチサイズ 1 では同一バッチ内に相手がいないため直前サンプルの
    表現/ラベルを buffer に保持し（勾配は切る）現サンプルと混ぜる buffer が空の初回や
    ``alpha<=0`` では混合せず素の表現を返し 損失も素の cross-entropy 規約に揃える

    Args:
        alpha: Beta(alpha, alpha) の形状 ``0`` 以下で無効
        n_cls: クラス数（ラベル補間の one-hot 次元）
        generator: 補間係数 λ のサンプリングに使う乱数生成器（決定性確保用）
    """

    def __init__(
        self,
        alpha: float,
        n_cls: int,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.alpha = alpha
        self.n_cls = n_cls
        self.generator = generator
        self._buf_repr: Optional[Tensor] = None
        self._buf_target: Optional[Tensor] = None

    @property
    def enabled(self) -> bool:
        """mixup が有効か（``alpha>0``）"""
        return self.alpha > MIXUP_DISABLED_ALPHA

    def _sample_lam(self, device: torch.device) -> Tensor:
        """Beta(alpha, alpha) から補間係数 λ を引く"""
        beta = torch.distributions.Beta(self.alpha, self.alpha)
        if self.generator is not None:
            # Beta は generator を取らないため一様 2 本から決定的に合成する
            u = torch.rand(2, generator=self.generator)
            g1 = u[0].clamp_min(torch.finfo(torch.float32).tiny).log().neg()
            g2 = u[1].clamp_min(torch.finfo(torch.float32).tiny).log().neg()
            lam = g1 / (g1 + g2)
            return lam.to(device)
        return beta.sample().to(device)

    def _one_hot(self, target: Tensor) -> Tensor:
        """ラベル整数 ``[B]`` を one-hot ``[B, n_cls]``（float）にする"""
        return F.one_hot(target, num_classes=self.n_cls).float()

    def mix(
        self, repr_bag: Tensor, target: Tensor
    ) -> Tuple[Tensor, Tensor, Callable[[Tensor, Tensor], Tensor]]:
        """bag 表現とラベルを直前サンプルと補間する

        ``enabled`` かつ buffer がある場合のみ ``repr' = λ·repr + (1-λ)·buf`` と
        ``soft' = λ·onehot(target) + (1-λ)·buf_target`` を作る 補間後に現サンプルの
        （勾配付き）表現/ラベルを buffer へ退避する（次回の相手）戻り値の損失計算は
        soft ラベルの cross-entropy を criterion で計算する callable

        Args:
            repr_bag: 融合後 bag 表現 ``[B, dim]``（勾配付き）
            target: 正解クラス ``[B]``

        Returns:
            ``(repr_mixed[B, dim], soft_target[B, n_cls], loss_fn)``
            ``loss_fn(criterion, logits)`` が混合損失スカラを返す 無効/初回は
            ``repr_bag`` そのものと one-hot と素 CE 相当の ``loss_fn`` を返す
        """
        soft_target = self._one_hot(target)
        if not self.enabled or self._buf_repr is None:
            self._store(repr_bag, soft_target)
            return repr_bag, soft_target, self._hard_loss(target)

        lam = self._sample_lam(repr_bag.device)
        repr_mixed = lam * repr_bag + (1.0 - lam) * self._buf_repr
        target_mixed = lam * soft_target + (1.0 - lam) * self._buf_target
        target_a, target_b = target, self._buf_target_idx
        self._store(repr_bag, soft_target)
        return repr_mixed, target_mixed, self._mixed_loss(lam, target_a, target_b)

    def _store(self, repr_bag: Tensor, soft_target: Tensor) -> None:
        """現サンプルの表現/ラベルを buffer へ退避する（勾配は切る）"""
        self._buf_repr = repr_bag.detach()
        self._buf_target = soft_target.detach()
        self._buf_target_idx = soft_target.argmax(dim=-1).detach()

    @staticmethod
    def _hard_loss(target: Tensor) -> Callable[[Tensor, Tensor], Tensor]:
        """素のラベルで criterion を呼ぶ loss_fn を返す（混合無効時）"""

        def loss_fn(criterion, logits: Tensor) -> Tensor:
            return criterion(logits, target)

        return loss_fn

    @staticmethod
    def _mixed_loss(
        lam: Tensor, target_a: Tensor, target_b: Tensor
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """混合損失 ``λ·CE(·,a) + (1-λ)·CE(·,b)`` を返す loss_fn を返す

        soft ラベルの cross-entropy はクラス頻度補正付き criterion とも両立する
        （2 つの hard ラベル損失の凸結合に分解する）
        """

        def loss_fn(criterion, logits: Tensor) -> Tensor:
            return lam * criterion(logits, target_a) + (1.0 - lam) * criterion(
                logits, target_b
            )

        return loss_fn


class OrdinalAuxLoss(nn.Module):
    """クラス順序を活かす ordinal 補助損失

    クラス index の並び（``0 < 1 < ... < n_cls-1``）を順序とみなし softmax 確率の
    期待ランク ``E[r] = Σ_k k·p_k`` と正解ランクの二乗距離を罰する 順序が遠い誤りほど
    大きく罰し 近い誤りは小さく罰する（順序の単調性）``weight`` で寄与をスケールし
    ``weight=0`` なら学習ループ側が呼ばないため寄与 0 とする

    ランクの絶対尺度を ``n_cls-1`` で割り正規化する（クラス数に依らず比較可能）

    Args:
        n_cls: クラス数
    """

    def __init__(self, n_cls: int) -> None:
        super().__init__()
        ranks = torch.arange(n_cls, dtype=torch.float32)
        self.register_buffer("ranks", ranks)
        # 単一クラスでは順序が無いため正規化を恒等にする
        self.scale = float(max(n_cls - 1, 1))

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """期待ランクと正解ランクの正規化二乗距離（平均）を返す

        Args:
            logits: 分類 logit ``[B, n_cls]``
            target: 正解クラス ``[B]``

        Returns:
            平均二乗距離スカラ
        """
        probs = F.softmax(logits, dim=-1)
        expected_rank = probs @ self.ranks
        true_rank = self.ranks[target]
        diff = (expected_rank - true_rank) / self.scale
        return (diff * diff).mean()
