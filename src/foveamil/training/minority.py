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

from typing import Callable, Optional, Sequence, Tuple

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
        """Beta(alpha, alpha) から補間係数 λ を引く

        ``torch.distributions.Beta`` は generator を取らず global RNG を汚すため
        generator 付きで Gamma(alpha, 1) を 2 本引き λ=g1/(g1+g2) として Beta(α,α) を
        合成する（Gamma 比＝Beta の定義）generator が無い場合のみ Beta.sample に委譲する
        """
        if self.generator is None:
            return torch.distributions.Beta(self.alpha, self.alpha).sample().to(device)
        g1 = self._sample_gamma(self.alpha)
        g2 = self._sample_gamma(self.alpha)
        total = (g1 + g2).clamp_min(torch.finfo(torch.float32).tiny)
        return (g1 / total).to(device)

    def _sample_gamma(self, shape: float) -> Tensor:
        """generator 付きで Gamma(shape, 1)（scale=1）を 1 サンプル引く

        Marsaglia-Tsang 法（shape>=1）で引き shape<1 は boost で補正する
        （Gamma(α)=Gamma(α+1)·U^(1/α)）global RNG を汚さず決定的に動く
        """
        # shape<1 は boost: Gamma(α+1) を引いて U^(1/α) を掛ける
        if shape < 1.0:
            g = self._sample_gamma(shape + 1.0)
            u = torch.rand(1, generator=self.generator)[0]
            u = u.clamp_min(torch.finfo(torch.float32).tiny)
            return g * u.pow(1.0 / shape)
        # Marsaglia-Tsang（shape>=1）棄却採択で 1 サンプル得るまで反復する
        d = shape - 1.0 / 3.0
        c = 1.0 / (9.0 * d) ** 0.5
        while True:
            x = torch.randn(1, generator=self.generator)[0]
            v = (1.0 + c * x) ** 3
            if v <= 0:
                continue
            u = torch.rand(1, generator=self.generator)[0]
            x2 = x * x
            # 採択条件（簡易版を先に試し 不成立なら対数版で判定）
            if u < 1.0 - 0.0331 * x2 * x2:
                return d * v
            if u.log() < 0.5 * x2 + d * (1.0 - v + v.log()):
                return d * v

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

    softmax 確率の期待ランクと正解ランクの正規化二乗距離を罰する 順序が遠い誤りほど
    大きく罰し 近い誤りは小さく罰する（順序の単調性）``weight`` で寄与をスケールし
    ``weight=0`` なら学習ループ側が呼ばないため寄与 0 とする

    ``class_order`` で**順序のあるクラス部分集合**を与えられる その index 列の並びを
    昇順ランク（0,1,...）とし 集合に属するクラスのみで期待ランクを測り（集合内質量で
    再正規化）正解が集合に属するサンプルにのみペナルティを課す 集合外クラスは名義として
    順序の対象外（分類 CE が扱う）``class_order=None`` は全クラスを index 順の全順序と
    みなす従来挙動で数値一致する

    ランクの絶対尺度を ``len-1`` で割り正規化する（集合長に依らず比較可能）

    Args:
        n_cls: クラス数
        class_order: 順序を課すクラス index の昇順列（``None`` で全クラス index 順）
    """

    def __init__(self, n_cls: int, class_order: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        if class_order is not None and not isinstance(class_order, (list, tuple)):
            raise ValueError(
                "class_order は list/tuple のクラス index 列を渡す"
                f"（sweep では list-of-lists にする）: {class_order!r}"
            )
        self.subset = class_order is not None
        if class_order is None:
            ranks = torch.arange(n_cls, dtype=torch.float32)
            in_chain = torch.ones(n_cls, dtype=torch.bool)
            # 単一クラスでは順序が無いため正規化を恒等にする
            self.scale = float(max(n_cls - 1, 1))
        else:
            order = [int(c) for c in class_order]
            if not order:
                raise ValueError("class_order が空")
            if len(set(order)) != len(order):
                raise ValueError(f"class_order に重複 index: {order}")
            if any(c < 0 or c >= n_cls for c in order):
                raise ValueError(f"class_order の index が範囲外(0..{n_cls-1}): {order}")
            # クラスごとのチェーン内昇順ランク（集合外は 0・マスクで無効化）
            ranks = torch.zeros(n_cls, dtype=torch.float32)
            in_chain = torch.zeros(n_cls, dtype=torch.bool)
            for rank, cls_idx in enumerate(order):
                ranks[cls_idx] = float(rank)
                in_chain[cls_idx] = True
            self.scale = float(max(len(order) - 1, 1))
        self.register_buffer("ranks", ranks)
        self.register_buffer("in_chain", in_chain)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """期待ランクと正解ランクの正規化二乗距離（平均）を返す

        Args:
            logits: 分類 logit ``[B, n_cls]``
            target: 正解クラス ``[B]``

        Returns:
            平均二乗距離スカラ（部分集合時は集合内サンプルが無ければ寄与 0）
        """
        probs = F.softmax(logits, dim=-1)
        if not self.subset:
            expected_rank = probs @ self.ranks
            true_rank = self.ranks[target]
            diff = (expected_rank - true_rank) / self.scale
            return (diff * diff).mean()
        # 部分集合: 集合内質量で再正規化した期待ランクを 集合内サンプルにのみ課す
        chain_mass = probs[:, self.in_chain].sum(dim=-1)
        weighted = probs @ (self.ranks * self.in_chain.to(self.ranks.dtype))
        expected_rank = weighted / chain_mass.clamp_min(1e-8)
        sample_mask = self.in_chain[target]
        if not bool(sample_mask.any()):
            # 集合内サンプルが無いバッチは順序信号なし＝寄与 0（grad graph は保持）
            return logits.sum() * 0.0
        true_rank = self.ranks[target]
        diff = (expected_rank[sample_mask] - true_rank[sample_mask]) / self.scale
        return (diff * diff).mean()
