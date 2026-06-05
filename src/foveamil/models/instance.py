"""インスタンス疑似ラベルによる補助損失

スライド単位ラベルしか無い弱教師下で，主アテンションが高い/低いパッチを per-class の
2 値分類器で pos/neg として検算する補助損失高アテンション上位 k を pos，低アテンション
下位 k を neg とする in-class 枝と，相互排他なサブタイプ向けに非正解クラスの上位 k を
neg とする out-of-class 枝からなる
パッチ選択は top-k の index 抽出で非微分なので勾配は**共有特徴射影**（パッチ表現 h）と
per-class 2 値分類器へ流れ，アテンション網は直接は監督しない（共有射影と bag 損失を介し
間接的に形作られる）bag 分類損失と重み付き和で結合して使う（単一倍率の全バッグが対象）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# 2 値分類器の出力次元（pos / neg）
_BINARY = 2
# 疑似ラベル
_POS_LABEL = 1
_NEG_LABEL = 0


class InstanceClusteringLoss(nn.Module):
    """高/低アテンションのパッチを per-class 2 値分類器で検算する補助損失

    Args:
        in_dim: パッチ表現の次元（主アテンションへの入力次元）
        n_cls: クラス数（per-class 2 値分類器の数）
        k: pos / neg に取る上位・下位パッチ数
        subtyping: 非正解クラスへの out-of-class 枝を加えるか
    """

    def __init__(
        self, in_dim: int, n_cls: int, k: int, subtyping: bool = True
    ) -> None:
        super().__init__()
        self.n_cls = n_cls
        self.k = k
        self.subtyping = subtyping
        self.classifiers = nn.ModuleList(
            nn.Linear(in_dim, _BINARY) for _ in range(n_cls)
        )

    def _targets(self, count: int, value: int, device: torch.device) -> Tensor:
        """``value`` で埋めた ``[count]`` の long ラベルを作る"""
        return torch.full((count,), value, dtype=torch.long, device=device)

    def _eval_in(
        self, h: Tensor, attention: Tensor, classifier: nn.Module, k: int
    ) -> Tensor:
        """正解クラス: 高アテンション上位 k を pos，低アテンション下位 k を neg で検算する"""
        top = torch.topk(attention, k).indices
        bottom = torch.topk(-attention, k).indices
        feats = torch.cat([h[top], h[bottom]], dim=0)
        targets = torch.cat(
            [
                self._targets(k, _POS_LABEL, h.device),
                self._targets(k, _NEG_LABEL, h.device),
            ]
        )
        return F.cross_entropy(classifier(feats), targets)

    def _eval_out(
        self, h: Tensor, attention: Tensor, classifier: nn.Module, k: int
    ) -> Tensor:
        """非正解クラス: 高アテンション上位 k をすべて neg で検算する"""
        top = torch.topk(attention, k).indices
        targets = self._targets(k, _NEG_LABEL, h.device)
        return F.cross_entropy(classifier(h[top]), targets)

    def _bag_loss(self, h: Tensor, attention: Tensor, label: int) -> Tensor:
        """1 バッグ分の補助損失を計算する（パッチ数が 2k 未満なら k を縮める）

        Args:
            h: パッチ表現 ``[N, in_dim]``
            attention: softmax 済み主アテンション ``[N]``
            label: 正解クラス整数

        Returns:
            スカラ補助損失
        """
        num_patches = h.shape[0]
        k = min(self.k, num_patches // 2)
        if k < 1:
            return h.new_zeros(())
        total = self._eval_in(h, attention, self.classifiers[label], k)
        if not self.subtyping:
            return total
        for cls_idx in range(self.n_cls):
            if cls_idx == label:
                continue
            total = total + self._eval_out(
                h, attention, self.classifiers[cls_idx], k
            )
        return total / self.n_cls

    def forward(self, h: Tensor, attention: Tensor, label: Tensor) -> Tensor:
        """バッグごとの補助損失の平均を返す

        Args:
            h: パッチ表現 ``[B, N, in_dim]``
            attention: softmax 済み主アテンション ``[B, N]``
            label: 正解クラス ``[B]``

        Returns:
            スカラ補助損失
        """
        batch = h.shape[0]
        total = h.new_zeros(())
        for b in range(batch):
            total = total + self._bag_loss(h[b], attention[b], int(label[b]))
        return total / batch
