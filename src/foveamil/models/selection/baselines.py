"""スコアを使わない非学習の選択コントローラ

補助アテンションも射影特徴も使わず，候補 ``N`` から k 個を選ぶ参照実装を与える
``random`` は per-controller の決定的乱数生成器で毎回ランダムに k 個を選び，
``uniform`` は等間隔の固定 index で決定的に k 個を選ぶいずれも勾配を要求せず，
学習あり選択器と同じ形状 ``[B, k, N]`` の hard one-hot 選択行列を返す``k`` が ``N``
を超える場合は ``min(N, k)`` に丸め，空バッグ（``N==0``）では ``[B, 0, 0]`` を返す
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from foveamil.models.selection import register_selection_controller
from foveamil.models.selection.base import SelectionController

# 既定の乱数シード（random の決定的再現に使う）
DEFAULT_SEED = 0


def _onehot_rows(indices: Tensor, num_elements: int) -> Tensor:
    """index 集合 ``[B, k]`` を行ごとの hard one-hot 選択行列 ``[B, k, N]`` へ写す"""
    return torch.nn.functional.one_hot(indices, num_classes=num_elements).to(
        torch.get_default_dtype()
    )


def _empty_selection(batch_size: int, device: torch.device) -> Tensor:
    """空バッグ（``N==0``）に対する選択行列 ``[B, 0, 0]`` を返す"""
    return torch.zeros(batch_size, 0, 0, device=device, dtype=torch.get_default_dtype())


@register_selection_controller("uniform")
class UniformSelectionController(SelectionController):
    """候補から等間隔の固定 index で k 個を選ぶ決定的な選択コントローラ

    入力に依らず ``0`` から ``N-1`` を k 分割した等間隔 index を選ぶスコアも特徴も
    使わず勾配を要求しない``k`` が ``N`` を超える場合は ``min(N, k)`` に丸める

    Args:
        k: 選択する要素数
    """

    def __init__(self, k: int) -> None:
        super().__init__(k)

    def select(self, scores: Tensor, features: Tensor) -> Tensor:
        """等間隔の固定 index から選択行列 ``[B, k, N]`` を返す（スコア・特徴は未使用）"""
        batch_size, num_elements = scores.shape
        if num_elements == 0:
            return _empty_selection(batch_size, scores.device)
        k = min(self.k, num_elements)
        # 等間隔の index は昇順（行を index 昇順にすると下流の argmax→sort→gather が
        # 選択重みを正しく拾える）
        positions = torch.linspace(
            0, num_elements - 1, steps=k, device=scores.device
        ).round().long()
        indices = positions.unsqueeze(0).expand(batch_size, -1)
        return _onehot_rows(indices, num_elements)


@register_selection_controller("random")
class RandomSelectionController(SelectionController):
    """候補から重複なしに k 個を一様ランダムに選ぶ非学習の選択コントローラ

    per-controller の乱数生成器で標本し，同じシードへ巻き戻すことで select の連続呼び出しでも
    再現するスコアも特徴も使わず勾配を要求せず，global RNG を汚さない``k`` が ``N`` を超える
    場合は ``min(N, k)`` に丸める

    Args:
        k: 選択する要素数
        seed: 標本の決定性を保つための乱数シード
    """

    def __init__(self, k: int, seed: int = DEFAULT_SEED) -> None:
        super().__init__(k)
        self.seed = seed
        # 標本専用の乱数生成器（global RNG を汚さない select の入力デバイスで作る）
        self._gen: Optional[torch.Generator] = None

    def _sync_generator_device(self, device: torch.device) -> None:
        """生成器を入力デバイスへ合わせ seed を巻き戻す

        ``torch.Generator`` はデバイス固有のため入力が別デバイスなら作り直す各 select で
        同 seed へ巻き戻し，連続呼び出しでも同一の標本を再現する
        """
        if self._gen is None or self._gen.device != device:
            self._gen = torch.Generator(device=device)
        self._gen.manual_seed(self.seed)

    def select(self, scores: Tensor, features: Tensor) -> Tensor:
        """重複なしランダムな k 個から選択行列 ``[B, k, N]`` を返す（スコア・特徴は未使用）"""
        batch_size, num_elements = scores.shape
        if num_elements == 0:
            return _empty_selection(batch_size, scores.device)
        k = min(self.k, num_elements)
        self._sync_generator_device(scores.device)
        noise = torch.rand(
            batch_size, num_elements, generator=self._gen, device=scores.device
        )
        # 重複なしに k 個を取り index 昇順へ並べる（行を index 昇順にすると下流の
        # argmax→sort→gather が選択重みを正しく拾える）
        indices = torch.sort(noise.topk(k, dim=-1).indices, dim=-1).values
        return _onehot_rows(indices, num_elements)
