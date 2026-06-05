"""ImageNet 学習済み ResNet50 による特徴抽出エンコーダ"""

from __future__ import annotations

from typing import Optional, Tuple

from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50

from foveamil.encoders.base import PatchEncoder

# ResNet50 を layer3 で打ち切ったときの出力チャネル数
_FEATURE_DIM = 1024
# layer3 までを残すため children の末尾から取り除く数（layer4, avgpool, fc）
_TRUNCATE_FROM_END = 3


class ResNet50Encoder(PatchEncoder):
    """ImageNet 学習済み ResNet50 を layer3 で打ち切った特徴抽出器

    cls 特徴は持たず，``[B, 1024, h, w]`` の空間特徴を空間平均して ``[B, 1024]`` を返す
    """

    name = "ResNet50"
    feature_dim = _FEATURE_DIM
    has_cls = False

    def _build_model(self) -> nn.Module:
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        return nn.Sequential(*list(backbone.children())[:-_TRUNCATE_FROM_END])

    def _forward_tokens(self, patches: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        feature_map = self._model(patches)
        pooled = feature_map.mean(dim=(2, 3))
        return pooled, None
