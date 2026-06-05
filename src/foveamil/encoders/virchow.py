"""Virchow / Virchow2（ViT 系病理基盤モデル）による特徴抽出エンコーダ"""

from __future__ import annotations

from typing import Optional, Tuple

import timm
import torch
from timm.layers import SwiGLUPacked
from torch import Tensor, nn

from foveamil.encoders.auth import ensure_hf_auth
from foveamil.encoders.base import PatchEncoder

# Virchow 系の特徴次元
_FEATURE_DIM = 1280
# Virchow の HuggingFace Hub モデル ID
_VIRCHOW_HF_MODEL_ID = "hf-hub:paige-ai/Virchow"
# Virchow2 の HuggingFace Hub モデル ID
_VIRCHOW2_HF_MODEL_ID = "hf-hub:paige-ai/Virchow2"
# Virchow: CLS 1 を除いた patch トークン開始位置
_VIRCHOW_PATCH_TOKEN_START = 1
# Virchow2: CLS 1 + register 4 を除いた patch トークン開始位置
_VIRCHOW2_PATCH_TOKEN_START = 5


class VirchowEncoder(PatchEncoder):
    """Virchow を実行する特徴抽出器

    出力 ``[B, 257, 1280]`` から cls=``[:, 0, :]``，patch ``[:, 1:, :]`` を
    空間平均して pooled を返す
    """

    name = "Virchow"
    feature_dim = _FEATURE_DIM
    has_cls = True

    def _build_model(self) -> nn.Module:
        ensure_hf_auth()
        return timm.create_model(
            _VIRCHOW_HF_MODEL_ID,
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

    def _forward_tokens(self, patches: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        tokens = self._model(patches)
        cls = tokens[:, 0, :]
        patch_tokens = tokens[:, _VIRCHOW_PATCH_TOKEN_START:, :]
        pooled = self._pool_patch_tokens(patch_tokens)
        return pooled, cls


class Virchow2Encoder(PatchEncoder):
    """Virchow2 を実行する特徴抽出器

    出力 ``[B, 261, 1280]`` から cls=``[:, 0, :]``，patch ``[:, 5:, :]`` を
    空間平均して pooled を返す（gated モデル）
    """

    name = "Virchow2"
    feature_dim = _FEATURE_DIM
    has_cls = True

    def _build_model(self) -> nn.Module:
        ensure_hf_auth()
        return timm.create_model(
            _VIRCHOW2_HF_MODEL_ID,
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

    def _forward_tokens(self, patches: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        tokens = self._model(patches)
        cls = tokens[:, 0, :]
        patch_tokens = tokens[:, _VIRCHOW2_PATCH_TOKEN_START:, :]
        pooled = self._pool_patch_tokens(patch_tokens)
        return pooled, cls
