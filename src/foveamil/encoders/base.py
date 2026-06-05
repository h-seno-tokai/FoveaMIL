"""パッチエンコーダの抽象基底

正規化済みパッチテンソル ``[B, 3, 224, 224]`` を受け，pooled 特徴 ``[B, dim]`` と
ViT 系の cls 特徴 ``[B, dim]``（無ければ ``None``）を返す共通インタフェースを定義する
モデルは :meth:`PatchEncoder.load` で遅延ロードする
"""

from __future__ import annotations

import abc
from typing import List, Optional, Tuple

import torch
from torch import Tensor

# ImageNet 正規化の平均（RGB）
NORMALIZER_MEAN: List[float] = [0.485, 0.456, 0.406]
# ImageNet 正規化の標準偏差（RGB）
NORMALIZER_STD: List[float] = [0.229, 0.224, 0.225]
# patch トークン列を空間配置へ戻すときの一辺（16x16=256 トークン）
TOKEN_GRID_SIDE = 16
# cls トークンのトークン列上の位置
CLS_TOKEN_INDEX = 0
# 既定のバッチサイズ
DEFAULT_BATCH_SIZE = 256
# 既定の DataLoader ワーカ数
DEFAULT_NUM_WORKERS = 4
# GPU 推論時の autocast dtype
_AUTOCAST_DTYPE = torch.float16


def _select_device() -> torch.device:
    """CUDA があれば現在の CUDA デバイス，無ければ CPU を返す"""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


class PatchEncoder(abc.ABC):
    """正規化済みパッチから pooled / cls 特徴を取り出すエンコーダの基底

    入力パッチは呼び出し側で正規化済みであることを前提とするモデルは
    :meth:`load` で初めて構築され，再呼び出しでは何もしない

    Attributes:
        name: エンコーダ名（canonical 名）
        feature_dim: 出力特徴次元
        has_cls: cls 特徴を返すか
        device: 推論デバイス
        normalizer_mean: 想定する正規化の平均（呼び出し側が利用）
        normalizer_std: 想定する正規化の標準偏差（呼び出し側が利用）
        batch_size: 推論バッチサイズ
        num_workers: DataLoader ワーカ数
    """

    name: str = ""
    feature_dim: int = 0
    has_cls: bool = False

    def __init__(
        self,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = _select_device()
        self.normalizer_mean = NORMALIZER_MEAN
        self.normalizer_std = NORMALIZER_STD
        self._model: Optional[torch.nn.Module] = None

    @abc.abstractmethod
    def _build_model(self) -> torch.nn.Module:
        """モデルを構築して返す（重みロードを含む，``eval`` 化は :meth:`load` が行う）"""

    def load(self) -> None:
        """モデルを遅延ロードする（既にロード済みなら何もしない）"""
        if self._model is not None:
            return
        model = self._build_model()
        model.eval()
        model.to(self.device)
        self._model = model

    @abc.abstractmethod
    def _forward_tokens(self, patches: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """モデルを実行し ``(pooled[B, dim], cls[B, dim] or None)`` を返す

        :meth:`forward` の ``no_grad`` / ``autocast`` 文脈内で呼ばれる
        """

    def forward(self, patches: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """正規化済みパッチ ``[B, 3, 224, 224]`` から pooled と cls を返す

        Args:
            patches: 正規化済みパッチテンソル ``[B, 3, 224, 224]``

        Returns:
            ``(pooled[B, feature_dim], cls[B, feature_dim] または None)``
        """
        if self._model is None:
            self.load()

        patches = patches.to(self.device)
        use_cuda = self.device.type == "cuda"
        with torch.no_grad():
            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=_AUTOCAST_DTYPE):
                    return self._forward_tokens(patches)
            return self._forward_tokens(patches)

    def _pool_patch_tokens(self, patch_tokens: Tensor) -> Tensor:
        """patch トークン列 ``[B, 256, dim]`` を ``[B, dim]`` に平均プールする

        ``[B, 16, 16, dim]`` に reshape し ``[B, dim, 16, 16]`` へ並べ替えてから
        :class:`~torch.nn.AdaptiveAvgPool2d` 相当の空間平均を取る

        Args:
            patch_tokens: patch トークン列 ``[B, TOKEN_GRID_SIDE**2, dim]``

        Returns:
            pooled 特徴 ``[B, dim]``
        """
        batch_size, _, dim = patch_tokens.shape
        spatial = patch_tokens.reshape(
            batch_size, TOKEN_GRID_SIDE, TOKEN_GRID_SIDE, dim
        ).permute(0, 3, 1, 2)
        return spatial.mean(dim=(2, 3))
