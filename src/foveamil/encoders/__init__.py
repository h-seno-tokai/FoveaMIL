from typing import Dict, Type

from foveamil.encoders.base import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    PatchEncoder,
)
from foveamil.encoders.resnet import ResNet50Encoder
from foveamil.encoders.uni import UNI2hEncoder
from foveamil.encoders.virchow import Virchow2Encoder, VirchowEncoder
from foveamil.encoders.virchow_mini import Virchow2MiniEncoder

ENCODERS: Dict[str, Type[PatchEncoder]] = {
    ResNet50Encoder.name: ResNet50Encoder,
    UNI2hEncoder.name: UNI2hEncoder,
    VirchowEncoder.name: VirchowEncoder,
    Virchow2Encoder.name: Virchow2Encoder,
    Virchow2MiniEncoder.name: Virchow2MiniEncoder,
}


def build_encoder(
    name: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> PatchEncoder:
    """``name`` の登録エンコーダを生成して返す

    Args:
        name: 登録名（``ENCODERS`` のキー）
        batch_size: 推論バッチサイズ
        num_workers: DataLoader ワーカ数

    Returns:
        構築済み（未ロード）の :class:`PatchEncoder`

    Raises:
        KeyError: 未登録の名前を与えた場合
    """
    if name not in ENCODERS:
        raise KeyError(f"unknown encoder '{name}'; available: {sorted(ENCODERS)}")
    return ENCODERS[name](batch_size=batch_size, num_workers=num_workers)


__all__ = ["ENCODERS", "build_encoder", "PatchEncoder"]
