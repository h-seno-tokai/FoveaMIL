"""学習済み FoveaMIL から階層アテンションのトレースを取り出す（推論専用）

最低倍率の全特徴を読み，段階 forward の選択結果に応じて高倍率の子パッチを
:class:`FeatureAccessor` で都度ロードする Lazy 駆動を no_grad で再現し，各倍率で
主アテンション重み・補助アテンション重み・選択されたパッチ・座標を集める子特徴へ
選択重みを掛ける処理は学習ループと揃え，モデルが実際に見た表現と一致させる返す
:class:`AttentionTrace` は各倍率の重みと選択，最終の予測クラスと確率を保持する
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from foveamil.models import FoveaMIL
from foveamil.training.accessor import FeatureAccessor
from foveamil.training.hierarchy import children_per_parent, compute_child_indices

# バッチ軸の添字（バッチサイズ 1 前提）
_BATCH = 0


@dataclass
class LayerTrace:
    """1 倍率分のアテンショントレース

    Attributes:
        magnification: 倍率
        global_indices: この倍率に存在するパッチの最低倍率基準 global index ``(N,)``
        coords: この倍率のパッチ座標 ``(N, 2)``
        primary: 主アテンションの正規化重み ``(N,)``
        aux: 補助アテンションの正規化重み ``(N,)``最終層は ``None``
        selected_local: 補助アテンションで選ばれた local index ``(k,)``最終層は ``None``
        selected_global: 選ばれたパッチの global index ``(k,)``最終層は ``None``
        select_weight: 選択重み ``(k,)``最終層は ``None``
    """

    magnification: float
    global_indices: np.ndarray
    coords: np.ndarray
    primary: np.ndarray
    aux: Optional[np.ndarray]
    selected_local: Optional[np.ndarray]
    selected_global: Optional[np.ndarray]
    select_weight: Optional[np.ndarray]


@dataclass
class AttentionTrace:
    """1 スライド分の階層アテンショントレースと予測

    Attributes:
        slide_id: スライド識別子
        magnifications: 倍率の列（低→高）
        layers: 各倍率の :class:`LayerTrace`
        y_hat: 予測クラス
        y_prob: クラス確率 ``(n_cls,)``
    """

    slide_id: str
    magnifications: List[float]
    layers: List[LayerTrace]
    y_hat: int
    y_prob: np.ndarray


def extract_attention_trace(
    model: FoveaMIL,
    feature_root: str,
    encoder: str,
    slide_id: str,
    magnifications: Sequence[float],
    feature_type: str,
    device: Optional[torch.device] = None,
) -> AttentionTrace:
    """学習済みモデルから 1 スライドの階層アテンショントレースを取り出す

    Args:
        model: 学習済み FoveaMIL
        feature_root: 特徴ルートディレクトリ
        encoder: エンコーダ名
        slide_id: スライド識別子
        magnifications: 倍率の列（低→高, モデルの倍率数と一致させる）
        feature_type: ``"mean"`` / ``"cls"`` / ``"concat"``
        device: 推論デバイス``None`` なら CPU

    Returns:
        :class:`AttentionTrace`
    """
    device = device or torch.device("cpu")
    model.eval()
    model.to(device)
    num_layers = len(magnifications)

    accessor = FeatureAccessor(feature_root, encoder, slide_id, feature_type)
    try:
        layers: List[LayerTrace] = []
        m_list: List[Tensor] = []
        x = accessor.load_all(magnifications[_BATCH]).float().unsqueeze(0).to(device)
        global_idx: Optional[np.ndarray] = None

        with torch.no_grad():
            for layer_idx in range(num_layers):
                mag = magnifications[layer_idx]
                if global_idx is None:
                    g_idx = np.arange(x.shape[1], dtype=np.int64)
                    coords = accessor.load_coords_all(mag)
                else:
                    g_idx = global_idx
                    coords = accessor.load_coords_indexed(mag, g_idx)

                A_primary, A_aux = model.layer_attention(x, layer_idx)
                primary = A_primary[_BATCH, 0].cpu().numpy()
                M, select_indices, select_weight = model.forward_layer(x, layer_idx)
                m_list.append(M)

                if layer_idx < num_layers - 1:
                    aux = A_aux[_BATCH].cpu().numpy()
                    sel_local = select_indices[_BATCH].cpu().numpy()
                    sel_global = g_idx[sel_local]
                    sw = select_weight[_BATCH].cpu().numpy()

                    next_mag = magnifications[layer_idx + 1]
                    cpp = children_per_parent(mag, next_mag)
                    child = compute_child_indices(
                        sel_local, global_idx, children=cpp
                    )
                    x_next = (
                        accessor.load_patches(next_mag, child)
                        .float()
                        .unsqueeze(0)
                        .to(device)
                    )
                    w_child = select_weight.repeat_interleave(cpp, dim=1)
                    x = x_next * w_child.unsqueeze(-1)
                    global_idx = child
                else:
                    aux = sel_local = sel_global = sw = None

                layers.append(
                    LayerTrace(
                        magnification=mag,
                        global_indices=g_idx,
                        coords=coords,
                        primary=primary,
                        aux=aux,
                        selected_local=sel_local,
                        selected_global=sel_global,
                        select_weight=sw,
                    )
                )

            _, Y_hat, Y_prob = model.forward_final(m_list)
    finally:
        accessor.close()

    return AttentionTrace(
        slide_id=slide_id,
        magnifications=list(magnifications),
        layers=layers,
        y_hat=int(Y_hat[_BATCH].item()),
        y_prob=Y_prob[_BATCH].cpu().numpy(),
    )
