"""階層的パッチ選択による多解像度 MIL モデル本体（段階 forward）

倍率ごとに特徴射影と 2 つのゲート付きアテンション（主・補助）を持つ主アテンション
は各倍率のプーリング表現を作り，補助アテンションのスコアから top-k セレクタで次倍率
へズームするパッチを選ぶ全パッチを一度に保持せず，1 倍率分の特徴を ``forward_layer``
へ渡し，返る選択結果に応じて学習ループ側が次倍率の子パッチを都度ロードして再び
``forward_layer`` へ渡す全倍率の表現を ``forward_final`` で融合し識別器ヘッドで分類する

倍率数・top-k 手法・融合器・識別器ヘッドを注入でき，部品を差し替えられる
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from foveamil.models.attention import GatedAttention
from foveamil.models.attention_norm import build_attention_norm
from foveamil.models.fusion import build_fusion
from foveamil.models.heads import LinearClassifierHead
from foveamil.models.instance import InstanceClusteringLoss
from foveamil.models.selection import build_selection_controller

# 既定の中間アテンション次元
DEFAULT_HIDDEN_DIM = 256
# 既定の出力特徴次元（特徴射影後の次元）
DEFAULT_OUT_DIM = 512
# 既定の射影段数（1 で従来の浅い 1 段）
DEFAULT_PROJ_NUM_LAYERS = 1
# 既定の射影 LayerNorm 有無（False で従来の正規化なし）
DEFAULT_PROJ_LAYER_NORM = False
# 既定のズーム選択数 k
DEFAULT_K_SAMPLE = 12
# 既定のクラス数
DEFAULT_N_CLS = 3
# 既定の倍率数
DEFAULT_NUM_LAYERS = 4
# 既定の top-k 手法名
DEFAULT_TOPK_METHOD = "perturbed"
# 既定の補助アテンション正規化器名
DEFAULT_AUX_NORM = "softmax"
# 既定の選択コントローラ名
DEFAULT_SELECTOR = "topk"
# 既定の融合名
DEFAULT_FUSION = "sum"
# 既定のインスタンス補助損失の pos/neg パッチ数
DEFAULT_INST_K = 8
# インスタンス補助損失を許す倍率数（単一倍率のみ）
_SINGLE_LAYER = 1
# 射影 MLP の最小段数
_MIN_PROJ_LAYERS = 1
# 主・補助アテンションのクラス数（1 スコア/要素）
_ATTENTION_N_CLS = 1
# アテンションスコアの次元（n_cls=1 の squeeze 対象軸）
_SCORE_AXIS = 1
# Y_hat 抽出時に取り出す上位ロジット数
_TOP1 = 1


def _build_projection(
    in_feat_dim: int,
    out_feat_dim: int,
    dropout: Optional[float],
    num_layers: int = DEFAULT_PROJ_NUM_LAYERS,
    layer_norm: bool = DEFAULT_PROJ_LAYER_NORM,
) -> nn.Sequential:
    """特徴射影 MLP を構築する

    ``num_layers`` 段の ``Linear (+ LayerNorm) + ReLU (+ Dropout)`` を積む先頭段が
    ``in_feat_dim -> out_feat_dim``，以降は ``out_feat_dim -> out_feat_dim``
    ``layer_norm`` が真なら各 ``Linear`` の直後に ``LayerNorm(out_feat_dim)`` を挟む
    既定（``num_layers=1`` / ``layer_norm=False``）は従来の ``Linear + ReLU (+ Dropout)``
    と部品の並びまで一致し数値も bit 互換になる

    Args:
        in_feat_dim: 入力特徴次元
        out_feat_dim: 出力特徴次元（2 段目以降の入出力次元でもある）
        dropout: Dropout 率``None`` なら Dropout なし
        num_layers: 射影段数（1 以上）
        layer_norm: 各 Linear 直後に LayerNorm を挟むか

    Raises:
        ValueError: ``num_layers`` が 1 未満の場合
    """
    if num_layers < _MIN_PROJ_LAYERS:
        raise ValueError(
            f"proj num_layers must be >= {_MIN_PROJ_LAYERS}; got {num_layers}"
        )
    layers: list = []
    for stage in range(num_layers):
        stage_in = in_feat_dim if stage == 0 else out_feat_dim
        layers.append(nn.Linear(stage_in, out_feat_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(out_feat_dim))
        layers.append(nn.ReLU())
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class FoveaMIL(nn.Module):
    """階層的パッチ選択による多解像度 MIL モデル

    Args:
        in_feat_dim: 入力特徴次元
        hidden_feat_dim: アテンション中間次元
        out_feat_dim: 特徴射影後の次元
        dropout: Dropout 率``None`` なら Dropout なし
        proj_num_layers: 特徴射影の段数（1 で従来の浅い 1 段）
        proj_layer_norm: 特徴射影の各 Linear 直後に LayerNorm を挟むか
        k_sample: 次倍率へズームする選択数 k
        n_cls: 出力クラス数
        num_layers: 倍率数
        topk_method: top-k 手法名（``build_topk`` のレジストリ）
        topk_kwargs: top-k セレクタへ渡す追加引数
        aux_norm: 補助アテンション正規化器名（``build_attention_norm`` のレジストリ）
        aux_norm_kwargs: 正規化器へ渡す追加引数
        selector: 選択コントローラ名（``build_selection_controller`` のレジストリ）
        selector_kwargs: 選択コントローラへ渡す追加引数
        fusion: 融合名（``build_fusion`` のレジストリ）
        instance_loss: インスタンス補助損失を持たせるか（単一倍率のみ）
        inst_k: インスタンス補助損失の pos/neg パッチ数
        inst_subtyping: インスタンス補助損失に out-of-class 枝を加えるか

    Raises:
        ValueError: ``instance_loss`` が真かつ ``num_layers`` が 1 でない場合
    """

    def __init__(
        self,
        in_feat_dim: int,
        hidden_feat_dim: int = DEFAULT_HIDDEN_DIM,
        out_feat_dim: int = DEFAULT_OUT_DIM,
        dropout: Optional[float] = None,
        proj_num_layers: int = DEFAULT_PROJ_NUM_LAYERS,
        proj_layer_norm: bool = DEFAULT_PROJ_LAYER_NORM,
        k_sample: int = DEFAULT_K_SAMPLE,
        n_cls: int = DEFAULT_N_CLS,
        num_layers: int = DEFAULT_NUM_LAYERS,
        topk_method: str = DEFAULT_TOPK_METHOD,
        topk_kwargs: Optional[dict] = None,
        aux_norm: str = DEFAULT_AUX_NORM,
        aux_norm_kwargs: Optional[dict] = None,
        selector: str = DEFAULT_SELECTOR,
        selector_kwargs: Optional[dict] = None,
        fusion: str = DEFAULT_FUSION,
        instance_loss: bool = False,
        inst_k: int = DEFAULT_INST_K,
        inst_subtyping: bool = True,
    ) -> None:
        super().__init__()
        if instance_loss and num_layers != _SINGLE_LAYER:
            raise ValueError(
                "instance_loss requires a single magnification "
                f"(num_layers=1); got num_layers={num_layers}"
            )
        self.num_layers = num_layers
        self.k_sample = k_sample
        self.n_cls = n_cls

        self.projections = nn.ModuleList(
            _build_projection(
                in_feat_dim, out_feat_dim, dropout, proj_num_layers, proj_layer_norm
            )
            for _ in range(num_layers)
        )
        self.attentions = nn.ModuleList(
            GatedAttention(out_feat_dim, hidden_feat_dim, dropout, n_cls=_ATTENTION_N_CLS)
            for _ in range(num_layers)
        )
        # 補助アテンションは最終層以外（次倍率へズームする層）に置く
        self.aux_attentions = nn.ModuleList(
            GatedAttention(out_feat_dim, hidden_feat_dim, dropout, n_cls=_ATTENTION_N_CLS)
            for _ in range(num_layers - 1)
        )

        self.aux_norm = build_attention_norm(aux_norm, **(aux_norm_kwargs or {}))
        self.selector = build_selection_controller(
            selector,
            k=k_sample,
            topk_method=topk_method,
            topk_kwargs=topk_kwargs,
            **(selector_kwargs or {}),
        )
        self.fusion = build_fusion(fusion, out_feat_dim, num_layers)
        self.head = LinearClassifierHead(self.fusion.out_dim, n_cls)
        self.instance_module = (
            InstanceClusteringLoss(out_feat_dim, n_cls, inst_k, inst_subtyping)
            if instance_loss
            else None
        )

    def _project_and_pool(
        self, x: Tensor, layer_idx: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """射影と主アテンションでプーリング表現を作り ``(M, x_fc, A_primary)`` を返す

        ``A_primary`` は softmax 済みの主アテンション ``[B, 1, N]````x_fc`` は射影後の
        特徴 ``[B, N, out_dim]``プーリングと補助損失で同一の ``x_fc`` / ``A_primary``
        を共有するための内部部品
        """
        x_fc = self.projections[layer_idx](x)
        A_primary, _ = self.attentions[layer_idx](x_fc)
        A_primary = F.softmax(A_primary.permute(0, 2, 1), dim=-1)
        M = A_primary @ x_fc
        return M, x_fc, A_primary

    def forward_layer(
        self, x: Tensor, layer_idx: int
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """1 倍率分の特徴を処理しプーリング表現と次倍率の選択を返す

        主アテンションで softmax 重み付き和を取りプーリング表現 ``M`` を作る最終層
        以外は補助アテンションのスコアから top-k セレクタで選択行列 ``[B, k, N]`` を
        作り，その argmax を index 昇順に並べ，選択行列から各選択 index の重み
        （学習時 soft / 推論時 hard）を ``select_weight`` として取り出す選択に使った
        正規化済み補助アテンション ``A_aux`` も併せて返す

        Args:
            x: 現倍率の特徴 ``[B, N, in_feat_dim]``
            layer_idx: 現倍率の添字

        Returns:
            ``(M, select_indices, select_weight, A_aux)``M は ``[B, 1, out_dim]``
            最終層では ``select_indices`` / ``select_weight`` / ``A_aux`` は ``None``
            それ以外は ``select_indices`` が ``[B, k]``（index 昇順, 勾配なし），
            ``select_weight`` が ``[B, k]``（学習時は勾配を保持），``A_aux`` が
            正規化済み補助アテンション ``[B, N]``
        """
        M, x_fc, _ = self._project_and_pool(x, layer_idx)

        if layer_idx >= self.num_layers - 1:
            return M, None, None, None

        A_aux, _ = self.aux_attentions[layer_idx](x_fc)
        A_aux = self.aux_norm(A_aux.squeeze(dim=-1))

        selection = self.selector.select(A_aux, x_fc)
        with torch.no_grad():
            select_indices = selection.argmax(dim=-1)
            select_indices = torch.sort(select_indices, dim=-1).values
        select_weight = selection.gather(
            -1, select_indices.unsqueeze(-1)
        ).squeeze(-1)
        return M, select_indices, select_weight, A_aux

    def layer_attention(
        self, x: Tensor, layer_idx: int
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """1 倍率分の主・補助アテンションの正規化重みを返す（可視化用）

        主アテンションは softmax 後のプーリング重み ``[B, 1, N]``，補助アテンション
        は softmax 後の選択スコア ``[B, N]`` を返す最終層は補助アテンションを持たない
        ため aux 側は ``None``選択や勾配は伴わず重みの参照のみを返す

        Args:
            x: 現倍率の特徴 ``[B, N, in_feat_dim]``
            layer_idx: 現倍率の添字

        Returns:
            ``(A_primary, A_aux)``A_primary は ``[B, 1, N]``，A_aux は ``[B, N]``
            または最終層では ``None``
        """
        x_fc = self.projections[layer_idx](x)
        A_primary, _ = self.attentions[layer_idx](x_fc)
        A_primary = F.softmax(A_primary.permute(0, 2, 1), dim=-1)
        if layer_idx >= self.num_layers - 1:
            return A_primary, None
        A_aux, _ = self.aux_attentions[layer_idx](x_fc)
        A_aux = self.aux_norm(A_aux.squeeze(dim=-1))
        return A_primary, A_aux

    def forward_with_instance_loss(
        self, x: Tensor, label: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """単一倍率の bag forward とインスタンス補助損失を同一テンソルから返す

        射影と主アテンションを 1 度だけ計算しプーリング表現と補助損失の双方で同じ
        ``x_fc`` / ``A_primary`` を共有する（dropout 実現を一致させ余分な forward を避ける）
        ``instance_module`` を持たないなら補助損失は ``None``単一倍率（``num_layers=1``）
        の学習時に使う

        Args:
            x: 最低倍率の特徴 ``[B, N, in_feat_dim]``
            label: 正解クラス ``[B]``

        Returns:
            ``(logits[B, n_cls], Y_hat[B, 1], Y_prob[B, n_cls], inst_loss)``
            ``inst_loss`` はスカラまたは ``None``
        """
        M, x_fc, A_primary = self._project_and_pool(x, 0)
        logits, Y_hat, Y_prob = self.forward_final([M])
        inst_loss = None
        if self.instance_module is not None:
            inst_loss = self.instance_module(x_fc, A_primary.squeeze(dim=1), label)
        return logits, Y_hat, Y_prob, inst_loss

    def forward_final(
        self, M_list: Sequence[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """各倍率のプーリング表現を融合し分類する

        Args:
            M_list: 各倍率のプーリング表現 ``[B, 1, out_dim]`` のリスト

        Returns:
            ``logits[B, n_cls]``，``Y_hat[B, 1]``（予測クラス），
            ``Y_prob[B, n_cls]``（softmax 確率）
        """
        fused = self.fusion(M_list)
        logits = self.head(fused)
        Y_hat = torch.topk(logits, _TOP1, dim=-1).indices
        Y_prob = F.softmax(logits, dim=-1)
        return logits, Y_hat, Y_prob
