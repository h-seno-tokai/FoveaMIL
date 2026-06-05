"""学習 1 回分の設定を保持する dataclass

最適化・データ・モデル・学習インタフェースの各設定をフィールドに持つ
``num_layers`` は ``magnifications`` の長さから導く``n_cls`` は ``classes`` や
ラベル辞書から上書きしてよい
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

# 既定の乱数シード
DEFAULT_SEED = 1
# 既定の最大エポック数
DEFAULT_MAX_EPOCHS = 100
# 既定の学習率
DEFAULT_LR = 1e-4
# 既定の重み減衰（weight_decay）
DEFAULT_REG = 0.0
# 既定の学習率スケジューラ減衰係数
DEFAULT_SCHEDULER_DECAY_RATE = 0.5
# 既定の学習率スケジューラ忍耐エポック数
DEFAULT_SCHEDULER_PATIENCE = 5
# 既定の feature_type
DEFAULT_FEATURE_TYPE = "mean"
# 既定の入力特徴次元
DEFAULT_IN_FEAT_DIM = 1024
# 既定のアテンション中間次元
DEFAULT_HIDDEN_FEAT_DIM = 256
# 既定の特徴射影後の次元
DEFAULT_OUT_FEAT_DIM = 512
# 既定のズーム選択数 k
DEFAULT_K_SAMPLE = 12
# 既定の top-k 平滑化パラメータ（perturbed の sigma / fast_sparse の epsilon）
DEFAULT_K_SIGMA = 0.002
# 既定の top-k 手法名
DEFAULT_TOPK_METHOD = "perturbed"
# 既定の補助アテンション正規化器名
DEFAULT_AUX_NORM = "softmax"
# 既定の選択コントローラ名
DEFAULT_SELECTOR = "topk"
# 既定の融合名
DEFAULT_FUSION = "sum"
# 既定のインスタンス補助損失の有効化
DEFAULT_INSTANCE_LOSS = False
# 既定の bag 損失とインスタンス補助損失の重み（bag 側）
DEFAULT_BAG_WEIGHT = 0.7
# 既定のインスタンス補助損失の pos/neg パッチ数
DEFAULT_INST_K = 8
# 既定のインスタンス補助損失の out-of-class 枝の有無
DEFAULT_INST_SUBTYPING = True
# 既定のクラス数
DEFAULT_N_CLS = 3
# 既定の DataLoader ワーカ数
DEFAULT_NUM_WORKERS = 4
# 既定の save_metric
DEFAULT_SAVE_METRIC = "loss"
# 既定のズーム駆動名（既定駆動は従来挙動を再現する）
DEFAULT_ZOOM_DRIVER = "differentiable"
# 既定の探索プランナ名（``zoom_driver="mcts"`` のときのみ有効）
DEFAULT_MCTS_PLANNER = "gumbel"
# 既定の模擬予算
DEFAULT_MCTS_SIMULATIONS = 16
# 既定の検討最大候補数 m（Gumbel top-m）
DEFAULT_MCTS_MAX_CONSIDERED = 8
# 既定の方策蒸留損失重み
DEFAULT_POLICY_LOSS_WEIGHT = 1.0
# 既定の価値回帰損失重み
DEFAULT_VALUE_LOSS_WEIGHT = 1.0
# 既定の方策エントロピー損失重み（探索の早期収束を抑える既定は無効）
DEFAULT_POLICY_ENTROPY_WEIGHT = 0.0


@dataclass
class TrainConfig:
    """学習 1 回分の設定

    Attributes:
        seed: 乱数シード
        max_epochs: 最大エポック数
        save_path: 出力先ディレクトリ
        lr: 学習率
        reg: 重み減衰（Adam の weight_decay）
        scheduler_decay_rate: ReduceLROnPlateau の減衰係数
        scheduler_patience: ReduceLROnPlateau の忍耐エポック数
        feature_root: 特徴ルートディレクトリ
        encoder: エンコーダ名
        labels_csv: ``slide_id,label`` の CSV パス
        magnifications: 倍率の列（低→高，倍率レイヤ順）
        feature_type: ``"mean"`` / ``"cls"`` / ``"concat"``
        classes: クラス名の並び（``None`` なら CSV から導く）
        in_feat_dim: 入力特徴次元
        hidden_feat_dim: アテンション中間次元
        out_feat_dim: 特徴射影後の次元
        drop_out: Dropout 率（``None`` なら Dropout なし）
        k_sample: ズーム選択数 k（単一倍率では無効）
        k_sigma: top-k 平滑化パラメータ（単一倍率では無効）
        topk_method: top-k 手法名（単一倍率では無効）
        aux_norm: 補助アテンション正規化器名（単一倍率では無効）
        selector: 選択コントローラ名（単一倍率では無効）
        fusion: 融合名
        instance_loss: インスタンス補助損失を加えるか（単一倍率のみ）
        bag_weight: bag 損失とインスタンス補助損失の重み（``bag·bag_weight + inst·(1-bag_weight)``）
        inst_k: インスタンス補助損失の pos/neg パッチ数
        inst_subtyping: インスタンス補助損失に out-of-class 枝を加えるか
        n_cls: クラス数
        is_weighted_sampler: train で WeightedRandomSampler を使うか
        num_workers: DataLoader ワーカ数
        pin_memory: DataLoader の pin_memory
        save_metric: best 保存基準（``"loss"`` / ``"f1"``）
        zoom_driver: ズーム駆動名（``"differentiable"`` で従来挙動，``"mcts"`` で探索）
        mcts_planner: 探索プランナ名（``"gumbel"`` / ``"puct"````mcts`` のみ有効）
        mcts_simulations: 模擬予算（``mcts`` のみ有効）
        mcts_max_considered: 検討最大候補数 m（Gumbel top-m``mcts`` のみ有効）
        policy_loss_weight: 方策蒸留損失の重み λ_π（``mcts`` のみ有効）
        value_loss_weight: 価値回帰損失の重み λ_v（``mcts`` のみ有効）
        policy_entropy_weight: 方策エントロピー損失の重み（``mcts`` のみ有効）
        mcts_hidden_dim: 方策・価値ネットの中間次元（``None`` なら ``hidden_feat_dim``）
    """

    seed: int = DEFAULT_SEED
    max_epochs: int = DEFAULT_MAX_EPOCHS
    save_path: Optional[str] = None

    lr: float = DEFAULT_LR
    reg: float = DEFAULT_REG
    scheduler_decay_rate: float = DEFAULT_SCHEDULER_DECAY_RATE
    scheduler_patience: int = DEFAULT_SCHEDULER_PATIENCE

    feature_root: Optional[str] = None
    encoder: Optional[str] = None
    labels_csv: Optional[str] = None
    magnifications: Optional[List[float]] = None
    feature_type: str = DEFAULT_FEATURE_TYPE
    classes: Optional[List[str]] = None

    in_feat_dim: int = DEFAULT_IN_FEAT_DIM
    hidden_feat_dim: int = DEFAULT_HIDDEN_FEAT_DIM
    out_feat_dim: int = DEFAULT_OUT_FEAT_DIM
    drop_out: Optional[float] = None
    k_sample: int = DEFAULT_K_SAMPLE
    k_sigma: float = DEFAULT_K_SIGMA
    topk_method: str = DEFAULT_TOPK_METHOD
    aux_norm: str = DEFAULT_AUX_NORM
    selector: str = DEFAULT_SELECTOR
    fusion: str = DEFAULT_FUSION
    instance_loss: bool = DEFAULT_INSTANCE_LOSS
    bag_weight: float = DEFAULT_BAG_WEIGHT
    inst_k: int = DEFAULT_INST_K
    inst_subtyping: bool = DEFAULT_INST_SUBTYPING
    n_cls: int = DEFAULT_N_CLS

    is_weighted_sampler: bool = True
    num_workers: int = DEFAULT_NUM_WORKERS
    pin_memory: bool = True
    save_metric: str = DEFAULT_SAVE_METRIC

    zoom_driver: str = DEFAULT_ZOOM_DRIVER
    mcts_planner: str = DEFAULT_MCTS_PLANNER
    mcts_simulations: int = DEFAULT_MCTS_SIMULATIONS
    mcts_max_considered: int = DEFAULT_MCTS_MAX_CONSIDERED
    policy_loss_weight: float = DEFAULT_POLICY_LOSS_WEIGHT
    value_loss_weight: float = DEFAULT_VALUE_LOSS_WEIGHT
    policy_entropy_weight: float = DEFAULT_POLICY_ENTROPY_WEIGHT
    mcts_hidden_dim: Optional[int] = None
