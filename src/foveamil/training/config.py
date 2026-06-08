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
# 既定の温度付き softmax の温度（aux_norm="temperature" のときのみ有効）
DEFAULT_AUX_NORM_TEMPERATURE = 1.0
# 既定の α-entmax の α（aux_norm="entmax" のときのみ有効）
DEFAULT_AUX_NORM_ALPHA = 1.5
# 既定の選択コントローラ名
DEFAULT_SELECTOR = "topk"
# 既定の DPP 類似度名（cosine / rbf）
DEFAULT_DPP_SIMILARITY = "cosine"
# 既定の DPP 緩和温度（soft argmax / Gumbel-softmax）
DEFAULT_DPP_TEMPERATURE = 1.0
# 既定の DPP 品質スケール q_i = exp(beta·scores_i)
DEFAULT_DPP_QUALITY_BETA = 1.0
# 既定の DPP RBF 帯域（similarity=="rbf" 時のみ）
DEFAULT_DPP_RBF_GAMMA = 1.0
# 既定の DPP Gumbel 利用（学習時の確率的選択）
DEFAULT_DPP_USE_GUMBEL = False
# 既定の DPP 多様性正則化重み（0 で無効）
DEFAULT_DPP_DIVERSITY_WEIGHT = 0.0
# 既定の融合名
DEFAULT_FUSION = "sum"
# 既定の集約器名（abmil は従来のゲート付きアテンションプーリングと bit 互換）
DEFAULT_AGGREGATOR = "abmil"
# 既定の自己アテンション集約器のヘッド数（aggregator="self_attn" のときのみ有効）
DEFAULT_AGGREGATOR_NUM_HEADS = 4
# 既定の自己アテンション集約器の landmark 数（aggregator="self_attn" のときのみ有効）
DEFAULT_AGGREGATOR_NUM_LANDMARKS = 64
# 既定の識別器ヘッド名（線形で従来挙動と一致）
DEFAULT_HEAD_TYPE = "linear"
# 既定のインスタンス補助損失の有効化
DEFAULT_INSTANCE_LOSS = False
# 既定の bag 損失とインスタンス補助損失の重み（bag 側）
DEFAULT_BAG_WEIGHT = 0.7
# 既定のインスタンス補助損失の pos/neg パッチ数
DEFAULT_INST_K = 8
# 既定のインスタンス補助損失の out-of-class 枝の有無
DEFAULT_INST_SUBTYPING = True
# 既定の不均衡対応損失種別（plain は素 cross-entropy で従来挙動と一致）
DEFAULT_LOSS_TYPE = "plain"
# 既定の logit-adjusted CE の補正強度 τ（loss_type="logit_adjusted" のときのみ有効）
DEFAULT_LOSS_TAU = 1.0
# 既定の class-balanced の有効標本数 β（loss_type="class_balanced" のときのみ有効）
DEFAULT_LOSS_CB_BETA = 0.999
# 既定の LDAM の最大マージン（loss_type="ldam" のときのみ有効）
DEFAULT_LOSS_LDAM_MAX_MARGIN = 0.5
# 既定のクラス数
DEFAULT_N_CLS = 3
# 既定の倍率間冗長性罰則の重み（0 で無効）
DEFAULT_DECORRELATION_WEIGHT = 0.0
# 既定の倍率間冗長性罰則の手法
DEFAULT_DECORRELATION_METHOD = "cosine"
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
        aux_norm_temperature: 温度付き softmax の温度（``aux_norm="temperature"`` のときのみ有効）
        aux_norm_alpha: α-entmax の α（``aux_norm="entmax"`` のときのみ有効）
        selector: 選択コントローラ名（単一倍率では無効）
        dpp_similarity: DPP 類似度名（``"cosine"`` / ``"rbf"``，``selector=="dpp"`` 時のみ）
        dpp_temperature: DPP 緩和温度（``selector=="dpp"`` 時のみ）
        dpp_quality_beta: DPP 品質スケール ``q_i = exp(beta·scores_i)``（``selector=="dpp"`` 時のみ）
        dpp_rbf_gamma: DPP RBF 帯域（``selector=="dpp"`` かつ ``dpp_similarity=="rbf"`` 時のみ）
        dpp_use_gumbel: DPP 学習時に Gumbel 雑音で確率的に選ぶか（``selector=="dpp"`` 時のみ）
        dpp_diversity_weight: DPP 多様性正則化の重み（0 で無効，``selector=="dpp"`` 多倍率時のみ）
        fusion: 融合名（``"sum"`` で従来挙動，``"gated"`` でスライド依存ゲート加重和，``"scale_attention"`` でスケール間自己アテンション集約）
        aggregator: 集約器名（``"abmil"`` で従来のゲート付きアテンションプーリングと bit 互換，``"self_attn"`` でパッチ間コンテキストを取り込む自己アテンション）
        aggregator_num_heads: 自己アテンション集約器の注意ヘッド数（``aggregator="self_attn"`` のときのみ有効）
        aggregator_num_landmarks: 自己アテンション集約器の Nyström landmark 数（``aggregator="self_attn"`` のときのみ有効パッチ数が landmark 以下なら厳密注意へ縮退）
        head_type: 識別器ヘッド名（``"linear"`` 既定で従来挙動，``"mlp"`` で容量増の小 MLP）
        head_hidden_dim: 小 MLP ヘッドの中間次元（``head_type="mlp"`` のときのみ有効，``None`` で既定値）
        instance_loss: インスタンス補助損失を加えるか（単一倍率のみ）
        bag_weight: bag 損失とインスタンス補助損失の重み（``bag·bag_weight + inst·(1-bag_weight)``）
        inst_k: インスタンス補助損失の pos/neg パッチ数
        inst_subtyping: インスタンス補助損失に out-of-class 枝を加えるか
        n_cls: クラス数
        loss_type: 分類損失種別（``"plain"`` で素 CE，``"logit_adjusted"`` / ``"ldam"`` / ``"class_balanced"`` で不均衡対応クラス頻度は train split から自動算出する）
        loss_tau: logit-adjusted CE の補正強度 τ（``loss_type="logit_adjusted"`` のときのみ有効）
        loss_cb_beta: class-balanced の有効標本数 β（``loss_type="class_balanced"`` のときのみ有効）
        loss_ldam_max_margin: LDAM の最大マージン（``loss_type="ldam"`` のときのみ有効）
        decorrelation_weight: 倍率間冗長性罰則の重み（0 で無効，多倍率のみ有効）
        decorrelation_method: 倍率間冗長性罰則の手法（``"cosine"`` / ``"covariance"``）
        is_weighted_sampler: train で WeightedRandomSampler を使うか
        num_workers: DataLoader ワーカ数
        pin_memory: DataLoader の pin_memory
        save_metric: best 保存基準（``"loss"`` / ``"f1"``）
        zoom_driver: ズーム駆動名（``"differentiable"`` で従来挙動，``"mcts"`` で探索）
        mcts_planner: 探索プランナ名（``"gumbel"`` / ``"puct"``，``mcts`` のみ有効）
        mcts_simulations: 模擬予算（``mcts`` のみ有効）
        mcts_max_considered: 検討最大候補数 m（Gumbel top-m，``mcts`` のみ有効）
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
    aux_norm_temperature: float = DEFAULT_AUX_NORM_TEMPERATURE
    aux_norm_alpha: float = DEFAULT_AUX_NORM_ALPHA
    selector: str = DEFAULT_SELECTOR
    dpp_similarity: str = DEFAULT_DPP_SIMILARITY
    dpp_temperature: float = DEFAULT_DPP_TEMPERATURE
    dpp_quality_beta: float = DEFAULT_DPP_QUALITY_BETA
    dpp_rbf_gamma: float = DEFAULT_DPP_RBF_GAMMA
    dpp_use_gumbel: bool = DEFAULT_DPP_USE_GUMBEL
    dpp_diversity_weight: float = DEFAULT_DPP_DIVERSITY_WEIGHT
    fusion: str = DEFAULT_FUSION
    aggregator: str = DEFAULT_AGGREGATOR
    aggregator_num_heads: int = DEFAULT_AGGREGATOR_NUM_HEADS
    aggregator_num_landmarks: int = DEFAULT_AGGREGATOR_NUM_LANDMARKS
    head_type: str = DEFAULT_HEAD_TYPE
    head_hidden_dim: Optional[int] = None
    instance_loss: bool = DEFAULT_INSTANCE_LOSS
    bag_weight: float = DEFAULT_BAG_WEIGHT
    inst_k: int = DEFAULT_INST_K
    inst_subtyping: bool = DEFAULT_INST_SUBTYPING
    n_cls: int = DEFAULT_N_CLS
    loss_type: str = DEFAULT_LOSS_TYPE
    loss_tau: float = DEFAULT_LOSS_TAU
    loss_cb_beta: float = DEFAULT_LOSS_CB_BETA
    loss_ldam_max_margin: float = DEFAULT_LOSS_LDAM_MAX_MARGIN
    decorrelation_weight: float = DEFAULT_DECORRELATION_WEIGHT
    decorrelation_method: str = DEFAULT_DECORRELATION_METHOD

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
