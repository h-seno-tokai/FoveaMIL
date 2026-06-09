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
# 既定の特徴射影の段数（1 で従来の浅い 1 段）
DEFAULT_PROJ_NUM_LAYERS = 1
# 既定の特徴射影の LayerNorm 有無（False で従来の正規化なし）
DEFAULT_PROJ_LAYER_NORM = False
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
# 既定の価値ターゲット種別（``"realised"`` で従来挙動＝最終 CE を全状態へ broadcast）
DEFAULT_MCTS_VALUE_TARGET = "realised"
# 既定の rollout 深さ（1 で従来挙動＝選んだ親の子を 1 段だけ評価し更に展開しない）
DEFAULT_MCTS_ROLLOUT_DEPTH = 1
# 既定の rollout 段プランナ模擬予算（``None`` で ``mcts_simulations`` に一致＝従来挙動）
DEFAULT_MCTS_ROLLOUT_SIMULATIONS = None
# 既定の確率的葉評価フラグ（False で従来挙動＝eval モード葉評価＋報酬 memoize）
DEFAULT_MCTS_EVAL_STOCHASTIC = False
# 既定の actor-critic 項スケール（1.0 で正規化 advantage を等倍で方策蒸留へ上乗せ，0 で無効）
DEFAULT_MCTS_ACTOR_CRITIC_WEIGHT = 1.0
# 既定のカリキュラム warmup epoch 数（機構L・0 で無効＝RL損失重みを epoch0 から定数＝従来挙動）
DEFAULT_CURRICULUM_WARMUP_EPOCHS = 0
# 既定の value 先行割合（warmup の何割で value_weight が full に達するか＝critic を actor に先行させる）
DEFAULT_CURRICULUM_VALUE_LEAD_FRAC = 0.5
# 既定の探索ネット LR スケール（1.0 で無効＝policy/value も主 LR と同一の単一 param group＝従来挙動）
DEFAULT_SEARCH_LR_SCALE = 1.0
# 既定の warm-start チェックポイント（機構M・None で無効＝差別化版背骨の流用なし＝従来挙動）
DEFAULT_WARM_START_CHECKPOINT = None
# 既定の背骨凍結 epoch 数（機構M・0 で無効＝凍結相なし）
DEFAULT_FREEZE_BACKBONE_EPOCHS = 0
# 既定の解凍後 LR スケール（0 で恒久凍結＝背骨は最後まで固定，>0 で相転移後に背骨を base×scale で共適応）
DEFAULT_UNFREEZE_LR_SCALE = 0.0
# 既定の探索カリキュラム warmup epoch 数（機構C・0 で無効＝探索予算を最初から full）
DEFAULT_SEARCH_WARMUP_EPOCHS = 0
# 既定の探索 warmup 中の模擬予算（warmup 中は depth=1＋この sim で探索を安くする）
DEFAULT_SEARCH_WARMUP_SIMS = 8
# 既定の bag 表現 mixup の Beta 形状 α（0 で無効＝従来挙動）
DEFAULT_MIXUP_ALPHA = 0.0
# 既定のバランスサンプラ温度（1.0 で現行＝重み不変）
DEFAULT_SAMPLER_TEMP = 1.0
# 既定の ordinal 補助損失の重み（0 で無効＝寄与なし）
DEFAULT_ORDINAL_AUX_WEIGHT = 0.0


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
        proj_num_layers: 特徴射影の段数（1 で従来の浅い 1 段）
        proj_layer_norm: 特徴射影の各 Linear 直後に LayerNorm を挟むか
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
        mixup_alpha: bag 表現 mixup の Beta 形状 α（0 で無効＝従来挙動，バッチサイズ 1 のため直前サンプルと混ぜる）
        sampler_temp: バランスサンプラ重みの温度（1.0 で現行，``<1`` で緩和 ``>1`` で強調，``is_weighted_sampler`` 時のみ有効）
        ordinal_aux_weight: クラス順序を活かす ordinal 補助損失の重み（0 で無効，クラス index の並びを順序とみなす）
        ordinal_class_order: ordinal で順序を課すクラス index の昇順列（None で全クラス index 順，部分集合を与えると集合内のみ順序ペナルティ・集合外は名義）
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
        mcts_value_target: 価値回帰目標の作り方（``"realised"`` で従来＝最終 CE を全状態へ broadcast，``"leaf_ce"`` で選択 j の結果状態を含む融合の負分類損失を状態依存リターンにする，``mcts`` のみ有効）
        mcts_rollout_depth: 葉評価で展開する rollout 深さ（``1`` で従来＝選んだ親の子を 1 段射影して評価し更に展開しない，``>1`` で子を更に次倍率へ再帰展開し最深状態を葉評価にする，``mcts`` のみ有効）
        mcts_rollout_simulations: rollout 各段の入れ子プランナ模擬予算（``None`` で ``mcts_simulations`` に一致＝従来挙動，``rollout_depth>1`` のときのみ意味を持ち入れ子探索の予算を最上層と分離して設定する，``mcts`` のみ有効）
        mcts_eval_stochastic: 葉評価を確率的にするか（``False`` で従来＝eval モード葉評価＋報酬 memoize で simulation 間同値，``True`` で MC dropout＋memoize 撤廃で simulation 間に分散を出す，``mcts`` のみ有効）
        mcts_actor_critic_weight: ``leaf_ce`` の actor-critic 項スケール（正規化 advantage × 選択 log 確率を方策蒸留へ上乗せする重み，``0`` で actor-critic 無効＝状態依存 value は価値回帰のみ残る，``mcts`` のみ有効）
        curriculum_warmup_epochs: 機構 L のカリキュラム warmup epoch 数（``0`` で無効＝RL 損失重みを epoch0 から定数＝従来挙動，``>0`` で value→policy→actor-critic の順に重みを 0 から目標へ ramp し分類器の報酬が安定してから探索の学習を立ち上げる，``mcts`` のみ有効）
        curriculum_value_lead_frac: value_weight が full に達する warmup 内の割合（``0.5`` で warmup の半分で value が full＝critic を actor に先行させ actor-critic 項はそこから ramp 開始，``curriculum_warmup_epochs>0`` のときのみ意味を持つ）
        search_lr_scale: 探索ネット（policy/value）専用 param group の LR 倍率（``1.0`` で無効＝単一 param group で従来挙動，``<1`` で探索ネットだけ学習を遅くし共有ヘッド・融合の立ち上がりを先行させる）
        warm_start_checkpoint: 機構M の背骨 warm-start チェックポイント（``None`` で無効，差別化版 best の ``.pt`` パス・``{fold}`` プレースホルダで fold 別に展開し背骨 projections/aggregators/head/aux_attentions を流用する search net は新規・CV リーク防止のため fold 一致必須，``mcts`` のみ有効）
        freeze_backbone_epochs: 機構M の背骨凍結 epoch 数（``0`` で無効，``>0`` で先頭 N epoch は背骨を requires_grad=False＋eval 固定し探索ネットのみ学習＝報酬 -CE を定常化する，``warm_start_checkpoint`` と併用）
        unfreeze_lr_scale: 機構M の解凍後 LR スケール（``0`` で恒久凍結＝背骨は最後まで固定，``>0`` で ``freeze_backbone_epochs`` 後に背骨を base×scale の低 LR で共適応させる）
        search_warmup_epochs: 機構C の探索カリキュラム warmup epoch 数（``0`` で無効＝探索予算を最初から full，``>0`` で先頭 N epoch は depth=1＋``search_warmup_sims`` の安い探索にし後で full へ昇格＝探索ネット未学習の序盤の重い探索を省く，L/M と独立で warm-start 有無に依らず効く，``mcts`` のみ有効）
        search_warmup_sims: 機構C の探索 warmup 中の模擬予算（``search_warmup_epochs>0`` の warmup 中に使う sim 数・``base`` より小さくして探索を安くする，``mcts`` のみ有効）
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
    proj_num_layers: int = DEFAULT_PROJ_NUM_LAYERS
    proj_layer_norm: bool = DEFAULT_PROJ_LAYER_NORM
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
    mixup_alpha: float = DEFAULT_MIXUP_ALPHA
    sampler_temp: float = DEFAULT_SAMPLER_TEMP
    ordinal_aux_weight: float = DEFAULT_ORDINAL_AUX_WEIGHT
    ordinal_class_order: Optional[List[int]] = None

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
    mcts_value_target: str = DEFAULT_MCTS_VALUE_TARGET
    mcts_rollout_depth: int = DEFAULT_MCTS_ROLLOUT_DEPTH
    mcts_rollout_simulations: Optional[int] = DEFAULT_MCTS_ROLLOUT_SIMULATIONS
    mcts_eval_stochastic: bool = DEFAULT_MCTS_EVAL_STOCHASTIC
    mcts_actor_critic_weight: float = DEFAULT_MCTS_ACTOR_CRITIC_WEIGHT
    curriculum_warmup_epochs: int = DEFAULT_CURRICULUM_WARMUP_EPOCHS
    curriculum_value_lead_frac: float = DEFAULT_CURRICULUM_VALUE_LEAD_FRAC
    search_lr_scale: float = DEFAULT_SEARCH_LR_SCALE
    warm_start_checkpoint: Optional[str] = DEFAULT_WARM_START_CHECKPOINT
    freeze_backbone_epochs: int = DEFAULT_FREEZE_BACKBONE_EPOCHS
    unfreeze_lr_scale: float = DEFAULT_UNFREEZE_LR_SCALE
    search_warmup_epochs: int = DEFAULT_SEARCH_WARMUP_EPOCHS
    search_warmup_sims: int = DEFAULT_SEARCH_WARMUP_SIMS
