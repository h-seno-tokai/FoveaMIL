# `foveamil.training` — 学習用の特徴量データセット・ステージング・学習ループ

学習に用いる多倍率の特徴量バッグを供給する層と，それを使った学習ループ・評価指標・
交差検証を提供する．特徴は正準レイアウト
`{feature_root}/{encoder}/{mag}x/{slide_id}.h5` に倍率ごと 1 ファイルで置かれている前提．
ネットワーク越しの NAS は遅いので，学習前に `foveamil-stage` で特徴量セットをローカル SSD へ
一括コピーしておき，学習はそのルートをそのまま読む（学習コード自体は自動ステージしない）．
ログ・結果は home 側（`save_path`），重み（`.pt`）は Dataset 側（`weights_dir`）へ分けて保存する．

## モジュール

| ファイル | 役割 |
|---|---|
| `dataset.py` | 多倍率特徴バッグを返す `FeatureBagDataset`，ラベル辞書ヘルパ `build_label_dict`． |
| `staging.py` | 特徴量セットを SSD へ一括コピーする `FeatureStager`（`foveamil-stage` が使う）． |
| `stage_cli.py` | 特徴セットを事前ステージする `foveamil-stage` コマンド． |
| `config.py` | 学習 1 回分の設定をまとめた dataclass `TrainConfig`． |
| `metrics.py` | 予測を蓄積して分類指標を集計する `MetricLogger`． |
| `saver.py` | 検証指標の改善時にモデル重みを保存する `ModelSaver`． |
| `trainer.py` | 学習・検証・評価を司る `Trainer`． |
| `cv.py` | 単一 fold 実行 `run_fold`，交差検証 `run_cross_validation`，fold 集計 `aggregate_folds`． |
| `hierarchy.py` | 親→子 index 計算 `compute_child_indices`，倍率比から子数 `children_per_parent`，倍率列の検証 `validate_magnification_hierarchy`． |
| `resolve.py` | sweep のパス・特徴次元・倍率を解決する（`resolve_paths`, `resolve_in_feat_dim`, `normalize_mags`）． |
| `sweep.py` | sweep 設定を combo へ展開し fold 並列で実行する `expand_combos` / `SweepRunner`． |
| `sweep_cli.py` | 多実験一括の `foveamil-sweep` コマンド． |

## `config.py` — `TrainConfig`

学習 1 回分の設定をフィールドに持つ dataclass．最適化（`lr` `reg`
`scheduler_decay_rate` `scheduler_patience`）・データ（`feature_root` `encoder`
`labels_csv` `magnifications` `feature_type` `classes`）・モデル（`in_feat_dim`
`hidden_feat_dim` `out_feat_dim` `drop_out` `k_sample` `k_sigma` `topk_method`
`fusion` `n_cls`）・学習インタフェース（`is_weighted_sampler` `num_workers`
`pin_memory` `save_metric`）を保持する．`num_layers` は
`magnifications` の長さから導かれ，`n_cls` は `classes` やラベル辞書から上書きされる．
`magnifications` は **昇順かつ隣接比が 2 のべき**なら任意の組を許す（`[2.5, 10, 40]` や
`[1.25, 2.5, 5, 40]` 等の飛ばし組も可比 `r` の段では親 1 つが `r^2` 個の子へ展開される）．
単一倍率（長さ 1）はズーム無しの attention pooling のみになる`Trainer` 構築時に
`validate_magnification_hierarchy` で検証する．
`k_sigma` は `topk_method` に応じて top-k セレクタへ渡す平滑化引数になる（`perturbed`
なら `sigma`，`fast_sparse` なら `epsilon`）．

## `metrics.py` — `MetricLogger`

`log(Y_hat, Y, Y_prob=None)` で 1 サンプルずつ予測クラス・正解クラス（任意で確率）を
蓄積し，`get_summary()` で accuracy / weighted・macro F1 / weighted・macro
precision・recall / kappa（二次重み付け）/ per-class f1・precision・recall を返す．
確率が与えられていれば OvR per-class AUC（両クラスが揃う場合のみ），多クラスでは
macro・weighted AUC も加える．`get_confusion_matrix()` で混同行列を返す．数値が安定して
出ない指標は安全に省きログに残す．

## `saver.py` — `ModelSaver`

`ModelSaver(save_path, save_metric, weights_dir=None)` の `__call__(model, {"val_loss":..,
"val_weighted_f1":..})` で，`save_metric="loss"` なら検証損失最小，`"f1"` なら検証
weighted F1 最大を更新したときに `model_best_{metric}.pt` を保存する．`save_model(model,
suffix)` で任意接尾辞の保存，`load_best_path()` で best 重みのパス（無ければ `None`）を返す．
重み（`.pt`）の保存・読み込みは `weights_dir` に対して行う（未指定なら `save_path` に
フォールバック）．

## `trainer.py` — `Trainer`

`Trainer(config, train_ds, val_ds, test_ds, save_path, weights_dir=None)` でモデル・最適化器・
スケジューラ・保存器を構築する．ログ・結果（tensorboard・混同行列）は `save_path`（home 側），
重み（`.pt`）は `weights_dir`（Dataset 側，未指定なら `save_path`）へ分けて保存する．
`build_topk`/`build_fusion` を介して `FoveaMIL` を組み，
バッチサイズ 1 の `DataLoader` を用いる．train は `is_weighted_sampler` ならクラス頻度の
逆数で重み付けした `WeightedRandomSampler`，そうでなければ `RandomSampler`，val/test は
`SequentialSampler`．`Adam(betas=(0.9,0.999), eps=1e-8, weight_decay=reg)` と
`ReduceLROnPlateau(mode='min')` を用いる．

- `train()`: 最大エポック数まで train→val を回し，検証損失で `scheduler.step` し，
  `ModelSaver` で best を更新する．最後に `model_last.pt` を保存する．
- `test()`: best 重みを読み直して（無ければ現状のまま）test 指標辞書を返し，混同行列を
  `confusion_matrix.npy` に保存する．

tensorboard と混同行列 PNG（matplotlib）は任意機能で，依存が無ければ自動的に省く（強依存に
しない）．有効時のみ scalar/PNG を出力する．

```python
from foveamil.training import Trainer, TrainConfig, FeatureBagDataset, build_label_dict

config = TrainConfig(
    feature_root="/path/to/features", encoder="ResNet50",
    labels_csv="cohort/labels/labels_3class.csv",
    magnifications=[1.25, 2.5, 5.0, 10.0], in_feat_dim=1024, n_cls=3,
)
label_dict = build_label_dict(config.labels_csv, classes=config.classes)
# ... train_ds / val_ds / test_ds を構築 ...
trainer = Trainer(
    config, train_ds, val_ds, test_ds,
    save_path="/tmp/run0", weights_dir="/tmp/run0_weights",
)
trainer.train()
metrics = trainer.test()
```

## `cv.py` — `run_fold`, `run_cross_validation`

`run_fold(config, split_csv, save_path, weights_dir=None)` は `train` / `val` / `test` 列を
持つ分割 CSV を読み，各列の slide_id（`dropna`）から 3 分割のデータセットを作り，
`Trainer` で学習・評価して test 指標辞書を返す．特徴は `config.feature_root` をそのまま
読む（事前に `foveamil-stage` でステージ済みである前提）．ログ・結果は `save_path`（home 側），
重み（`.pt`）は `weights_dir`（Dataset 側，未指定なら `save_path`）へ保存する．

`run_cross_validation(config, split_paths, save_root, weights_root=None)` は各 fold を
ログ・結果 `{save_root}/fold{i}`（home 側）・重み `{weights_root}/fold{i}`（Dataset 側，
未指定なら `save_root`）で実行し，主要指標（accuracy / weighted_f1 / macro_f1 / kappa /
macro_auc）の fold 間 mean±std を計算する．per-fold の指標も保持し，集計を
`{save_root}/cv_summary.json`（home 側）に保存する．集計そのものは `aggregate_folds(per_fold)`
（純関数）が担い，`foveamil-sweep` も同じ関数で combo の CV を集計する．

```python
from foveamil.training import run_cross_validation, TrainConfig

config = TrainConfig(...)
summary = run_cross_validation(
    config, ["fold0.csv", "fold1.csv"], "/tmp/cv", weights_root="/tmp/cv_weights"
)
# summary["aggregate"]["weighted_f1"] == {"mean": ..., "std": ...}
```

### 事前ステージ（`foveamil-stage`）

学習コードは特徴を自動ステージしない．NAS への同時アクセスを減らすには，学習前に一度
`foveamil-stage` で対象スライドの特徴をローカル SSD へ一括コピーし，表示された staged root を
`feature_root` に指定する．既存ファイルは再利用するため再実行は冪等．SSD に収まらない場合は
警告を出して元の `feature_root` を返す（NAS 直読）．キャッシュ先は `--cache-dir` または環境変数
`FOVEAMIL_STAGE_DIR` で指定できる．

```bash
foveamil-stage \
    --feature-root /path/to/features --encoder ResNet50 \
    --magnifications 1.25 2.5 5.0 --splits-dir /path/to/splits \
    --cache-dir /tmp/foveamil_feat_stage
```

## `resolve.py` / `sweep.py` — `foveamil-sweep`（多実験一括）

1 個の YAML（`resolve` / `sweep` / `fixed` / `parallel`）で多数の実験を一括実行する．

`resolve.py` は解決の単一責任を負う．`resolve_paths(n_cls, folds, cohort_root,
feature_root_base)` が labels CSV（`labels_{N}class.csv`）と splits（`{N}class/cv{folds}/`）を
解決し存在を検証する（`folds` は `5` / `10` のみ）．`resolve_in_feat_dim(encoder,
feature_type)` がエンコーダのクラス属性 `feature_dim` を真実源に入力特徴次元を解決する
（`concat` は 2 倍）．`normalize_mags` が `"1.25x"` / `1.25` を float へ正規化する．

`sweep.py` の `expand_combos(sweep, fixed, resolved)` は `(encoder, feature_type)` を制約付き
join（`cls` / `concat` は `has_cls=True` のみ`ResNet50` は `mean` のみ），他の軸を `ParameterGrid`
で直積展開し，各 combo に解決済み `in_feat_dim` / `feature_root` / `labels_csv` / `n_cls` を載せる．
`SweepRunner` は `(combo, fold)` をジョブにフラット化して `foveamil-train --split` のサブプロセス
として GPU へ割り当て並列実行する．`test_metrics.json` が既にある fold はスキップ（resume），
fold 失敗は記録して継続する．**combo ランキングは validation 指標で行い，その test を報告する**
（test 1 位は oracle 上限として別途併記＝楽観バイアス回避）．combo ごとに val/test の per-fold＋
mean±std＋信頼区間を `cv_summary.json` に，sweep ルートに `sweep_summary.{json,md}`（val 選定
best＋その test・oracle）と `sweep_detailed.csv`（全 combo×fold×split）を出す．
重みは `{weights_root}/{combo_name}/fold{i}`．CLI は `foveamil-sweep`（`--dry-run` で展開だけ確認）．

## 学習時の保存物（再学習なしで何でも再生成できる）

各 fold（`{save_path}/`）に，`Trainer.evaluate_best` が best-val 重みで val/test を評価して
生データを残す: `predictions_{val,test}.csv`（slide_id・正解・予測・全クラス確率・logit），
`history.csv`（per-epoch），`metrics_{val,test}.json`，混同行列（生/正規化・クラス名）．
`run_fold` が後方互換の `test_metrics.json` と再現情報 `run_meta.json`（config・seed・環境・
git・データ指紋・split 別クラス内訳）を書く．`metrics.py` の指標は accuracy / balanced /
F1 / precision・recall / specificity・sensitivity / kappa / MCC / AUC(OvR・OvO) / AUPRC を
含む．保存済み予測からの図・有意差検定・レポートは `foveamil.evaluation`（`foveamil-eval`）が担う．

## `dataset.py` — `FeatureBagDataset`, `build_label_dict`

`FeatureBagDataset` は倍率ごとに特徴 H5 を読み，倍率テンソルとラベル整数のタプル
`(feat_mag0, feat_mag1, ..., label_int)` を返す．`magnifications` は低→高の順を保ち，
これが倍率レイヤ順になる．`DataLoader` を `batch_size=1` で回すと各特徴は `[1, N_i, dim]`
となり，`FoveaMIL.forward(x)`（倍率ごとのテンソルのタプルを受ける）にそのまま渡せる．

`feature_type` は次の 3 種から選ぶ．

- `"mean"`: pooled 特徴（dataset `patches`）`[N, dim]`．
- `"cls"`: cls 特徴（dataset `patches_cls`）`[N, dim]`．`has_cls=True` のエンコーダのみ．
- `"concat"`: pooled と cls を特徴次元で連結 `[N, 2*dim]`．

`build_label_dict(labels_csv, classes=None)` は `slide_id,label` の CSV から
クラス名→整数の辞書を作る．`classes` 未指定なら CSV のユニークな `label` をソートして
`0..K-1` を割り当て，指定時はその順序で割り当てる．`label_dict` に無いラベルの行は
データセット構築時に除外する．

```python
from foveamil.training import FeatureBagDataset, build_label_dict

label_dict = build_label_dict("cohort/labels/labels_3class.csv")
ds = FeatureBagDataset(
    feature_root="/path/to/features",
    encoder="ResNet50",
    magnifications=[1.25, 2.5, 5.0, 10.0],
    slide_ids=train_ids,
    labels_csv="cohort/labels/labels_3class.csv",
    label_dict=label_dict,
    feature_type="mean",
)
*feats, label = ds[0]   # feats[i]: [N_i, dim], label: int
```

## `staging.py` — `FeatureStager`

`stage_set(feature_root, encoder, magnifications, slide_ids)` で対象 slide_id × 倍率の
特徴 H5（実在するもの）をローカルキャッシュへ一括コピーし，新しいルート（キャッシュ先）を
返す．コピー前に必要容量と SSD 空き容量を確認し，必要容量が空きの `1 - free_space_margin`
以内なら `{cache_dir}/{encoder}/{mag}x/` 構造へアトミックにコピーする（既存ファイルは再利用）．
収まらない場合は警告を出し，元の `feature_root` を返す（NAS 直読フォールバック・学習は遅くなる）．

キャッシュ先は環境変数 `FOVEAMIL_STAGE_DIR` で指定でき，未指定時は `/tmp` 配下の
プロセス固有ディレクトリを使う．`cleanup()` でステージングディレクトリを削除する．
context manager に対応しており，`with` ブロックを抜けると自動で `cleanup()` する．

```python
from foveamil.training import FeatureStager, FeatureBagDataset

with FeatureStager(free_space_margin=0.1) as stager:
    root = stager.stage_set("/path/to/features", "ResNet50", [1.25, 2.5], train_ids)
    ds = FeatureBagDataset(
        feature_root=root, encoder="ResNet50", magnifications=[1.25, 2.5],
        slide_ids=train_ids, labels_csv="cohort/labels/labels_3class.csv",
        label_dict=label_dict, feature_type="mean",
    )
    # ... 学習 ...
```

返ったルートを `FeatureBagDataset` の `feature_root` に渡す運用が主．補助的に
`localize(path)` で単一ファイルをキャッシュへコピーして読むこともでき，
`FeatureBagDataset(stager=...)` に渡すと `__getitem__` がファイル単位でキャッシュ経由になる．
