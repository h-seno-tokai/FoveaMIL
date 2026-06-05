# scripts

## extract_features.sh

`foveamil-features` の薄いラッパ（単一エンコーダ）．引数を組み立てて 1 回呼ぶだけで，
複数 GPU への動的割当・既存出力のスキップ再開・要約メールは core が担う．
パス・GPU は全て引数/環境変数で受ける．

### 引数

| 引数 | 必須 | 説明 |
|------|------|------|
| `--encoder NAME` | 必須 | エンコーダ名（`ResNet50` / `UNI2-h` / `Virchow` / `Virchow2` / `Virchow2-mini-dinov2`） |
| `--coords-dir DIR` | 必須 | 座標 H5 のディレクトリ（`{mag}x/{slide_id}.h5`） |
| `--out DIR` | 必須 | 特徴 H5 の出力ルート（`{out}/{encoder}/{mag}x/{slide_id}.h5`） |
| `--slides PATH` | どちらか一方 | slide_id ファイル（CSV/テキスト）．`WSI_BASE_PATH` でパス解決 |
| `--wsi-dir DIR` | どちらか一方 | ディレクトリ内の対応 WSI を全件処理 |
| `--mags "1.25 2.5 ..."` | 任意 | 倍率列（未指定時は既定の倍率列） |
| `--batch-size N` | 任意 | 推論バッチサイズ（未指定時は `PREPROCESS_BATCH_SIZE` か既定値） |
| `--num-workers N` | 任意 | GPU ワーカあたりのパッチ I/O スレッド数（未指定時は `PREPROCESS_NUM_WORKERS` か既定値） |
| `--gpu-ids "0,1,2"` | 任意 | スライドを動的割当する物理 GPU（未指定なら可視 GPU 全て，無ければ CPU） |
| `--skip-background` | 任意 | 背景パッチの順伝播を省きダミー特徴で埋める |
| `--stage` | 任意 | WSI をローカル SSD に退避してから読む |
| `--overrides CSV` | 任意 | `slide_id,path` のパス対応表 |
| `--wsi-base-path DIR` | 任意 | slide_id 解決ルート（`:` 区切りで複数可） |
| `--notify` | 任意 | 開始/完了/エラーの要約メール |

`--slides` と `--wsi-dir` は排他．**再開は自動**で，全 `--mags` の出力が揃ったスライドと
既存倍率を core がスキップするので途中中断後も同じコマンドで再開できる（専用フラグ不要）．
複数 GPU は `--gpu-ids` で指定すると各 GPU が空き次第スライドを取りに行く（動的割当）ため
スライドごとの計算量がばらついても GPU が遊ばない．
バッチは大きくしても推論スループットは頭打ちになりやすいので，モデル規模に応じ VRAM に収まる値にする
（大型 ViT は控えめ，小型は大きく可）．

### 環境変数

| 変数 | 用途 |
|------|------|
| `WSI_BASE_PATH` | `--slides` 使用時の slide_id 解決ルート |
| `PREPROCESS_BATCH_SIZE` | `--batch-size` 未指定時の既定上書き |
| `PREPROCESS_NUM_WORKERS` | `--num-workers` 未指定時の既定上書き |
| `FOVEAMIL_STAGE_DIR` | `--stage` 使用時のローカル退避先（未設定なら `/tmp` 配下） |
| `CUDA_VISIBLE_DEVICES` | `--gpu-ids` 未指定時に使う可視 GPU の集合 |
| `VIRCHOW_MINI_CHECKPOINT` | `Virchow2-mini` の重みチェックポイント |
| `GMAIL_USER` / `GMAIL_APP_PASSWORD` / `RECEIVE_USER` | `--notify` 使用時の認証情報 |

### 実行例

```bash
# slide_id ファイルから解決して ResNet50 で抽出（通知あり）
./scripts/extract_features.sh \
    --encoder ResNet50 \
    --coords-dir /path/to/coords \
    --out /path/to/features \
    --slides /path/to/slides.csv \
    --mags "1.25 2.5 5.0 10.0 20.0 40.0" \
    --num-workers 8 \
    --notify

# 本番: 3 GPU へ動的割当 + 背景スキップ + SSD 退避（途中中断後も同じコマンドで再開）
./scripts/extract_features.sh \
    --encoder Virchow2 \
    --coords-dir /path/to/coords \
    --out /path/to/features \
    --slides /path/to/slides.csv \
    --gpu-ids "0,1,2" --skip-background --stage \
    --batch-size 2048 --num-workers 8 --notify
```

## extract_coords.sh

`foveamil-coords` を本番運用するラッパ（座標抽出）．既存出力のスキップ再開・異常終了時の再試行・
要約メールを持つ．座標抽出は CPU（`--num-workers` で並列）のため GPU 分割は持たない．

### 引数

| 引数 | 必須 | 説明 |
|------|------|------|
| `--out DIR` | 必須 | 座標 H5 の出力ディレクトリ（`{out}/{mag}x/{slide_id}.h5`） |
| `--slides PATH` | どちらか一方 | slide_id ファイル（CSV/テキスト）．`WSI_BASE_PATH` でパス解決 |
| `--wsi-dir DIR` | どちらか一方 | ディレクトリ内の対応 WSI を全件処理（`--resume` 非対応） |
| `--mags "1.25 2.5 ..."` | 任意 | 倍率列（未指定時は既定の倍率列） |
| `--num-workers N` | 任意 | 並列ワーカー数（未指定時は既定値） |
| `--resume` | 任意 | 全 `--mags` の座標が揃う slide を除外して残りだけ処理する |
| `--retries N` | 任意 | 異常終了時の再試行回数（既定 2．再試行は残り slide のみ） |
| `--notify` | 任意 | 完了・エラー時に要約メールを送る |

`--slides` と `--wsi-dir` は排他．`--resume` は `--slides` 指定時のみ有効．

### 環境変数

| 変数 | 用途 |
|------|------|
| `WSI_BASE_PATH` | `--slides` 使用時の slide_id 解決ルート |
| `GMAIL_USER` | `--notify` 使用時の送信元アドレス |
| `GMAIL_APP_PASSWORD` | `--notify` 使用時のアプリパスワード |
| `RECEIVE_USER` | `--notify` 使用時の宛先アドレス |

### 環境変数の読み込み

`.env` を用意し，`direnv` を使うか以下のように読み込む．

```bash
set -a
source .env
set +a
```

### 実行例

```bash
# slide_id ファイルから解決して抽出（通知あり）
./scripts/extract_coords.sh \
    --out /path/to/output \
    --slides /path/to/slides.csv \
    --mags "1.25 2.5 5.0 10.0" \
    --num-workers 8 \
    --notify

# ディレクトリ内の WSI を全件処理（通知なし）
./scripts/extract_coords.sh \
    --out /path/to/output \
    --wsi-dir /path/to/wsi
```

## train.sh

`foveamil-train` の引数を組み立てて学習を実行する薄いラッパ．SMTP/メールロジックは
持たず，`--notify` を `foveamil-train` にそのまま渡す（通知は Python 側の責任）．

### 引数

| 引数 | 必須 | 説明 |
|------|------|------|
| `--config PATH` | 必須 | 学習設定 YAML |
| `--out DIR` | 必須 | 結果の出力ルート |
| `--split PATH` | どちらか一方 | 単一 fold の分割 CSV（train/val/test 列） |
| `--splits-dir DIR` | どちらか一方 | `split_fold*.csv` のディレクトリ（交差検証） |
| `--folds "1,2,3"` | 任意 | `--splits-dir` 使用時に対象 fold 番号を絞る |
| `--notify` | 任意 | 開始・完了・エラー時にメールを送る |

`--split` と `--splits-dir` は排他．どちらか一方を指定する．

### 環境変数

| 変数 | 用途 |
|------|------|
| `FOVEAMIL_STAGE_DIR` | `stage: true` 使用時のローカル退避先（未設定なら `/tmp` 配下） |
| `GMAIL_USER` / `GMAIL_APP_PASSWORD` / `RECEIVE_USER` | `--notify` 使用時の認証情報 |

### 実行例

```bash
# 単一 fold の学習
./scripts/train.sh \
    --config configs/train.example.yaml \
    --split /path/to/split_fold0.csv \
    --out /path/to/out

# 交差検証（fold を絞る・通知あり）
./scripts/train.sh \
    --config configs/train.example.yaml \
    --splits-dir /path/to/splits \
    --folds "1,2,3" \
    --out /path/to/out \
    --notify
```

## sweep.sh

`foveamil-sweep` の引数を組み立てて多実験一括を実行する薄いラッパ．SMTP/メールロジックは
持たず，`--notify` を `foveamil-sweep` にそのまま渡す（通知は Python 側の責任）．

### 引数

| 引数 | 必須 | 説明 |
|------|------|------|
| `--config PATH` | 必須 | sweep 設定 YAML（`resolve`/`sweep`/`fixed`/`parallel` の 4 ブロック） |
| `--out DIR` | 必須 | combo のログ・結果の出力ルート（home） |
| `--weights-out DIR` | 任意 | 重み（`.pt`）の出力ルート（Dataset 未指定なら `--out`） |
| `--gpu-ids "0,1"` | 任意 | `parallel.gpu_ids` を上書き |
| `--jobs-per-gpu N` | 任意 | `parallel.jobs_per_gpu` を上書き |
| `--dry-run` | 任意 | 展開結果と job 数・解決値を表示し実行しない |
| `--notify` | 任意 | 開始・完了・エラー時にメールを送る |

`splits` と `n_cls` / `folds` / `feature_root` は YAML の `resolve` ブロックから解決するため
コマンド引数では受けない．

### 環境変数

| 変数 | 用途 |
|------|------|
| `FEATURE_ROOT` | 特徴ルートの base（`resolve.feature_root` を `${FEATURE_ROOT}` で渡す場合） |
| `FOVEAMIL_STAGE_DIR` | 事前ステージ先（`foveamil-stage` 使用時 未設定なら `/tmp` 配下） |
| `GMAIL_USER` / `GMAIL_APP_PASSWORD` / `RECEIVE_USER` | `--notify` 使用時の認証情報 |

### 実行例

```bash
# 展開結果と job 数・解決値だけ確認（実行しない）
FEATURE_ROOT=/path/to/features ./scripts/sweep.sh \
    --config configs/sweep.example.yaml \
    --out /path/to/out \
    --dry-run

# 4 GPU 並列・通知あり（ログ=home, 重み=Dataset）
FEATURE_ROOT=/path/to/features ./scripts/sweep.sh \
    --config configs/sweep.example.yaml \
    --out /path/to/out \
    --weights-out /path/to/weights \
    --gpu-ids "0,1,2,3" \
    --jobs-per-gpu 12 \
    --notify
```

## eval.sh

`foveamil-eval` の引数を組み立てて評価レポートを生成する薄いラッパ．sweep の出力に保存された
予測から **再学習なしで** ROC/PR/キャリブレーション図・combo 間有意差検定・markdown レポートを作る．

### 引数

| 引数 | 必須 | 説明 |
|------|------|------|
| `--in DIR` | 必須 | sweep の出力ルート（`foveamil-sweep --out` に渡したディレクトリ） |
| `--out DIR` | 任意 | レポート出力先（未指定なら `{in}/report`） |
| `--split NAME` | 任意 | report する split（`val`/`test`/`train` 既定 `test`） |
| `--metric NAME` | 任意 | selection/報告指標（既定 `macro_auc`） |
| `--compare "A:B"` | 任意 | combo 名の対を Wilcoxon と Nadeau-Bengio 補正 t で検定（繰り返し可） |
| `--bins N` | 任意 | キャリブレーションの bin 数（既定 10） |
| `--all-combos` | 任意 | 全 combo の図を作る（既定は best のみ） |
| `--no-plots` | 任意 | 図を作らない（指標・レポートのみ） |

combo ランキングは **val 指標で選び test を報告**する方針（test 1 位は oracle 上限として併記）．
出力は `{out}/`（`roc_*.png` / `pr_*.png` / `calibration_*.png` / `significance_*.json` / `report.md`）．

### 実行例

```bash
# best combo の図・ECE・レポート
./scripts/eval.sh --in /path/to/out --split test

# combo 間の有意差検定つき
./scripts/eval.sh --in /path/to/out --split test \
    --compare "combo_000__A:combo_001__B" --metric macro_auc
```

## visualize.sh

`foveamil-visualize` の薄いラッパ．サブコマンド `overview` / `zoom` / `compare` をそのまま渡す．
保存済み予測・重みから WSI に attention を重ねた図を **再学習なしで** 作る（学習はしない）．

### サブコマンド

| view | 内容 |
|------|------|
| `overview` | 倍率 × {主primary, 補助aux} の WSI 全体オーバーレイ格子 |
| `zoom` | 階層ズーム照明（親拡大→子を primary 連続明度で照らす）`--chain` で中心窩経路図 |
| `compare` | 成功 vs 失敗 症例の対比格子 |

### 主な引数

| 引数 | 必須 | 説明 |
|------|------|------|
| `--sweep-root DIR` | 任意 | sweep の出力ルート（best_by_val を解決 `--combo-dir` で上書き可） |
| `--feature-root DIR` | 必須 | `{encoder}/{mag}x/{slide}.h5` のルート |
| `--out-dir DIR` | 必須 | 図の保存先（home） |
| `--weights-root DIR` | 任意 | 重み（`.pt`）のルート（Dataset 側 未指定なら combo 配下） |
| `--coords-root DIR` | 任意 | `actual_max_mag` を取る座標ルート（特徴 H5 には無いため） |
| `--fold N` | 任意 | 可視化に使う fold（既定 1） |
| `--split / --outcome / --slide-id / --per-class / --n / --target-class` | 任意 | 症例選択（成功/失敗は `y_true==y_pred`） |
| `--dry-run` | 任意 | 解決した combo/fold/モデル設定/症例だけ表示し描画しない |

zoom 固有: `--parent-mag` / `--parent-pick {top_aux,top_primary,index}` / `--n-parents` / `--zoom-px` / `--dim-factor` / `--chain`．

### 環境変数

| 変数 | 用途 |
|------|------|
| `WSI_BASE_PATH` | `--wsi-base-path` 未指定時の slide_id→WSI 解決ルート |

### 実行例

```bash
# best combo の成功/失敗症例の overview（クラスごと 1 件）
WSI_BASE_PATH=/path/to/wsi ./scripts/visualize.sh overview \
    --sweep-root /path/to/out --feature-root /path/to/features \
    --weights-root /path/to/weights --coords-root /path/to/coords \
    --per-class 1 --split test --out-dir /path/to/viz

# 特定症例の階層ズーム経路（中心窩）
WSI_BASE_PATH=/path/to/wsi ./scripts/visualize.sh zoom \
    --sweep-root /path/to/out --feature-root /path/to/features \
    --weights-root /path/to/weights --slide-id SAMPLE_0001 --chain \
    --out-dir /path/to/viz
```
