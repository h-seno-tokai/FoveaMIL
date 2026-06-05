# FoveaMIL

病理 WSI（Whole Slide Image）向けの多重解像度 Multiple Instance Learning．低倍率で広く薄く見て，
注視すべき少数パッチだけを高倍率で精査する中心窩（fovea）型のパッチ選定を核にする
（病理医の弱拡大 → 強拡大に対応）．実装は多解像度のアテンション pooling と，次の倍率へズームする
パッチを選ぶ微分可能 top-k からなる（[models](src/foveamil/models/README.md)）．WSI とスライド単位
ラベルさえあれば，任意の N クラス分類タスクで学習・評価・可視化まで一通り回せる．

## パイプライン全体像

```
                       cohort（labels / splits）
                                │
  WSI ─► 座標抽出 ─► 特徴抽出 ─► (SSD ステージング) ─► 学習 / sweep ─► 評価 ─► 可視化
         coords     features      stage               train / sweep    eval     visualize
```

各段は独立した console script で，前段の出力を次段が読む（H5・CSV・JSON でやり取りする）．
パス・データ・認証情報はすべて引数か環境変数で受け，コードにハードコードしない．

| 段 | console script | 入力 → 出力 | 詳細 |
|---|---|---|---|
| コホート | `foveamil-cohort` | master ラベル → 絞り込みラベル CSV・層化 CV split | [cohort](src/foveamil/cohort/README.md) |
| 座標抽出 | `foveamil-coords` | WSI → 多重解像度の組織パッチ座標 H5 | [preprocessing](src/foveamil/preprocessing/README.md) |
| 特徴抽出 | `foveamil-features` | 座標 H5 ＋ WSI → エンコーダ特徴 H5 | [preprocessing](src/foveamil/preprocessing/README.md) |
| ステージング | `foveamil-stage` | NAS の特徴 → ローカル SSD（任意・I/O 負荷低減） | [training](src/foveamil/training/README.md) |
| 学習 | `foveamil-train` | 特徴 ＋ split → 学習済み重み・予測・指標 | [training](src/foveamil/training/README.md) |
| 多実験一括 | `foveamil-sweep` | 1 YAML → 多 combo × fold を並列実行 | [training](src/foveamil/training/README.md) |
| 評価 | `foveamil-eval` | 保存済み予測 → 図・有意差検定・レポート（再学習なし） | [evaluation](src/foveamil/evaluation/README.md) |
| 可視化 | `foveamil-visualize` | 保存済み予測・重み → アテンション図（再学習なし） | [visualization](src/foveamil/visualization/README.md) |
| 補助 | `foveamil-notify` | 件名・本文 → 完了/エラーのメール通知 | [utils](src/foveamil/utils/README.md) |

## ディレクトリ構造

```
FoveaMIL/
├── src/foveamil/        # パッケージ本体
│   ├── wsi/             # WSI アクセス層（パス解決・倍率/レベル・組織マスク・ステージング）
│   ├── preprocessing/   # 座標抽出・特徴抽出（foveamil-coords / foveamil-features）
│   ├── encoders/        # パッチ特徴抽出器（ResNet50 / UNI2-h / Virchow / Virchow2 / mini）
│   ├── models/          # 多解像度 MIL の部品（attention / topk / fusion / heads / mil）
│   ├── training/        # データセット・学習ループ・交差検証・sweep（train / stage / sweep）
│   ├── evaluation/      # 指標・図・有意差検定レポート（foveamil-eval）
│   ├── visualization/   # アテンション可視化（foveamil-visualize）
│   ├── cohort/          # ラベル絞り込み・層化 CV split 生成（foveamil-cohort）
│   └── utils/           # メール通知・メモリ・再現情報
├── cohort/              # コホート定義の置き場（labels / splits・実データは含まれない）
├── configs/             # 学習・sweep の設定 YAML 例
├── scripts/             # console script の薄いラッパ（.sh・再開/再試行/通知）
├── environments/        # conda 環境定義（cu118 / cu128）
├── pretrained/          # 学習済み重みの置き場
├── tests/               # ユニットテスト
└── pyproject.toml       # パッケージ・依存・console script 定義
```

各ディレクトリに README を置き，そのディレクトリ固有の事実（仕様・引数・出力）を書く．
全体像はこの README が，部品の詳細は各 README が責任を持つ．

## セットアップ

GPU 世代に応じて CUDA 版ごとの conda 環境を使う（詳細 [environments](environments/README.md)）．

```bash
conda env create -f environments/environment-cu118.yml   # Pascal / Ampere
conda activate foveamil-cu118
pip install -e .

cp .env.example .env      # WSI_BASE_PATH 等を自分の環境に合わせて編集
```

環境変数（`.env`）は WSI 置き場・特徴ルート・メール認証・ステージング先などを与える
（雛形 [.env.example](.env.example)）．

## クイックスタート

以下は一連の流れの例（パス・設定はプレースホルダ）．`configs/` の YAML と `cohort/` のラベルは
自分のタスク（クラス数・倍率・fold）に合わせて編集する．

```bash
# 1. 層化 CV split を生成（自分のラベル CSV から）
foveamil-cohort splits --labels cohort/labels/labels_3class.csv \
    --output-dir cohort/splits/3class/cv10 --k 10 --seed 42

# 2. 多重解像度の組織パッチ座標を抽出
foveamil-coords --slides cohort/labels/labels_3class.csv \
    --out /path/to/coords --mags 1.25 2.5 5.0 10.0 20.0 40.0

# 3. エンコーダ特徴を抽出
foveamil-features --encoder Virchow2 --coords-dir /path/to/coords \
    --out /path/to/features --slides cohort/labels/labels_3class.csv \
    --mags 1.25 2.5 5.0 10.0 20.0 40.0

# 4. 特徴をローカル SSD へ事前ステージ（任意・I/O 負荷低減）
foveamil-stage --feature-root /path/to/features --encoder Virchow2 \
    --magnifications 1.25 2.5 5.0 --splits-dir cohort/splits/3class/cv10

# 5. 多実験 sweep（resolve の n_cls / folds / feature_root と sweep 軸を YAML で指定）
foveamil-sweep --config configs/sweep.example.yaml \
    --out /path/to/out --weights-out /path/to/weights

# 6. 評価レポート（再学習なし）
foveamil-eval --in /path/to/out --split test

# 7. アテンション可視化（再学習なし）
foveamil-visualize overview --sweep-root /path/to/out \
    --feature-root /path/to/features --weights-root /path/to/weights \
    --out-dir /path/to/viz --per-class 1 --split test
```

単一学習は `foveamil-train`（[configs](configs/README.md)）．本番運用では `scripts/` の薄いラッパが
再開・再試行・複数 GPU 動的割当・通知を備える（[scripts](scripts/README.md)）．

## データについて

本リポジトリにコホートの実データは含まれない．`cohort/labels/labels.template.csv` と
`cohort/splits/split.template.csv` に倣って，スライド単位ラベル（`slide_id,label`）と交差検証 split を
自分のデータで用意すれば，そのまま学習・評価できる（[cohort](cohort/README.md)）．WSI は `slide_id` と
環境変数 `WSI_BASE_PATH` から解決するため，パス一覧ファイルは要らない．

## テスト

```bash
pip install -e ".[dev]"
pytest tests/
```

CI（GitHub Actions [.github/workflows/tests.yml](.github/workflows/tests.yml)）は依存の重さで二段に分ける．
`core` はシステムライブラリ不要（pip wheel のみ）で非 WSI テストを回し，`full` は OpenSlide を入れて
全テストを回す．
