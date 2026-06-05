# configs

学習・sweep の設定 YAML 例と書き方をまとめる．パスはすべてプレースホルダなので
環境に合わせて差し替える．

## train.example.yaml

`foveamil-train` 用の単一学習設定．トップレベルは辞書で，キーは `TrainConfig` の
フィールド名に対応する．未知キーは警告ログを出して無視される．

```bash
# 単一 fold の学習
foveamil-train \
    --config configs/train.example.yaml \
    --split /path/to/split_fold0.csv \
    --out /path/to/out \
    --weights-out /path/to/weights

# 交差検証（split_fold*.csv をまとめて実行，fold は任意で絞る）
foveamil-train \
    --config configs/train.example.yaml \
    --splits-dir /path/to/splits \
    --folds 1,2,3 \
    --out /path/to/out \
    --weights-out /path/to/weights
```

`--out`（home 側）にはログ・config・結果（JSON / tensorboard / 混同行列）を，
`--weights-out`（Dataset 側）にはモデル重み（`.pt`）を分けて保存する．`--weights-out`
未指定なら重みも `--out` に保存する（後方互換）．`feature_root` は事前に `foveamil-stage`
でステージ済みである前提で，学習はそのルートをそのまま読む．

`--split`（単一 fold）と `--splits-dir`（交差検証）は排他．`--splits-dir` では
`split_fold*.csv` を fold 番号順に集め，`--folds` で対象 fold 番号を絞れる．

`--override key=value` で任意の `TrainConfig` フィールドを上書きできる．値は YAML
リテラルとして解釈される（繰り返し可）．

```bash
foveamil-train --config configs/train.example.yaml --split s.csv --out o \
    --override lr=1e-3 --override 'magnifications=[1.25, 2.5]'
```

`--notify` で開始・完了・エラー時に日本語のメールを送る（認証情報は環境変数）．

## sweep.example.yaml

`foveamil-sweep` 用の多実験一括設定．`resolve` / `sweep` / `fixed` / `parallel` の 4 ブロックからなる．

- `resolve`: 解決の起点（スカラ）．`n_cls` から labels CSV と splits ディレクトリ，`folds`
  （`5` か `10` のみ 推奨 `10`）から `cv{folds}/` を解決する．`feature_root` は base のみ与え，
  `encoder` と倍率は自動付与される（`${FEATURE_ROOT}` のように環境変数で渡せる）．
- `sweep`: 展開する軸（リスト）．`encoder` と `feature_type` は直積でなく妥当な組合せのみ
  残す（`cls` / `concat` は cls を持つエンコーダのみ`ResNet50` は `mean` のみ）．`magnifications`
  は list-of-lists で倍率セット自体を軸にできる．他の軸は直積展開する．
- `fixed`: 全 combo 共通のスカラ（`TrainConfig` フィールド）．
- `parallel`: `gpu_ids`（使用 GPU 一覧）・`jobs_per_gpu`（GPU あたり並列数）．

**構成に無関係なパラメータは自動で畳まれ重複 combo は統合される**（直積の無駄を断つ・統合時は警告）．
単一倍率では `k_sample` / `k_sigma` / `topk_method`（ズーム系）が学習に無関係なので畳んで記録しない．
インスタンス補助損失 `instance_loss=true` は単一倍率のみ有効で，多倍率の combo では無効化して統合する
（`bag_weight` / `inst_k` / `inst_subtyping` は `instance_loss=true` のときだけ意味を持ち，無効時は畳む）．
例えば `instance_loss: [false, true]` と `magnifications: [[1.25, 5.0], [40]]` を与えると，多倍率の
`[1.25, 5.0]` は false の 1 件，単一倍率の `[40]` は false/true の 2 件＝計 3 combo に展開される．

`in_feat_dim` / `labels_csv` / 完全形の `feature_root` / `splits_dir` は自動解決されるため
設定に書かない（書くとエラー）．`in_feat_dim` は `encoder` と `feature_type` から解決され，
`concat` は素の次元の 2 倍になる．

```bash
foveamil-sweep \
    --config configs/sweep.example.yaml \
    --out /path/to/out \
    --weights-out /path/to/weights \
    --gpu-ids 0,1 \
    --jobs-per-gpu 12

# 展開結果と job 数・解決値だけ確認（実行しない）
foveamil-sweep --config configs/sweep.example.yaml --out /path/to/out --dry-run
```

`--gpu-ids` / `--jobs-per-gpu` は `parallel` ブロックを上書きする．各 combo は
`{out}/{combo_name}/config.yaml` に解決済み設定を書き，`(combo, fold)` を `foveamil-train
--split` のサブプロセスとして GPU へ割り当て並列実行する．`test_metrics.json` が既にある
fold はスキップするため再実行は冪等（resume）．ログ・結果は `{out}/{combo_name}`（home 側），
重みは `{weights-out}/{combo_name}`（Dataset 側）へ分けて保存する（`--weights-out` 未指定なら
`--out` にフォールバック）．fold が失敗しても全体は止めず（continue-and-report）．

### model selection と保存物
combo のランキングは **validation 指標で行い，その combo の test を報告する**
（test で直接選ぶと楽観バイアスになるため）．test 指標 1 位は oracle 上限として別途併記する．
`fixed.save_metric`（`loss` / `f1`）が val でのチェックポイント選択基準になる．

各 fold（`{out}/{combo_name}/fold{n}/`）には再学習なしで何でも再生成できるよう生データを残す:
`predictions_{val,test}.csv`（slide_id・正解・予測・全クラス確率・logit），`history.csv`
（per-epoch），`metrics_{val,test}.json`，`test_metrics.json`（後方互換），`run_meta.json`
（config・seed・環境・git・データ指紋・class 内訳），混同行列（生/正規化・クラス名）．
combo ごとに `cv_summary.json`（val/test の per-fold＋mean±std＋信頼区間），sweep ルートに
`sweep_summary.{json,md}`（val 選定 best＋その test・oracle）と `sweep_detailed.csv`
（全 combo×fold×split の指標）を保存する．

### レポート生成（`foveamil-eval`・再学習なし）
保存済み予測から ROC/PR/キャリブレーション図・combo 間有意差検定・人間可読レポートを作る:

```bash
# best combo の図・ECE・レポート
foveamil-eval --in /path/to/out --split test

# combo 間を Wilcoxon と Nadeau-Bengio 補正 t で比較
foveamil-eval --in /path/to/out --split test \
    --compare combo_000__A:combo_001__B --metric macro_auc
```

出力は `{in}/report/`（`roc_*.png` / `pr_*.png` / `calibration_*.png` / `significance_*.json` /
`report.md`）．`--all-combos` で全 combo の図，`--no-plots` で図を省く．

## SSD ステージング（事前に手動実行）

学習・sweep は `feature_root` をそのまま読むため，NAS への同時アクセスを減らすには
学習前に一度だけ `foveamil-stage` で対象スライドの特徴をローカル SSD へ一括コピーしておく．

```bash
# splits 配下の split_fold*.csv の train/val/test 全 slide をステージ
foveamil-stage \
    --feature-root /path/to/features \
    --encoder ResNet50 \
    --magnifications 1.25 2.5 5.0 \
    --splits-dir /path/to/splits \
    --cache-dir /tmp/foveamil_feat_stage

# slide_id 列の CSV / 1 行 1 個のテキストで対象を指定することもできる
foveamil-stage \
    --feature-root /path/to/features --encoder ResNet50 \
    --magnifications 1.25 2.5 5.0 --slides /path/to/slides.csv
```

既存ファイルは再利用するため再実行は冪等．表示された staged root を学習設定の
`feature_root` に指定する．`--cache-dir` 未指定時は環境変数 `FOVEAMIL_STAGE_DIR`，
それも無ければ `/tmp` 配下の既定ディレクトリを使う．
