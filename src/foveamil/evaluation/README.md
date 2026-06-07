# `foveamil.evaluation` — 指標・図・有意差検定レポート（再学習なし）

sweep が保存した予測・集計を二次利用し，ROC/PR/キャリブレーション図・combo 間の有意差検定・
人間可読レポートを生成する．学習は一切しない（保存済み予測を読むだけ）．

## モジュール

| ファイル | 役割 |
|---|---|
| `report.py` | 予測・集計から評価成果物（図・指標・レポート）を生成する本体． |
| `stats.py` | 区間推定（t 分布・ブートストラップ）と有意差検定（Wilcoxon・Nadeau-Bengio 補正 t・反復 CV 補正 t・多重比較補正 Holm/BH）． |
| `group_metrics.py` | クラス部分集合の非加重平均 F1（group-F1）を保存済み per-fold から算出する． |
| `calibration.py` | pooled-val で temperature scaling と クラス別ロジット補正 δ_c を当て test に適用する（事後較正・再学習なし）． |
| `redundancy.py` | 倍率間表現の冗長性診断の本体（融合入力ベクトルの収集・指標計算）． |
| `ablation.py` | sweep 出力を手法タグ付けしてベースライン比 Δ 表に集計する本体． |
| `stability.py` | 学習履歴から振動・過学習・best epoch・分散比を診断する本体． |
| `report_cli.py` | `foveamil-eval` コマンド． |
| `calibration_cli.py` | `foveamil-calibrate` コマンド． |
| `redundancy_cli.py` | `foveamil-redundancy` コマンド． |
| `ablation_cli.py` | `foveamil-ablation` コマンド． |
| `stability_cli.py` | `foveamil-stability` コマンド． |

## 入出力

入力は sweep の出力ルート（`sweep_summary.json` / 各 combo の `cv_summary.json` /
`fold*/predictions_{split}.csv` / `run_meta.json`）．combo の選定は validation 指標で行い，
その combo の test を報告する（test 指標 1 位は oracle 上限として併記＝楽観バイアス回避）．
出力は `--out`（既定 `{in}/report/`）に `roc_*.png` / `pr_*.png` / `calibration_*.png` /
`significance_*.json` / `report.md`．matplotlib が無ければ図は省く．

## `stats.py`

fold 間平均の信頼区間（t 分布・ブートストラップ），2 手法の対比較（Wilcoxon 符号順位），
交差検証の fold 間相関を補正した対 t 検定（Nadeau-Bengio）を提供する．標本が少ない・差が
全て 0 等の縮退時は `nan` を返し例外を投げない．

`repeated_cv_corrected_t` は複数 seed × fold の反復 CV 向けの補正リサンプル t 検定
（Bouckaert-Frank）で，全リサンプル差を平坦化して総数 `m` で補正する．訓練集合の重なり由来の
相関のみを補正し seed×fold の二段相関は補正しないため，seed 効果が支配的だと anti-conservative に
なり得る（主張用の p はプール予測の並べ替え/ブートストラップか seed 単位の二段検定で併せて確認する）．
`adjust_pvalues` は複数 p に Holm-Bonferroni（FWER）/ Benjamini-Hochberg（FDR）補正をかける．

## 使い方

```bash
# best combo の図・ECE・レポート
foveamil-eval --in /path/to/out --split test

# combo 間を Wilcoxon と Nadeau-Bengio 補正 t で比較
foveamil-eval --in /path/to/out --split test \
    --compare combo_000__A:combo_001__B --metric macro_auc
```

`--all-combos` で全 combo の図，`--no-plots` で図を省く．

## `redundancy.py` / `foveamil-redundancy`

val 選定の best combo と fold 重みを解決し，対象 split の各スライドで融合へ入る各倍率の
プーリング表現 `M_i`（識別器ヘッド直前，和を取る前の入力）を収集して倍率間の冗長性を
診断する．`collect_magnification_vectors` が `FeatureAccessor` + `model.forward_layer` で
Lazy 駆動を no_grad で再現し，スライドごとに `[L, D]` 行列を返す．学習はしない．

指標：余弦類似度（生・中心化），Pearson 相関，線形 CKA，積み上げ行列の特異値スペクトル・
実効ランク（スペクトルエントロピーの指数）．余弦・相関は `L×L` 行列の上三角平均で 1 値に，
CKA・Pearson は倍率対ごとの `L×L` 行列にまとめる．

```bash
foveamil-redundancy --in /path/to/out --feature-root /path/to/features --split test
```

出力は `--out`（既定 `{in}/redundancy/`）に `redundancy.json`（指標要約）と
`cka_heatmap.png` / `pearson_heatmap.png`．matplotlib が無ければ図は省く．特徴ルートは
`--feature-root` か環境変数 `FOVEAMIL_FEATURE_ROOT` で渡す．`--weights-root` で重みの
別ルートを指定できる．

## `ablation.py` / `foveamil-ablation`

1 つ以上の sweep 出力ルートを受け，各 combo の `config.yaml` から手法タグ（ABMIL / CLAM /
`FoveaMIL(no-A/B/C/D)` / `FoveaMIL+A`・`+B`・`+D` の組合せ / `FoveaMIL+MCTS(C)`）と倍率レジームを
判定し，`cv_summary.json` の指標集計（mean±std・CI）を読む．同一倍率レジーム内で多倍率
ベースラインとの差分 Δ を付けた markdown 表を出す．A/B/D と MCTS を別ルートで回した場合も
複数ルートをまとめて 1 表にできる．学習はしない．

```bash
foveamil-ablation --in experiments/11class_virchow2/abd experiments/11class_virchow2/mcts \
    --metric weighted_f1 --split test --out experiments/11class_virchow2/ablation.md
```

`--baseline <手法タグ>` を渡すと そのタグを基準に各手法の Δ・対応 t（Nadeau-Bengio 補正）・
`--adjust holm|fdr_bh` の補正後 p を per-fold から算出して列に付ける（`--n-train` / `--n-test` で
補正係数の標本数を渡す）．`--metric group_f1 --group-classes 4,5,6` でクラス部分集合の group-F1 を
同じ Δ・p 経路で集計できる（aggregate に無いため per-fold から算出する）．

`--pooled --baseline <手法タグ> --group-classes 4,5,6` を渡すと 全 fold（複数ルートも）の保存済み
予測（`predictions_{split}.csv`）をプールし，baseline と各手法を `slide_id` で対応付けた同一テスト
症例集合上で，プール group-F1 の Δ・対応あり並べ替え検定の p・クラス層化 bootstrap の差の CI を出す．
fold 平均ではなく全症例を 1 つに束ねた単一値で，クラス不均衡時に少数クラスへ寄与した予測を直接評価
できる．並べ替え/bootstrap は `--n-perm` / `--n-boot` / `--seed`（決定的）で制御する．

## `calibration.py` / `foveamil-calibrate`

保存済み予測の logit/prob を二次利用し，全 fold の val をプールした 1 標本で事後較正パラメタを
当てて test に適用する（学習・モデル再推論はしない）．較正は 2 段：(1) temperature scaling は val の
NLL を最小化する温度 `T` を 1 次元最適化し確率の鋭さを直す（argmax 不変＝分類指標は変えない）．
(2) クラス別ロジット補正 `δ_c` は T 適用後のロジットへクラスごとの加算項を入れ，pooled-val の macro-F1
（`--group-classes` 指定時はその group-F1）を座標降下で最大化する（argmax を動かすので少数クラス
recall を引き上げ得る）．

temperature scaling は argmax 不変なので分類指標（macro/group-F1・recall）を構造的に変えない．
よって段階表の T 段は F1 寄与が 0 になるが，これは仕様である．T の本来効用は確率較正にあるため，
段階表は各段の test **NLL・ECE** も併記し（限界効用は負＝改善），T の価値を NLL↓/ECE↓ で示す．
macro-F1・group-F1 は **present-only 平均**（`y_true` に不在のクラスを除外）で出す——per-class recall の
nan 規約・per-fold 側 `group_f1_from_fold` の「キー不在クラスを除外」と定義を一致させ，不在クラスを
F1=0 で誤減点しない（`δ_c` の最適化目的も同様に present-only で，val 不在クラスによる希釈を避ける）．

過適合回避：較正は必ず pooled-val で当て，`δ_c` は L2（`--l2`）で 0 方向へ縮める．test 指標は
`baseline → +temperature → +temperature+δ` の段階で出し，各段の限界効用（前段との差）を併記する
（`δ_c` に効が無ければ T 止まりで足りると判る）．少数クラス（支持標本数が中央値未満）の recall と
主要流出先混同を before/after で返す．縮退（空クラス・logit 欠損・標本不足）では恒等変換へ落とし
nan を返して例外を投げない（決定的シード）．

```bash
# best combo を pooled-val で較正し test に適用 macro-F1 目的
foveamil-calibrate --in /path/to/out --split test

# 少数クラス集合の group-F1 を目的に δ_c を当てる
foveamil-calibrate --in /path/to/out --split test --group-classes 4,5,6 --all-combos
```

出力は `--out`（既定 `{in}/calibration/`）に `calibration.json`（T・δ_c・段階指標）と
`calibration.md`．`--all-combos` で全 combo，`--l2` で δ_c の正則強度を変える．

## `stability.py` / `foveamil-stability`

学習履歴（各 `fold*/history.csv`）を読み，fold 平均で検証指標の終盤振動（tail std）・val_loss 最小後の
上昇量（過学習兆候）・best epoch を算出する．`--compare` で 2 構成の per-fold 標準偏差の分散比を
ブートストラップ（決定的シード）で区間推定する．学習はしない．

```bash
foveamil-stability --in /path/to/out --metric macro_f1
```
