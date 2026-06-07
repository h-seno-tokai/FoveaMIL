# `foveamil.evaluation` — 指標・図・有意差検定レポート（再学習なし）

sweep が保存した予測・集計を二次利用し，ROC/PR/キャリブレーション図・combo 間の有意差検定・
人間可読レポートを生成する．学習は一切しない（保存済み予測を読むだけ）．

## モジュール

| ファイル | 役割 |
|---|---|
| `report.py` | 予測・集計から評価成果物（図・指標・レポート）を生成する本体． |
| `stats.py` | 区間推定（t 分布・ブートストラップ）と有意差検定（Wilcoxon・Nadeau-Bengio 補正 t・反復 CV 補正 t・多重比較補正 Holm/BH）． |
| `group_metrics.py` | クラス部分集合の非加重平均 F1（group-F1）を保存済み per-fold から算出する． |
| `redundancy.py` | 倍率間表現の冗長性診断の本体（融合入力ベクトルの収集・指標計算）． |
| `ablation.py` | sweep 出力を手法タグ付けしてベースライン比 Δ 表に集計する本体． |
| `stability.py` | 学習履歴から振動・過学習・best epoch・分散比を診断する本体． |
| `report_cli.py` | `foveamil-eval` コマンド． |
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

## `stability.py` / `foveamil-stability`

学習履歴（各 `fold*/history.csv`）を読み，fold 平均で検証指標の終盤振動（tail std）・val_loss 最小後の
上昇量（過学習兆候）・best epoch を算出する．`--compare` で 2 構成の per-fold 標準偏差の分散比を
ブートストラップ（決定的シード）で区間推定する．学習はしない．

```bash
foveamil-stability --in /path/to/out --metric macro_f1
```
