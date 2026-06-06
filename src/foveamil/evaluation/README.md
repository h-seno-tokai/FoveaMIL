# `foveamil.evaluation` — 指標・図・有意差検定レポート（再学習なし）

sweep が保存した予測・集計を二次利用し，ROC/PR/キャリブレーション図・combo 間の有意差検定・
人間可読レポートを生成する．学習は一切しない（保存済み予測を読むだけ）．

## モジュール

| ファイル | 役割 |
|---|---|
| `report.py` | 予測・集計から評価成果物（図・指標・レポート）を生成する本体． |
| `stats.py` | 区間推定（t 分布・ブートストラップ）と有意差検定（Wilcoxon・Nadeau-Bengio 補正 t）． |
| `redundancy.py` | 倍率間表現の冗長性診断の本体（融合入力ベクトルの収集・指標計算）． |
| `ablation.py` | sweep 出力を手法タグ付けしてベースライン比 Δ 表に集計する本体． |
| `report_cli.py` | `foveamil-eval` コマンド． |
| `redundancy_cli.py` | `foveamil-redundancy` コマンド． |
| `ablation_cli.py` | `foveamil-ablation` コマンド． |

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
`ZoomMIL(baseline)` / `ZoomMIL+A`・`+B`・`+D` の組合せ / `ZoomMIL+MCTS(C)`）と倍率レジームを
判定し，`cv_summary.json` の指標集計（mean±std・CI）を読む．同一倍率レジーム内で多倍率
ベースラインとの差分 Δ を付けた markdown 表を出す．A/B/D と MCTS を別ルートで回した場合も
複数ルートをまとめて 1 表にできる．学習はしない．

```bash
foveamil-ablation --in experiments/11class_virchow2/abd experiments/11class_virchow2/mcts \
    --metric weighted_f1 --split test --out experiments/11class_virchow2/ablation.md
```
