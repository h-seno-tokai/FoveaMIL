# `foveamil.evaluation` — 指標・図・有意差検定レポート（再学習なし）

sweep が保存した予測・集計を二次利用し，ROC/PR/キャリブレーション図・combo 間の有意差検定・
人間可読レポートを生成する．学習は一切しない（保存済み予測を読むだけ）．

## モジュール

| ファイル | 役割 |
|---|---|
| `report.py` | 予測・集計から評価成果物（図・指標・レポート）を生成する本体． |
| `stats.py` | 区間推定（t 分布・ブートストラップ）と有意差検定（Wilcoxon・Nadeau-Bengio 補正 t）． |
| `report_cli.py` | `foveamil-eval` コマンド． |

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
