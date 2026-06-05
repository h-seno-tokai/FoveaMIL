# cohort/splits/ — 交差検証の分割定義

## 分割方式
**層化 10-fold CV（test 非重複・各症例ちょうど 1 回 test）＋ 各 fold で train プールから
val を層化抽出（≈ train 80 / val 10 / test 10）**

## ディレクトリ構成
クラス数と fold 数で解決する: `{N}class/cv{K}/split_fold{1..K}.csv`．
- `{N}class` はクラス数（labels の `labels_{N}class.csv` に対応）．
- `cv{K}` は fold 数別のサブディレクトリ．**`cv5` と `cv10` を用意**する（`foveamil-sweep` の
  `resolve.folds` が `5` / `10` を解決 推奨 `10`）．

## 必要な形式（あなたのデータ）
fold ごとに 1 CSV・ヘッダ **`train,val,test`**．雛形: [`split.template.csv`](split.template.csv)．
- 各列に `slide_id` を縦に並べる．**列ごとに長さが異なってよい**（
- `slide_id` は `../labels/` に一致．3 列の和集合が対象症例全体．
- K-fold なら `cv{K}/split_fold1.csv … split_foldK.csv` を置く

```csv
train,val,test
SAMPLE_0001,SAMPLE_0003,SAMPLE_0004
SAMPLE_0002,,
```

## 再生成手順（`foveamil-cohort`）
ラベル CSV から `cv{K}/split_fold1.csv … split_foldK.csv` を再生成する（`cv5` と `cv10` の両方）:

```bash
for K in 5 10; do
  foveamil-cohort splits \
      --labels     ../labels/labels_3class.csv \
      --output-dir 3class/cv${K} \
      --k ${K} --seed 42        # --val-frac は省略時 1/(k-1)
done
```
