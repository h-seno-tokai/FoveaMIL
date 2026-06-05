# cohort/labels/ — スライド単位のサブタイプ(病型)ラベル

## 必要な形式
CSV・ヘッダ **`slide_id,label`**．雛形: [`labels.template.csv`](labels.template.csv)．
- `slide_id`: WSI ファイル名（拡張子なし）と一致させる．
- `label`: クラス名（文字列）．本パイプラインの既定は 3 クラス `DLBCL` / `FL` / `Reactive`．

```csv
slide_id,label
SAMPLE_0001,DLBCL
SAMPLE_0002,FL
SAMPLE_0003,Reactive
```

## 生成手順（`foveamil-cohort`）
master から対象クラスを抽出し，必要なら手元にある WSI 集合と積を取って学習用ラベルを作る:

```bash
foveamil-cohort labels \
    --input       subtype_labels_master.csv \
    --output      labels_3class.csv \
    --classes     DLBCL FL Reactive \
    --restrict-to available_slides.txt
    # 必要なら --exclude SAMPLE_0001 SAMPLE_0002 ... で個別除外
```

`--restrict-to` は「実際に WSI を持っている slide_id 集合」に絞るための任意指定で，
1 行 1 要素のテキストか `slide_id` 列を持つ CSV（または WSI パス一覧）を渡せる
（WSI パスは basename から拡張子を除いて slide_id 化される）．master の出現順を維持し，出力は
ヘッダ付き `slide_id,label`．
