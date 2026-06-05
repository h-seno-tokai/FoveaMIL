# splits/11class/

11 クラス
`DLBCL` / `Reactive` / `FL_G1-2` / `FL_G3a` / `FL_G3b` / `MALT` / `CHL` / `AITL` / `ATLL` / `PTCL_NOS` / `MCL`
の交差検証分割．ラベルは `../../labels/labels_11class.csv`．形式・生成手順は [`../README.md`](../README.md)．

## 分割
- `cv10/`（`split_fold1.csv … split_fold10.csv`）と `cv5/`（`split_fold1.csv … split_fold5.csv`）．
  `foveamil-sweep` の `resolve.folds` が `10` / `5` を解決する．
- **10 を推奨する理由**: 各 fold の test が全体の約 1/10 になる．fold 数が少ないと
  性能推定の分散が大きく，多いと test が小さくなりクラスごとの指標が不安定になる．10 はこの中間で,
  各 test fold にクラスごと十分な症例を残しつつ推定の分散を抑えられる`cv5` は短時間で回す用途．

## 厳密性（分割で担保している不変条件）
- **test は fold 間で互いに素**で，和集合が全症例＝各 slide はちょうど 1 つの fold で test になる．
- **ラベルで層化**し，各 fold がクラス比を保つ．クラス間で症例数に偏りがあるため，
  層化に加えて評価はクラスごとの指標（macro-F1 等）を用いる．
- **val は同じ fold の train プールからのみ**層化抽出する（test と重ならない）．
- 1 slide = 1 症例 = 1 患者のため，患者単位のグループ化は不要．
- `labels`／`k`／`seed`／`val-frac` が同じなら**決定的に同一**の分割になる．
