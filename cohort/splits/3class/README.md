# splits/3class/

クラス `DLBCL` / `FL` / `Reactive` の交差検証分割．


## 分割
- `cv10/`（`split_fold1.csv … split_fold10.csv`）と `cv5/`（`split_fold1.csv … split_fold5.csv`）．
  `foveamil-sweep` の `resolve.folds` が `10` / `5` を解決する 推奨 `10`．
- `_legacy_zoommil/` は旧 5-fold（過去実験の再現用 新規 `cv5` とは別物）．


## 厳密性（分割で担保している不変条件）
- **test は fold 間で互いに素**で，和集合が全症例＝各 slide はちょうど 1 つの fold で test になる．
- **ラベルで層化**し，各 fold がクラス比を保つ．
- **val は同じ fold の train プールからのみ**層化抽出する（test と重ならない）．

