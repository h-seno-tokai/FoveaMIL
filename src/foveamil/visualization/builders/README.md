# `foveamil.visualization.builders` — ビュー組立（組立層）

`core` のアテンショントレースと `render` の素材を束ね，3 つのビューを 1 図に組み上げる．
ビューを増やすときは builder を 1 ファイル足す（全体像は [visualization](../README.md)）．

| ファイル | 役割 |
|---|---|
| `overview.py` | View A：倍率 × {主primary, 補助aux} の WSI 全体オーバーレイ格子． |
| `zoom.py` | View B：階層ズーム照明（WSI 全体図では潰れる高倍率の選択を解く）．`--chain` で中心窩経路図． |
| `compare.py` | View C：成功（`y_true==y_pred`）vs 失敗 症例の対比格子． |
