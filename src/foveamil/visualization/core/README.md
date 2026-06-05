# `foveamil.visualization.core` — アテンション抽出（データ層）

学習済み FoveaMIL を推論し，倍率ごとの主アテンション（pooling 寄与）・補助アテンション
（選択スコア）・選択 index・親子対応を `AttentionTrace` / `LayerTrace` に取り出す．これが
可視化の唯一の入力源で，以降の `render` 素材と `builders` 組立はすべてこのトレースを材料にする
（全体像は [visualization](../README.md)）．

| ファイル | 役割 |
|---|---|
| `extraction.py` | `extract_attention_trace` / `AttentionTrace` / `LayerTrace`．推論専用（学習はしない）． |
