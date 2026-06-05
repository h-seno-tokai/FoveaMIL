# `foveamil.models.topk` — 微分可能 top-k セレクタ

スコア `[B, N]` から選択行列 `[B, k, N]` を作る部品群．学習時は手法ごとの soft な選択行列，
推論時は基底共通の hard 実装（`torch.topk` の上位 k を index 昇順で one-hot 化）を返す．
`k` が `N` を超える場合は `min(N, k)` に丸める．レジストリ `TOPK_METHODS` と
`build_topk(name, k, **kwargs)` で差し替える（全体像は [models](../README.md)）．

| ファイル | 役割 |
|---|---|
| `base.py` | 抽象基底 `TopKSelector`．`k` を保持し `forward` で学習/推論を分岐．サブクラスは `soft_select(scores, k)` のみ実装する． |
| `perturbed.py` | `PerturbedTopK(k, num_samples, sigma)`．ガウス摂動標本の hard top-k の平均で soft 選択行列を作り，期待値勾配で逆伝播する． |
| `sparse.py` | `FastSparseTopK(k, epsilon, max_iter)`．Permutahedron 射影（二分探索）でスパースマスクを求め，上位 k 値を soft 選択行列に詰める． |

新しい手法は `TopKSelector` を継承して `soft_select(scores, k)` を実装し，`__init__.py` の
`TOPK_METHODS` に `"name": クラス` を 1 行登録すれば `build_topk("name", k, ...)` から使える．
