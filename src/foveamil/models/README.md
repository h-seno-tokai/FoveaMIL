# models

病理 WSI 向け多解像度 MIL モデルのコンポーネント群．部品を細分化し，どの部品を使うかを引数で選べる設計とする．

## 構成

| ファイル | 役割 |
|---|---|
| `attention.py` | ゲート付きアテンション部品 |
| `topk/` | 微分可能 top-k セレクタ（レジストリで差し替え） |
| `fusion.py` | 多解像度表現の融合（インタフェース，現状 Sum のみ） |
| `heads.py` | 識別器ヘッド（融合と分離） |
| `mil.py` | 本体組立（倍率数可変，部品注入） |

## attention.py

`GatedAttention(L, D, dropout=None, n_cls=1)`．Tanh 枝と Sigmoid 枝を要素積し，`Linear(D, n_cls)` でスコアへ写す独立した再利用可能部品．

- `forward(x: [B, N, L]) -> (A: [B, N, n_cls], x)`．A は正規化前の生スコア．

## topk/（微分可能 top-k）

スコア `[B, N]` から選択行列 `[B, k, N]` を作る部品群．学習時は各手法の soft な選択行列，推論時は基底共通の hard 実装（`torch.topk` の上位 k を index 昇順で one-hot 化）を返す．`k` が `N` を超える場合は `min(N, k)` に丸める．

- `base.py`: `TopKSelector(nn.Module)` 抽象基底．`k` を保持し，`forward(scores)` で学習・推論を分岐する．サブクラスは `soft_select(scores, k)` のみ実装する．
- `perturbed.py`: `PerturbedTopK(k, num_samples=100, sigma=0.002)`．ガウス摂動標本の hard top-k の平均で soft 選択行列を作り，期待値勾配で逆伝播する．
- `sparse.py`: `FastSparseTopK(k, epsilon=0.002, max_iter=50)`．Permutahedron 射影（二分探索）でスパースマスク `[B, N]` を求め，上位 k 値を `scatter_` で soft 選択行列に詰める．clamp と四則演算のみで自動微分される．

### 公開 API

- `build_topk(name, k, **kwargs) -> TopKSelector`：レジストリ `TOPK_METHODS` から手法を構築する．未登録名は `KeyError`．
- 登録済み：`"perturbed"`，`"fast_sparse"`．

### 新しい top-k 手法の追加

1. `topk/` に新ファイルを作り，`TopKSelector` を継承して `soft_select(scores, k)` を実装する（推論時の hard 実装は基底に任せてよい）．
2. `topk/__init__.py` の `TOPK_METHODS` に `"name": クラス` を 1 行追加する．

これだけで `build_topk("name", k, ...)` から使える．

## fusion.py（多解像度融合）

各倍率のプーリング表現 `[B, 1, dim]` のリストを `[B, out_dim]` に融合する．融合は識別器と分離し，`out_dim` 属性で後段ヘッドの入力次元を宣言する．

- `Fusion(nn.Module)` 基底：属性 `out_dim`，`forward(M_list) -> [B, out_dim]`．
- `SumFusion(dim, num_layers)`：`out_dim = dim`，総和して squeeze する．

### 公開 API

- `build_fusion(name, dim, num_layers) -> Fusion`：レジストリ `FUSION_METHODS` から融合器を構築する．未登録名は `KeyError`．
- 登録済み：`"sum"`．`"concat"`，`"attention_pooling"` は将来追加．

### 新しい融合の追加

1. `Fusion` を継承し，コンストラクタで `out_dim` を設定，`forward(M_list)` を実装する（`out_dim` を融合方式に合わせて宣言する）．
2. `fusion.py` の `FUSION_METHODS` に `"name": クラス` を追加する．

ヘッドは `fusion.out_dim` から別途構築されるため，融合方式を変えてもヘッド側の手当ては不要．

## heads.py（識別器）

`LinearClassifierHead(in_dim, n_cls)`．`forward(x: [B, in_dim]) -> logits: [B, n_cls]`．融合と分離した独立部品．

## mil.py（組立）

`FoveaMIL(in_feat_dim, hidden_feat_dim=256, out_feat_dim=512, dropout=None, k_sample=12, n_cls=3, num_layers=4, topk_method="perturbed", topk_kwargs=None, fusion="sum")`．

倍率ごとに `nn.ModuleList` で特徴射影（`Linear + ReLU (+ Dropout)`），主アテンション，補助アテンション（最終層以外）を持つ．top-k セレクタ 1 つ・融合器・ヘッドを注入する．

- `forward(x) -> (logits, Y_hat, Y_prob)`．`x` は倍率ごとのテンソルのタプル（各 `[B, N_i, in_feat_dim]`，`N_i` は親の 4 倍）．
- 各倍率でこれまでの選択行列を `kron(selection, eye(ratio))` と `einsum` で適用 → 射影 → 主アテンションで softmax プーリング表現を蓄積 → 補助アテンションのスコアから top-k で次倍率の選択行列を得る．最後に融合 → ヘッド → `logits`，`Y_hat`（予測クラス `[B, 1]`），`Y_prob`（softmax `[B, n_cls]`）．
- 学習・推論の分岐は top-k セレクタの `.training` に委ね，モデルの `train()` / `eval()` が伝播する．デバイス移動は標準の `.to()` で動く．
