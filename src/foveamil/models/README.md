# models

病理 WSI 向け多解像度 MIL モデルのコンポーネント群．部品を細分化し，どの部品を使うかを引数で選べる設計とする．

## 構成

| ファイル | 役割 |
|---|---|
| `attention.py` | ゲート付きアテンション部品 |
| `attention_norm/` | アテンションスコアの正規化器（レジストリで差し替え，既定 softmax） |
| `topk/` | 微分可能 top-k セレクタ（レジストリで差し替え） |
| `selection/` | パッチ選択コントローラ（スコア・特徴 → 選択行列，レジストリで差し替え） |
| `regularizers/` | 補助損失（正則化項）と forward 文脈（レジストリで差し替え） |
| `fusion.py` | 多解像度表現の融合（インタフェース，現状 Sum のみ） |
| `heads.py` | 識別器ヘッド（融合と分離） |
| `instance.py` | インスタンス疑似ラベルによる補助損失（単一倍率の主アテンション向け） |
| `mil.py` | 本体組立（倍率数可変，部品注入） |

## attention.py

`GatedAttention(L, D, dropout=None, n_cls=1)`．Tanh 枝と Sigmoid 枝を要素積し，`Linear(D, n_cls)` でスコアへ写す独立した再利用可能部品．

- `forward(x: [B, N, L]) -> (A: [B, N, n_cls], x)`．A は正規化前の生スコア．

## attention_norm/（アテンション正規化）

アテンションスコア `[B, N]` を最終軸で正規化する部品群．補助アテンションの正規化を差し替え，密な softmax から温度付き・スパース系（sparsemax / entmax 等）へ拡張できるようにする．

- 公開 API：`build_attention_norm(name, **kwargs) -> Callable[[Tensor], Tensor]`．レジストリ `ATTENTION_NORMS` から構築する．未登録名は `KeyError`．
- 登録済み：
  - `"softmax"`：密な既定 `F.softmax(scores, dim=-1)`（パラメータなし）．
  - `"temperature"`：`softmax(scores / temperature)`（`temperature` を引数に取る．`temperature=1` で `"softmax"` と一致し，小さいほど鋭く大きいほど平坦になる）．
  - `"sparsemax"`：確率単体への Euclidean 射影（Martins and Astudillo, 2016）．ソート・累積和の閾値で求め，非負・和 1 で鋭い入力に対し厳密に 0 を含むスパースな分布を返す．
  - `"entmax"`：α-entmax（Peters et al., 2019）．Tsallis α-エントロピー正則化下の単体射影を二分探索で解く（`alpha` を引数に取る）．`alpha=1` で `"softmax"`，`alpha=2` で `"sparsemax"` に一致し，`1<alpha<2`（既定 1.5）でその中間のスパース性を取る．
- いずれもパラメータを持たず（`nn.Parameter` なし）checkpoint 互換を保つ．`temperature` / `alpha` はモデル構築時に固定する定数．
- 追加方法：`attention_norm/` に新ファイルを作り，`@register_attention_norm("name")` でファクトリを登録する（パッケージ自動探索で読み込まれるため共有リストの編集は不要）．

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

## selection/（選択コントローラ）

補助アテンションの正規化スコア `[B, N]` と射影特徴 `[B, N, D]` から，次倍率へズームするパッチの選択行列 `[B, k, N]` を作る部品群．スコアのみで選ぶ手法（top-k）と，特徴の多様性も見る手法（DPP 等）を同一インタフェースで差し替える．

- `base.py`: `SelectionController(nn.Module)` 抽象基底．`k` を保持し，`select(scores, features) -> [B, k, N]` を実装する．
- `topk_controller.py`: `TopKSelectionController(k, topk_method, topk_kwargs)`．`build_topk` の微分可能 top-k をスコアに適用する既定コントローラ（特徴は未使用）．
- 公開 API：`build_selection_controller(name, k, topk_method="perturbed", topk_kwargs=None, **kwargs) -> SelectionController`．レジストリ `SELECTION_CONTROLLERS` から構築する．未登録名は `KeyError`．
- 登録済み：`"topk"`．
- 追加方法：`selection/` に新ファイルを作り，`SelectionController` を継承して `@register_selection_controller("name")` を付ける（自動探索で読み込まれる）．

## regularizers/（補助損失）

スライド分類損失（CE）に加える補助損失（正則化項）の部品群と，段階 forward の中間量を運ぶ文脈．

- `base.py`: `ForwardContext`（各倍率のプーリング表現 `m_list`，各層の正規化補助アテンション `layer_aux`，各層の選択 `selections`，名前付きスカラ損失 `extra_losses`）と `Regularizer` 抽象基底（`__call__(context, label) -> scalar`，`weight`，`from_config(config) -> Optional[Regularizer]`）．
- 公開 API：`iter_active_regularizers(config) -> List[Regularizer]`．登録済み各クラスの `from_config` を呼び有効な項を集める．学習ループは `CE + Σ w_i·reg_i + Σ extra_losses` を最小化する．
- 登録済み：なし（具体項は各機能ブランチが追加する）．
- 追加方法：`regularizers/` に新ファイルを作り，`Regularizer` を継承して `name` を定め，`@register_regularizer` を付ける（自動探索で読み込まれる）．無効化は `from_config` が `None` を返すことで表す．

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

## instance.py（インスタンス補助損失）

`InstanceClusteringLoss(in_dim, n_cls, k, subtyping=True)`．スライド単位ラベルしか無い弱教師下で，主アテンションが高い/低いパッチを per-class の 2 値分類器（クラスごとに `Linear(in_dim, 2)`）で pos/neg として検算し，アテンションをクラス方向へ追加で鍛える補助損失．

- `forward(h: [B, N, in_dim], attention: [B, N], label: [B]) -> scalar`．`attention` は softmax 済みの主アテンション重み．
- in-class 枝：正解クラスで高アテンション上位 k を pos，低アテンション下位 k を neg として検算する．
- out-of-class 枝（`subtyping=True`）：非正解クラスの分類器に高アテンション上位 k をすべて neg として与える（相互排他なサブタイプ向け）．`subtyping=True` のときは枝の和をクラス数で割る．
- パッチ数が `2k` 未満なら k を縮める（`k = min(self.k, N // 2)`）．

bag 分類損失（CE）と `bag·bag_weight + inst·(1-bag_weight)` で重み付き和を取って使う．**単一倍率（ズーム無しの attention pooling）の全バッグ主アテンションが対象**で，`k`（`inst_k`）はズーム選択数 `k_sample` とは別物．

## mil.py（組立）

`FoveaMIL(in_feat_dim, hidden_feat_dim=256, out_feat_dim=512, dropout=None, k_sample=12, n_cls=3, num_layers=4, topk_method="perturbed", topk_kwargs=None, aux_norm="softmax", aux_norm_kwargs=None, selector="topk", selector_kwargs=None, fusion="sum", instance_loss=False, inst_k=8, inst_subtyping=True)`．

倍率ごとに `nn.ModuleList` で特徴射影（`Linear + ReLU (+ Dropout)`），主アテンション，補助アテンション（最終層以外）を持つ．補助アテンションの正規化器 1 つ（`aux_norm`）・選択コントローラ 1 つ（`selector`）・融合器・ヘッドを注入する．`aux_norm` / `selector` は既定で従来挙動（softmax 正規化・top-k 選択）と一致する．`instance_loss=True` のとき `InstanceClusteringLoss` を持ち，`forward_with_instance_loss(x, label)` が単一倍率の bag forward（logits / Y_hat / Y_prob）と補助損失を**同一の射影・主アテンションから**返す（無効なら補助損失は `None`）．`instance_loss=True` は `num_layers=1`（単一倍率）のみ許し，多倍率では `ValueError`．

- `forward(x) -> (logits, Y_hat, Y_prob)`．`x` は倍率ごとのテンソルのタプル（各 `[B, N_i, in_feat_dim]`，`N_i` は親の 4 倍）．
- 各倍率でこれまでの選択行列を `kron(selection, eye(ratio))` と `einsum` で適用 → 射影 → 主アテンションで softmax プーリング表現を蓄積 → 補助アテンションのスコアから top-k で次倍率の選択行列を得る．最後に融合 → ヘッド → `logits`，`Y_hat`（予測クラス `[B, 1]`），`Y_prob`（softmax `[B, n_cls]`）．
- 学習・推論の分岐は top-k セレクタの `.training` に委ね，モデルの `train()` / `eval()` が伝播する．デバイス移動は標準の `.to()` で動く．
