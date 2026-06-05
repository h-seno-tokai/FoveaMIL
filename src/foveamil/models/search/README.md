# search（探索ベースのズーム決定）

倍率ごとのズーム先を，補助アテンションの一括 top-k 確定ではなく，学習した方策・価値による look-ahead で決める部品群．ズームの木（根＝最低倍率の視野，あるノードを展開＝次倍率の子を materialise）を探索し，どの親を高解像度の子へ展開するかを選ぶ．

## 構成

| ファイル | 役割 |
|---|---|
| `policy.py` | 候補親上の事前方策 `π(a|state)` を返す方策ネット． |
| `value.py` | 部分選択状態のスカラ価値 `v(state)` を返す価値ネット． |
| `mcts.py` | ズーム木の探索プランナ（Gumbel-AlphaZero / PUCT）と探索問題インタフェース． |
| `driver.py` | 探索ベースのズーム駆動 `MCTSZoomDriver` と学習目的． |

## policy.py — `PolicyNetwork`

`PolicyNetwork(feat_dim, hidden_dim, dropout=None)`．現倍率の射影特徴 `[B, N, D]` から候補親 N 個上の事前分布を返す．ゲート付きアテンションを状態符号化器に流用し，要素ごとのスコアを候補軸 softmax で正規化する（容量を基線の補助アテンションと揃える）．

- `logits(features: [B, N, D]) -> [B, N]`：正規化前スコア．
- `forward(features: [B, N, D]) -> [B, N]`：候補軸の和が 1 の事前方策．

## value.py — `ValueNetwork`

`ValueNetwork(feat_dim, hidden_dim, dropout=None)`．射影特徴 `[B, N, D]` をゲート付きアテンションでプーリングし，小 MLP でスカラへ写す．現在の部分選択のもとでスライドがどれだけよく分類されるかの推定で，探索の葉評価に用いる．

- `forward(features: [B, N, D]) -> [B]`：状態のスカラ価値．

## mcts.py — プランナ

候補親（＝単一パッチ展開アクション）上で，事前方策を prior，葉評価を価値として探索し，訪問・スコア由来の改良方策（非負・和 1）と選択アクションを返す．

- `SearchProblem`（抽象）：`num_actions()` / `prior() -> (N,)` / `evaluate(action) -> float`．報酬評価は純粋でよく，プランナはこのインタフェースのみに依存する（toy 問題で単体検証できる）．
- `GumbelAlphaZeroPlanner(simulations, max_considered, seed, c_scale)`：Gumbel で上位 m 候補を標本化し，sequential halving で予算を配分し，完了 Q 値から決定的な改良方策 `softmax(logits + σ(completedQ))` を作る．PUCT 定数の手調整を要しない（Danihelka et al., ICLR 2022）．
- `PuctPlanner(simulations, max_considered, seed, c_puct)`：`argmax_a Q(a) + c_puct · prior(a) · √ΣN / (1 + N(a))` で逐次探索し，訪問数で改良方策を作る AlphaZero 流の参照実装．
- `PlannerResult`：`improved_policy` `chosen_actions`（改良方策上位 k，昇順）`q_values`（完了 Q）`visit_counts` `prior`．
- 公開 API：`build_planner(name, simulations, max_considered, seed, **kwargs)`（`"gumbel"` / `"puct"`）．未登録名は `KeyError`．
- すべての乱数はシードで固定し，同一シードでは決定的に動く（改良方策は完了 Q から決定的，Gumbel は探索する候補集合を変える）．

## driver.py — `MCTSZoomDriver`

`foveamil.training.zoom_driver.ZoomDriver` を実装し，各倍率で `Planner` を回してどの親を展開するかを選ぶ．`from_config(config, model, num_layers)` で `TrainConfig` から構築する．

- 各倍率：方策ネットで事前を作り，`SearchProblem.evaluate(action)` がその親の子を `child_loader` で materialise し次倍率へ射影・プーリングした状態の価値推定（look-ahead）を返す．プランナが改良方策と選択親を返し，選んだ親の子を方策確率で重み付けして次倍率へ進める．
- 学習損失（`ForwardContext.extra_losses` に名前付きで積む）：分類損失（CE，学習ループが加える）に加え `λ_π · 方策蒸留`（π から改良方策への交差エントロピー，目標は detach）と `λ_v · 価値回帰`（探索表現の実現報酬＝負分類損失への MSE，目標は detach）．任意でエントロピー項．
- 方策・価値ネットは `model` の子モジュール（`search_policy` / `search_value`）として登録し，`model.parameters()` で基線の識別器ヘッド・射影と同一 optimizer で最適化される（並列分類器は持たず，利得を探索に帰属させる）．
- 推論：ラベル無しで同じ探索を回しズーム先を選ぶ（`extra_losses` は積まない）．
- `context.selections[i]['select_weight']`：MCTS 駆動では選んだ親の方策確率（`PolicyNetwork` 出力）であり，differentiable 駆動の補助アテンション重みとは別物．アテンションベースの正則化器など下流の消費者はこの違いに留意する．
- 価値ネットは価値回帰項からのみ勾配を受けるため，`value_loss_weight > 0` のときに限り学習される（0 なら初期重みのまま固定）．`zoom_driver="mcts"` かつ `value_loss_weight==0` のとき 1 度警告を出す．

## 他シームとの両立

`ForwardContext.m_list` は駆動に依らず埋まる．方策の事前はスパース正規化器で整形でき，候補展開は多様性基準で形作れる．駆動は config でゲートされ，既定 `differentiable` では従来挙動を完全に再現するため，他のシームと組み合わせても影響しない．

## 参考

- Silver et al., 2018, *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*（AlphaZero）．
- Schrittwieser et al., 2020, *Mastering Atari, Go, chess and shogi by planning with a learned model*（MuZero）．
- Danihelka et al., ICLR 2022, *Policy improvement by planning with Gumbel*（Gumbel-AlphaZero / Gumbel-MuZero）．
