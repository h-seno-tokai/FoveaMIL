# `foveamil.encoders` — パッチ特徴抽出器プラグイン

正規化済みパッチ `[B, 3, 224, 224]` を受け，pooled 特徴 `[B, dim]`（ViT 系は cls 特徴 `[B, dim]` も）を返す共通インタフェースのエンコーダ群．モデルは遅延ロードする．

## 共通インタフェース（`base.PatchEncoder`）
- 属性: `name`，`feature_dim`，`has_cls`，`device`，`normalizer_mean`/`normalizer_std`（ImageNet），`batch_size`，`num_workers`
- `load()`: モデルを遅延ロードする（再呼び出しは二重ロードしない）
- `forward(patches) -> (pooled, cls)`: `no_grad`＋CUDA 時 `autocast(float16)`．`cls` は `has_cls=False` のとき `None`．入力は正規化済み前提（正規化は呼び出し側の責務）

## 登録済みエンコーダ
| name | feature_dim | has_cls | 重みの取得元 |
|---|---|---|---|
| `ResNet50` | 1024 | False | torchvision ImageNet（layer3 出力を空間平均） |
| `UNI2-h` | 1536 | True | HuggingFace `MahmoodLab/UNI2-h`（gated） |
| `Virchow` | 1280 | True | HuggingFace `paige-ai/Virchow` |
| `Virchow2` | 1280 | True | HuggingFace `paige-ai/Virchow2`（gated） |
| `Virchow2-mini-dinov2` | 384 | True | ローカル蒸留チェックポイント（DINOv2 ViT-S） |

ViT 系は patch トークンを 16×16 に並べ替えて空間平均し pooled 特徴とする（cls はトークン位置 0）．

## レジストリ
```python
from foveamil.encoders import build_encoder, ENCODERS

enc = build_encoder("UNI2-h", batch_size=256, num_workers=4)
enc.load()
pooled, cls = enc.forward(patches)  # patches は正規化済み [B, 3, 224, 224]
```

## 重み・認証
- HuggingFace 系（`UNI2-h`/`Virchow`/`Virchow2`）は HuggingFace 標準キャッシュ（既定 `~/.cache/huggingface`，`HF_HOME` で変更可）に保存される．gated モデルは初回のみ認証が必要で，`auth.ensure_hf_auth()` が環境変数 `HF_TOKEN` か既存のログイン情報を用いる
- `Virchow2-mini` はローカルの `.pt` を読む．パスは環境変数 `VIRCHOW_MINI_CHECKPOINT` で与える（[`../../../pretrained/`](../../../pretrained/) を参照）
