# pretrained

学習済み重みの置き場．`.gitignore` 済みで重みファイル自体は追跡しない（この
README と `.gitkeep` のみ追跡する）．

- `Virchow2-mini` の自家蒸留チェックポイント `.pt` をここに置き，パスを環境変数
  `VIRCHOW_MINI_CHECKPOINT` で指す．
- HuggingFace 系（`UNI2-h` / `Virchow` / `Virchow2`）は HuggingFace 標準キャッシュ
  （既定 `~/.cache/huggingface`，`HF_HOME` で変更可）に保存される．gated モデルは
  初回のみ HF ログインが必要で，`huggingface-cli login` か環境変数 `HF_TOKEN` で行う．
