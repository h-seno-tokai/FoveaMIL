# cohort/ — コホート(症例ラベル・分割)を定義する

FoveaMIL は **WSI（Whole Slide Image）＋スライド単位ラベル** さえあれば，任意のデータセットで学習・評価できる．
このディレクトリは「**どの症例を・どのラベルで・どう分割して使うか**」という**コホート定義のみ**を置く場所である．

> **本リポジトリにコホートの実データは含まれない．** 下記の手順とテンプレートに従って
> 自分のデータを同じ形式で用意すれば，そのまま学習・評価できる．

## あなたのデータで動かす手順
1. `labels/labels.template.csv` に倣って，スライド単位のラベル CSV を用意する（`slide_id,label`）．
2. WSI 置き場を環境変数 `WSI_BASE_PATH`（`.env`）で指す．各 `slide_id` は
   `{WSI_BASE_PATH}/{slide_id}.{ext}`（OpenSlide 対応拡張子）として自動解決される
   （`foveamil.wsi.WSIResolver`）．ファイル名が `slide_id` と一致しない等の例外は，
   `slide_id,path` の2列 CSV を**オーバーライド表**として渡せる．
   → **WSI パス一覧ファイルは不要**（コホート定義＝labels と splits だけ）．
3. `splits/split.template.csv` に倣って，交差検証の分割を用意する（または付属スクリプトで生成）．
4. ラベル/分割 CSV を `cohort/` 配下に置く場合，実データは `.gitignore` 済みなので
   **commit されない**．

## ディレクトリ
```
cohort/
├── labels/        # スライド単位のサブタイプラベル        → labels/README.md
└── splits/        # 交差検証の分割定義                     → splits/README.md
```
