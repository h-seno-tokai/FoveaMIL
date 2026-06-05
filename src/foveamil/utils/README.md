# `foveamil.utils` — 補助ユーティリティ

パイプライン各段から使う小さな補助部品．

## モジュール

| ファイル | 役割 |
|---|---|
| `notify.py` | Gmail SMTP のプレーンテキストメール通知（`send_email` と CLI `foveamil-notify`）． |
| `memory.py` | `gc.collect()` の実行と RSS の DEBUG ログ（`psutil` があるときのみ）． |
| `provenance.py` | git リビジョン・実行環境・入力ファイルハッシュを集めた再現情報 `run_meta` の組み立て． |

## `notify.py` — メール通知

`send_email(subject, body)` は認証情報を環境変数 `GMAIL_USER` / `GMAIL_APP_PASSWORD` /
`RECEIVE_USER` から取る（引数で上書き可）．認証情報が揃わない・送信失敗時は例外を投げず
警告ログを出して `False` を返す．CLI `foveamil-notify` は `--subject` と `--body`（または標準入力）を
受け，成功で 0・失敗で 1 を返す．各段の `--notify` フラグがこれを使う．

```bash
echo "本文" | foveamil-notify --subject "完了"
```

## `provenance.py` — 再現情報

git リビジョン・hostname・Python/ライブラリ版・入力ファイルのハッシュを集め `run_meta` 辞書に
する．git 未初期化や取得失敗時は例外を投げず `None` を入れる（再現情報の欠落で学習を止めない）．
学習時に各 fold へ保存され，後から実験条件を引き当てられる．
