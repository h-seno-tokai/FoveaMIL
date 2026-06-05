"""Gmail SMTP によるプレーンテキストメール通知

送信関数 :func:`send_email` と，薄い CLI エントリ :func:`notify_main` を提供する
認証情報は環境変数 ``GMAIL_USER`` / ``GMAIL_APP_PASSWORD`` / ``RECEIVE_USER`` から
取得し，引数で明示的に上書きもできる認証情報が揃わない場合や送信に失敗した場合は
例外を送出せず ``logging`` で警告して ``False`` を返す
"""

from __future__ import annotations

import argparse
import logging
import os
import smtplib
import sys
from email.mime.text import MIMEText
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# Gmail SMTP の接続先ホスト・ポート（STARTTLS）
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
# SMTP 接続のタイムアウト（秒）
SMTP_TIMEOUT = 30
# 認証情報を読む環境変数名
ENV_GMAIL_USER = "GMAIL_USER"
ENV_GMAIL_APP_PASSWORD = "GMAIL_APP_PASSWORD"
ENV_RECEIVE_USER = "RECEIVE_USER"


def send_email(
    subject: str,
    body: str,
    *,
    gmail_user: Optional[str] = None,
    gmail_app_password: Optional[str] = None,
    receive_user: Optional[str] = None,
) -> bool:
    """件名・本文のプレーンテキストメールを Gmail SMTP 経由で送る

    認証情報の各値は引数優先で，``None`` のときに対応する環境変数を読む
    認証情報が 1 つでも欠けている場合は送信を試みず ``False`` を返す
    送信中に例外が起きた場合も送出せず ``False`` を返す

    Args:
        subject: メールの件名
        body: メールの本文（プレーンテキスト）
        gmail_user: 送信元アドレス``None`` なら ``GMAIL_USER``
        gmail_app_password: アプリパスワード``None`` なら ``GMAIL_APP_PASSWORD``
        receive_user: 宛先アドレス``None`` なら ``RECEIVE_USER``

    Returns:
        送信に成功すれば ``True``，認証情報不足・送信失敗なら ``False``
    """
    user = gmail_user if gmail_user is not None else os.environ.get(ENV_GMAIL_USER)
    password = (
        gmail_app_password
        if gmail_app_password is not None
        else os.environ.get(ENV_GMAIL_APP_PASSWORD)
    )
    receiver = (
        receive_user if receive_user is not None else os.environ.get(ENV_RECEIVE_USER)
    )

    if not (user and password and receiver):
        logger.warning("email credentials incomplete; skipping notification")
        return False

    message = MIMEText(body, "plain", "utf-8")
    message["From"] = user
    message["To"] = receiver
    message["Subject"] = subject

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(message)
    except Exception as exc:  # noqa: BLE001 - 通知失敗でジョブを巻き込まない
        logger.warning("email send failed: %s", exc)
        return False

    logger.info("email sent: %s", subject)
    return True


def _load_dotenv_if_available() -> None:
    """``.env`` があれば読み込む``python-dotenv`` が無ければ何もしない"""
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    """``foveamil-notify`` の引数パーサを構築する"""
    parser = argparse.ArgumentParser(
        prog="foveamil-notify",
        description="Send a plain-text email via Gmail SMTP.",
    )
    parser.add_argument("--subject", required=True, help="Email subject.")
    parser.add_argument(
        "--body",
        default=None,
        help="Email body; read from stdin when omitted.",
    )
    return parser


def notify_main(argv: Optional[Sequence[str]] = None) -> int:
    """``foveamil-notify`` コンソールスクリプトのエントリポイント

    Returns:
        送信成功なら 0，失敗なら 1
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    _load_dotenv_if_available()

    body = args.body if args.body is not None else sys.stdin.read()
    ok = send_email(args.subject, body)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(notify_main())
