"""利用者からの誤り報告。公式データ（spots）は変えずに別テーブルへ保存し、管理画面で確認する。"""

from __future__ import annotations

import sqlite3

from src.db import repository

MAX_BODY_CHARS = 500


class ErrorReportError(ValueError):
    pass


def post_error_report(conn: sqlite3.Connection, spot_id: str, body: str) -> int:
    body = body.strip()
    if not body:
        raise ErrorReportError("内容を入力してください")
    if len(body) > MAX_BODY_CHARS:
        raise ErrorReportError(f"内容は{MAX_BODY_CHARS}文字以内にしてください")
    if repository.get_spot(conn, spot_id) is None:
        raise ErrorReportError(f"スポット {spot_id} は存在しません")
    return repository.add_error_report(conn, spot_id, body)
