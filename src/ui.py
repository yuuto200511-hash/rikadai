"""Streamlit の画面（app.py と pages/）で共有する小物。"""

from __future__ import annotations

import sqlite3

from src.db import repository
from src.rag.answer import location, user_location


def conn() -> sqlite3.Connection:
    # Streamlit はリクエストごとにスレッドが変わるので、接続は都度作る
    return repository.connect()


def spot_label(s: dict) -> str:
    """利用者向けの表示（空欄の建物・階は出さない）。"""
    return user_location(s)


def admin_spot_label(s: dict) -> str:
    """管理画面向けの表示（spot_id と空欄の明示つき）。"""
    return f"{s['spot_id']} {location(s)}"
