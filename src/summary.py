"""「今日のまとめ」：チャットの上に出す、今日のキッチンカーと今日使えない場所。

開くたびに表示するので LLM は使わず、DB の値からプログラムで作る。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date
from typing import Any

from src.db import repository
from src.rag.kitchen_car import groups

NO_INFO = "今日の情報はありません"


@dataclass
class TodaySummary:
    day: date
    # 今日のキッチンカー（店ごと。kitchen_car.groups の形）
    kitchen_cars: list[dict[str, Any]]
    # 今日使えない場所：今日が有効期間に入っている、確認済みの「閉鎖」のお知らせ
    closed: list[dict[str, Any]]


def today_summary(conn: sqlite3.Connection, today: date) -> TodaySummary:
    closed = [n for n in repository.active_notices(conn, today) if n["impact_type"] == "閉鎖"]
    return TodaySummary(today, groups(repository.list_kitchen_cars(conn, day=today)), closed)
