"""キッチンカーの出店情報（管理者が毎月用意する CSV）の取込。

- 1行目が列名。列: date, weekday, open, close, location, vendor, item, price_yen
  （1行 = 1日・1店・1メニュー）。
- 次を確かめ、1行でも問題があれば何も保存しない（問題のある行と理由を返す）:
  列がそろっているか（列名・各行の列の数・空欄）／日付と曜日が合っているか／値段が0以上の整数か
  （ほかに、時刻が "HH:MM" で open < close か）。
- 保存は管理画面で内容を確認してから（parse → 一覧表示 → save）。取込データにある月の分を置き換える。
"""

from __future__ import annotations

import csv
import io
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from src.choices import WEEKDAYS
from src.db import repository

COLUMNS = repository.KITCHEN_CAR_COLUMNS
TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
PRICE_RE = re.compile(r"^\d+$")


@dataclass
class KitchenCarResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    # (行番号, 理由)。行番号は CSV の行（1行目が列名）。0 はファイル全体の問題
    errors: list[tuple[int, str]] = field(default_factory=list)
    # 行番号 → CSV の行の値（問題のある行を画面に示すため）
    raw: dict[int, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.rows)

    @property
    def months(self) -> dict[str, int]:
        months: dict[str, int] = {}
        for r in self.rows:
            months[r["date"][:7]] = months.get(r["date"][:7], 0) + 1
        return months


def _decode(data: bytes) -> str:
    # Excel で保存した CSV（Shift_JIS）も受け付ける
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp932")


def _time(v: str) -> str:
    m = TIME_RE.match(v)
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        raise ValueError
    return f"{int(m.group(1)):02d}:{m.group(2)}"


def validate_row(raw: dict[str, str]) -> tuple[dict[str, Any] | None, str | None]:
    """戻り値は (行, エラー理由)。"""
    if blank := [c for c in COLUMNS if not raw.get(c)]:
        return None, f"空欄の列がある: {', '.join(blank)}"
    try:
        day = date.fromisoformat(raw["date"])
    except ValueError:
        return None, f"date が日付（YYYY-MM-DD）でない: {raw['date']!r}"
    weekday = WEEKDAYS[day.weekday()]
    if raw["weekday"].rstrip("曜日") != weekday:
        return None, f"weekday が date の曜日（{weekday}）と違う: {raw['weekday']!r}"
    try:
        open_, close = _time(raw["open"]), _time(raw["close"])
    except ValueError:
        return None, f"open / close が時刻（HH:MM）でない: {raw['open']!r} / {raw['close']!r}"
    if open_ >= close:
        return None, f"open が close 以降: {open_} / {close}"
    if not PRICE_RE.match(raw["price_yen"]):
        return None, f"price_yen が0以上の整数でない: {raw['price_yen']!r}"
    return {
        "date": day.isoformat(),
        "weekday": weekday,
        "open": open_,
        "close": close,
        "location": raw["location"],
        "vendor": raw["vendor"],
        "item": raw["item"],
        "price_yen": int(raw["price_yen"]),
    }, None


def parse(data: bytes | str) -> KitchenCarResult:
    """CSV を読んで確かめる（保存はしない）。"""
    text = _decode(data) if isinstance(data, bytes) else data
    result = KitchenCarResult()
    lines = list(csv.reader(io.StringIO(text)))
    if not lines:
        result.errors.append((0, "ファイルが空です"))
        return result
    header = [c.strip() for c in lines[0]]
    missing = [c for c in COLUMNS if c not in header]
    if missing:
        result.errors.append((1, f"必須の列がありません: {', '.join(missing)}（列名: {', '.join(header)}）"))
        return result
    for i, values in enumerate(lines[1:], start=2):
        if not any(v.strip() for v in values):
            continue  # 空行
        result.raw[i] = values
        if len(values) != len(header):
            result.errors.append((i, f"列の数が {len(values)} 個（列名は {len(header)} 個）"))
            continue
        row, err = validate_row({h: v.strip() for h, v in zip(header, values, strict=True)})
        if err:
            result.errors.append((i, err))
        else:
            result.rows.append(row)
    if not result.rows and not result.errors:
        result.errors.append((0, "データの行がありません"))
    return result


def save(conn: sqlite3.Connection, result: KitchenCarResult) -> dict[str, int]:
    """確認済みの取込結果を保存する（その月の分を置き換える）。問題のある結果は保存しない。"""
    if not result.ok:
        raise ValueError("問題のある行があるため保存できません")
    return repository.replace_kitchen_cars(conn, result.rows)


if __name__ == "__main__":
    import sys

    res = parse(Path(sys.argv[1]).read_bytes())
    for row, reason in res.errors:
        print(f"  行{row}: {reason}")
    if not res.ok:
        sys.exit("問題があるため取り込みませんでした")
    for month, n in save(repository.connect(), res).items():
        print(f"{month}: {n} 行を取り込みました（この月の分を置き換え）")
