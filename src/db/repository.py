"""SQLite へのアクセスはすべてここに集約する。"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.choices import UNSURVEYED

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.environ.get("SPOT_DB_PATH", ROOT / "data" / "spots.db"))
SCHEMA_PATH = Path(__file__).with_name("schema.sql")

SPOT_COLUMNS = (
    "spot_id", "building", "floor", "area_name", "seats", "outlets", "chair_type",
    "standing", "quietness", "food", "task_light", "outlet_location", "note", "photo",
    "surveyed_at", "surveyor", "source",
)  # fmt: skip

# 検索条件のうち「リストのどれかに一致」で絞り込むキー
_IN_KEYS = ("chair_type", "quietness", "food")


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn


def _spot_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    # 電源確保率はカラムに持たず取得時に計算する（seats / outlets が空欄なら計算しない）
    seats, outlets = d["seats"], d["outlets"]
    d["outlet_ratio"] = round(outlets / seats, 2) if seats and outlets is not None else None
    return d


# ---------------------------------------------------------------- spots


def replace_spots(conn: sqlite3.Connection, spots: Iterable[dict[str, Any]]) -> int:
    """spots を取込データで置き換える（調査データの再取込用）。

    口コミ・誤り報告が参照しているスポットを消さないよう、既存 ID は更新し、取込データにない ID だけ削除する。
    口コミ・誤り報告のあるスポットが取込データから消えた場合は ValueError。
    """
    rows = [tuple(s.get(c) for c in SPOT_COLUMNS) for s in spots]
    new_ids = {r[0] for r in rows}
    placeholders = ", ".join("?" for _ in SPOT_COLUMNS)
    updates = ", ".join(f"{c} = excluded.{c}" for c in SPOT_COLUMNS if c != "spot_id")
    with conn:
        conn.executemany(
            f"INSERT INTO spots ({', '.join(SPOT_COLUMNS)}) VALUES ({placeholders}) "
            f"ON CONFLICT(spot_id) DO UPDATE SET {updates}",
            rows,
        )
        gone = [r[0] for r in conn.execute("SELECT spot_id FROM spots") if r[0] not in new_ids]
        referenced = [
            sid for sid in gone
            if conn.execute("SELECT 1 FROM reviews WHERE spot_id = ?", (sid,)).fetchone()
            or conn.execute("SELECT 1 FROM error_reports WHERE spot_id = ?", (sid,)).fetchone()
        ]  # fmt: skip
        if referenced:
            raise ValueError(
                f"口コミ・誤り報告のあるスポットが取込データにありません: {', '.join(referenced)}"
            )
        conn.executemany("DELETE FROM spots WHERE spot_id = ?", [(sid,) for sid in gone])
    return len(rows)


def get_spot(conn: sqlite3.Connection, spot_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM spots WHERE spot_id = ?", (spot_id,)).fetchone()
    return _spot_dict(row) if row else None


def list_spots(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [_spot_dict(r) for r in conn.execute("SELECT * FROM spots ORDER BY spot_id")]


def search_spots(conn: sqlite3.Connection, cond: dict[str, Any]) -> list[dict[str, Any]]:
    """検索条件で spots を厳密に絞り込む。値が None / 空のキーは条件に使わない。

    cond のキー: building, room（エリア名と完全一致）, floor, outlets_required, min_seats, standing, task_light,
                chair_type / quietness / food（いずれも候補リスト）
    空欄（NULL）や「未調査」のスポットは、その項目の条件には一致しない。
    """
    where: list[str] = []
    params: list[Any] = []
    if cond.get("building"):
        where.append("building = ?")
        params.append(cond["building"])
    if cond.get("room"):
        where.append("area_name = ?")
        params.append(cond["room"])
    if cond.get("floor") is not None:
        where.append("floor = ?")
        params.append(int(cond["floor"]))
    if cond.get("outlets_required"):
        where.append("outlets > 0")
    if cond.get("min_seats"):
        where.append("seats >= ?")
        params.append(int(cond["min_seats"]))
    for key in ("standing", "task_light"):
        if cond.get(key) and cond[key] != UNSURVEYED:
            where.append(f"{key} = ?")
            params.append(cond[key])
    for key in _IN_KEYS:
        values = [v for v in cond.get(key) or [] if v != UNSURVEYED]
        if cond.get(key) and not values:
            where.append("0")  # 「未調査」だけが指定された条件には何も一致させない
        elif values:
            where.append(f"{key} IN ({', '.join('?' for _ in values)})")
            params.extend(values)
    sql = "SELECT * FROM spots"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY spot_id"
    return [_spot_dict(r) for r in conn.execute(sql, params)]


def list_buildings(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT building FROM spots WHERE building IS NOT NULL ORDER BY building"
    )
    return [r[0] for r in rows]


# ---------------------------------------------------------------- reviews


def add_review(
    conn: sqlite3.Connection, spot_id: str, body: str, rating: str | None = None
) -> int:
    posted_at = datetime.now().isoformat(timespec="seconds")
    with conn:
        cur = conn.execute(
            "INSERT INTO reviews (spot_id, body, rating, posted_at) VALUES (?, ?, ?, ?)",
            (spot_id, body, rating, posted_at),
        )
    return int(cur.lastrowid)


def list_reviews(conn: sqlite3.Connection, spot_id: str | None = None) -> list[dict[str, Any]]:
    if spot_id:
        rows = conn.execute(
            "SELECT * FROM reviews WHERE spot_id = ? ORDER BY posted_at DESC", (spot_id,)
        )
    else:
        rows = conn.execute("SELECT * FROM reviews ORDER BY posted_at DESC")
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- error reports


def add_error_report(conn: sqlite3.Connection, spot_id: str, body: str) -> int:
    posted_at = datetime.now().isoformat(timespec="seconds")
    with conn:
        cur = conn.execute(
            "INSERT INTO error_reports (spot_id, body, posted_at) VALUES (?, ?, ?)",
            (spot_id, body, posted_at),
        )
    return int(cur.lastrowid)


def list_error_reports(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM error_reports ORDER BY resolved, posted_at DESC")
    return [dict(r) for r in rows]


def set_error_report_resolved(conn: sqlite3.Connection, report_id: int, resolved: bool) -> None:
    with conn:
        conn.execute(
            "UPDATE error_reports SET resolved = ? WHERE report_id = ?", (int(resolved), report_id)
        )


# ---------------------------------------------------------------- reservations

RESERVATION_COLUMNS = ("date", "weekday", "room", "start", "end", "periods", "duplicate", "spot_id")


def replace_reservations(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    """教室の利用状況を取込データで全件入れ替える。"""
    values = [tuple(r.get(c) for c in RESERVATION_COLUMNS) for r in rows]
    with conn:
        conn.execute("DELETE FROM reservations")
        conn.executemany(
            f"INSERT INTO reservations ({', '.join(RESERVATION_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in RESERVATION_COLUMNS)})",
            values,
        )
    return len(values)


def reservation_dates(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT DISTINCT date FROM reservations ORDER BY date")]


def overlapping_reservations(
    conn: sqlite3.Connection, day: str, start: str, end: str
) -> list[dict[str, Any]]:
    """day の [start, end) と時間が重なる予約（時刻は "HH:MM" の文字列で比較）。"""
    rows = conn.execute(
        "SELECT * FROM reservations WHERE date = ? AND start < ? AND end > ? ORDER BY room, start",
        (day, end, start),
    )
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- notices


def upsert_notice(conn: sqlite3.Connection, notice: dict[str, Any]) -> int:
    cols = (
        "source_file", "title", "raw_body", "is_relevant", "impact_type", "target_building",
        "target_floor", "target_area", "valid_from", "valid_to", "confirmed",
    )  # fmt: skip
    values = [notice.get(c) for c in cols]
    values[cols.index("is_relevant")] = int(bool(notice.get("is_relevant")))
    values[cols.index("confirmed")] = int(bool(notice.get("confirmed")))
    updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c != "source_file")
    with conn:
        conn.execute(
            f"INSERT INTO notices ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)}) "
            f"ON CONFLICT(source_file) DO UPDATE SET {updates}",
            values,
        )
    row = conn.execute(
        "SELECT notice_id FROM notices WHERE source_file = ?", (notice["source_file"],)
    ).fetchone()
    return int(row[0])


def notice_exists(conn: sqlite3.Connection, source_file: str) -> bool:
    row = conn.execute("SELECT 1 FROM notices WHERE source_file = ?", (source_file,)).fetchone()
    return row is not None


def list_notices(conn: sqlite3.Connection, relevant_only: bool = False) -> list[dict[str, Any]]:
    sql = "SELECT * FROM notices"
    if relevant_only:
        sql += " WHERE is_relevant = 1"
    sql += " ORDER BY notice_id"
    return [dict(r) for r in conn.execute(sql)]


def set_notice_confirmed(conn: sqlite3.Connection, notice_id: int, confirmed: bool) -> None:
    with conn:
        conn.execute(
            "UPDATE notices SET confirmed = ? WHERE notice_id = ?", (int(confirmed), notice_id)
        )


def active_notices(conn: sqlite3.Connection, today: date | None = None) -> list[dict[str, Any]]:
    """有効期間内・確認済み・関連ありのお知らせ。期間の片側が空なら無制限とみなす。"""
    d = (today or date.today()).isoformat()
    rows = conn.execute(
        "SELECT * FROM notices WHERE is_relevant = 1 AND confirmed = 1 "
        "AND (valid_from IS NULL OR valid_from <= ?) AND (valid_to IS NULL OR valid_to >= ?) "
        "ORDER BY notice_id",
        (d, d),
    )
    return [dict(r) for r in rows]
