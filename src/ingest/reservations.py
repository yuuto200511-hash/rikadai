"""教室の利用状況（survey.xlsx の reservations シート、または同じ列の CSV）の取込。

- 1行目が列名、2行目は日本語ラベル（読み飛ばす）。列: date, weekday, room, start, end, periods,
  duplicate, spot_id
- 授業教室は学期中ほぼ固定なので、ある日の利用状況を毎週同じ曜日の利用状況として扱う
  （曜日は date から決める。weekday 列が date の曜日と食い違う行は弾く）。
- 表の下の注記行（date が日付でなく、ほかの列が空欄の行）は読み飛ばす。
- 日付・時刻が読めない行は弾き、理由を返す。取込は全件入れ替え。
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.choices import WEEKDAYS
from src.db import repository

SHEET = "reservations"
COLUMNS = repository.RESERVATION_COLUMNS
TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
FIRST_DATA_ROW = 3


@dataclass
class ReservationResult:
    accepted: list[dict[str, Any]] = field(default_factory=list)
    # (行番号, 理由)
    rejected: list[tuple[int, str]] = field(default_factory=list)


def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == ""


def _time(v: Any) -> str:
    m = TIME_RE.match(str(v).strip())
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        raise ValueError
    return f"{int(m.group(1)):02d}:{m.group(2)}"


def validate_row(raw: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """戻り値は (行, エラー理由)。注記行は (None, None)。"""
    others_blank = all(_blank(raw.get(c)) for c in COLUMNS if c != "date")
    try:
        day = date.fromisoformat(str(raw.get("date")).strip()[:10])
    except ValueError:
        return (None, None) if others_blank else (None, f"date が日付でない: {raw.get('date')!r}")
    weekday = WEEKDAYS[day.weekday()]
    if not _blank(raw.get("weekday")) and str(raw["weekday"]).strip().rstrip("曜日") != weekday:
        return None, f"weekday が date の曜日（{weekday}）と違う: {raw['weekday']!r}"
    if _blank(raw.get("room")):
        return None, "room が空欄"
    try:
        start, end = _time(raw.get("start")), _time(raw.get("end"))
    except ValueError:
        return None, f"start / end が時刻でない: {raw.get('start')!r} / {raw.get('end')!r}"
    if start >= end:
        return None, f"start が end 以降: {start} / {end}"
    periods = None if _blank(raw.get("periods")) else re.sub(r"\s", "", str(raw["periods"]))
    return {
        "date": day.isoformat(),
        "weekday": weekday,
        "room": str(raw["room"]).strip(),
        "start": start,
        "end": end,
        "periods": periods,
        "duplicate": 0 if _blank(raw.get("duplicate")) else 1,
        "spot_id": None if _blank(raw.get("spot_id")) else str(raw["spot_id"]).strip(),
    }, None


def _read(path: Path) -> pd.DataFrame | None:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    else:
        try:
            df = pd.read_excel(path, sheet_name=SHEET, dtype=str)
        except ValueError:
            return None  # シートがない
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"reservations の必須列がありません: {', '.join(missing)}")
    return df.iloc[1:]  # 2行目（日本語ラベル）を読み飛ばす


def load(path: Path | str) -> ReservationResult | None:
    """シートがなければ None。"""
    df = _read(Path(path))
    if df is None:
        return None
    result = ReservationResult()
    for i, raw in enumerate(df.to_dict(orient="records"), start=FIRST_DATA_ROW):
        row, err = validate_row(raw)
        if row:
            result.accepted.append(row)
        elif err:
            result.rejected.append((i, err))
    return result


def ingest(path: Path | str, conn: sqlite3.Connection) -> ReservationResult | None:
    result = load(path)
    if result is not None:
        repository.replace_reservations(conn, result.accepted)
    return result


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else repository.ROOT / "data" / "survey.xlsx"
    res = ingest(src, repository.connect())
    if res is None:
        print("reservations シートがありません")
    else:
        print(f"利用状況: 取込 {len(res.accepted)} 件 / 除外 {len(res.rejected)} 件")
        for row, reason in res.rejected:
            print(f"  除外 行{row}: {reason}")
