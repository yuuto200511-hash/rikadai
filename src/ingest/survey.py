"""調査データ(Excel)の取込。

- spots シートを読む。1行目が列名、2行目は日本語ラベル（読み飛ばす）、3行目以降がデータ。
- 「_」で始まる列（_power_ratio など参考表示）は取り込まない。
- reservations シート（教室の利用状況）があれば、src/ingest/reservations.py で一緒に取り込む。
  それ以外のシートは取り込まない。
- 選択肢外の値、必須列の空欄は行ごと弾く。「未調査」は有効な値として受け付ける。
- NULLABLE の列の空欄はエラーにせず NULL とし、取込後に一覧を返す。
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.choices import SPOT_ENUMS
from src.db import repository
from src.ingest import reservations as reservations_ingest

SHEET = "spots"
CHOICES_SHEET = "選択肢"
REQUIRED = ("spot_id", "area_name", "chair_type", "standing", "quietness", "food", "task_light")
NULLABLE = ("building", "floor", "seats", "outlets", "surveyed_at", "surveyor")
OPTIONAL = ("outlet_location", "note", "photo", "source")
INT_COLUMNS = ("floor", "seats", "outlets")
# 「閲覧室（建物未記入）」のように建物名の代わりに書かれた値
BUILDING_UNKNOWN_RE = re.compile(r"^(?P<place>.*?)\s*[（(]建物未記入[)）]$")
FIRST_DATA_ROW = 3


@dataclass
class IngestResult:
    accepted: list[dict[str, Any]] = field(default_factory=list)
    # (Excel の行番号, spot_id, 理由)
    rejected: list[tuple[int, str, str]] = field(default_factory=list)
    # (Excel の行番号, spot_id, 空欄だった列)
    blanks: list[tuple[int, str, list[str]]] = field(default_factory=list)
    # 「選択肢」シートと src/choices.py の差異
    warnings: list[str] = field(default_factory=list)
    # reservations シートの取込結果（シートがなければ None）
    reservations: reservations_ingest.ReservationResult | None = None


def _is_blank(v: Any) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == ""


def _to_date(v: Any) -> str:
    if isinstance(v, datetime | pd.Timestamp):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    return date.fromisoformat(str(v).strip()[:10]).isoformat()


def _to_int(col: str, v: Any) -> int:
    f = float(v)
    if not f.is_integer() or (col != "floor" and f < 0):
        raise ValueError
    return int(f)


def validate_row(raw: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    """1行を検証・正規化する。戻り値は (spot, エラー理由, 空欄だった NULLABLE 列)。"""
    errors: list[str] = []
    blanks: list[str] = []
    spot: dict[str, Any] = {}
    for col in (*REQUIRED, *NULLABLE, *OPTIONAL):
        v = raw.get(col)
        if _is_blank(v):
            if col in REQUIRED:
                errors.append(f"{col} が空欄")
            elif col in NULLABLE:
                blanks.append(col)
            spot[col] = None
        else:
            spot[col] = v.strip() if isinstance(v, str) else v

    if spot["building"] is not None and (m := BUILDING_UNKNOWN_RE.match(str(spot["building"]))):
        # 建物名は NULL にし、「閲覧室」などの場所の種類はエリア名の先頭に残す
        place = m.group("place").strip()
        spot["building"] = None
        blanks.insert(0, "building")
        area = str(spot["area_name"] or "")
        if place and area and not area.startswith(place):
            spot["area_name"] = f"{place} {area}"

    for col in INT_COLUMNS:
        if spot[col] is not None:
            try:
                spot[col] = _to_int(col, spot[col])
            except (TypeError, ValueError):
                errors.append(f"{col} が整数でない: {spot[col]!r}")
    for col, choices in SPOT_ENUMS.items():
        if spot[col] is not None and spot[col] not in choices:
            errors.append(f"{col} が選択肢外: {spot[col]!r}（{' / '.join(choices)}）")
    if spot["surveyed_at"] is not None:
        try:
            spot["surveyed_at"] = _to_date(spot["surveyed_at"])
        except ValueError:
            errors.append(f"surveyed_at が日付でない: {spot['surveyed_at']!r}")
    for col in ("spot_id", "building", "area_name", "surveyor", "source"):
        if spot[col] is not None:
            spot[col] = str(spot[col])
    if errors:
        return None, "; ".join(errors), blanks
    if not spot["photo"]:
        spot["photo"] = f"{spot['spot_id']}.jpg"
    return spot, None, blanks


def check_choices_sheet(path: Path | str) -> list[str]:
    """「選択肢」シートと src/choices.py が一致しているかを確かめる。"""
    try:
        sheet = pd.read_excel(path, sheet_name=CHOICES_SHEET, dtype=object)
    except ValueError:
        return []  # シートがなければ確認しない
    warnings = []
    for col, choices in SPOT_ENUMS.items():
        if col not in sheet.columns:
            warnings.append(f"選択肢シートに {col} 列がありません")
            continue
        # 列の先頭から最初の空欄までを選択肢とみなす（下の注記は含めない）
        in_sheet: list[str] = []
        for v in sheet[col]:
            if _is_blank(v):
                break
            in_sheet.append(str(v).strip())
        in_sheet = tuple(in_sheet)
        if in_sheet != choices:
            warnings.append(
                f"{col} の選択肢が src/choices.py と異なります: "
                f"シート={' / '.join(in_sheet)} ／ コード={' / '.join(choices)}"
            )
    return warnings


def load_excel(path: Path | str) -> IngestResult:
    # 2行目（日本語ラベル）を読み飛ばす
    df = pd.read_excel(path, sheet_name=SHEET, dtype=object, skiprows=[1])
    df.columns = [str(c).strip() for c in df.columns]
    df = df[[c for c in df.columns if not c.startswith("_")]]
    missing = [c for c in (*REQUIRED, *NULLABLE) if c not in df.columns]
    if missing:
        raise ValueError(f"必須列がありません: {', '.join(missing)}")

    result = IngestResult(warnings=check_choices_sheet(path))
    seen: set[str] = set()
    for i, raw in enumerate(df.to_dict(orient="records"), start=FIRST_DATA_ROW):
        if all(_is_blank(v) for v in raw.values()):
            continue
        spot, err, blanks = validate_row(raw)
        sid = "" if _is_blank(raw.get("spot_id")) else str(raw.get("spot_id"))
        if spot and spot["spot_id"] in seen:
            spot, err = None, "spot_id が重複"
        if spot is None:
            result.rejected.append((i, sid, err or ""))
            continue
        seen.add(spot["spot_id"])
        result.accepted.append(spot)
        if blanks:
            result.blanks.append((i, sid, blanks))
    return result


def ingest(path: Path | str, conn: sqlite3.Connection) -> IngestResult:
    result = load_excel(path)
    result.reservations = reservations_ingest.load(path)
    repository.replace_spots(conn, result.accepted)
    if result.reservations is not None:
        repository.replace_reservations(conn, result.reservations.accepted)
    return result


def format_report(res: IngestResult) -> str:
    lines = [f"取込: {len(res.accepted)} 件 / 除外: {len(res.rejected)} 件"]
    lines += [f"[警告] {w}" for w in res.warnings]
    for row, sid, reason in res.rejected:
        lines.append(f"  除外 行{row} {sid or '(IDなし)'}: {reason}")
    if res.blanks:
        lines.append(f"空欄（NULL として取込）: {len(res.blanks)} 件")
        for row, sid, cols in res.blanks:
            lines.append(f"  行{row} {sid}: {', '.join(cols)}")
    if (rv := res.reservations) is not None:
        lines.append(f"教室の利用状況: 取込 {len(rv.accepted)} 件 / 除外 {len(rv.rejected)} 件")
        lines += [f"  除外 行{row}: {reason}" for row, reason in rv.rejected]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else repository.ROOT / "data" / "survey.xlsx"
    print(format_report(ingest(src, repository.connect())))
