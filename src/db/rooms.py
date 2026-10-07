"""実在する教室の一覧（data/rooms.csv）。

公式の設備一覧と CLASS の予約データから作った一覧。列: room, in_official_list, in_class_reservations,
spot_id（調査済みの教室だけ）。一覧にない教室は「実在しない」、spot_id が空なら「実在するが未調査」とする。
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

from src.db.repository import ROOT

ROOMS_PATH = Path(os.environ.get("SPOT_ROOMS_PATH", ROOT / "data" / "rooms.csv"))


@dataclass(frozen=True)
class Room:
    room: str
    spot_id: str | None  # 調査済みならスポット ID


def load_rooms(path: Path | str | None = None) -> dict[str, Room]:
    """教室番号 → Room。ファイルがなければ空。"""
    path = Path(path) if path else ROOMS_PATH
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {
            r["room"].strip(): Room(r["room"].strip(), (r.get("spot_id") or "").strip() or None)
            for r in csv.DictReader(f)
            if (r.get("room") or "").strip()
        }
