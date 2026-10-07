"""お知らせ(メール)取込。

入口（NoticeSource）は差し替え可能にしてある。今はディレクトリ内の .txt を読むが、
将来は大学の配信元から取得する Source に置き換える。
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from src import prompts
from src.choices import NOTICE_IMPACT_TYPES
from src.db import repository
from src.llm import claude_cli

logger = logging.getLogger(__name__)

NOTICE_DIR = repository.ROOT / "data" / "notices"
# デモ用の架空メールの先頭行。画面では「（デモ用）」と表示する
DEMO_MARK = "【デモ用・架空】"


def is_demo(notice: dict[str, Any]) -> bool:
    return (notice.get("raw_body") or "").lstrip().startswith(DEMO_MARK)


def demo_label(notice: dict[str, Any]) -> str:
    """画面に添える印（デモ用のお知らせなら「（デモ用）」）。"""
    return "（デモ用）" if is_demo(notice) else ""


def strip_demo_mark(body: str) -> str:
    """判定・抽出に渡す本文（デモ用の印の行は除く）。"""
    return body.lstrip().removeprefix(DEMO_MARK).lstrip()


@dataclass
class RawNotice:
    source_id: str  # 取込元での一意な ID（ファイル名など）
    body: str


class NoticeSource(Protocol):
    def fetch(self) -> Iterator[RawNotice]: ...


class DirectorySource:
    """1件1ファイル（.txt）のメール本文を読む。"""

    def __init__(self, directory: Path | str = NOTICE_DIR) -> None:
        self.directory = Path(directory)

    def fetch(self) -> Iterator[RawNotice]:
        for path in sorted(self.directory.glob("*.txt")):
            yield RawNotice(path.name, path.read_text(encoding="utf-8"))


def judge_relevance(body: str) -> bool:
    res = claude_cli.ask_json(
        prompts.load("notice_relevance", body=body),
        prompts.load("notice_relevance_schema"),
        system=prompts.load("notice_system"),
    )
    return res.get("is_relevant") is True


def _iso_or_none(v: Any) -> str | None:
    try:
        return date.fromisoformat(str(v)[:10]).isoformat() if v else None
    except ValueError:
        return None


def extract_fields(body: str, buildings: list[str], today: date | None = None) -> dict[str, Any]:
    res = claude_cli.ask_json(
        prompts.load(
            "notice_extract",
            body=body,
            buildings=" / ".join(buildings),
            impact_types=" / ".join(NOTICE_IMPACT_TYPES),
            today=(today or date.today()).isoformat(),
        ),
        prompts.load("notice_extract_schema"),
        system=prompts.load("notice_system"),
    )
    impact = res.get("impact_type")
    try:
        floor = int(res["target_floor"]) if res.get("target_floor") is not None else None
    except (TypeError, ValueError):
        floor = None
    return {
        "title": res.get("title"),
        "target_building": res.get("target_building") or None,
        "target_floor": floor,
        "target_area": res.get("target_area") or None,
        "impact_type": impact if impact in NOTICE_IMPACT_TYPES else None,
        "valid_from": _iso_or_none(res.get("valid_from")),
        "valid_to": _iso_or_none(res.get("valid_to")),
    }


def process(raw: RawNotice, conn: sqlite3.Connection) -> dict[str, Any]:
    """1件を判定・抽出して confirmed=false で保存する。"""
    notice: dict[str, Any] = {
        "source_file": raw.source_id,
        "raw_body": raw.body,
        "confirmed": False,
    }
    body = strip_demo_mark(raw.body)
    notice["is_relevant"] = judge_relevance(body)
    if notice["is_relevant"]:
        notice.update(extract_fields(body, repository.list_buildings(conn)))
    else:
        first = body.strip().splitlines()[0] if body.strip() else ""
        notice["title"] = first.removeprefix("件名:").removeprefix("件名：").strip()[:80]
    notice["notice_id"] = repository.upsert_notice(conn, notice)
    return notice


def ingest(
    conn: sqlite3.Connection, source: NoticeSource | None = None, *, skip_existing: bool = True
) -> list[dict[str, Any]]:
    results = []
    for raw in (source or DirectorySource()).fetch():
        if skip_existing and repository.notice_exists(conn, raw.source_id):
            continue
        try:
            results.append(process(raw, conn))
        except claude_cli.ClaudeError:
            logger.exception("notice %s の処理に失敗", raw.source_id)
    return results


if __name__ == "__main__":
    import sys

    for n in ingest(repository.connect(), skip_existing="--all" not in sys.argv):
        mark = "関連" if n["is_relevant"] else "無関係"
        detail = (
            f"{n.get('target_building')} {n.get('target_floor')}階 {n.get('target_area')} "
            f"{n.get('impact_type')} {n.get('valid_from')}〜{n.get('valid_to')}"
            if n["is_relevant"]
            else ""
        )
        print(f"[{mark}] {n['source_file']}: {n.get('title')} {detail}")
