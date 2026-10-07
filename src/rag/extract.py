"""質問文 → 検索条件。"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from src import prompts
from src.choices import (
    CHAIR_TYPES, FOOD, PERIOD_TIMES, QUIETNESS, STANDING, TASK_LIGHT, WEEKDAYS, searchable,
)  # fmt: skip
from src.db import repository
from src.llm import claude_cli

# 条件として指定できる値（「未調査」は含めない）
LIST_KEYS: dict[str, tuple[str, ...]] = {
    "chair_type": searchable(CHAIR_TYPES),
    "food": searchable(FOOD),
}
# 静粛性は段階（私語厳禁 < 小声可 < 会話可 < 通話可）。LLM には範囲の両端だけを出させ、
# 範囲に入る段階をコードで条件にする
QUIETNESS_LEVELS = searchable(QUIETNESS)
SINGLE_KEYS: dict[str, tuple[str, ...]] = {
    "standing": searchable(STANDING),
    "task_light": searchable(TASK_LIGHT),
}


INTENTS = ("spot", "notice", "chat", "review")


@dataclass
class Extraction:
    supported: bool
    unsupported_reason: str | None
    conditions: dict[str, Any] = field(default_factory=dict)
    # 日付（YYYY-MM-DD）・曜日（月〜日。日付の指定がないとき）・時限（6限以降もそのまま残し、
    # 回答側で対応外とする）。now=True は「今」（現在日時から判定する）
    date: str | None = None
    weekday: str | None = None
    period: int | None = None
    now: bool = False
    # 質問文に書かれた教室番号（表記のまま。実在するかは回答側で確かめる）
    room: str | None = None
    # 質問の種類: "spot"（作業スポット探し）/ "notice"（お知らせの確認）/ "chat"（挨拶・雑談）/
    # "review"（口コミの確認）
    intent: str = "spot"
    # お知らせの確認で指定された建物・場所（建物なら conditions["building"] にも入る）
    notice_place: str | None = None
    # 前の質問の続き（前の条件を引き継いだうえで、今回の発言を反映した条件になっている）
    follow_up: bool = False
    # 口コミの確認の対象：会話で紹介したスポットの ID / 質問文に書かれた場所の名前
    review_spot_ids: list[str] = field(default_factory=list)
    review_place: str | None = None
    # スポット探しで、口コミの内容も知りたいか
    want_reviews: bool = False


def _to_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


def quietness_range(raw: dict[str, Any]) -> list[str]:
    """quietness_min（少なくともこの段階まで話せる）〜 quietness_max（この段階までのにぎやかさなら可）
    に入る段階。どちらもなければ、旧形式の quietness（リスト）の最小〜最大を範囲とする。"""
    lo, hi = raw.get("quietness_min"), raw.get("quietness_max")
    if lo not in QUIETNESS_LEVELS and hi not in QUIETNESS_LEVELS:
        values = raw.get("quietness")
        values = [values] if isinstance(values, str) else values
        given = [v for v in QUIETNESS_LEVELS if isinstance(values, list) and v in values]
        if not given:
            return []
        lo, hi = given[0], given[-1]
    i = QUIETNESS_LEVELS.index(lo) if lo in QUIETNESS_LEVELS else 0
    j = QUIETNESS_LEVELS.index(hi) if hi in QUIETNESS_LEVELS else len(QUIETNESS_LEVELS) - 1
    return list(QUIETNESS_LEVELS[i : j + 1])  # 矛盾（i > j）なら空 = 条件にしない


def normalize_room(v: Any) -> str | None:
    """教室番号の表記をそろえる（全角→半角、英字は大文字、空白を除く）。"""
    if not isinstance(v, str):
        return None
    room = re.sub(r"\s", "", unicodedata.normalize("NFKC", v)).upper()
    return room or None


def _text(v: Any) -> str | None:
    return v.strip() if isinstance(v, str) and v.strip() else None


def normalize(raw: dict[str, Any]) -> Extraction:
    """LLM 出力を検証し、選択肢外の値を捨てる。値のある条件だけを残す。"""
    cond: dict[str, Any] = {}
    if isinstance(raw.get("building"), str) and raw["building"].strip():
        cond["building"] = raw["building"].strip()
    if (floor := _to_int(raw.get("floor"))) is not None:
        cond["floor"] = floor
    if raw.get("outlets_required") is True:
        cond["outlets_required"] = True
    if (n := _to_int(raw.get("min_seats"))) and n > 0:
        cond["min_seats"] = n
    for key, choices in SINGLE_KEYS.items():
        if raw.get(key) in choices:
            cond[key] = raw[key]
    for key, choices in LIST_KEYS.items():
        values = raw.get(key)
        if isinstance(values, str):
            values = [values]
        if isinstance(values, list):
            kept = [v for v in choices if v in values]  # 選択肢の順に揃える
            if kept:
                # 全選択肢でも条件として残す（「未調査」のスポットは一致させない）
                cond[key] = kept
    if quiet := quietness_range(raw):
        cond["quietness"] = quiet
    supported = raw.get("supported") is not False
    reason = raw.get("unsupported_reason") if not supported else None
    weekday = raw.get("weekday") if raw.get("weekday") in WEEKDAYS else None
    period = _to_int(raw.get("period"))
    try:
        day = date.fromisoformat(str(raw.get("date"))).isoformat() if raw.get("date") else None
    except ValueError:
        day = None
    return Extraction(
        supported,
        reason or (None if supported else "対応外の質問"),
        cond,
        date=day,
        weekday=weekday,
        period=period if period is not None and period > 0 else None,
        now=raw.get("now") is True,
        room=normalize_room(raw.get("room")),
        intent=raw["intent"] if raw.get("intent") in INTENTS else "spot",
        notice_place=_text(raw.get("notice_place")),
        follow_up=raw.get("follow_up") is True,
        review_spot_ids=[
            v for v in raw.get("review_spot_ids") or [] if isinstance(v, str) and v.strip()
        ],
        review_place=_text(raw.get("review_place")),
        want_reviews=raw.get("want_reviews") is True,
    )


def as_raw(ext: Extraction) -> dict[str, Any]:
    """抽出結果を LLM の出力形式（extract_schema の項目名）に戻す。会話の履歴として次の抽出に渡す。"""
    raw: dict[str, Any] = {}
    for key, value in ext.conditions.items():
        if key == "quietness":
            # 範囲の両端に戻す（quietness_range と同じ意味になる）
            if value[0] != QUIETNESS_LEVELS[0]:
                raw["quietness_min"] = value[0]
            if value[-1] != QUIETNESS_LEVELS[-1] or "quietness_min" not in raw:
                raw["quietness_max"] = value[-1]
        else:
            raw[key] = value
    for key in ("date", "weekday", "period"):
        if (value := getattr(ext, key)) is not None:
            raw[key] = value
    if ext.now:
        raw["now"] = True
    return raw


def extract(
    question: str, conn: sqlite3.Connection, today: date | None = None, history: str = ""
) -> Extraction:
    """history はこれまでの会話（answer.format_history）。続きの質問なら前の条件を引き継がせる。"""
    today = today or date.today()
    prompt = prompts.load(
        "extract",
        question=question,
        history=history or "（なし）",
        today=f"{today.isoformat()}（{WEEKDAYS[today.weekday()]}曜日）",
        buildings=" / ".join(repository.list_buildings(conn)),
        chair_types=" / ".join(LIST_KEYS["chair_type"]),
        standing=" / ".join(SINGLE_KEYS["standing"]),
        quietness=" < ".join(QUIETNESS_LEVELS),
        food=" / ".join(LIST_KEYS["food"]),
        task_light=" / ".join(SINGLE_KEYS["task_light"]),
        weekdays=" / ".join(WEEKDAYS),
        period_times=" / ".join(f"{p}限 {s}-{e}" for p, (s, e) in PERIOD_TIMES.items()),
    )
    raw = claude_cli.ask_json(
        prompt, prompts.load("extract_schema"), system=prompts.load("extract_system")
    )
    return normalize(raw)
