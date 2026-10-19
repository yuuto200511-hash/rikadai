"""キッチンカーについての質問。該当する日と店はプログラムが表（kitchen_cars）から引き、
LLM には返事の文章だけを書かせる（answer._converse。値段・日付の数値は reply.checked で照合する）。

- 日付・曜日の指定があればその日、メニュー・店の名前だけなら今日以降でそれが出る日、どちらもなければ今日
- 場所は質問文から拾う（「3号館A」→ その場所だけ、「3号館」→ 3号館の出店場所すべて）
- 出店日でない日（土日・データのない月）は「その日の出店予定はありません」
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from src.db import repository
from src.rag import extract as extract_mod
from src.rag import when
from src.rag.when import day_label

NO_CARS = "その日の出店予定はありません"


@dataclass
class KitchenCarLookup:
    text: str  # 定型文（LLM の返事を使えないとき）
    facts: str  # 返事を書かせるときに渡すデータ
    # 画面の「メニューを見る」に出す、日ごとの店（groups の形）
    days: list[tuple[date, list[dict[str, Any]]]] = field(default_factory=list)


def _norm(text: str) -> str:
    return re.sub(r"\s", "", unicodedata.normalize("NFKC", text)).casefold()


def groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """1行 = 1メニューの出店情報を、店ごと（日付・場所・店・時間）にまとめる（行の順のまま）。"""
    out: dict[tuple[str, ...], dict[str, Any]] = {}
    for r in rows:
        key = (r["date"], r["location"], r["vendor"], r["open"], r["close"])
        g = out.setdefault(key, {**{k: r[k] for k in ("date", "location", "vendor", "open", "close")}, "items": []})
        g["items"].append((r["item"], r["price_yen"]))
    return list(out.values())


def menu_line(item: str, price: int) -> str:
    return f"{item} {price}円"


def vendor_line(g: dict[str, Any]) -> str:
    """「3号館A：KALTENA（10:00〜14:00）」"""
    return f"{g['location']}：{g['vendor']}（{g['open']}〜{g['close']}）"


def _place(question: str, locations: list[str]) -> tuple[str | None, list[str]]:
    """質問文に書かれた出店場所（表示名, 対象の location）。書かれていなければ (None, [])。

    「3号館A」のように location そのものが書かれていればそれだけ、「3号館」なら 3号館の location すべて。
    出店場所でない「◯号館」なら対象は空。
    """
    q = _norm(question)
    if exact := [loc for loc in locations if _norm(loc) in q]:
        return exact[0], exact[:1]
    if written := extract_mod.written_buildings(question):
        b = written[0]
        return b, [loc for loc in locations if b in extract_mod.written_buildings(loc)]
    return None, []


def _matches(r: dict[str, Any], item: str | None, targets: list[str] | None) -> bool:
    if targets is not None and r["location"] not in targets:
        return False
    return item is None or _norm(item) in _norm(r["item"]) or _norm(item) in _norm(r["vendor"])


def _day(ext: extract_mod.Extraction, today: date) -> date:
    if ext.date:
        return date.fromisoformat(ext.date)
    if ext.weekday:
        return when.next_weekday(ext.weekday, today)
    return today


def _day_block(gs: list[dict[str, Any]], with_menu: bool) -> list[str]:
    lines = []
    for g in gs:
        line = f"- {vendor_line(g)}"
        if with_menu:
            line += " " + "、".join(menu_line(*it) for it in g["items"])
        lines.append(line)
    return lines


def _facts(head: str, days: list[tuple[date, list[dict[str, Any]]]], months: dict[str, int]) -> str:
    lines = [head]
    for day, gs in days:
        lines.append(f"[{day_label(day)}]")
        for g in gs:
            lines.append(f"  - {vendor_line(g)}")
            lines += [f"      {menu_line(*it)}" for it in g["items"]]
    registered = " / ".join(months) or "（なし）"
    lines.append(f"登録済みの出店データの月: {registered}")
    return "\n".join(lines)


def lookup(
    question: str, ext: extract_mod.Extraction, conn: sqlite3.Connection, today: date
) -> KitchenCarLookup:
    months = repository.kitchen_car_months(conn)
    locations = list(dict.fromkeys(r["location"] for r in repository.list_kitchen_cars(conn)))
    place, targets = _place(question, locations)
    item = ext.kitchen_item
    where = f"{place}の" if place else ""
    what = f"「{item}」を出す" if item else ""

    if place and not targets:
        text = (
            f"{place}のキッチンカーの出店情報はありません。"
            + (f"\n\nキッチンカーの出店場所：{' / '.join(locations)}" if locations else "")
        )
        return KitchenCarLookup(text, _facts(f"案内文:\n{text}", [], months))

    if item and not (ext.date or ext.weekday):
        # 「タコライスはいつ来る？」: 今日以降でそれが出る日
        rows = [r for r in repository.list_kitchen_cars(conn, since=today) if _matches(r, item, targets or None)]
        days = _by_day(rows)
        if not days:
            text = f"{day_label(today)}以降、{where}{what}キッチンカーの出店予定はありません。"
        else:
            lines = [f"{day_label(today)}以降、{where}{what}キッチンカーの出店予定です。", ""]
            for day, gs in days:
                lines.append(f"**{day_label(day)}**")
                lines += _day_block(gs, with_menu=True)
            text = "\n".join(lines)
        head = f"質問の対象: {where}{item}（{day_label(today)}以降。出店日・店・場所・時間・メニューは表から引いたもの）"
        return KitchenCarLookup(text, _facts(head, days, months), days)

    day = _day(ext, today)
    all_rows = repository.list_kitchen_cars(conn, day=day)
    rows = [r for r in all_rows if _matches(r, item, targets or None)]
    gs = groups(rows)
    label = day_label(day)
    if not all_rows:
        text = f"{label}のキッチンカーは、{NO_CARS}。"
    elif not gs:
        text = f"{label}は、{where}{what}キッチンカーの出店予定はありません。"
    else:
        lines = [f"{label}の{where}キッチンカーです（メニューは「メニューを見る」）。", ""]
        lines += _day_block(gs, with_menu=bool(item))
        text = "\n".join(lines)
    head = f"質問の対象: {label}の{where}キッチンカー" + (f"（{item}）" if item else "")
    if not all_rows:
        head += f"\n{label}は出店予定なし（{NO_CARS}）"
    days = [(day, gs)] if gs else []
    return KitchenCarLookup(text, _facts(head, days, months), days)


def _by_day(rows: list[dict[str, Any]]) -> list[tuple[date, list[dict[str, Any]]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for g in groups(rows):
        out.setdefault(g["date"], []).append(g)
    return [(date.fromisoformat(d), gs) for d, gs in out.items()]
