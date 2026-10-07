"""絞り込み + お知らせによる除外 + 意味検索による並べ替え。"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from src.choices import LECTURE_BUILDING, UNSURVEYED
from src.db import repository, vector_store
from src.ingest.notice import demo_label
from src.rag.when import Slot

# 回答生成に渡す候補数（利用者の目的に合わせて LLM が順位を決めるため、表示する件数より多く渡す）
TOP_K = 10
# おすすめにしない講義棟の教室を、詳細カードにだけ載せる件数
MAX_CLASSROOM_CARDS = 2
MAX_REVIEWS_PER_SPOT = 3

# 0件のとき緩める順（場所の指定 → 付帯条件 → 本質的な条件）
RELAX_ORDER = (
    "room", "floor", "building", "chair_type", "standing", "task_light", "food", "min_seats",
    "quietness", "outlets_required",
)  # fmt: skip

Similarity = Callable[[str, list[str]], dict[str, float]]


@dataclass
class Retrieval:
    conditions: dict[str, Any]
    spots: list[dict[str, Any]] = field(default_factory=list)
    # 除外したスポット: {"spot": {...}, "reason": str, "notice_id": int}
    excluded: list[dict[str, Any]] = field(default_factory=list)
    # 候補に関係するお知らせ（除外に使ったもの以外も含む）
    notices: list[dict[str, Any]] = field(default_factory=list)
    # 0件だったため緩めた条件（緩めていなければ None）
    relaxed_key: str | None = None
    relaxed_from: Any = None
    # 元の条件のうち「未調査」「空欄」の項目があるため一致させなかったスポット
    # {"spot": {...}, "keys": [条件キー]}
    unknown: list[dict[str, Any]] = field(default_factory=list)
    # 質問・画面で指定された日付・時限（指定なしなら None）
    slot: Slot | None = None
    # slot の日付の利用状況データがあるか（False なら講義棟の教室は外していない）
    has_reservation_data: bool = False
    # 時間が重なる予約があるため外した教室: {"spot": {...}, "reservations": [...]}
    class_excluded: list[dict[str, Any]] = field(default_factory=list)
    # 空いていると確認できないためおすすめにしない講義棟の教室（詳細カードにだけ載せる）
    classrooms: list[dict[str, Any]] = field(default_factory=list)

    @property
    def classrooms_confirmed(self) -> bool:
        """日時の指定があり、その日の予約データで空いているか確かめられるか。"""
        return self.slot is not None and self.has_reservation_data


def is_classroom(spot: dict[str, Any]) -> bool:
    return spot["building"] == LECTURE_BUILDING


def reserved_spots(conn: sqlite3.Connection, slot: Slot) -> dict[str, list[dict[str, Any]]]:
    """slot と時間が重なる予約がある spot_id → 予約。

    予約の spot_id を使い、空欄なら講義棟のスポットのエリア名と教室名（room）で対応づける。
    """
    rooms = {s["area_name"]: s["spot_id"] for s in repository.list_spots(conn) if is_classroom(s)}
    busy: dict[str, list[dict[str, Any]]] = {}
    for r in repository.overlapping_reservations(conn, slot.day.isoformat(), slot.start, slot.end):
        if sid := r["spot_id"] or rooms.get(r["room"]):
            busy.setdefault(sid, []).append(r)
    return busy


def _apply_reservations(
    spots: list[dict[str, Any]], busy: dict[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept, excluded = [], []
    for s in spots:
        if s["spot_id"] in busy:
            excluded.append({"spot": s, "reservations": busy[s["spot_id"]]})
        else:
            kept.append(s)
    return kept, excluded


def notice_matches(notice: dict[str, Any], spot: dict[str, Any]) -> bool:
    """お知らせの対象範囲（建物・階・エリア）にスポットが含まれるか。空欄は「全体」扱い。

    建物が書かれていないお知らせ（「閲覧室」など）は、エリア名だけで対応づける。
    """
    area = notice.get("target_area")
    if notice.get("target_building"):
        if notice["target_building"] != spot["building"]:
            return False
    elif not area:
        return False
    if notice.get("target_floor") is not None and notice["target_floor"] != spot["floor"]:
        return False
    return not area or area in spot["area_name"] or spot["area_name"] in area


def notice_place(notice: dict[str, Any]) -> str:
    """お知らせの対象の場所（建物・階・エリア。空欄は省く）。"""
    floor = None if notice.get("target_floor") is None else f"{notice['target_floor']}階"
    parts = (notice.get("target_building"), floor, notice.get("target_area"))
    return " ".join(p for p in parts if p) or "場所の記載なし"


def _describe(notice: dict[str, Any]) -> str:
    period = f"{notice.get('valid_from') or ''}〜{notice.get('valid_to') or ''}"
    return (
        f"{demo_label(notice)}お知らせ「{notice.get('title') or '(無題)'}」により "
        f"{notice_place(notice)} は{notice['impact_type']}中（{period}）"
    )


def _apply_notices(
    spots: list[dict[str, Any]], notices: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    kept, excluded, related = [], [], {}
    for s in spots:
        hits = [n for n in notices if notice_matches(n, s)]
        closure = next((n for n in hits if n["impact_type"] == "閉鎖"), None)
        if closure:
            excluded.append({"spot": s, "reason": _describe(closure), "notice_id": closure["notice_id"]})
        else:
            kept.append(s)
        for n in hits:
            related[n["notice_id"]] = n
    return kept, excluded, list(related.values())


# 条件キー → spots の列
_COND_COLUMN = {
    "building": "building", "room": "area_name", "floor": "floor", "outlets_required": "outlets",
    "min_seats": "seats", "standing": "standing", "task_light": "task_light",
    "chair_type": "chair_type", "quietness": "quietness", "food": "food",
}  # fmt: skip


def _known_match(key: str, value: Any, spot_value: Any) -> bool:
    if key == "outlets_required":
        return spot_value > 0
    if key == "min_seats":
        return spot_value >= value
    if isinstance(value, list):
        return spot_value in value
    return spot_value == value


def unknown_blocked(
    spots: list[dict[str, Any]], cond: dict[str, Any]
) -> list[dict[str, Any]]:
    """分かっている値では条件を満たすが、条件の項目が「未調査」か空欄のため一致させなかったスポット。"""
    blocked = []
    for s in spots:
        keys, ok = [], True
        for key, value in cond.items():
            v = s[_COND_COLUMN[key]]
            if v is None or v == UNSURVEYED:
                keys.append(key)
            elif not _known_match(key, value, v):
                ok = False
                break
        if ok and keys:
            blocked.append({"spot": s, "keys": keys})
    return blocked


def _rank(question: str, spots: list[dict[str, Any]], similarity: Similarity) -> list[dict]:
    scores = similarity(question, [s["spot_id"] for s in spots])
    for s in spots:
        s["similarity"] = round(scores.get(s["spot_id"], -1.0), 4)
    # 類似度が同じなら電源確保率の高い順
    return sorted(spots, key=lambda s: (s["similarity"], s["outlet_ratio"] or 0), reverse=True)


def full_power(spot: dict[str, Any]) -> bool:
    """全席で電源が使える（コンセント数が座席数以上。講義棟では机にコンセントがある教室）。"""
    return bool(spot["outlets"]) and spot["seats"] is not None and spot["outlets"] >= spot["seats"]


def classrooms_by_power(spots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """講義棟の教室だけを、全席で電源が使える教室 → 電源確保率の高い順に並べ替える。
    講義棟以外のスポットの位置と順番は変えない（教室どうしの並び順だけ入れ替える）。"""
    rooms = sorted(
        (s for s in spots if is_classroom(s)),
        key=lambda s: (full_power(s), s["outlet_ratio"] or 0),
        reverse=True,
    )  # sorted は安定なので、同じ段階の中では元の順番（類似度順）のまま
    it = iter(rooms)
    return [next(it) if is_classroom(s) else s for s in spots]


def _search(conn, cond, notices, busy):
    spots, class_excluded = _apply_reservations(repository.search_spots(conn, cond), busy)
    return (*_apply_notices(spots, notices), class_excluded)


def retrieve(
    question: str,
    conditions: dict[str, Any],
    conn: sqlite3.Connection,
    *,
    today: date | None = None,
    similarity: Similarity | None = None,
    slot: Slot | None = None,
) -> Retrieval:
    similarity = similarity or vector_store.similarity_by_spot
    notices = repository.active_notices(conn, today)
    result = Retrieval(conditions=dict(conditions), slot=slot)
    busy: dict[str, list[dict[str, Any]]] = {}
    if slot is not None:
        # 予約データがない日付なら、講義棟の教室は外さない（回答で「データなし」と注記する）
        result.has_reservation_data = slot.day.isoformat() in repository.reservation_dates(conn)
        if result.has_reservation_data:
            busy = reserved_spots(conn, slot)

    kept, excluded, related, result.class_excluded = _search(conn, conditions, notices, busy)
    result.excluded, result.notices = excluded, related
    if conditions:
        candidates, _ = _apply_reservations(repository.list_spots(conn), busy)
        candidates, _, _ = _apply_notices(candidates, notices)
        result.unknown = unknown_blocked(candidates, conditions)

    if not kept:
        # 条件を1つだけ緩めて再検索し、候補を示す
        for key in RELAX_ORDER:
            if key not in conditions:
                continue
            relaxed = {k: v for k, v in conditions.items() if k != key}
            k2, e2, r2, c2 = _search(conn, relaxed, notices, busy)
            if k2:
                kept, result.class_excluded = k2, c2
                result.relaxed_key, result.relaxed_from = key, conditions[key]
                result.excluded += [e for e in e2 if e not in result.excluded]
                result.notices += [n for n in r2 if n not in result.notices]
                break

    ranked = _rank(question, kept, similarity)
    if conditions.get("outlets_required"):
        # 電源を求められたら、講義棟はコンセントの多い（机にコンセントがある）教室を優先する
        ranked = classrooms_by_power(ranked)
    # 日時の指定がない（または予約データがない日付の）ときは、講義棟の教室をおすすめの候補にせず、
    # 詳細カードにだけ載せる。教室番号を指定された質問はその教室について答える
    if not result.classrooms_confirmed and "room" not in conditions:
        result.classrooms = [s for s in ranked if is_classroom(s)][:MAX_CLASSROOM_CARDS]
        ranked = [s for s in ranked if not is_classroom(s)]
    ranked = ranked[:TOP_K]
    for s in ranked + result.classrooms:
        s["reviews"] = repository.list_reviews(conn, s["spot_id"])[:MAX_REVIEWS_PER_SPOT]
    result.spots = ranked
    # お知らせは提示するスポット・除外したスポットに関係するものだけに絞る
    shown = ranked + result.classrooms + [e["spot"] for e in result.excluded]
    result.notices = [n for n in result.notices if any(notice_matches(n, s) for s in shown)]
    return result
