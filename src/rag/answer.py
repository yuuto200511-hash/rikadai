"""回答生成（質問応答の全体の流れもここで束ねる）。"""

from __future__ import annotations

import difflib
import json
import logging
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from collections.abc import Callable
from typing import Any
from zoneinfo import ZoneInfo

from src import prompts
from src.choices import NOTICE_IMPACT_TYPES, UNSURVEYED
from src.db import repository
from src.db.rooms import Room, load_rooms
from src.ingest.notice import demo_label, is_demo, strip_demo_mark
from src.llm import claude_cli
from src.rag import extract as extract_mod
from src.rag import reply as reply_mod
from src.rag import retrieve as retrieve_mod
from src.rag import when

logger = logging.getLogger(__name__)

LABELS = {
    "building": "建物", "room": "教室", "floor": "階", "outlets_required": "電源あり", "min_seats": "座席数",
    "chair_type": "椅子", "standing": "立ち作業", "quietness": "静粛性", "food": "飲食",
    "task_light": "作業灯",
}  # fmt: skip
# 回答に渡すスポットの項目（source は記録用なので渡さない）
SPOT_FIELD_LABELS = {
    "building": "建物", "floor": "階", "seats": "座席数", "outlets": "コンセント数",
    "chair_type": "椅子", "standing": "立ち作業", "quietness": "静粛性", "food": "飲食",
    "task_light": "作業灯", "surveyed_at": "調査日", "surveyor": "調査者",
}  # fmt: skip
BLANK = "未記入"
# おすすめの最大件数（「詳細を見る」のカードも同じ件数・同じ順番）
MAX_PICKS = 3
PICK_LABELS = tuple(f"おすすめ{i}" for i in range(1, MAX_PICKS + 1))
# 次の質問の条件抽出・返事に渡す、直近の会話のやりとりの数
MAX_HISTORY_TURNS = 3
# 口コミの確認で答えるスポット数と、1スポットあたりの口コミ数
MAX_REVIEW_SPOTS = 5
MAX_REVIEWS_SHOWN = 5
# 対象を指定されなかったときに答える、最近の口コミの件数
MAX_RECENT_REVIEWS = 10


@dataclass
class Answer:
    text: str
    # normal / relaxed / no_result / unsupported / clarify / unsurveyed / notice / chat / error
    kind: str
    conditions: dict[str, Any] = field(default_factory=dict)
    retrieval: retrieve_mod.Retrieval | None = None
    # 「おすすめ1〜3」: {"spot": {...}, "reason": str}
    picks: list[dict[str, Any]] = field(default_factory=list)
    # 聞き返し（clarify）・未調査（unsurveyed）のとき、候補に置き換えた質問文
    suggestions: list[str] = field(default_factory=list)
    # 「詳細を見る」に出すお知らせ（お知らせの確認・対応外の質問で、返事が触れたもの）
    notices: list[dict[str, Any]] = field(default_factory=list)
    # 返事を書かせるときに LLM に渡したデータ（評価で根拠情報として使う）
    context: str = ""
    # 質問から抽出した条件など（次の質問で、会話の続きとして条件を引き継ぐために使う）
    extraction: extract_mod.Extraction | None = None
    # 意味検索に使った文（会話の続きなら、前の質問をつなげたもの）
    search_text: str = ""
    # 口コミの確認で答えた口コミ: {"spot": {...}, "reviews": [...]}
    reviews: list[dict[str, Any]] = field(default_factory=list)

    @property
    def spot_ids(self) -> list[str]:
        """画面に出すスポット（カードの順。おすすめ → 授業がない時間のみ利用できる教室）。"""
        return [s["spot_id"] for s, _ in cards(self)]


def format_conditions(cond: dict[str, Any]) -> str:
    if not cond:
        return "（指定なし）"
    parts = []
    for k, v in cond.items():
        if isinstance(v, list):
            v = " / ".join(v)
        elif v is True:
            v = "必要"
        elif k == "min_seats":
            v = f"{v}席以上"
        elif k == "floor":
            v = f"{v}階"
        parts.append(f"{LABELS.get(k, k)}: {v}")
    return "、".join(parts)


def location(s: dict[str, Any]) -> str:
    """建物・階・エリア名。空欄は「建物未記入」「階未記入」と明示する。"""
    building = s["building"] or "建物未記入"
    floor = f"{s['floor']}階" if s["floor"] is not None else "階未記入"
    return f"{building} {floor} {s['area_name']}"


def power_level(s: dict[str, Any]) -> str:
    """利用者に見せる電源の段階（電源確保率の%は利用者には出さない。管理画面の集計だけで使う）。"""
    seats, outlets = s["seats"], s["outlets"]
    if outlets == 0:
        return "なし"
    if outlets is None or seats is None:
        return BLANK
    return "全席で使える" if outlets >= seats else "一部の席で使える"


def _v(s: dict[str, Any], key: str) -> str:
    return BLANK if s[key] is None else str(s[key])


def _format_spot(i: int, s: dict[str, Any]) -> str:
    unsurveyed = [label for k, label in SPOT_FIELD_LABELS.items() if s[k] == UNSURVEYED]
    blank = [label for k, label in SPOT_FIELD_LABELS.items() if s[k] is None]
    lines = [
        f"[{i}] {s['spot_id']} {location(s)}",
        f"  表示名: {user_location(s)}",
        f"  座席数: {_v(s, 'seats')} / コンセント数: {_v(s, 'outlets')} / 電源: {power_level(s)}",
        f"  コンセントの位置: {s['outlet_location'] or BLANK}",
        f"  椅子: {s['chair_type']} / 立ち作業: {s['standing']} / 静粛性: {s['quietness']}"
        f" / 飲食: {s['food']} / 作業灯: {s['task_light']}",
        f"  調査者の備考（公式）: {s['note'] or '（なし）'}",
        f"  調査日: {_v(s, 'surveyed_at')}",
        f"  未調査の項目: {'、'.join(unsurveyed) or '（なし）'}",
        f"  未記入の項目: {'、'.join(blank) or '（なし）'}",
    ]
    reviews = s.get("reviews") or []
    if reviews:
        lines.append("  利用者の声（口コミ。公式データではない）:")
        for r in reviews:
            rating = f"［{r['rating']}］" if r.get("rating") else ""
            lines.append(f"    - {rating}{r['body']}（{r['posted_at'][:10]} 投稿）")
    return "\n".join(lines)


def _format_notice(n: dict[str, Any]) -> str:
    return (
        f"- {demo_label(n)}{n.get('title') or '(無題)'}：{retrieve_mod.notice_place(n)} / "
        f"{n['impact_type']} / {n.get('valid_from') or ''}〜{n.get('valid_to') or ''}"
    )


def user_location(s: dict[str, Any]) -> str:
    """利用者に見せる場所。空欄の建物・階は書かない（「未記入」も出さない）。"""
    floor = f"{s['floor']}階" if s["floor"] is not None else None
    return " ".join(p for p in (s["building"], floor, s["area_name"]) if p)


CLASSROOM_NOTE = "授業がない時間のみ利用可"


def no_data_note(r: retrieve_mod.Retrieval) -> str | None:
    if r.slot is None or r.has_reservation_data:
        return None
    return f"{r.slot.weekday}曜日の利用状況データがありません"


def classroom_note(s: dict[str, Any], r: retrieve_mod.Retrieval) -> str | None:
    """講義棟の教室についての利用状況の注記（講義棟以外は None）。"""
    if not retrieve_mod.is_classroom(s):
        return None
    if r.slot is None:
        return CLASSROOM_NOTE
    if not r.has_reservation_data:
        return f"{no_data_note(r)}（{CLASSROOM_NOTE}）"
    return f"{r.slot.label}は予約なし"


def format_unknown(r: retrieve_mod.Retrieval) -> str:
    """未調査・未記入のため条件に一致させなかったスポットを1文にまとめる。

    回答に出すスポットと、建物・階が空欄なだけのものは数えない（空欄の建物は利用者に出さない）。
    例: 「図書館の2件・その他の3件は静粛性が未調査のため、条件に合うか分かりません。」
    """
    shown = {s["spot_id"] for s in r.spots + r.classrooms}
    counts: dict[str, int] = {}
    labels: dict[str, None] = {}
    states: set[str] = set()
    for u in r.unknown:
        s = u["spot"]
        if s["spot_id"] in shown or {"building", "floor"} & set(u["keys"]):
            continue
        key = s["building"] or "その他"
        counts[key] = counts.get(key, 0) + 1
        for k in u["keys"]:
            labels[LABELS.get(k, k)] = None
            states.add("未調査" if s[retrieve_mod._COND_COLUMN[k]] == UNSURVEYED else "未記入")
    if not counts:
        return ""
    # 「その他」は最後に
    where = "・".join(
        f"{b}の{n}件" for b, n in sorted(counts.items(), key=lambda kv: kv[0] == "その他")
    )
    state = "・".join(x for x in ("未調査", "未記入") if x in states)
    return f"{where}は{'・'.join(labels)}が{state}のため、条件に合うか分かりません。"


def note_line(r: retrieve_mod.Retrieval) -> str:
    """冒頭の下に添える1行（条件を緩めた場合・未調査で一致させなかったスポットがある場合・
    指定された日の利用状況データがない場合）。"""
    parts = []
    if note := no_data_note(r):
        parts.append(f"{note}（講義棟の教室は予約の有無を確認できていません）。")
    if r.relaxed_key:
        parts.append(
            f"「{format_conditions({r.relaxed_key: r.relaxed_from})}」に合う場所はなかったため、"
            "この条件を外した候補です。"
        )
    if unknown := format_unknown(r):
        parts.append(("ほかに、" if r.relaxed_key else "") + unknown)
    return "".join(parts)


def _format_candidate(i: int, s: dict[str, Any], r: retrieve_mod.Retrieval) -> str:
    text = _format_spot(i, s)
    if note := classroom_note(s, r):
        text += f"\n  教室の利用状況: {note}"
    return text


REVIEW_REQUEST = (
    "学生は口コミ（利用者の声）の内容も知りたがっています。紹介する場所に利用者の声があれば、"
    "その内容を「口コミでは〜」のように公式データと区別して伝えてください。"
    "利用者の声がない場所は、口コミがまだないことを短く伝えてください。"
)


def build_prompt(
    question: str,
    r: retrieve_mod.Retrieval,
    today: date | None = None,
    *,
    history: str = "",
    follow_up: bool = False,
    want_reviews: bool = False,
) -> str:
    if r.relaxed_key:
        status = (
            "元の条件に完全一致するスポットはなかったため、"
            f"条件「{format_conditions({r.relaxed_key: r.relaxed_from})}」を外した候補です。"
        )
    elif r.conditions:
        status = "条件に一致したスポットです。"
    else:
        status = "質問から検索条件は抽出されなかったため、全スポットが対象です。"
    if follow_up:
        status = "会話の続きの質問のため、前の条件に今回の発言を反映した条件で検索しました。" + status
    return prompts.load(
        "answer",
        today=(today or date.today()).isoformat(),
        history=history or "（なし）",
        question=question,
        conditions=format_conditions(r.conditions),
        slot=r.slot.label if r.slot else "（指定なし）",
        status=status,
        spots="\n".join(_format_candidate(i, s, r) for i, s in enumerate(r.spots, 1)) or "（なし）",
        notices="\n".join(_format_notice(n) for n in r.notices) or "（なし）",
        notes=note_line(r) or "（なし）",
        reviews=REVIEW_REQUEST if want_reviews else "（特になし）",
    )


_NUMBER_RE = re.compile(r"\d+")
# 利用者に出さない表現（電源確保率）
_FORBIDDEN_IN_REASON = re.compile(r"[%％]|確保率")


def _data_reason(s: dict[str, Any]) -> str:
    """LLM の理由が使えないときの、DB の値だけで作る理由。"""
    if s["outlets"]:
        return f"電源：{power_level(s)}（コンセント{s['outlets']}口）"
    if s["quietness"] not in (None, UNSURVEYED):
        return f"静粛性：{s['quietness']}"
    return f"座席{_v(s, 'seats')}席"


def _checked_reason(reason: Any, s: dict[str, Any], context: str) -> str:
    """理由に出てくる数値が、そのスポットのデータにすべてあるか確かめる。なければ DB の値で作り直す。"""
    if not isinstance(reason, str) or not reason.strip():
        return _data_reason(s)
    reason = reason.strip().splitlines()[0]
    if _FORBIDDEN_IN_REASON.search(reason) or any(
        n not in context for n in _NUMBER_RE.findall(reason)
    ):
        logger.warning("reason has numbers not in data: %s / %s", s["spot_id"], reason)
        return _data_reason(s)
    return reason


def is_standing_only(s: dict[str, Any]) -> bool:
    """立ち作業の席（椅子がない）。"""
    return s["chair_type"] == "なし"


def _standing_not_first(
    picks: list[dict[str, Any]], r: retrieve_mod.Retrieval
) -> list[dict[str, Any]]:
    """立ち作業を望んでいないなら、立ち作業の席をおすすめ1にしない。

    おすすめの中に座れる席があれば先頭に上げ、なければ候補から座れる席を1件目に足す。
    """
    if r.conditions.get("standing") == "可" or not picks or not is_standing_only(picks[0]["spot"]):
        return picks
    seated = next((p for p in picks if not is_standing_only(p["spot"])), None)
    if seated is None:
        s = next((s for s in r.spots if not is_standing_only(s)), None)
        if s is None:
            return picks
        seated = {"spot": s, "reason": _data_reason(s)}
        picks = [seated, *picks][:MAX_PICKS]
    return [seated, *[p for p in picks if p is not seated]]


def _power_classrooms_first(
    picks: list[dict[str, Any]], r: retrieve_mod.Retrieval
) -> list[dict[str, Any]]:
    """電源を求められたのに、全席で電源が使える教室を差し置いて一部の席だけの教室をおすすめにしていたら、
    候補にある全席で電源が使える教室に置き換える（講義棟以外のおすすめは変えない）。"""
    if not r.conditions.get("outlets_required"):
        return picks
    picked = {p["spot"]["spot_id"] for p in picks}
    better = iter(
        s for s in r.spots
        if retrieve_mod.is_classroom(s) and retrieve_mod.full_power(s) and s["spot_id"] not in picked
    )  # fmt: skip
    out, replaced = [], []
    for p in picks:
        s = p["spot"]
        if retrieve_mod.is_classroom(s) and not retrieve_mod.full_power(s):
            if (b := next(better, None)) is not None:
                p = {"spot": b, "reason": _data_reason(b)}
                replaced.append(p)
        out.append(p)
    # 全席で電源が使える教室を、一部の席だけの教室より前にする（LLM が選んだ教室は置き換えた教室より前。
    # 講義棟以外の位置は変えない）
    rooms = sorted(
        (p for p in out if retrieve_mod.is_classroom(p["spot"])),
        key=lambda p: (retrieve_mod.full_power(p["spot"]), all(p is not x for x in replaced)),
        reverse=True,
    )
    it = iter(rooms)
    return [next(it) if retrieve_mod.is_classroom(p["spot"]) else p for p in out]


def _llm_pick_ids(raw: dict[str, Any], r: retrieve_mod.Retrieval) -> list[str]:
    """LLM が選んだ候補の spot_id（候補外・重複を除き、最大 MAX_PICKS 件）。"""
    ids = {s["spot_id"] for s in r.spots}
    out: list[str] = []
    for item in raw.get("picks") or []:
        sid = item.get("spot_id") if isinstance(item, dict) else None
        if sid in ids and sid not in out:
            out.append(sid)
    return out[:MAX_PICKS]


def spot_reply(
    raw: dict[str, Any], r: retrieve_mod.Retrieval, picks: list[dict[str, Any]], context: str,
    question: str = "",
) -> str | None:  # fmt: skip
    """LLM が書いた返事。データにない数値・場所の名前がある、またはおすすめの順番をコードで変えた
    （返事とカードの順番が合わない）ときは None（定型文を使う）。"""
    if _llm_pick_ids(raw, r) != [p["spot"]["spot_id"] for p in picks]:
        return None
    return reply_mod.checked(raw.get("reply"), context, question)


def choose_picks(raw: dict[str, Any], r: retrieve_mod.Retrieval) -> list[dict[str, Any]]:
    """LLM の出力から「おすすめ1〜3」（最大 MAX_PICKS 件）を作る。候補外の spot_id は捨てる。

    候補が少なければその件数だけにする（無理に3件そろえない）。
    """
    by_id = {s["spot_id"]: s for s in r.spots}
    picks: list[dict[str, Any]] = []
    for item in raw.get("picks") or []:
        if not isinstance(item, dict):
            continue
        s = by_id.get(item.get("spot_id"))
        if s is None or any(p["spot"] is s for p in picks):
            continue
        context = _format_candidate(0, s, r)
        picks.append({"spot": s, "reason": _checked_reason(item.get("reason"), s, context)})
        if len(picks) == MAX_PICKS:
            break
    if not picks and r.spots:
        picks = [{"spot": r.spots[0], "reason": _data_reason(r.spots[0])}]
    return _standing_not_first(_power_classrooms_first(picks, r), r)


CLASSROOMS_ONLY = (
    "日時の指定がないため、講義棟の教室はおすすめに含めていません（{note}）。"
    "「詳細を見る」に候補を載せています。日付と時限を指定すると、空いている教室を確かめられます。"
)


def _review_line(rv: dict[str, Any]) -> str:
    rating = f"［{rv['rating']}］" if rv.get("rating") else ""
    return f"{rating}{rv['body']}（{rv['posted_at'][:10]}）"


def format_headline(
    picks: list[dict[str, Any]], r: retrieve_mod.Retrieval, want_reviews: bool = False
) -> str:
    if not picks and r.classrooms:
        text = CLASSROOMS_ONLY.format(note=classroom_note(r.classrooms[0], r))
        if note := note_line(r):
            text += f"\n\n※ {note}"
        return text
    lines = []
    for label, p in zip(PICK_LABELS, picks, strict=False):
        s = p["spot"]
        note = classroom_note(s, r)
        suffix = f"（{CLASSROOM_NOTE}）" if note and CLASSROOM_NOTE in note else ""
        line = f"**{label}：{user_location(s)}**{suffix}  \n{p['reason']}"
        if want_reviews:
            reviews = s.get("reviews") or []
            line += "".join(f"  \n口コミ：{_review_line(rv)}" for rv in reviews) or "  \n口コミ：まだありません"
        lines.append(line)
    text = "\n\n".join(lines)
    if note := note_line(r):
        text += f"\n\n※ {note}"
    return text


def _card_value(s: dict[str, Any], key: str) -> str:
    return BLANK if s[key] is None else str(s[key])


def format_card(s: dict[str, Any], r: retrieve_mod.Retrieval, label: str | None = None) -> str:
    """スポットの詳細カード（DB の値からのテンプレート。LLM は使わない）。"""
    outlets = f"（コンセント{s['outlets']}口）" if s["outlets"] else ""
    seats = BLANK if s["seats"] is None else f"{s['seats']}席"
    lines = [
        f"#### {label + '｜' if label else ''}{user_location(s)}",
        f"- 電源：{power_level(s)}{outlets} ／ 座席：{seats}",
    ]
    if s["outlet_location"]:
        lines.append(f"- コンセントの位置：{s['outlet_location']}")
    lines += [
        f"- 椅子：{s['chair_type']} ／ 立ち作業：{s['standing']} ／ 作業灯：{s['task_light']}",
        f"- 静粛性：{s['quietness']} ／ 飲食：{s['food']}",
    ]
    if s["note"]:
        lines.append(f"- 調査者の備考：{s['note']}")
    if note := classroom_note(s, r):
        lines.append(f"- 教室の利用状況：{note}")
    for n in r.notices:
        if retrieve_mod.notice_matches(n, s):
            lines.append(f"- お知らせ：{_format_notice(n)[2:]}")
    for rv in s.get("reviews") or []:
        lines.append(f"- 利用者の声（口コミ）：{_review_line(rv)}")
    lines.append(f"- 調査日：{_card_value(s, 'surveyed_at')}")
    return "\n".join(lines)


def cards(ans: Answer) -> list[tuple[dict[str, Any], str]]:
    """回答に添えるカード。おすすめと同じスポットを同じ順番で出し、日時の指定がないときは
    講義棟の教室を「授業がない時間のみ利用可」として後ろに足す（おすすめにはしない）。"""
    r = ans.retrieval
    if r is None:
        return []
    out = [
        (p["spot"], format_card(p["spot"], r, label))
        for label, p in zip(PICK_LABELS, ans.picks, strict=False)
    ]
    out += [(s, format_card(s, r, CLASSROOM_NOTE)) for s in r.classrooms]
    return out


def side_notes(r: retrieve_mod.Retrieval) -> list[str]:
    """カードの下に添える補足（予約・お知らせで外したスポット）。"""
    notes = []
    if r.class_excluded:
        rooms = "、".join(e["spot"]["area_name"] for e in r.class_excluded)
        notes.append(
            f"{r.slot.label}と時間が重なる予約があるため、教室{len(r.class_excluded)}件"
            f"（{rooms}）を候補から外しました。"
        )
    for e in r.excluded:
        notes.append(f"{user_location(e['spot'])} は候補から外しました：{e['reason']}")
    return notes


ROOM_RE = extract_mod.ROOM_RE
MAX_SUGGESTIONS = 3
# あることは分かっているが調査していない場所（建物以外）。質問に書かれていたら未調査と答え、
# 名前や用途の近い調査済みの場所の情報で答えない（談話室を閲覧室と混同しない）
UNSURVEYED_PLACES = ("談話室",)


def _close(word: str, choices: list[str]) -> list[str]:
    """word に近い候補（表記の近さ順）。"""
    return difflib.get_close_matches(word, choices, n=MAX_SUGGESTIONS, cutoff=0.4)


def _room_tokens(question: str, ext: extract_mod.Extraction, real_rooms: set[str]) -> list[str]:
    """質問に書かれた教室番号（LLM の抽出・「K503」形式の表記・一覧にある「7401」などの表記）。"""
    q = extract_mod.normalize_room(question) or ""
    written = ROOM_RE.findall(q)
    written += [r for r in sorted(real_rooms, key=len, reverse=True)
                if not ROOM_RE.fullmatch(r) and r in q]  # fmt: skip
    return list(dict.fromkeys([*written, *([ext.room] if ext.room else [])]))


def _replaced(question: str, written: str, candidates: list[str]) -> list[str]:
    """質問文に書かれた表記を候補に置き換えた質問（表記が見つからなければ空）。"""
    span = _find_written(question, written)
    return [question[: span[0]] + c + question[span[1] :] for c in candidates] if span else []


def _room_number(room: str) -> tuple[int, int] | None:
    """講義棟の教室番号（K203 → (2, 3)：階と番号）。"""
    m = re.fullmatch(r"K(\d)(\d{2})", room)
    return (int(m[1]), int(m[2])) if m else None


def _nearby_classrooms(room: str, spots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """未調査の教室に近い、調査済みの講義棟の教室（階が近い順 → 番号が近い順）。"""
    target = _room_number(room)
    if target is None:
        return []
    near = [
        (abs(n[0] - target[0]), abs(n[1] - target[1]), s)
        for s in spots
        if retrieve_mod.is_classroom(s) and (n := _room_number(s["area_name"]))
    ]
    return [s for *_, s in sorted(near, key=lambda t: t[:2])][:MAX_SUGGESTIONS]


def _unsurveyed_room(
    question: str, room: str, spots: list[dict[str, Any]], places: list[str],
    cond: dict[str, Any],
) -> Answer:  # fmt: skip
    near = _nearby_classrooms(room, spots)
    if near:
        lines = ["調査済みの講義棟の教室："] + [f"- {user_location(s)}" for s in near]
        suggestions = _replaced(question, room, [s["area_name"] for s in near])
    else:
        lines = [surveyed_line(places)]
        suggestions = []
    text = f"{room} は未調査のため、設備の情報がありません。\n\n" + "\n".join(lines)
    return Answer(text, "unsurveyed", cond, suggestions=suggestions)


def _unsurveyed_building(
    question: str, building: str, places: list[str], cond: dict[str, Any]
) -> Answer:
    # 「◯号館」には、番号が似ているだけの「◯号館」を代わりに勧めない（23号館 → 21号館 など）
    candidates = places
    if extract_mod.written_buildings(building):
        candidates = [p for p in places if not extract_mod.written_buildings(p)]
    text = f"{building}は未調査のため情報がありません。\n\n{surveyed_line(places)}"
    return Answer(text, "unsurveyed", cond, suggestions=_replaced(question, building, candidates))


def _use_written_building(ext: extract_mod.Extraction, building: str) -> None:
    """条件・お知らせ・口コミの対象の建物名を、質問文に書かれた表記にそろえる。"""
    previous = ext.conditions.get("building")
    ext.conditions["building"] = building
    for key in ("notice_place", "review_place"):
        place = getattr(ext, key)
        if place and (place == previous or extract_mod.written_buildings(place)):
            setattr(ext, key, building)


def check_building(
    question: str, ext: extract_mod.Extraction, conn: sqlite3.Connection
) -> Answer | None:
    """建物名を質問文と照合する（教室番号と同じ扱い。LLM の読み替えを信じない）。

    - 質問文に書かれた「◯号館」が調査済みの建物になければ、未調査と答える（kind="unsurveyed"）
    - 質問文の「◯号館」と LLM が返した「◯号館」が食い違ったら、質問文のほうを正とする
    - 「体育館」のような、番号のない未調査の建物名（LLM の抽出）も未調査と答える
    調査済みかどうかは building 列で判定し、利用者に見せる一覧は surveyed_places（閲覧室・自習室も含む）。
    """
    buildings = repository.list_buildings(conn)
    written = extract_mod.written_buildings(question)
    for name in written:
        if name not in buildings:
            _use_written_building(ext, name)
            return _unsurveyed_building(question, name, surveyed_places(conn), ext.conditions)
    building = ext.conditions.get("building")
    if written and building and building not in written and extract_mod.written_buildings(building):
        # LLM が別の番号の「◯号館」に読み替えた（質問文の「図書館」などの建物名は上書きしない）
        _use_written_building(ext, written[0])
        building = written[0]
    if building and building not in buildings and building not in surveyed_places(conn):
        return _unsurveyed_building(question, building, surveyed_places(conn), ext.conditions)
    return None


def surveyed_places(conn: sqlite3.Connection) -> list[str]:
    """調査済みの場所を、利用者に通じる名前で並べる：建物と、建物名が未記入の場所
    （「閲覧室 6人掛けデスク」の「閲覧室」）。未調査と答えるときの一覧・候補ボタンに使う。"""
    places = [s["area_name"].split()[0] for s in repository.list_spots(conn) if not s["building"]]
    return list(dict.fromkeys([*repository.list_buildings(conn), *places]))


def surveyed_line(places: list[str]) -> str:
    return f"調査済みの建物・場所：{' / '.join(places)}"


def unsurveyed_place(question: str, ext: extract_mod.Extraction) -> str | None:
    """質問（または抽出した建物・場所）に書かれた、調査していない場所の名前。"""
    written = [
        question, ext.conditions.get("building"), ext.room, ext.review_place, ext.notice_place
    ]  # fmt: skip
    text = _norm("".join(w for w in written if w))
    return next((p for p in UNSURVEYED_PLACES if p in text), None)


def _unsurveyed_place(
    question: str, place: str, conn: sqlite3.Connection, cond: dict[str, Any]
) -> Answer:
    places = surveyed_places(conn)
    text = f"{place}は未調査のため情報がありません。\n\n{surveyed_line(places)}"
    return Answer(text, "unsurveyed", cond, suggestions=_replaced(question, place, places))


def check_places(
    question: str,
    ext: extract_mod.Extraction,
    conn: sqlite3.Connection,
    rooms: dict[str, Room] | None = None,
) -> Answer | None:
    """質問の教室番号・建物名を確かめる。

    - 調査済みの教室 → 条件（room）に加えて None を返す
    - 実在する（data/rooms.csv にある）が未調査の教室 → 未調査と伝え、調査済みの講義棟の教室（階・番号が近い順）を示す
    - 実在しない教室 → 推測で決めつけず、近い候補を示して聞き返す（kind="clarify"）
    - 調査済みの建物にない建物 → 聞き返さず、未調査と伝えて調査済みの建物・場所の一覧を示す
    - 調査していない場所（UNSURVEYED_PLACES。建物以外）→ 未調査と伝えて調査済みの建物・場所の一覧を示す
    教室番号・建物名は LLM の抽出に加えて質問文からも拾い、食い違ったら質問文を正とする
    （LLM が近い教室・近い番号の建物に直してしまっても見逃さない）。建物名の照合は check_building。
    """
    if place := unsurveyed_place(question, ext):
        return _unsurveyed_place(question, place, conn, ext.conditions)
    if unsurveyed := check_building(question, ext, conn):
        return unsurveyed
    if ext.room and not re.search(r"\d", ext.room):
        # 「閲覧室」のような数字のない場所の名前は教室番号ではない（条件にせず、意味検索に任せる）
        ext.room = None
    spots = repository.list_spots(conn)
    surveyed = {s["area_name"] for s in spots if retrieve_mod.is_classroom(s)}
    rooms = load_rooms() if rooms is None else rooms
    real = set(rooms) | surveyed
    for token in _room_tokens(question, ext, real):
        if token in surveyed:
            continue
        if token in rooms:
            places = surveyed_places(conn)
            return _unsurveyed_room(question, token, spots, places, ext.conditions)
        return _clarify(question, "教室", token, _close(token, sorted(real)), ext.conditions)
    if ext.room:
        ext.conditions["room"] = ext.room
    return None


def _find_written(question: str, written: str) -> tuple[int, int] | None:
    """質問文の中で、表記をそろえると written になる部分（全角・小文字・空白の違いを許す）。"""
    for i in range(len(question)):
        acc = ""
        for j in range(i, len(question)):
            acc += re.sub(r"\s", "", unicodedata.normalize("NFKC", question[j])).upper()
            if acc == written:
                return i, j + 1
            if not written.startswith(acc):
                break
    return None


def _clarify(
    question: str, kind: str, written: str, candidates: list[str], cond: dict[str, Any]
) -> Answer:
    # 質問文に書かれた表記を候補に置き換えた質問（表記が見つからなければ候補ボタンは出さない）
    suggestions = _replaced(question, written, candidates)
    text = prompts.load(
        "clarify",
        kind=kind,
        written=written,
        candidates="\n".join(f"- {c}" for c in candidates) or "（近い候補はありません）",
    )
    return Answer(text, "clarify", cond, suggestions=suggestions)


def _notice_about(n: dict[str, Any], place: str) -> bool:
    """お知らせが指定された建物・場所についてのものか。"""
    fields = [n.get("target_building"), n.get("target_area"), n.get("title")]
    return any(f and (place in f or f in place) for f in fields)


def _md(day: str | None) -> str:
    if not day:
        return ""
    d = date.fromisoformat(day)
    return f"{d.month}/{d.day}"


def _notice_period(n: dict[str, Any]) -> str:
    start, end = _md(n.get("valid_from")), _md(n.get("valid_to"))
    if not end:
        return f"{start}〜（終了日の記載なし）" if start else "期間の記載なし"
    return f"{start}〜{end}"


# お知らせの影響の種類を、利用者向けの言い方で添える
IMPACT_MEANINGS = {
    "閉鎖": "期間中は使えない", "時間変更": "利用できる時間が変わる",
    "開放": "新たに使えるようになる", "設備変更": "設備が変わる",
}  # fmt: skip
# 場所の名前ではない一般的な言い方（お知らせの確認で、これらでは対象を絞らない）
GENERIC_PLACES = frozenset(
    {"教室", "空き教室", "部屋", "場所", "ところ", "所", "席", "座席", "スペース", "施設", "建物",
     "エリア", "キャンパス", "作業スポット", "スポット"}
)  # fmt: skip


def notice_body(n: dict[str, Any]) -> str:
    """利用者に見せるメール本文（デモ用の印の行は除く）。"""
    return strip_demo_mark(n.get("raw_body") or "").strip()


def notice_facts(n: dict[str, Any]) -> str:
    """返事を書かせるときに渡すお知らせ1件のデータ。"""
    impact = n["impact_type"]
    lines = [
        f"[お知らせID {n['notice_id']}]" + ("（デモ用のお知らせ）" if is_demo(n) else ""),
        f"  件名: {n.get('title') or '(無題)'}",
        f"  対象の場所: {retrieve_mod.notice_place(n)}",
        f"  影響: {impact}（{IMPACT_MEANINGS.get(impact, impact)}）",
        f"  期間: {_notice_period(n)}",
        "  メール本文:",
        *(f"    {line}" for line in notice_body(n).splitlines()),
    ]
    return "\n".join(lines)


def notice_details(n: dict[str, Any]) -> str:
    """「詳細を見る」に出すお知らせ1件（DB の値からのテンプレート。LLM は使わない）。"""
    return "\n".join(
        [
            f"#### {demo_label(n)}{n.get('title') or '(無題)'}",
            f"- 対象：{retrieve_mod.notice_place(n)}",
            f"- 影響：{n['impact_type']}（{IMPACT_MEANINGS.get(n['impact_type'], n['impact_type'])}）",
            f"- 期間：{_notice_period(n)}",
        ]
    )


def notice_target(ext: extract_mod.Extraction, question: str) -> str | None:
    """お知らせを絞り込む建物・場所。名前が明示されたときだけで、「教室」「部屋」などの言い方では絞らない。

    LLM が notice_place を空にして building だけ入れた場合は、建物名が質問文に書かれているときだけ使う
    （「教室」→「講義棟」のような置き換えで絞らない）。
    """
    place = ext.notice_place
    if place and place not in GENERIC_PLACES:
        return place
    building = ext.conditions.get("building")
    q = unicodedata.normalize("NFKC", question)
    if building and building not in GENERIC_PLACES and building in q:
        return building
    return None


def _sorted_notices(notices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {t: i for i, t in enumerate(NOTICE_IMPACT_TYPES)}
    return sorted(notices, key=lambda n: (order.get(n["impact_type"], len(order)), n["notice_id"]))


def answer_notices(
    ext: extract_mod.Extraction, conn: sqlite3.Connection, today: date, question: str = ""
) -> Answer:
    """お知らせの確認：確認済みで、今日が有効期間内のお知らせ（未確認のものは出さない）。

    text は定型文（返事は answer_question で LLM が書き直す）。notices は「詳細を見る」に出す。
    """
    notices = _sorted_notices(repository.active_notices(conn, today))
    place = notice_target(ext, question)
    if place:
        notices = [n for n in notices if _notice_about(n, place)]
    about = f"{place}について、" if place else ""
    if not notices:
        return Answer(f"{about}現在、利用に影響するお知らせはありません。", "notice", ext.conditions)
    lines = [f"{about}現在、利用に影響するお知らせは次の{len(notices)}件です（{_md(today.isoformat())}時点）。", ""]
    lines += [
        f"- {demo_label(n)}{retrieve_mod.notice_place(n)} ／ {n['impact_type']} ／ {_notice_period(n)}"
        for n in notices
    ]
    return Answer("\n".join(lines), "notice", ext.conditions, notices=notices)


def _spot_line(s: dict[str, Any]) -> str:
    return (
        f"- {user_location(s)}：電源 {power_level(s)} ／ 静粛性 {s['quietness']} ／ 飲食 {s['food']}"
    )


def _capabilities(conn: sqlite3.Connection) -> str:
    return (
        "できること:\n"
        f"- 野田キャンパスの調査済みの作業スポット（{'・'.join(surveyed_places(conn))}）を、"
        "静かさ・電源・座席数・飲食・椅子などの条件で探す（例:「静かで電源がある場所は？」）\n"
        "- 日付と時限を指定して、授業・予約で使われていない講義棟の教室を探す\n"
        "- 施設のお知らせ（工事・閉鎖・開放など）を確認する"
    )


def _unsupported_facts(
    ext: extract_mod.Extraction, conn: sqlite3.Connection, today: date, question: str
) -> tuple[str, list[dict[str, Any]]]:
    """対応外の質問に添えるデータ：有効なお知らせ（建物・場所が書かれていればその場所のものだけ）と、
    質問に書かれた建物のスポット。"""
    notices = _sorted_notices(repository.active_notices(conn, today))
    place = notice_target(ext, question)
    if place:
        notices = [n for n in notices if _notice_about(n, place)]
    parts = ["## 現在有効なお知らせ", "\n".join(notice_facts(n) for n in notices) or "（なし）"]
    if place:
        spots = [s for s in repository.list_spots(conn) if s["building"] == place]
        parts += [f"## {place}の調査済みスポット", "\n".join(map(_spot_line, spots)) or "（なし）"]
    parts += ["## このチャットについて", _capabilities(conn)]
    return "\n\n".join(parts), notices


Turn = tuple[str, Answer]
INTENT_LABELS = {"notice": "お知らせの確認", "chat": "挨拶・雑談", "review": "口コミの確認"}


def shown_spots(ans: Answer) -> list[dict[str, Any]]:
    """回答で紹介したスポット（おすすめ → 詳細カードの教室 → 口コミを答えたスポット）。"""
    spots = [s for s, _ in cards(ans)] + [x["spot"] for x in ans.reviews]
    return list({s["spot_id"]: s for s in spots}.values())


def format_history(turns: list[Turn] | None) -> str:
    """直近の会話（最大 MAX_HISTORY_TURNS 回）を、条件抽出・返事の LLM に渡す形にする。

    検索条件は extract_schema の項目名（JSON）で渡し、続きの質問ではそのまま引き継げるようにする。
    """
    lines = []
    for i, (question, ans) in enumerate((turns or [])[-MAX_HISTORY_TURNS:], 1):
        ext = ans.extraction
        lines.append(f"[{i}] 学生: {question}")
        if ext is None:
            lines.append("    （回答できませんでした）")
            continue
        if ext.intent == "spot" and ext.supported:
            raw = json.dumps(extract_mod.as_raw(ext), ensure_ascii=False)
            lines.append(f"    種類: スポット探し / 検索条件: {raw}")
        elif not ext.supported:
            lines.append(f"    種類: 対応外の質問（{ext.unsupported_reason}）")
        else:
            lines.append(f"    種類: {INTENT_LABELS.get(ext.intent, ext.intent)}")
        if spots := shown_spots(ans):
            lines.append(
                "    紹介したスポット: "
                + " / ".join(
                    f"{label}={s['spot_id']} {user_location(s)}"
                    for label, s in zip(_spot_labels(ans), spots, strict=False)
                )
            )
    return "\n".join(lines)


def _spot_labels(ans: Answer) -> list[str]:
    """shown_spots の順の呼び名（「おすすめ1」など。おすすめ以外は「候補」）。"""
    labels = list(PICK_LABELS[: len(ans.picks)])
    return labels + ["候補"] * (len(shown_spots(ans)) - len(labels))


def _norm(text: str) -> str:
    return re.sub(r"\s", "", unicodedata.normalize("NFKC", text))


def review_targets(ext: extract_mod.Extraction, conn: sqlite3.Connection) -> list[dict[str, Any]] | None:
    """口コミを答えるスポット。対象の指定がなければ None（最近の口コミを答える）。

    会話で紹介したスポットの ID → 場所の名前（エリア名・表示名の一部）→ 建物 の順で決める。
    """
    spots = repository.list_spots(conn)
    by_id = {s["spot_id"]: s for s in spots}
    if ids := [i for i in ext.review_spot_ids if i in by_id]:
        return [by_id[i] for i in dict.fromkeys(ids)]
    place = ext.review_place
    building = ext.conditions.get("building")
    if place:
        # 「閲覧室のソファ」「図書館 壁沿いデスク」のように、語がすべて表示名に含まれるスポット
        words = [w for w in re.split(r"[\sの・、]+", unicodedata.normalize("NFKC", place)) if w]
        q = _norm(place)
        hits = [
            s for s in spots
            if all(w in _norm(user_location(s)) for w in words) or _norm(s["area_name"]) in q
        ]  # fmt: skip
        if building and building != place:
            hits = [s for s in hits if s["building"] == building] or hits
        return hits
    if building:
        return [s for s in spots if s["building"] == building]
    return None


def answer_reviews(ext: extract_mod.Extraction, conn: sqlite3.Connection) -> tuple[Answer, str]:
    """口コミの確認：対象のスポットの口コミ（指定がなければ最近の口コミ）。

    text は定型文（返事は answer_question で LLM が書き直す）。返り値の2つ目は返事を書かせるときのデータ。
    """
    targets = review_targets(ext, conn)
    if targets is None:
        by_id = {s["spot_id"]: s for s in repository.list_spots(conn)}
        found: dict[str, dict[str, Any]] = {}
        for rv in repository.list_reviews(conn)[:MAX_RECENT_REVIEWS]:
            found.setdefault(rv["spot_id"], {"spot": by_id[rv["spot_id"]], "reviews": []})["reviews"].append(rv)
        items, empty = list(found.values()), []
        head = "最近の口コミ（利用者の声。公式データではありません）です。"
        if not items:
            head = "まだ口コミはありません。「口コミ・誤り報告」のページから投稿できます。"
    else:
        if not targets:
            text = (
                f"「{ext.review_place}」というスポットは見つかりませんでした。"
                "場所の名前を確認して、もう一度質問してください。"
            )
            return Answer(text, "review", ext.conditions), f"案内文:\n{text}"
        items, empty = [], []
        for s in targets:
            reviews = repository.list_reviews(conn, s["spot_id"])[:MAX_REVIEWS_SHOWN]
            (items if reviews else empty).append({"spot": s, "reviews": reviews})
        items = items[:MAX_REVIEW_SPOTS]
        head = "口コミ（利用者の声。公式データではありません）です。" if items else ""
    lines = [head] if head else []
    for x in items:
        lines += ["", f"**{user_location(x['spot'])}**"] + [f"- {_review_line(rv)}" for rv in x["reviews"]]
    if empty:
        names = "、".join(user_location(x["spot"]) for x in empty[:MAX_REVIEW_SPOTS])
        more = f"ほか{len(empty) - MAX_REVIEW_SPOTS}件" if len(empty) > MAX_REVIEW_SPOTS else ""
        if lines:
            lines.append("")
        lines.append(
            f"{names}{more}には、まだ口コミがありません。「口コミ・誤り報告」のページから投稿できます。"
        )
    facts = ["口コミ（利用者の声。公式データではない）:"]
    for x in items:
        facts.append(f"- {user_location(x['spot'])}")
        facts += [f"    - {_review_line(rv)}" for rv in x["reviews"]]
    if not items:
        facts.append("（なし）")
    if empty:
        facts.append("口コミがまだないスポット: " + "、".join(user_location(x["spot"]) for x in empty))
    return Answer("\n".join(lines), "review", ext.conditions, reviews=items), "\n".join(facts)


def _converse(
    question: str,
    ans: Answer,
    guide: str,
    facts: str,
    today: date,
    notices: list[dict[str, Any]] | None = None,
    history: str = "",
) -> Answer:
    """ans の返事を LLM に書かせる（失敗したら ans.text の定型文のまま）。

    notices を渡したときは、返事が触れたお知らせを ans.notices にする（「詳細を見る」に出す）。
    """
    r = reply_mod.write(
        question, guide, facts, ans.text, today=today,
        notice_ids=[n["notice_id"] for n in notices or []], history=history,
    )  # fmt: skip
    ans.text, ans.context = r.text, facts
    if notices is not None and not r.fallback:
        ans.notices = [n for n in notices if n["notice_id"] in r.notice_ids]
    return ans


def _rewrite(question: str, ans: Answer, today: date, history: str = "") -> Answer:
    """定型文の内容を変えずに、話し言葉に書き直させる（聞き返し・未調査・該当なしなど）。"""
    return _converse(
        question, ans, prompts.load("reply_rewrite"), f"案内文:\n{ans.text}", today, history=history
    )


def answer_question(
    question: str,
    conn: sqlite3.Connection,
    *,
    today: date | None = None,
    now: datetime | None = None,
    similarity: retrieve_mod.Similarity | None = None,
    slot_choice: tuple[date, int] | None = None,
    history: list[Turn] | None = None,
    progress: Callable[[str], None] | None = None,
) -> Answer:
    """slot_choice は画面で選んだ（日付, 時限）。指定されれば質問文の日時より優先する。

    history はこれまでの会話（質問と回答の組。古い順）。続きの質問なら前の条件を引き継ぐ。
    progress は処理の段階（画面に出す文言）を受け取る。
    """
    progress = progress or (lambda _: None)
    now = now or datetime.now(ZoneInfo("Asia/Tokyo"))
    today = today or now.date()
    context = format_history(history)
    progress("質問を読み取っています…")
    try:
        ext = extract_mod.extract(question, conn, today, history=context)
    except claude_cli.ClaudeError:
        logger.exception("condition extraction failed")
        return Answer(prompts.load("error"), "error")
    if not history:
        ext.follow_up = False
    search_text = question
    if ext.follow_up and history and history[-1][1].search_text:
        search_text = f"{history[-1][1].search_text} {question}"
    progress("探して、回答を書いています…")
    ans = _answer(question, ext, conn, today, now, similarity, slot_choice, context, search_text)
    ans.extraction, ans.search_text = ext, search_text
    return ans


def _answer(
    question: str,
    ext: extract_mod.Extraction,
    conn: sqlite3.Connection,
    today: date,
    now: datetime,
    similarity: retrieve_mod.Similarity | None,
    slot_choice: tuple[date, int] | None,
    history: str,
    search_text: str,
) -> Answer:
    # 調査していない場所・建物は、質問の種類にかかわらず先に確かめる。「23号館」「談話室」なら、
    # お知らせ・口コミ・対応外の質問でも未調査と答える（番号・名前の近い場所のことを答えない）
    if place := unsurveyed_place(question, ext):
        ans = _unsurveyed_place(question, place, conn, ext.conditions)
        return _rewrite(question, ans, today, history)
    if unsurveyed := check_building(question, ext, conn):
        return _rewrite(question, unsurveyed, today, history)
    if not ext.supported:
        ans = Answer(prompts.load("unsupported", reason=ext.unsupported_reason), "unsupported")
        facts, notices = _unsupported_facts(ext, conn, today, question)
        guide = prompts.load("reply_unsupported", reason=ext.unsupported_reason)
        return _converse(question, ans, guide, facts, today, notices, history)
    if ext.intent == "chat":
        ans = Answer(prompts.load("chat"), "chat")
        return _converse(
            question, ans, prompts.load("reply_chat"), _capabilities(conn), today, history=history
        )
    if ext.intent == "notice":
        ans = answer_notices(ext, conn, today, question)
        place = notice_target(ext, question)
        facts = (f"質問で指定された場所: {place}\n\n" if place else "") + (
            "\n".join(notice_facts(n) for n in ans.notices) or "（なし）"
        )
        notices = ans.notices
        ans = _converse(question, ans, prompts.load("reply_notice"), facts, today, history=history)
        ans.notices = notices  # 詳細は該当したお知らせをすべて出す
        return ans
    if ext.intent == "review":
        ans, facts = answer_reviews(ext, conn)
        return _converse(question, ans, prompts.load("reply_review"), facts, today, history=history)
    if clarify := check_places(question, ext, conn):
        return _rewrite(question, clarify, today, history)
    try:
        if slot_choice is not None:
            slot = when.period_slot(*slot_choice)
        else:
            slot = when.resolve(ext.date, ext.weekday, ext.period, ext.now, now)
    except when.UnsupportedPeriod as e:
        ans = Answer(prompts.load("unsupported", reason=str(e)), "unsupported", ext.conditions)
        return _rewrite(question, ans, today, history)

    r = retrieve_mod.retrieve(
        search_text, ext.conditions, conn, today=today, similarity=similarity, slot=slot
    )
    if not r.spots and r.classrooms:
        # 候補が講義棟の教室だけで、空いているか確かめられない（おすすめにしない）
        ans = Answer(format_headline([], r), "normal", ext.conditions, r)
        return _rewrite(question, ans, today, history)
    if not r.spots:
        unknown = format_unknown(r)
        text = prompts.load(
            "no_result",
            conditions=format_conditions(ext.conditions) + (f"、時間: {slot.label}" if slot else ""),
            excluded="".join(f"\n※ {n}" for n in side_notes(r)),
            unknown=f"\n※ ほかに、{unknown}" if unknown else "",
        )
        return _rewrite(question, Answer(text, "no_result", ext.conditions, r), today, history)

    prompt = build_prompt(
        question, r, today, history=history, follow_up=ext.follow_up, want_reviews=ext.want_reviews
    )
    try:
        raw = claude_cli.ask_json(
            prompt, prompts.load("answer_schema"), system=prompts.load("answer_system")
        )
    except claude_cli.ClaudeError:
        logger.exception("answer generation failed")
        return Answer(prompts.load("error"), "error", ext.conditions, r)
    picks = choose_picks(raw, r)
    text = spot_reply(raw, r, picks, prompt, question) or format_headline(picks, r, ext.want_reviews)
    kind = "relaxed" if r.relaxed_key else "normal"
    return Answer(text, kind, ext.conditions, r, picks, context=prompt)
