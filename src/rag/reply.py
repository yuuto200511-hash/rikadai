"""会話の文章（返事）を LLM に書かせる。

事実の表示（スポットの詳細カード・お知らせの詳細）はデータからテンプレートで作り、返事の文章だけを LLM に任せる。
LLM に守らせるのは2つ：(a) 渡したデータの値を書き換えない (b) 渡したデータにないことを事実として書かない。
(a) はコードでも確かめ、返事にデータにない数値・場所の名前が出てきたら、定型文（fallback）を使う。
場所の名前は、番号を含む固有名詞（「◯号館」「K503」）を質問文・データと照合する（数値の照合と同じ仕組み）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date

from src import prompts
from src.llm import claude_cli
from src.rag import extract as extract_mod

logger = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"\d+")
# 利用者に出さない表現（電源確保率）
_FORBIDDEN_RE = re.compile(r"[%％]|確保率")


@dataclass
class Reply:
    text: str
    # 返事で触れたお知らせの ID（データに渡したものだけ）
    notice_ids: list[int] = field(default_factory=list)
    # LLM の返事を使えず fallback にしたか
    fallback: bool = False


def wrong_places(text: str, question: str, context: str) -> list[str]:
    """返事に出してはいけない場所の名前（番号を含む固有名詞「◯号館」「K503」）。空なら使ってよい。

    - 質問文かデータ（context）にある場所だけ出してよい（データにない建物・教室を出させない）
    - 質問文で指定された場所以外の「◯号館」「K503」を出すときは、指定された場所の話もしていること。
      「23号館はある？」に、23号館に触れず「21号館はあります」と答えるのを防ぐ
      （調査済みの建物の一覧として 21号館 を挙げるのはよい）
    """
    asked = extract_mod.place_tokens(question)
    allowed = {*asked, *extract_mod.place_tokens(context)}
    written = extract_mod.place_tokens(text)
    others = [p for p in written if p not in asked]
    wrong = [p for p in others if p not in allowed]
    if others:
        wrong += [p for p in asked if p not in written]  # 質問された場所に触れていない
    return wrong


def checked(text: object, context: str, question: str = "") -> str | None:
    """返事の数値・場所の名前がすべてデータ（context）にあり、禁止の表現がなければ返事を、
    そうでなければ None（定型文を使う）を返す。"""
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()
    if _FORBIDDEN_RE.search(text):
        return None
    if missing := [n for n in _NUMBER_RE.findall(text) if n not in context]:
        logger.warning("reply has numbers not in data: %s / %s", missing, text)
        return None
    if wrong := wrong_places(text, question, context):
        logger.warning("reply has places not asked about or not in data: %s / %s", wrong, text)
        return None
    return text


def write(
    question: str,
    guide: str,
    facts: str,
    fallback: str,
    *,
    today: date | None = None,
    notice_ids: list[int] | None = None,
    history: str = "",
) -> Reply:
    """guide（状況と書き方）と facts（データ）から返事を書かせる。失敗したら fallback を返す。

    notice_ids は facts に載せたお知らせの ID（返事で触れたものとして受け付ける範囲）。
    history はこれまでの会話（話の流れに合わせた言い回しにするため。事実の根拠には使わせない）。
    """
    today = today or date.today()
    prompt = prompts.load(
        "reply",
        today=today.isoformat(),
        history=history or "（なし）",
        question=question,
        guide=guide,
        facts=facts or "（なし）",
    )
    try:
        raw = claude_cli.ask_json(
            prompt, prompts.load("reply_schema"), system=prompts.load("reply_system")
        )
    except claude_cli.ClaudeError:
        logger.exception("reply generation failed")
        return Reply(fallback, fallback=True)
    text = checked(
        raw.get("reply"),
        f"{today.isoformat()} {today.month}/{today.day}\n{history}\n{question}\n{facts}",
        question,
    )
    if text is None:
        return Reply(fallback, fallback=True)
    allowed = set(notice_ids or [])
    used = [i for i in raw.get("used_notice_ids") or [] if isinstance(i, int) and i in allowed]
    return Reply(text, list(dict.fromkeys(used)))
