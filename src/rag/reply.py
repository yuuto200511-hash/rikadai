"""会話の文章（返事）を LLM に書かせる。

事実の表示（スポットの詳細カード・お知らせの詳細）はデータからテンプレートで作り、返事の文章だけを LLM に任せる。
LLM に守らせるのは2つ：(a) 渡したデータの値を書き換えない (b) 渡したデータにないことを事実として書かない。
(a) はコードでも確かめ、返事にデータにない数値が出てきたら、定型文（fallback）を使う。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date

from src import prompts
from src.llm import claude_cli

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


def checked(text: object, context: str) -> str | None:
    """返事の数値がすべてデータ（context）にあり、禁止の表現がなければ返事を、そうでなければ None を返す。"""
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()
    if _FORBIDDEN_RE.search(text):
        return None
    if missing := [n for n in _NUMBER_RE.findall(text) if n not in context]:
        logger.warning("reply has numbers not in data: %s / %s", missing, text)
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
        raw.get("reply"), f"{today.isoformat()} {today.month}/{today.day}\n{history}\n{question}\n{facts}"
    )
    if text is None:
        return Reply(fallback, fallback=True)
    allowed = set(notice_ids or [])
    used = [i for i in raw.get("used_notice_ids") or [] if isinstance(i, int) and i in allowed]
    return Reply(text, list(dict.fromkeys(used)))
