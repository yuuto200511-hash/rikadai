"""チャットの入力の制限（OWASP LLM10: 際限のない消費）。

AI を呼ぶ前に確かめ、引っかかったら AI を呼ばずに返す文を返す。
"""

from __future__ import annotations

import math

MAX_QUESTION_CHARS = 300
MAX_CALLS_PER_MINUTE = 10
WINDOW_S = 60.0

TOO_LONG = f"質問は{MAX_QUESTION_CHARS}文字以内でお願いします"


def too_many(seconds: int) -> str:
    return f"短い時間に質問が続いたため、少し待ってからもう一度お試しください（あと{seconds}秒ほど）"


def check(question: str, calls: list[float], now: float) -> str | None:
    """問題がなければ None を返し、AI を呼ぶ時刻として now を calls に記録する。

    calls はこのセッションで AI を呼んだ時刻（time.monotonic()）。直近1分の分だけ残す。
    1回の質問を AI を1回呼ぶものとして数える（条件の読み取りと返事の文章をまとめて1回）。
    """
    if len(question) > MAX_QUESTION_CHARS:
        return TOO_LONG
    calls[:] = [t for t in calls if now - t < WINDOW_S]
    if len(calls) >= MAX_CALLS_PER_MINUTE:
        return too_many(max(1, math.ceil(calls[0] + WINDOW_S - now)))
    calls.append(now)
    return None
