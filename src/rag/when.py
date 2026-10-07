"""質問の「いつ」（日付・時限）を判定する。教室の利用状況（reservations）と照らし合わせるために使う。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from src.choices import PERIOD_TIMES, WEEKDAYS


class UnsupportedPeriod(ValueError):
    """時刻が未確認の時限（6限以降）が指定された。"""


@dataclass(frozen=True)
class Slot:
    """日付と時間帯 [start, end)。時刻は "HH:MM"。"""

    day: date
    start: str
    end: str
    label: str


def day_label(d: date) -> str:
    return f"{d.month}/{d.day}({WEEKDAYS[d.weekday()]})"


def period_slot(day: date, period: int) -> Slot:
    if period not in PERIOD_TIMES:
        raise UnsupportedPeriod(
            f"{period}限は時刻が未確認です。"
            f"{min(PERIOD_TIMES)}〜{max(PERIOD_TIMES)}限で指定してください"
        )
    start, end = PERIOD_TIMES[period]
    return Slot(day, start, end, f"{day_label(day)}{period}限")


def next_weekday(weekday: str, today: date) -> date:
    """その曜日のうち、今日以降で一番近い日付。"""
    return today + timedelta(days=(WEEKDAYS.index(weekday) - today.weekday()) % 7)


def resolve(
    day: str | None, weekday: str | None, period: int | None, now: bool, at: datetime
) -> Slot | None:
    """抽出した日付・曜日・時限から Slot を作る。時限がなければ None（時間の指定なし）。

    - now=True: 現在の時刻（1分間）で判定する
    - 日付 > 曜日（今日以降で一番近い日付）> 今日 の順で日付を決める
    - 6限以降は UnsupportedPeriod
    """
    if now:
        t0 = at.strftime("%H:%M")
        t1 = (at + timedelta(minutes=1)).strftime("%H:%M")
        return Slot(at.date(), t0, max(t1, t0), f"今（{day_label(at.date())} {t0}）")
    if period is None:
        return None
    if day:
        d = date.fromisoformat(day)
    elif weekday in WEEKDAYS:
        d = next_weekday(weekday, at.date())
    else:
        d = at.date()
    return period_slot(d, period)
