"""選択肢の定義（取込・条件抽出・表示で共有する唯一の定義場所）。

data/survey.xlsx の「選択肢」シートと一致させること（取込時に差異があれば警告する）。
"""

from __future__ import annotations

# 取込では有効な値として受け付けるが、条件検索ではどの条件にも一致させない
UNSURVEYED = "未調査"

# 通常椅子＝背もたれのある普通の椅子、スツール＝背もたれなし
CHAIR_TYPES = (
    "木製椅子", "通常椅子", "クッション椅子", "キャスター付き椅子", "机一体型", "ソファ", "スツール", "なし",
    UNSURVEYED,
)  # fmt: skip
STANDING = ("可", "不可")
# 静かな順
QUIETNESS = ("私語厳禁", "小声可", "会話可", "通話可", UNSURVEYED)
# 制限が厳しい順
FOOD = ("不可", "飲み物のみ", "食事可", UNSURVEYED)
TASK_LIGHT = ("あり", "なし", UNSURVEYED)

REVIEW_RATINGS = ("静かだった", "普通", "騒がしかった")

NOTICE_IMPACT_TYPES = ("閉鎖", "時間変更", "開放", "設備変更")

# spots の列ごとの選択肢（取込時の検証に使う）
SPOT_ENUMS: dict[str, tuple[str, ...]] = {
    "chair_type": CHAIR_TYPES,
    "standing": STANDING,
    "quietness": QUIETNESS,
    "food": FOOD,
    "task_light": TASK_LIGHT,
}


def searchable(choices: tuple[str, ...]) -> tuple[str, ...]:
    """条件検索で指定できる値（「未調査」を除く）。"""
    return tuple(c for c in choices if c != UNSURVEYED)

# 曜日（日付から求める表示用）
WEEKDAYS = ("月", "火", "水", "木", "金", "土", "日")
# 時限ごとの開始・終了時刻（確定値）。6限以降は時刻が未確認のため扱わない（指定されたら対応外）
PERIOD_TIMES: dict[int, tuple[str, str]] = {
    1: ("08:50", "10:20"),
    2: ("10:30", "12:00"),
    3: ("13:00", "14:30"),
    4: ("14:40", "16:10"),
    5: ("16:20", "17:50"),
}
# 教室の利用状況（reservations）の影響を受ける建物
LECTURE_BUILDING = "講義棟"
