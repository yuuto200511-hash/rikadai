"""動作確認用の仮データ（調査 Excel 10 件）を生成する。実在の調査結果ではない。

本物の調査データ（data/survey.xlsx）を上書きしないよう、既定では data/sample/survey_sample.xlsx に出力する。
形式は本物と同じ（spots シート、2行目は日本語ラベル）なので、次のように取り込める:
    uv run python scripts/make_sample_data.py
    SPOT_DB_PATH=... uv run python -m src.ingest.survey data/sample/survey_sample.xlsx
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

COLUMNS = [
    "spot_id", "building", "floor", "area_name", "seats", "outlets", "chair_type",
    "standing", "quietness", "food", "task_light", "outlet_location", "note", "photo",
    "surveyed_at", "surveyor", "source",
]  # fmt: skip
LABELS = [
    "スポットID", "建物", "階", "エリア名", "座席数", "コンセント数（使える差込口）", "椅子の種類",
    "立ち作業", "静粛性", "飲食可否", "作業灯", "コンセントの位置", "備考", "写真", "調査日",
    "調査者", "元データ",
]  # fmt: skip

SPOTS = [
    ("S001", "図書館", 2, "窓側自習エリア", 24, 24, "木製椅子", "不可", "私語厳禁", "不可",
     "全席に電源あり。窓際で明るく、キーボードの音も気になるほど静か。", "2026-09-10", "調査A"),
    ("S002", "図書館", 1, "グループ学習室", 16, 8, "木製椅子", "不可", "会話可", "飲み物のみ",
     "ホワイトボードあり。複数人での議論向け。", "2026-09-10", "調査A"),
    ("S003", "図書館", 3, "個人ブース", 12, 12, "木製椅子", "不可", "私語厳禁", "不可",
     "仕切り付きの個人席。集中したいとき向け。", "2026-09-11", "調査A"),
    ("S004", "講義棟", 1, "ラウンジ", 40, 6, "ソファ", "不可", "会話可", "食事可",
     "ソファ席が多くくつろげる。昼休みは混みやすい。電源は壁際のみ。", "2026-09-12", "調査B"),
    ("S005", "講義棟", 2, "廊下スタンディングカウンター", 0, 4, "なし", "可", "会話可",
     "飲み物のみ", "立ったまま短時間作業できるカウンター。電源は4口。", "2026-09-12", "調査B"),
    ("S006", "1号館", 3, "学生ホール", 30, 0, "木製椅子", "不可", "小声可", "食事可",
     "電源なし。広めのテーブルで弁当を食べながら勉強する人が多い。", "2026-09-13", "調査B"),
    ("S007", "7号館", 2, "談話スペース", 10, 5, "キャスター付き椅子", "可", "通話可", "飲み物のみ",
     "オンライン会議や通話ができる。スツール席と立ち席が混在。", "2026-09-13", "調査C"),
    ("S008", "7号館", 4, "静音自習室", 20, 10, "木製椅子", "不可", "小声可", "不可",
     "空調が効いていて静か。半分の席に電源。", "2026-09-14", "調査C"),
    ("S009", "食堂棟", 1, "カフェテリア", 80, 10, "木製椅子", "不可", "会話可", "食事可",
     "食事時間帯以外は作業に使える。窓側の一部の席に電源。", "2026-09-14", "調査C"),
    ("S010", "10号館", 1, "エントランスロビー", 8, 2, "ソファ", "可", "小声可", "飲み物のみ",
     "待ち合わせ用のソファ。人の出入りがあり少しざわつく。", "2026-09-15", "調査A"),
]  # fmt: skip


def main() -> None:
    rows = []
    for s in SPOTS:
        (sid, building, floor, area, seats, outlets, chair, standing, quiet, food, note,
         surveyed_at, surveyor) = s  # fmt: skip
        rows.append(
            dict(zip(COLUMNS, (
                sid, building, floor, area, seats, outlets, chair, standing, quiet, food,
                "未調査", None, note, f"{sid}.jpg", surveyed_at, surveyor, "仮データ",
            ), strict=True))  # fmt: skip
        )
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "sample" / "survey_sample.xlsx"
    if out.resolve() == (ROOT / "data" / "survey.xlsx").resolve():
        sys.exit("data/survey.xlsx は本物の調査データなので上書きしません")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([dict(zip(COLUMNS, LABELS, strict=True)), *rows], columns=COLUMNS)
    df.to_excel(out, sheet_name="spots", index=False)
    print(f"wrote {out} ({len(rows)} spots)")


if __name__ == "__main__":
    main()
