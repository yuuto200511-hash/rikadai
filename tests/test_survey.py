import pandas as pd
import pytest

from src.db import repository
from src.ingest import survey

BASE = {
    "spot_id": "S001", "building": "図書館", "floor": 2, "area_name": "窓側", "seats": 10,
    "outlets": 5, "chair_type": "木製椅子", "standing": "不可", "quietness": "私語厳禁",
    "food": "不可", "task_light": "あり", "outlet_location": "壁際", "note": "", "photo": None,
    "surveyed_at": "2026-09-10", "surveyor": "A", "source": "テスト",
}  # fmt: skip


@pytest.fixture
def conn(tmp_path):
    return repository.connect(tmp_path / "t.db")


def write_xlsx(path, rows, choices=None):
    labels = {c: f"ラベル{c}" for c in rows[0]}
    df = pd.DataFrame([labels, *rows])
    df["_power_ratio"] = ["（参考）", *[0.5] * len(rows)]
    with pd.ExcelWriter(path) as w:
        pd.DataFrame({"x": ["凡例"]}).to_excel(w, sheet_name="凡例", index=False)
        df.to_excel(w, sheet_name="spots", index=False)
        if choices is not None:
            pd.DataFrame({k: pd.Series(v) for k, v in choices.items()}).to_excel(
                w, sheet_name="選択肢", index=False
            )


def test_valid_row():
    spot, err, blanks = survey.validate_row(dict(BASE))
    assert err is None and blanks == []
    assert spot["photo"] == "S001.jpg" and spot["note"] is None and spot["source"] == "テスト"


def test_unsurveyed_is_valid():
    row = {**BASE, "chair_type": "未調査", "quietness": "未調査", "food": "未調査",
           "task_light": "未調査"}  # fmt: skip
    spot, err, _ = survey.validate_row(row)
    assert err is None and spot["quietness"] == "未調査"


def test_nullable_blanks_become_null():
    row = {**BASE, "floor": None, "seats": None, "outlets": "", "surveyed_at": None,
           "surveyor": None}  # fmt: skip
    spot, err, blanks = survey.validate_row(row)
    assert err is None
    assert blanks == ["floor", "seats", "outlets", "surveyed_at", "surveyor"]
    assert spot["seats"] is None and spot["outlets"] is None


def test_building_unknown_marker():
    spot, err, blanks = survey.validate_row(
        {**BASE, "building": "閲覧室（建物未記入）", "area_name": "6人掛けデスク"}
    )
    assert err is None and spot["building"] is None and blanks == ["building"]
    assert spot["area_name"] == "閲覧室 6人掛けデスク"
    spot, _, _ = survey.validate_row(
        {**BASE, "building": "自習室（建物未記入）", "area_name": "自習室"}
    )
    assert spot["area_name"] == "自習室"


@pytest.mark.parametrize(
    "patch, reason",
    [
        ({"quietness": "静か"}, "quietness が選択肢外"),
        ({"chair_type": "パイプ椅子"}, "chair_type が選択肢外"),
        ({"standing": "未調査"}, "standing が選択肢外"),
        ({"task_light": "たぶん"}, "task_light が選択肢外"),
        ({"area_name": ""}, "area_name が空欄"),
        ({"food": None}, "food が空欄"),
        ({"outlets": "たくさん"}, "outlets が整数でない"),
        ({"seats": -1}, "seats が整数でない"),
        ({"surveyed_at": "昨日"}, "surveyed_at が日付でない"),
    ],
)
def test_invalid_rows_rejected(patch, reason):
    spot, err, _ = survey.validate_row({**BASE, **patch})
    assert spot is None and reason in err


def test_ingest_excel(tmp_path, conn):
    rows = [
        BASE,
        {**BASE, "spot_id": "S002", "chair_type": "椅子"},
        {**BASE, "spot_id": "S001"},  # 重複
        {**BASE, "spot_id": "S003", "outlets": 0, "seats": None},
    ]
    path = tmp_path / "s.xlsx"
    write_xlsx(path, rows)
    res = survey.ingest(path, conn)
    assert [s["spot_id"] for s in res.accepted] == ["S001", "S003"]
    # 2行目はラベルなのでデータは3行目から
    assert [(r[0], r[1], r[2].split(":")[0]) for r in res.rejected] == [
        (4, "S002", "chair_type が選択肢外"),
        (5, "S001", "spot_id が重複"),
    ]
    assert res.blanks == [(6, "S003", ["seats"])]
    spots = {s["spot_id"]: s for s in repository.list_spots(conn)}
    assert spots["S001"]["outlet_ratio"] == 0.5 and spots["S001"]["task_light"] == "あり"
    assert spots["S003"]["outlet_ratio"] is None
    assert "_power_ratio" not in spots["S001"]


def test_choices_sheet_mismatch_warns(tmp_path, conn):
    from src.choices import SPOT_ENUMS

    path = tmp_path / "s.xlsx"
    # 空欄の下の注記は選択肢に含めない
    with_note = {**SPOT_ENUMS, "chair_type": (*SPOT_ENUMS["chair_type"], None, "注記")}
    write_xlsx(path, [BASE], choices=with_note)
    assert survey.ingest(path, conn).warnings == []
    write_xlsx(path, [BASE], choices={**SPOT_ENUMS, "food": ("不可", "可")})
    assert "food" in survey.ingest(path, conn).warnings[0]


def test_reingest_keeps_reviews(tmp_path, conn):
    path = tmp_path / "s.xlsx"
    write_xlsx(path, [BASE, {**BASE, "spot_id": "S002"}])
    survey.ingest(path, conn)
    repository.add_review(conn, "S001", "よい")
    write_xlsx(path, [{**BASE, "note": "更新"}])
    survey.ingest(path, conn)
    assert [s["spot_id"] for s in repository.list_spots(conn)] == ["S001"]
    assert repository.get_spot(conn, "S001")["note"] == "更新"
    write_xlsx(path, [{**BASE, "spot_id": "S009"}])
    with pytest.raises(ValueError, match="S001"):
        survey.ingest(path, conn)


def test_search_excludes_null_and_unsurveyed(conn):
    repository.replace_spots(
        conn,
        [
            {**BASE},
            {**BASE, "spot_id": "S002", "quietness": "会話可", "outlets": 0},
            {**BASE, "spot_id": "S003", "quietness": "小声可", "building": "講義棟"},
            {**BASE, "spot_id": "S004", "quietness": "未調査", "food": "未調査",
             "outlets": None, "seats": None, "building": None, "task_light": "未調査"},
        ],
    )  # fmt: skip
    ids = lambda c: [s["spot_id"] for s in repository.search_spots(conn, c)]  # noqa: E731
    assert ids({}) == ["S001", "S002", "S003", "S004"]
    assert ids({"outlets_required": True}) == ["S001", "S003"]
    assert ids({"min_seats": 1}) == ["S001", "S002", "S003"]
    assert ids({"quietness": ["私語厳禁", "小声可", "会話可", "通話可"]}) == ["S001", "S002", "S003"]
    assert ids({"quietness": ["私語厳禁", "小声可"], "building": "講義棟"}) == ["S003"]
    assert ids({"food": ["不可", "飲み物のみ", "食事可"]}) == ["S001", "S002", "S003"]
    assert ids({"task_light": "あり"}) == ["S001", "S002", "S003"]
    assert ids({"food": ["未調査"]}) == []
    assert sorted(repository.list_buildings(conn)) == sorted(["講義棟", "図書館"])
