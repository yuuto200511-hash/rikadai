from datetime import date, datetime

import pandas as pd
import pytest

from src.db import repository
from src.ingest import error_report, reservations, survey
from src.rag import retrieve, when
from src.rag.answer import (
    Answer, cards, choose_picks, format_headline, note_line, power_level,
)  # fmt: skip
from src.rag.extract import normalize
from tests.test_retrieve import BASE, fake_similarity

FRI = date(2026, 10, 2)
ROWS = [
    {"date": "2026-10-02", "weekday": "金", "room": "K501", "start": "13:00", "end": "16:30",
     "periods": "3,4,5", "duplicate": None, "spot_id": "S001"},
    {"date": "2026-10-02", "weekday": "金", "room": "K502", "start": "10:30", "end": "12:00",
     "periods": "2", "duplicate": "あり", "spot_id": None},
    {"date": "2026-10-02", "weekday": "金", "room": "K503", "start": "18:00", "end": "21:00",
     "periods": None, "duplicate": None, "spot_id": None},
]  # fmt: skip


@pytest.fixture
def conn(tmp_path):
    c = repository.connect(tmp_path / "t.db")
    repository.replace_spots(
        c,
        [
            {**BASE, "spot_id": "S001", "building": "講義棟", "area_name": "K501"},
            {**BASE, "spot_id": "S002", "building": "講義棟", "area_name": "K502"},
            {**BASE, "spot_id": "S003", "area_name": "K502"},  # 講義棟以外は room で対応づけない
            {**BASE, "spot_id": "S004", "building": None, "floor": None, "area_name": "閲覧室"},
        ],
    )
    repository.replace_reservations(c, [reservations.validate_row(r)[0] for r in ROWS])
    return c


def test_validate_row():
    row, err = reservations.validate_row({**ROWS[1], "start": "9:00"})
    assert err is None and row["start"] == "09:00" and row["duplicate"] == 1
    assert reservations.validate_row({"date": "この表に載っていない教室は…"}) == (None, None)
    assert reservations.validate_row({**ROWS[0], "date": "明日"})[1].startswith("date")
    assert reservations.validate_row({**ROWS[0], "end": "12:00"})[1].startswith("start が end")
    assert reservations.validate_row({**ROWS[0], "room": ""})[1] == "room が空欄"


def test_survey_ingest_reads_reservations_sheet(tmp_path):
    from tests.test_survey import BASE as SURVEY_BASE

    path = tmp_path / "s.xlsx"
    labels = {c: "ラベル" for c in ROWS[0]}
    with pd.ExcelWriter(path) as w:
        pd.DataFrame([{c: "ラベル" for c in SURVEY_BASE}, SURVEY_BASE]).to_excel(
            w, sheet_name="spots", index=False
        )
        pd.DataFrame([labels, *ROWS, {"date": "出典：CLASS 教室利用状況表"}]).to_excel(
            w, sheet_name="reservations", index=False
        )
    c = repository.connect(tmp_path / "t.db")
    res = survey.ingest(path, c)
    assert len(res.accepted) == 1 and len(res.reservations.accepted) == 3
    assert res.reservations.rejected == []
    assert repository.reservation_dates(c) == ["2026-10-02"]


def test_resolve_slot():
    thu = datetime(2026, 10, 1, 13, 10)
    assert when.resolve("2026-10-02", None, 4, False, thu) == when.Slot(FRI, "14:40", "16:10", "10/2(金)4限")
    # 曜日だけなら、その曜日のうち一番近い日付
    assert when.resolve(None, "金", 3, False, thu).day == FRI
    assert when.resolve(None, "木", 3, False, thu).day == date(2026, 10, 1)
    assert when.resolve(None, None, 2, False, thu).day == date(2026, 10, 1)  # 時限だけなら今日
    assert when.resolve("2026-10-02", None, None, False, thu) is None  # 時限なしは指定なし
    now = when.resolve(None, None, None, True, thu)
    assert (now.day, now.start, now.end) == (date(2026, 10, 1), "13:10", "13:11")
    with pytest.raises(when.UnsupportedPeriod):
        when.resolve(None, "金", 6, False, thu)


def test_normalize_time_fields():
    e = normalize({"supported": True, "date": "2026-10-02", "period": "3"})
    assert (e.date, e.weekday, e.period, e.now) == ("2026-10-02", None, 3, False)
    e = normalize({"supported": True, "date": "明日", "weekday": "金曜", "period": 6, "now": True})
    assert (e.date, e.weekday, e.period, e.now) == (None, None, 6, True)


def test_overlapping_reservations_exclude_classrooms(conn):
    sim = fake_similarity(["S001", "S002", "S003", "S004"])
    r = retrieve.retrieve("q", {}, conn, similarity=sim, slot=when.period_slot(FRI, 3))
    # K501 は 13:00-16:30 の予約と重なる。K503 の夜間予約・K502 の2限は重ならない
    assert [s["spot_id"] for s in r.spots] == ["S002", "S003", "S004"]
    assert [e["spot"]["spot_id"] for e in r.class_excluded] == ["S001"]
    r = retrieve.retrieve("q", {}, conn, similarity=sim, slot=when.period_slot(FRI, 2))
    # spot_id が空欄の予約は、講義棟のエリア名（K502）で対応づける
    assert [e["spot"]["spot_id"] for e in r.class_excluded] == ["S002"]
    r = retrieve.retrieve("q", {}, conn, similarity=sim, slot=when.period_slot(FRI, 5))
    assert [e["spot"]["spot_id"] for e in r.class_excluded] == ["S001"]  # 16:20-17:50 も重なる


def test_no_data_date_keeps_classrooms_and_notes_it(conn):
    sim = fake_similarity(["S001", "S004"])
    r = retrieve.retrieve("q", {}, conn, similarity=sim, slot=when.period_slot(date(2026, 10, 3), 3))
    assert not r.has_reservation_data and r.class_excluded == []
    # 予約データがない日付は日時の指定なしと同じ: 教室は外さないが、おすすめにせずカードにだけ載せる
    assert [s["spot_id"] for s in r.classrooms] == ["S001", "S002"]
    assert [s["spot_id"] for s in r.spots] == ["S004", "S003"]
    assert note_line(r).startswith("10/3(土)の利用状況データがありません")
    picks = choose_picks({"picks": [{"spot_id": "S001", "reason": "コンセント5口"}]}, r)
    assert [p["spot"]["spot_id"] for p in picks] == ["S004"]  # 候補外（教室）は捨てる
    ans = Answer(format_headline(picks, r), "normal", {}, r, picks)
    assert ans.spot_ids == ["S004", "S001", "S002"]
    assert cards(ans)[1][1].startswith("#### 授業がない時間のみ利用可｜講義棟 2階 K501")


def test_classrooms_are_recommended_only_when_confirmed_free(conn):
    sim = fake_similarity(["S002", "S004"])
    # 日時の指定なし: 教室はおすすめの候補にしない
    r = retrieve.retrieve("q", {}, conn, similarity=sim)
    assert "S002" not in [s["spot_id"] for s in r.spots]
    assert "S002" in [s["spot_id"] for s in r.classrooms]
    # 予約データで空いていると確かめられた教室は候補にする
    r = retrieve.retrieve("q", {}, conn, similarity=sim, slot=when.period_slot(FRI, 3))
    assert r.spots[0]["spot_id"] == "S002" and r.classrooms == []
    # 候補が教室だけで日時の指定がないときは、おすすめなしでカードにだけ載せる
    r = retrieve.retrieve("q", {"building": "講義棟"}, conn, similarity=sim)
    assert r.spots == [] and len(r.classrooms) == 2
    text = format_headline([], r)
    assert text.startswith("日時の指定がないため、講義棟の教室はおすすめに含めていません（授業がない時間のみ利用可）")
    # 教室番号を指定された質問は、その教室について答える
    r = retrieve.retrieve("q", {"room": "K501"}, conn, similarity=sim)
    assert [s["spot_id"] for s in r.spots] == ["S001"]


def test_power_level():
    assert power_level({"seats": 10, "outlets": 10}) == "全席で使える"
    assert power_level({"seats": 10, "outlets": 3}) == "一部の席で使える"
    assert power_level({"seats": 10, "outlets": 0}) == "なし"
    assert power_level({"seats": 10, "outlets": None}) == "未記入"
    assert power_level({"seats": None, "outlets": 4}) == "未記入"


def test_reasons_and_cards_have_no_percent(conn):
    r = retrieve.retrieve("q", {}, conn, similarity=fake_similarity(["S004", "S001"]),
                          slot=when.period_slot(FRI, 2))  # fmt: skip
    picks = choose_picks({"picks": [
        {"spot_id": "S004", "reason": "電源確保率50%"},
        {"spot_id": "S001", "reason": "電源：一部の席で使える"},
    ]}, r)  # fmt: skip
    assert [p["reason"] for p in picks] == ["電源：一部の席で使える（コンセント5口）", "電源：一部の席で使える"]
    ans = Answer(format_headline(picks, r), "normal", {}, r, picks)
    rendered = [card for _, card in cards(ans)]
    assert "- 電源：一部の席で使える（コンセント5口） ／ 座席：10席" in rendered[0]
    assert "教室の利用状況：10/2(金)2限は予約なし" in rendered[1]
    assert not any("%" in x or "確保率" in x for x in [ans.text, *rendered])


def test_error_report(conn):
    rid = error_report.post_error_report(conn, "S001", " コンセントが2口だった ")
    [rep] = repository.list_error_reports(conn)
    assert rep["report_id"] == rid and rep["body"] == "コンセントが2口だった" and not rep["resolved"]
    repository.set_error_report_resolved(conn, rid, True)
    assert repository.list_error_reports(conn)[0]["resolved"] == 1
    for spot_id, body in [("S001", " "), ("S999", "x"), ("S001", "あ" * 501)]:
        with pytest.raises(error_report.ErrorReportError):
            error_report.post_error_report(conn, spot_id, body)
    # 誤り報告のあるスポットは再取込で消さない
    with pytest.raises(ValueError):
        repository.replace_spots(conn, [{**BASE, "spot_id": "S002"}])
