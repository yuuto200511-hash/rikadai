"""キッチンカーの取込（確かめてから保存・月ごとの置き換え）・質問への答え・今日のまとめ。"""

from datetime import date

import pytest

from src.db import repository
from src.ingest import kitchen_cars as kc_ingest
from src.rag import answer
from src.rag.extract import normalize
from src.summary import today_summary
from tests.test_answer import _llm_fails, _notice
from tests.test_retrieve import BASE

TODAY = date(2026, 10, 20)  # 火
HEADER = "date,weekday,open,close,location,vendor,item,price_yen\n"
CSV = HEADER + (
    "2026-10-19,月,10:00,14:00,3号館B,sinamonn kitchen,タコライス,600\n"
    "2026-10-20,火,10:00,14:00,3号館A,Cafe' yori-michi,海鮮丼,600\n"
    "2026-10-20,火,10:00,14:00,3号館A,Cafe' yori-michi,豚丼,600\n"
    "2026-10-20,火,10:00,14:00,3号館B,KALTENA,ジャークチキンライス（Sサイズ）,600\n"
    "2026-10-23,金,10:00,14:00,3号館A,Kene's Diner,オリジナルタコライス,600\n"
    "2026-10-23,金,10:00,14:00,3号館A,Kene's Diner,ペットボトルドリンク,0\n"
)


@pytest.fixture
def conn(tmp_path):
    c = repository.connect(tmp_path / "t.db")
    repository.replace_spots(c, [{**BASE, "spot_id": "S001"}])
    kc_ingest.save(c, kc_ingest.parse(CSV.encode()))
    return c


def test_parse_reports_problem_rows_and_does_not_save(conn):
    bad = HEADER + (
        "2026-10-20,水,10:00,14:00,3号館A,X,天丼,600\n"  # 曜日が違う
        "2026-10-20,火,10:00,14:00,3号館A,X,天丼,-1\n"  # 値段が負
        "2026-10-20,火,10:00,14:00,3号館A,X,天丼,1.5\n"  # 値段が整数でない
        "2026-10-20,火,10:00,14:00,3号館A,X,天丼\n"  # 列が足りない
        "2026-10-20,火,10:00,14:00,3号館A,X,天丼,600\n"  # 問題なし
    )
    res = kc_ingest.parse(bad.encode())
    assert [row for row, _ in res.errors] == [2, 3, 4, 5]
    assert "曜日（火）" in res.errors[0][1] and "0以上の整数" in res.errors[1][1]
    assert res.raw[5] == ["2026-10-20", "火", "10:00", "14:00", "3号館A", "X", "天丼"]
    with pytest.raises(ValueError):
        kc_ingest.save(conn, res)
    assert repository.kitchen_car_months(conn) == {"2026-10": 6}
    # 列名が足りなければファイル全体を止める
    res = kc_ingest.parse("date,weekday,open,close,location,vendor,item\n".encode())
    assert res.errors[0][0] == 1 and "price_yen" in res.errors[0][1]
    # 値段0は受け付ける。Excel で保存した Shift_JIS も読める
    assert kc_ingest.parse(CSV.encode("cp932")).ok


def test_same_month_is_replaced(conn):
    kc_ingest.save(conn, kc_ingest.parse(HEADER + "2026-10-21,水,10:00,14:00,3号館A,ナシーブ,ケバブ丼,600\n"))
    kc_ingest.save(conn, kc_ingest.parse(HEADER + "2026-11-02,月,10:00,14:00,3号館A,ナシーブ,ケバブ丼,600\n"))
    assert repository.kitchen_car_months(conn) == {"2026-10": 1, "2026-11": 1}


def _ask(conn, monkeypatch, question, **raw):
    monkeypatch.setattr(
        answer.extract_mod, "extract", lambda q, c, t, **kw: normalize({"supported": True, **raw})
    )
    monkeypatch.setattr(answer.reply_mod.claude_cli, "ask_json", _llm_fails)
    return answer.answer_question(question, conn, today=TODAY)


def test_today_and_weekend(conn, monkeypatch):
    ans = _ask(conn, monkeypatch, "今日のキッチンカーは？", intent="kitchen_car")
    assert ans.kind == "kitchen_car"
    assert "- 3号館A：Cafe' yori-michi（10:00〜14:00）" in ans.text
    assert "- 3号館B：KALTENA（10:00〜14:00）" in ans.text
    assert [len(g["items"]) for g in ans.kitchen_cars[0][1]] == [2, 1]
    # 土曜・データのない月
    for day in ("2026-10-24", "2026-12-01"):
        ans = _ask(conn, monkeypatch, "その日のキッチンカー", intent="kitchen_car", date=day)
        assert "その日の出店予定はありません" in ans.text and not ans.kitchen_cars


def test_item_lists_upcoming_days(conn, monkeypatch):
    ans = _ask(conn, monkeypatch, "タコライスはいつ来る？", intent="kitchen_car", kitchen_item="タコライス")
    # 今日以降だけ（10/19 は出さない）。「オリジナルタコライス」も一致する
    assert [d for d, _ in ans.kitchen_cars] == [date(2026, 10, 23)]
    assert "オリジナルタコライス 600円" in ans.text and "10/19" not in ans.text


def test_building_3_is_a_kitchen_car_place_but_unsurveyed_for_study(conn, monkeypatch):
    ans = _ask(conn, monkeypatch, "3号館に何が来てる？", intent="kitchen_car")
    assert ans.kind == "kitchen_car" and "3号館A" in ans.text
    ans = _ask(conn, monkeypatch, "3号館Bに何が来てる？", intent="kitchen_car")
    assert "KALTENA" in ans.text and "Cafe' yori-michi" not in ans.text
    ans = _ask(conn, monkeypatch, "23号館に何が来てる？", intent="kitchen_car")
    assert ans.text.startswith("23号館のキッチンカーの出店情報はありません")
    # 作業場所を聞かれたら、今までどおり未調査
    ans = _ask(conn, monkeypatch, "3号館で勉強できる場所", intent="spot", building="3号館")
    assert ans.kind == "unsurveyed" and ans.text.startswith("3号館は未調査のため情報がありません。")


def test_reply_numbers_are_checked(conn, monkeypatch):
    monkeypatch.setattr(
        answer.extract_mod, "extract", lambda q, c, t, **kw: normalize({"intent": "kitchen_car"})
    )
    reply = "今日は3号館AにCafe' yori-michi、3号館BにKALTENAが10:00〜14:00に来ます。"
    monkeypatch.setattr(answer.reply_mod.claude_cli, "ask_json", lambda *a, **kw: {"reply": reply})
    assert answer.answer_question("今日のキッチンカーは？", conn, today=TODAY).text == reply
    # データにない値段を書いたら定型文に戻す
    monkeypatch.setattr(
        answer.reply_mod.claude_cli, "ask_json", lambda *a, **kw: {"reply": "海鮮丼は650円です。"}
    )
    text = answer.answer_question("今日のキッチンカーは？", conn, today=TODAY).text
    assert text.startswith("10/20(火)のキッチンカーです")


def test_today_summary(conn):
    _notice(conn, "lib", target_building="図書館", valid_from="2026-10-15", valid_to="2026-10-25")
    _notice(conn, "unconfirmed", confirmed=False, target_building="図書館", valid_to="2026-10-25")
    _notice(conn, "past", target_building="図書館")  # 10/9 まで
    _notice(conn, "open", impact_type="開放", valid_to="2026-10-25")
    s = today_summary(conn, TODAY)
    assert [g["vendor"] for g in s.kitchen_cars] == ["Cafe' yori-michi", "KALTENA"]
    assert [n["title"] for n in s.closed] == ["lib"]
    s = today_summary(conn, date(2026, 10, 31))  # 土曜・お知らせの期間外
    assert s.kitchen_cars == [] and s.closed == []
