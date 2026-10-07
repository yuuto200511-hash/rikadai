"""会話の続き（条件の引き継ぎ）・講義棟で電源を求められたときの教室の優先・口コミの確認。"""

import json
from datetime import date

import pytest

from src.db import repository
from src.rag import answer, extract, retrieve, when
from src.rag.answer import Answer, answer_reviews, choose_picks, format_headline, format_history
from src.rag.extract import normalize
from tests.test_retrieve import BASE, fake_similarity

TODAY = date(2026, 10, 1)
ROOM = {**BASE, "building": "講義棟", "floor": 5, "quietness": "会話可", "chair_type": "木製椅子"}


@pytest.fixture
def conn(tmp_path):
    c = repository.connect(tmp_path / "t.db")
    repository.replace_spots(
        c,
        [
            {**BASE, "spot_id": "S001", "area_name": "壁沿いデスク", "seats": 112, "outlets": 76},
            {**BASE, "spot_id": "S002", "building": None, "floor": None, "area_name": "閲覧室 ソファ（3人掛け）",
             "outlets": 0},
            {**ROOM, "spot_id": "S005", "area_name": "K501", "seats": 142, "outlets": 10},
            {**ROOM, "spot_id": "S006", "area_name": "K502", "seats": 142, "outlets": 10},
            {**ROOM, "spot_id": "S023", "floor": 7, "area_name": "K701", "seats": 160, "outlets": 220,
             "chair_type": "机一体型"},
            {**ROOM, "spot_id": "S024", "floor": 7, "area_name": "K702", "seats": 160, "outlets": 220,
             "chair_type": "机一体型"},
        ],
    )  # fmt: skip
    return c


ORDER = ["S005", "S006", "S001", "S023", "S024", "S002"]


# ---------------------------------------------------------------- 講義棟で電源を求められたとき


def test_classrooms_with_desk_outlets_come_first_when_power_is_wanted(conn):
    sim = fake_similarity(ORDER)
    # 電源の条件があれば、教室どうしは全席で電源が使える教室が先（講義棟以外の位置は変えない）
    order = [s["spot_id"] for s in retrieve.classrooms_by_power(retrieve._rank("q", repository.list_spots(conn), sim))]
    assert order == ["S023", "S024", "S001", "S005", "S006", "S002"]
    # 日時の指定がない（教室はカードだけ）ときも、カードに載るのは机にコンセントがある教室
    r = retrieve.retrieve("q", {"outlets_required": True}, conn, similarity=sim)
    assert [s["spot_id"] for s in r.classrooms] == ["S023", "S024"]
    # 電源の条件がなければ、これまでどおり類似度順
    r = retrieve.retrieve("q", {}, conn, similarity=sim)
    assert [s["spot_id"] for s in r.classrooms] == ["S005", "S006"]


def test_partial_power_classroom_pick_is_replaced(conn):
    r = retrieve.retrieve("q", {"outlets_required": True, "building": "講義棟"}, conn, similarity=fake_similarity(ORDER))
    # 空いていると確かめられた教室（おすすめにできる）
    r.slot, r.has_reservation_data = when.period_slot(TODAY, 3), True
    r.spots = [s for s in repository.list_spots(conn) if retrieve.is_classroom(s)]
    for s in r.spots:
        s["reviews"] = []
    raw = {"picks": [{"spot_id": "S005", "reason": "座席142席"}, {"spot_id": "S023", "reason": "電源：全席で使える"}]}
    picks = choose_picks(raw, r)
    assert [p["spot"]["spot_id"] for p in picks] == ["S023", "S024"]
    assert picks[1]["reason"] == "電源：全席で使える（コンセント220口）"
    # 電源を求めていなければ LLM の選んだとおり
    r.conditions = {"building": "講義棟"}
    assert [p["spot"]["spot_id"] for p in choose_picks(raw, r)] == ["S005", "S023"]


# ---------------------------------------------------------------- 会話の続き


@pytest.mark.parametrize(
    "raw",
    [
        {"outlets_required": True, "quietness_max": "小声可", "food": ["飲み物のみ", "食事可"]},
        {"quietness_min": "会話可", "building": "講義棟", "date": "2026-10-02", "period": 3},
        {"quietness_min": "私語厳禁", "quietness_max": "通話可", "min_seats": 3, "now": True},
        {"weekday": "金", "period": 2, "chair_type": ["ソファ"]},
    ],
)
def test_conditions_round_trip_for_history(raw):
    ext = normalize({"supported": True, **raw})
    again = normalize({"supported": True, **extract.as_raw(ext)})
    assert again.conditions == ext.conditions
    assert (again.date, again.weekday, again.period, again.now) == (ext.date, ext.weekday, ext.period, ext.now)


def test_follow_up_question_gets_previous_conditions(conn, monkeypatch):
    seen = []
    first = normalize({"supported": True, "outlets_required": True, "quietness_max": "小声可"})
    second = normalize(
        {"supported": True, "follow_up": True, "outlets_required": True, "quietness_max": "小声可",
         "food": ["飲み物のみ", "食事可"]}
    )  # fmt: skip
    results = iter([first, second])

    def fake_extract(q, c, t, history=""):
        seen.append(history)
        return next(results)

    queries = []

    def sim(q, ids):
        queries.append(q)
        return fake_similarity(ORDER)(q, ids)

    monkeypatch.setattr(answer.extract_mod, "extract", fake_extract)
    monkeypatch.setattr(
        answer.claude_cli, "ask_json", lambda *a, **kw: {"picks": [{"spot_id": "S001", "reason": "コンセント76口"}]}
    )
    a1 = answer.answer_question("静かで電源がある場所", conn, today=TODAY, similarity=sim)
    assert seen == [""] and a1.extraction is first and a1.search_text == "静かで電源がある場所"
    a2 = answer.answer_question(
        "飲み物も飲みたい", conn, today=TODAY, similarity=sim, history=[("静かで電源がある場所", a1)]
    )
    # 前の質問の条件（抽出の形式）と紹介したスポットを渡す
    assert '"outlets_required": true' in seen[1] and '"quietness_max": "小声可"' in seen[1]
    assert "おすすめ1=S001 図書館 2階 壁沿いデスク" in seen[1]
    # 続きの質問は、前の質問とつなげて意味検索する
    assert queries[-1] == "静かで電源がある場所 飲み物も飲みたい" and a2.search_text == queries[-1]
    assert a2.conditions["food"] == ["飲み物のみ", "食事可"]
    assert "会話の続きの質問" in a2.context and "飲み物も飲みたい" in a2.context


def test_follow_up_is_ignored_without_history(conn, monkeypatch):
    monkeypatch.setattr(
        answer.extract_mod, "extract", lambda q, c, t, **kw: normalize({"supported": True, "follow_up": True})
    )
    monkeypatch.setattr(answer.claude_cli, "ask_json", lambda *a, **kw: {"picks": []})
    ans = answer.answer_question("飲み物も飲みたい", conn, today=TODAY, similarity=fake_similarity(ORDER))
    assert ans.extraction.follow_up is False and "会話の続き" not in ans.context


def test_history_keeps_only_recent_turns_and_marks_kinds():
    turns = [(f"質問{i}", Answer("", "chat", extraction=normalize({"supported": True, "intent": "chat"})))
             for i in range(5)]  # fmt: skip
    turns.append(("エラー", Answer("", "error")))
    text = format_history(turns)
    assert "質問0" not in text and "質問1" not in text and "質問2" not in text
    assert "[1] 学生: 質問3" in text and "種類: 挨拶・雑談" in text and "（回答できませんでした）" in text
    assert format_history(None) == ""


# ---------------------------------------------------------------- 口コミの確認


def _review_ext(**raw):
    return normalize({"supported": True, "intent": "review", **raw})


def test_reviews_by_place_name_id_and_recent(conn):
    repository.add_review(conn, "S002", "机がないから、勉強しにくい", "騒がしかった")
    repository.add_review(conn, "S001", "コンセントが多くて助かる")
    # 場所の名前（「の」でつないだ言い方も可）
    ans, facts = answer_reviews(_review_ext(review_place="閲覧室のソファ"), conn)
    assert ans.kind == "review" and [x["spot"]["spot_id"] for x in ans.reviews] == ["S002"]
    assert "［騒がしかった］机がないから、勉強しにくい" in ans.text and "机がないから" in facts
    # 会話で紹介したスポットの ID（口コミがないスポットはそう伝える）
    ans, facts = answer_reviews(_review_ext(review_spot_ids=["S001", "S023", "S999"]), conn)
    assert [x["spot"]["spot_id"] for x in ans.reviews] == ["S001"]
    assert "講義棟 7階 K701には、まだ口コミがありません" in ans.text and "口コミがまだないスポット: 講義棟 7階 K701" in facts
    # 対象の指定がなければ最近の口コミ
    ans, _ = answer_reviews(_review_ext(), conn)
    assert ans.text.startswith("最近の口コミ") and {x["spot"]["spot_id"] for x in ans.reviews} == {"S001", "S002"}
    # 見つからない場所
    ans, _ = answer_reviews(_review_ext(review_place="屋上テラス"), conn)
    assert ans.reviews == [] and "「屋上テラス」というスポットは見つかりませんでした" in ans.text


def test_no_reviews_yet(conn):
    ans, facts = answer_reviews(_review_ext(), conn)
    assert ans.text.startswith("まだ口コミはありません") and "（なし）" in facts
    ans, _ = answer_reviews(_review_ext(building="講義棟", review_place="講義棟"), conn)
    assert ans.reviews == [] and "まだ口コミがありません" in ans.text


def test_review_question_is_answered_from_reviews(conn, monkeypatch):
    repository.add_review(conn, "S002", "机がないから、勉強しにくい", "騒がしかった")
    monkeypatch.setattr(
        answer.extract_mod, "extract", lambda q, c, t, **kw: _review_ext(review_place="閲覧室 ソファ")
    )
    seen = {}

    def fake(prompt, *a, **kw):
        seen["prompt"] = prompt
        return {"reply": "利用者の声では、机がないので勉強しにくいそうです。", "used_notice_ids": []}

    monkeypatch.setattr(answer.reply_mod.claude_cli, "ask_json", fake)
    ans = answer.answer_question("閲覧室ソファの口コミを教えて", conn, today=TODAY)
    assert ans.kind == "review" and ans.text.startswith("利用者の声では")
    assert "机がないから、勉強しにくい" in seen["prompt"] and len(ans.reviews) == 1
    # 返事を書けなければ定型文（口コミの本文をそのまま出す）
    monkeypatch.setattr(answer.reply_mod.claude_cli, "ask_json", lambda *a, **kw: {"reply": "星5つです。"})
    ans = answer.answer_question("閲覧室ソファの口コミを教えて", conn, today=TODAY)
    assert "［騒がしかった］机がないから、勉強しにくい" in ans.text


def test_spot_search_with_reviews_wanted(conn):
    repository.add_review(conn, "S001", "コンセントが多くて助かる")
    r = retrieve.retrieve("q", {"outlets_required": True}, conn, similarity=fake_similarity(ORDER))
    picks = choose_picks({"picks": [{"spot_id": "S001", "reason": "コンセント76口"}]}, r)
    assert "口コミ：コンセントが多くて助かる" in format_headline(picks, r, want_reviews=True)
    assert "口コミ" not in format_headline(picks, r)
    prompt = answer.build_prompt("q", r, TODAY, want_reviews=True)
    assert "口コミ（利用者の声）の内容も知りたがっています" in prompt
    assert "（特になし）" in answer.build_prompt("q", r, TODAY)


def test_history_json_is_valid(conn, monkeypatch):
    ext = normalize({"supported": True, "quietness_min": "会話可", "building": "講義棟", "period": 3})
    text = format_history([("q", Answer("", "normal", extraction=ext))])
    raw = json.loads(text.split("検索条件: ")[1].splitlines()[0])
    assert raw == {"building": "講義棟", "quietness_min": "会話可", "period": 3}


def test_place_name_without_digits_is_not_a_room(conn):
    ext = normalize({"supported": True, "room": "閲覧室", "quietness_min": "会話可"})
    assert answer.check_places("閲覧室で話せる場所", ext, conn, rooms={}) is None
    assert "room" not in ext.conditions
    # 教室番号の形なら、これまでどおり確かめる
    ext = normalize({"supported": True, "room": "K53"})
    assert answer.check_places("K53で電源ある？", ext, conn, rooms={}).kind == "clarify"


def test_unsurveyed_place_is_not_confused_with_surveyed_one(conn):
    # 談話室は未調査。閲覧室（調査済み）の情報では答えない
    ext = normalize({"supported": True, "quietness_min": "会話可"})
    ans = answer.check_places("談話室で話せる場所", ext, conn, rooms={})
    assert ans.kind == "unsurveyed" and ans.text.startswith("談話室は未調査のため情報がありません")
    assert "調査済みの建物・場所：図書館 / 講義棟 / 閲覧室" in ans.text
    assert "閲覧室で話せる場所" in ans.suggestions
    # LLM が建物として抜き出しても同じ
    ext = normalize({"supported": True, "building": "談話室"})
    assert answer.check_places("だんわしつで作業したい", ext, conn, rooms={}).kind == "unsurveyed"
    # 調査済みの場所の名前なら止めない
    ext = normalize({"supported": True})
    assert answer.check_places("閲覧室で作業できる？", ext, conn, rooms={}) is None


def test_review_of_unsurveyed_place(conn, monkeypatch):
    repository.add_review(conn, "S002", "机がないから、勉強しにくい", "騒がしかった")
    monkeypatch.setattr(
        answer.extract_mod, "extract", lambda q, c, t, **kw: _review_ext(review_place="談話室")
    )
    monkeypatch.setattr(answer.reply_mod.claude_cli, "ask_json", lambda *a, **kw: {})
    ans = answer.answer_question("談話室の口コミを教えて", conn, today=TODAY)
    assert ans.kind == "unsurveyed" and "談話室は未調査" in ans.text and not ans.reviews
