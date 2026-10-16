"""静粛性の段階・誤字・実在しない教室/建物の聞き返し・質問文の「◯号館」との照合。

LLM を呼ぶテスト（誤字入りの質問の条件抽出）は RUN_LLM_TESTS=1 のときだけ実行する:
    RUN_LLM_TESTS=1 uv run pytest tests/test_typos.py
"""

import os

import pytest

from src.db import repository
from src.db.rooms import Room, load_rooms
from src.rag import answer as answer_mod
from src.rag import extract
from src.rag import reply as reply_mod
from src.rag.answer import check_building, check_places
from src.rag.extract import normalize
from tests.test_retrieve import BASE

QUIET = ["私語厳禁", "小声可"]
TALK = ["会話可", "通話可"]
CALL = ["通話可"]


# 実在する教室の一覧（data/rooms.csv の代わり）。spot_id が空なら未調査
ROOMS = {r: Room(r, f"S{i:03d}") for i, r in
         enumerate(["K501", "K502", "K503", "K509", "K603", "K701"], 1)}  # fmt: skip
ROOMS |= {r: Room(r, None) for r in ["K203", "K510", "7401", "2131PC室"]}


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(answer_mod, "load_rooms", lambda: ROOMS)
    c = repository.connect(tmp_path / "t.db")
    rooms = ["K501", "K502", "K503", "K509", "K603", "K701"]
    repository.replace_spots(
        c,
        [{**BASE, "spot_id": f"S{i:03d}", "building": "講義棟", "area_name": r}
         for i, r in enumerate(rooms, 1)]
        + [{**BASE, "spot_id": "S100", "building": "図書館"},
           {**BASE, "spot_id": "S101", "building": "21号館"}],
    )  # fmt: skip
    return c


@pytest.mark.parametrize(
    "raw, expected",
    [
        ({"quietness_max": "小声可"}, QUIET),  # 静か・集中
        ({"quietness_max": "私語厳禁"}, ["私語厳禁"]),
        ({"quietness_min": "会話可"}, TALK),  # 話しながら・グループで
        ({"quietness_min": "通話可"}, CALL),  # 通話・電話・オンライン面接
        ({"quietness_min": "小声可", "quietness_max": "会話可"}, ["小声可", "会話可"]),
        ({"quietness_min": "通話可", "quietness_max": "小声可"}, []),  # 矛盾は条件にしない
        ({"quietness": ["通話可", "私語厳禁"]}, ["私語厳禁", "小声可", "会話可", "通話可"]),  # 旧形式
        ({"quietness_max": "静か"}, []),
    ],
)
def test_quietness_is_ordinal(raw, expected):
    assert extract.quietness_range(raw) == expected
    cond = extract.normalize({"supported": True, **raw}).conditions
    assert cond.get("quietness", []) == expected


def test_normalize_room():
    assert extract.normalize_room("ｋ５０３") == "K503"
    assert extract.normalize_room(" k 53 ") == "K53"
    assert extract.normalize_room(None) is None


def _ext(**kw):
    return extract.normalize({"supported": True, **kw})


def test_existing_room_becomes_condition(conn):
    ext = _ext(room="k503")
    assert check_places("K503で電源ある？", ext, conn) is None
    assert ext.conditions["room"] == "K503"


@pytest.mark.parametrize(
    "question, raw",
    [
        ("K53で作業できる？", {"room": "K53"}),
        ("K53で作業できる？", {"room": "K503"}),  # LLM が近い教室に直しても、質問文の表記で確かめる
        ("K53で作業できる？", {}),  # LLM が拾わなくても質問文から拾う
        ("Ｋ５３で作業できる？", {"room": "Ｋ５３"}),
    ],
)
def test_unknown_room_asks_back_with_close_candidates(conn, question, raw):
    ans = check_places(question, _ext(**raw), conn)
    assert ans is not None and ans.kind == "clarify"
    assert ans.text.startswith("「K53」という教室は見つかりませんでした")
    assert "- K503" in ans.text
    assert ans.suggestions[0] == "K503で作業できる？"


def test_unsurveyed_building_is_not_asked_back(conn):
    ans = check_places("1号館で電源がある所", _ext(building="1号館", outlets_required=True), conn)
    assert ans.kind == "unsurveyed"
    assert ans.text == "1号館は未調査のため情報がありません。\n\n調査済みの建物・場所：21号館 / 図書館 / 講義棟"
    # 番号が似ているだけの「◯号館」は代わりに勧めない（一覧には出す）
    assert ans.suggestions == ["図書館で電源がある所", "講義棟で電源がある所"]
    ans = check_places("体育館で作業", _ext(building="体育館"), conn)
    assert ans.kind == "unsurveyed" and ans.text.startswith("体育館は未調査のため")


@pytest.mark.parametrize("question, raw", [("K203で電源ある？", {"room": "K203"}),
                                           ("ｋ２０３で電源ある？", {})])  # fmt: skip
def test_unsurveyed_room_suggests_nearby_surveyed(conn, question, raw):
    ans = check_places(question, _ext(**raw), conn)
    assert ans.kind == "unsurveyed"
    assert ans.text.splitlines()[0] == "K203 は未調査のため、設備の情報がありません。"
    # 階が近い順（5階 → 6階）、同じ階なら番号が近い順
    assert "- 講義棟 2階 K502" in ans.text  # fixture の BASE は floor=2
    assert ans.suggestions[0].endswith("で電源ある？") and len(ans.suggestions) == 3


def test_unsurveyed_room_outside_lecture_building(conn):
    ans = check_places("7401で作業できる？", _ext(room="7401"), conn)
    assert ans.kind == "unsurveyed" and ans.text.startswith("7401 は未調査のため")
    assert "調査済みの建物・場所：21号館 / 図書館 / 講義棟" in ans.text
    # LLM が拾わなくても、一覧にある表記を質問文から拾う
    ans = check_places("2131PC室は静か？", _ext(), conn)
    assert ans.kind == "unsurveyed" and ans.text.startswith("2131PC室 は未調査")


def test_rooms_csv_is_loaded():
    rooms = load_rooms()
    assert len(rooms) == 76
    assert rooms["K203"].spot_id is None and rooms["K501"].spot_id == "S005"


def test_digits_only_and_known_buildings_pass(conn):
    assert check_places("10/2の3限に電源", _ext(period=3), conn) is None
    assert check_places("としょかんで勉強", _ext(building="図書館"), conn) is None


# ---------------------------------------------------------------- 質問文の「◯号館」との照合


def test_written_buildings_picks_up_all_notations():
    assert extract.written_buildings("23号館はある？") == ["23号館"]
    assert extract.written_buildings("２３号館") == ["23号館"]  # 全角数字
    assert extract.written_buildings("二十三号館で電源") == ["23号館"]  # 漢数字
    assert extract.written_buildings("2号館の静かな場所") == ["2号館"]
    assert extract.written_buildings("図書館で勉強") == []


# LLM が「23号館」を「21号館」に読み替えた（中間発表の不具合）ときも、質問文の表記で確かめる
UNSURVEYED_BUILDING_CASES = [
    ("23号館はある？", {"intent": "notice", "building": "21号館", "notice_place": "21号館"}, "23号館"),
    ("23号館はあったけ？", {"intent": "notice", "building": "21号館"}, "23号館"),
    ("23号館で電源がある場所", {"building": "21号館", "outlets_required": True}, "23号館"),
    ("23号館で電源がある場所", {"outlets_required": True}, "23号館"),  # LLM が拾わなくても拾う
    ("２３号館", {"building": "21号館"}, "23号館"),
    ("2号館の静かな場所", {"building": "21号館", "quietness_max": "小声可"}, "2号館"),
    ("2号館の静かな場所", {"building": "2号館", "quietness_max": "小声可"}, "2号館"),
]


@pytest.mark.parametrize("question, raw, written", UNSURVEYED_BUILDING_CASES)
def test_unsurveyed_building_from_question_text(conn, question, raw, written):
    ext = _ext(**raw)
    ans = check_places(question, ext, conn)
    assert ans is not None and ans.kind == "unsurveyed"
    assert ans.text.startswith(f"{written}は未調査のため情報がありません。")
    assert "調査済みの建物・場所：21号館 / 図書館 / 講義棟" in ans.text
    # 条件・お知らせの対象も質問文の建物にそろえる（21号館のお知らせ・スポットを答えない）
    assert ext.conditions["building"] == written
    assert ext.notice_place in (None, written)
    # 番号が似ているだけの「21号館」は代わりに勧めない
    assert not any("21号館" in sg for sg in ans.suggestions)


def test_surveyed_list_uses_place_names_not_only_buildings(tmp_path, monkeypatch):
    """未調査の答えに出す一覧・候補ボタンは、建物名が空欄の場所（閲覧室・自習室）も含める。"""
    monkeypatch.setattr(answer_mod, "load_rooms", lambda: ROOMS)
    c = repository.connect(tmp_path / "p.db")
    repository.replace_spots(
        c,
        [{**BASE, "spot_id": "S100"},  # 図書館
         {**BASE, "spot_id": "S101", "building": "21号館"},
         {**BASE, "spot_id": "S102", "building": None, "floor": None, "area_name": "閲覧室 6人掛けデスク"},
         {**BASE, "spot_id": "S103", "building": None, "floor": None, "area_name": "自習室 窓側デスク"}],
    )  # fmt: skip
    assert answer_mod.surveyed_places(c) == ["21号館", "図書館", "閲覧室", "自習室"]
    ans = check_places("23号館はある？", _ext(building="21号館"), c)
    assert ans.text == (
        "23号館は未調査のため情報がありません。\n\n調査済みの建物・場所：21号館 / 図書館 / 閲覧室 / 自習室"
    )
    # 候補ボタンも同じ一覧から（番号が似ているだけの「21号館」は除く）
    assert ans.suggestions == ["図書館はある？", "閲覧室はある？", "自習室はある？"]


def test_surveyed_building_is_answered_as_before(conn):
    ext = _ext(building="21号館", outlets_required=True)
    assert check_places("21号館で電源がある場所", ext, conn) is None
    assert ext.conditions == {"building": "21号館", "outlets_required": True}
    # 表記のゆれ（「21号」「２１号館」）を LLM が揃えたのは上書きしない
    for question in ("21号で電源がある場所", "２１号館で電源がある場所"):
        ext = _ext(building="21号館", outlets_required=True)
        assert check_places(question, ext, conn) is None
        assert ext.conditions["building"] == "21号館"


def test_llm_building_wins_only_when_question_has_no_number(conn):
    # 「◯号館」以外の建物名は上書きしない（「21号館の近くの図書館」のような質問のため）
    ext = _ext(building="図書館")
    assert check_building("21号館の近くの図書館で勉強", ext, conn) is None
    assert ext.conditions["building"] == "図書館"


def test_unsurveyed_building_answer_for_notice_question(conn, monkeypatch):
    """「23号館はある？」（お知らせの確認と読み取られる質問）の答え。"""
    monkeypatch.setattr(
        answer_mod.extract_mod, "extract",
        lambda q, c, t, **kw: normalize({"supported": True, "intent": "notice", "building": "21号館",
                                         "notice_place": "21号館"}),
    )  # fmt: skip
    # 返事を書く LLM が「21号館はあります」と書いても、定型文に差し替える
    monkeypatch.setattr(
        answer_mod.reply_mod.claude_cli, "ask_json",
        lambda *a, **kw: {"reply": "21号館はありますよ。"},
    )  # fmt: skip
    ans = answer_mod.answer_question("23号館はある？", conn)
    assert ans.kind == "unsurveyed"
    assert ans.text == (
        "23号館は未調査のため情報がありません。\n\n調査済みの建物・場所：21号館 / 図書館 / 講義棟"
    )


def test_unsurveyed_place_answer_for_notice_question(conn, monkeypatch):
    """「談話室は使える？」（お知らせの確認と読み取られる質問）も、未調査と答える。"""
    monkeypatch.setattr(
        answer_mod.extract_mod, "extract",
        lambda q, c, t, **kw: normalize({"supported": True, "intent": "notice", "notice_place": "談話室"}),
    )  # fmt: skip
    monkeypatch.setattr(answer_mod.reply_mod.claude_cli, "ask_json", lambda *a, **kw: {})
    ans = answer_mod.answer_question("談話室は使える？", conn)
    assert ans.kind == "unsurveyed" and ans.text.startswith("談話室は未調査のため情報がありません。")


# ---------------------------------------------------------------- 返事の場所の名前の照合


FACTS = "案内文:\n23号館は未調査のため情報がありません。\n\n調査済みの建物・場所：21号館 / 図書館 / 講義棟"


def test_reply_about_another_building_is_replaced():
    """質問された建物に触れず、別の「◯号館」のことを答えた返事は使わない（定型文に差し替える）。"""
    assert reply_mod.checked("21号館はありますよ。", FACTS, "23号館はある？") is None
    assert reply_mod.checked("21号館で電源が使えます。", FACTS, "23号館で電源がある場所") is None
    assert reply_mod.checked("21号館の静かな席がおすすめです。", FACTS, "2号館の静かな場所") is None
    # 質問された建物に触れていれば、調査済みの建物の一覧として挙げるのはよい
    assert reply_mod.checked("23号館は調査していないので情報がないんです。", FACTS, "23号館はある？")
    assert reply_mod.checked(
        "23号館はまだ調べていません。調査済みは21号館・図書館・講義棟です。", FACTS, "23号館はある？"
    )
    # 質問文に場所の指定がなければ、データにある建物・教室は出してよい
    assert reply_mod.checked("21号館の1階がおすすめです。", "建物: 21号館 / 階: 1", "電源がある場所")
    # データにない建物・教室は出させない
    assert reply_mod.checked("2号館はありますよ。", "建物: 21号館", "電源がある場所") is None
    assert reply_mod.checked("K709が空いています。", "K503 K507", "電源がある場所") is None
    assert reply_mod.checked("K503が空いています。", "K503 K507", "電源がある場所")
    # 聞き返しも同じ（「K53」に触れずに K503 のことを答えない）。context は質問文も含む
    clarify = "K53で電源ある？\n案内文:\n「K53」という教室は見つかりませんでした。\n- K503\n- K507"
    assert reply_mod.checked("K503のことですか？", clarify, "K53で電源ある？") is None
    assert reply_mod.checked("K53は見つかりません。K503ですか？", clarify, "K53で電源ある？")


# ---------------------------------------------------------------- LLM（誤字入りの質問）

llm = pytest.mark.skipif(not os.environ.get("RUN_LLM_TESTS"), reason="RUN_LLM_TESTS=1 で実行")

TYPO_CASES = [
    ("でんげんがあるところ", {"outlets_required": True}),
    ("コンセンとがある静かな場所", {"outlets_required": True, "quietness": QUIET}),
    ("としょかんでべんきょうしたい", {"building": "図書館"}),
    ("こうぎとうで充電したい", {"building": "講義棟", "outlets_required": True}),
    ("しずかで集中できるとこ", {"quietness": QUIET}),
    ("グループで話しながら作業したい", {"quietness": TALK}),
    ("ぐるーぷではなしながら作業", {"quietness": TALK}),
    ("でんわできる場所", {"quietness": CALL}),
    ("オンライン面接ができるところ", {"quietness": CALL}),
]


@llm
@pytest.mark.parametrize("question, expected", TYPO_CASES)
def test_typo_questions_map_to_conditions(conn, question, expected):
    ext = extract.extract(question, conn)
    assert ext.supported
    for key, value in expected.items():
        assert ext.conditions.get(key) == value, ext.conditions


@llm
@pytest.mark.parametrize("question, written", [("K53で電源ある？", "K53"), ("Ｋ５３は静か？", "K53")])
def test_unknown_rooms_are_not_guessed(conn, question, written):
    """実在しない教室番号は、近い教室に直さず聞き返す。"""
    ext = extract.extract(question, conn)
    assert ext.room is None or extract.normalize_room(ext.room) == written  # 近い教室に直さない
    ans = check_places(question, ext, conn)
    assert ans is not None and ans.kind == "clarify" and f"「{written}」" in ans.text


@llm
@pytest.mark.parametrize(
    "question, written",
    [("23号館はある？", "23号館"), ("23号館で電源がある場所", "23号館"), ("２３号館", "23号館"),
     ("2号館の静かな場所", "2号館"), ("1号館で電源がある所", "1号館")],
)  # fmt: skip
def test_unsurveyed_buildings_are_not_replaced(conn, question, written):
    """調査していない「◯号館」は、番号の近い建物（21号館）に読み替えない。"""
    ext = extract.extract(question, conn)
    assert ext.conditions.get("building") in (None, written)  # 抽出で番号を変えない
    ans = check_places(question, ext, conn)
    assert ans is not None and ans.kind == "unsurveyed"
    assert ans.text.startswith(f"{written}は未調査のため情報がありません。")
    assert not any("21号館" in sg for sg in ans.suggestions)  # 代わりに勧めない


@llm
def test_surveyed_building_question_is_unchanged(conn):
    ext = extract.extract("21号館で電源がある場所", conn)
    assert ext.conditions == {"building": "21号館", "outlets_required": True}
    assert check_places("21号館で電源がある場所", ext, conn) is None
