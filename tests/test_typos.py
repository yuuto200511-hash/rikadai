"""静粛性の段階・誤字・実在しない教室/建物の聞き返し。

LLM を呼ぶテスト（誤字入りの質問の条件抽出）は RUN_LLM_TESTS=1 のときだけ実行する:
    RUN_LLM_TESTS=1 uv run pytest tests/test_typos.py
"""

import os

import pytest

from src.db import repository
from src.db.rooms import Room, load_rooms
from src.rag import answer as answer_mod
from src.rag import extract
from src.rag.answer import check_places
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
    assert ans.text == "1号館は未調査のため情報がありません。\n\n調査済みの建物：21号館 / 図書館 / 講義棟"
    assert ans.suggestions == ["21号館で電源がある所", "図書館で電源がある所", "講義棟で電源がある所"]
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
    assert "調査済みの建物：21号館 / 図書館 / 講義棟" in ans.text
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
@pytest.mark.parametrize(
    "question, written", [("K53で電源ある？", "K53"), ("1号館で電源がある所", "1号館")]
)
def test_unknown_places_are_not_guessed(conn, question, written):
    ext = extract.extract(question, conn)
    ans = check_places(question, ext, conn)
    assert ans is not None and ans.kind == "clarify" and f"「{written}」" in ans.text
