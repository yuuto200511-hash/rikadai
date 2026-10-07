from datetime import date

import pytest

from src.db import repository
from src.rag import retrieve
from src.rag.extract import normalize

BASE = {
    "building": "図書館", "floor": 2, "area_name": "窓側自習エリア", "seats": 10, "outlets": 5,
    "chair_type": "木製椅子", "standing": "不可", "quietness": "私語厳禁", "food": "不可",
    "task_light": "なし", "outlet_location": None, "note": None, "photo": None,
    "surveyed_at": "2026-09-10", "surveyor": "A", "source": "テスト",
}  # fmt: skip


def fake_similarity(order):
    def sim(_q, ids):
        return {sid: 1.0 - order.index(sid) * 0.1 for sid in ids if sid in order}

    return sim


@pytest.fixture
def conn(tmp_path):
    c = repository.connect(tmp_path / "t.db")
    repository.replace_spots(
        c,
        [
            {**BASE, "spot_id": "S001"},
            {**BASE, "spot_id": "S002", "floor": 3, "area_name": "個人ブース"},
            {**BASE, "spot_id": "S003", "building": "講義棟", "quietness": "会話可"},
            {**BASE, "spot_id": "S004", "building": "10号館", "quietness": "小声可"},
            {**BASE, "spot_id": "S005", "building": "7号館", "outlets": 0},
        ],
    )
    return c


def notice(c, name, **kw):
    n = {"source_file": name, "raw_body": "x", "is_relevant": True, "confirmed": True,
         "impact_type": "閉鎖", "title": name, **kw}  # fmt: skip
    return repository.upsert_notice(c, n)


def test_strict_filter_and_rank(conn):
    r = retrieve.retrieve(
        "q",
        {"quietness": ["私語厳禁", "小声可"], "outlets_required": True},
        conn,
        similarity=fake_similarity(["S004", "S002", "S001"]),
    )
    assert [s["spot_id"] for s in r.spots] == ["S004", "S002", "S001"]
    assert r.relaxed_key is None and r.excluded == []


def test_closure_notice_excludes_with_reason(conn):
    notice(conn, "n1", target_building="図書館", target_floor=2, target_area="窓側自習エリア",
           valid_from="2026-09-20", valid_to="2026-10-10")  # fmt: skip
    notice(conn, "n2", target_building="10号館", valid_from="2026-09-26", valid_to="2026-09-26")
    notice(conn, "past", target_building="講義棟", valid_from="2026-08-01", valid_to="2026-08-02")
    notice(conn, "unconfirmed", target_building="7号館", confirmed=False)
    notice(conn, "hours", target_building="図書館", target_floor=3, impact_type="時間変更")
    r = retrieve.retrieve("q", {}, conn, today=date(2026, 9, 26), similarity=fake_similarity([]))
    assert sorted(e["spot"]["spot_id"] for e in r.excluded) == ["S001", "S004"]
    assert "閉鎖" in r.excluded[0]["reason"]
    assert {s["spot_id"] for s in r.spots} <= {"S002", "S003", "S005"}
    assert {n["source_file"] for n in r.notices} == {"n1", "n2", "hours"}
    # 期間外なら除外されない
    r2 = retrieve.retrieve("q", {}, conn, today=date(2026, 10, 11), similarity=fake_similarity([]))
    assert r2.excluded == []


def test_relax_one_condition_when_empty(conn):
    r = retrieve.retrieve(
        "q",
        {"building": "10号館", "quietness": ["私語厳禁"]},
        conn,
        similarity=fake_similarity(["S001", "S002", "S005"]),
    )
    assert r.relaxed_key == "building"
    assert [s["spot_id"] for s in r.spots] == ["S001", "S002", "S005"]


def test_no_result_even_after_relax(conn):
    r = retrieve.retrieve(
        "q", {"food": ["食事可"], "chair_type": ["ソファ"]}, conn, similarity=fake_similarity([])
    )
    assert r.spots == [] and r.relaxed_key is None


def test_normalize_drops_out_of_choice_values():
    e = normalize(
        {"supported": True, "quietness": ["静か", "小声可", "未調査"], "food": ["未調査"],
         "outlets_required": None, "floor": "3", "chair_type": "ソファ", "standing": "たぶん",
         "task_light": "あり"}
    )  # fmt: skip
    assert e.conditions == {
        "quietness": ["小声可"], "floor": 3, "chair_type": ["ソファ"], "task_light": "あり"
    }
    # 全選択肢が指定されても条件として残す（未調査のスポットを出さないため）
    e = normalize({"supported": True, "food": ["不可", "飲み物のみ", "食事可"]})
    assert e.conditions == {"food": ["不可", "飲み物のみ", "食事可"]}
    assert normalize({"supported": False, "unsupported_reason": "混雑"}).supported is False


def test_answer_prompt_marks_unsurveyed_and_blank_and_hides_source(conn):
    from src.rag.answer import build_prompt

    repository.replace_spots(
        conn,
        [{**BASE, "spot_id": "S001", "building": None, "floor": None, "seats": None,
          "quietness": "未調査", "food": "未調査", "outlet_location": "柱に2か所",
          "surveyed_at": None, "source": "秘密の元データ.xlsx"}],
    )  # fmt: skip
    r = retrieve.retrieve("q", {}, conn, similarity=fake_similarity([]))
    p = build_prompt("q", r)
    assert "建物未記入 階未記入 窓側自習エリア" in p
    assert "未調査の項目: 静粛性、飲食" in p
    assert "未記入の項目: 建物、階、座席数、調査日" in p
    assert "電源: 未記入" in p and "確保率" not in p
    assert "柱に2か所" in p
    assert "秘密の元データ" not in p
    assert "全スポットが対象" in p


def test_unknown_blocked_spots_are_reported(conn):
    repository.replace_spots(
        conn,
        [
            {**BASE, "spot_id": "S001", "food": "未調査"},
            {**BASE, "spot_id": "S002", "food": "未調査", "building": "講義棟"},
            {**BASE, "spot_id": "S003", "food": "食事可", "outlets": None},
        ],
    )
    r = retrieve.retrieve(
        "q", {"building": "図書館", "food": ["食事可"]}, conn, similarity=fake_similarity([])
    )
    assert [(u["spot"]["spot_id"], u["keys"]) for u in r.unknown] == [("S001", ["food"])]
    r = retrieve.retrieve(
        "q", {"food": ["食事可"], "outlets_required": True}, conn, similarity=fake_similarity([])
    )
    assert [(u["spot"]["spot_id"], u["keys"]) for u in r.unknown] == [
        ("S001", ["food"]), ("S002", ["food"]), ("S003", ["outlets_required"]),
    ]  # fmt: skip


def test_unknown_spots_are_summarized_in_one_line(conn):
    from src.rag.answer import format_unknown, note_line

    repository.replace_spots(
        conn,
        [
            {**BASE, "spot_id": "S001", "food": "未調査"},
            {**BASE, "spot_id": "S002", "food": "未調査"},
            {**BASE, "spot_id": "S003", "food": "未調査", "building": None},
            {**BASE, "spot_id": "S004", "outlets": None, "food": "飲み物のみ"},
            {**BASE, "spot_id": "S006", "building": "講義棟", "food": "飲み物のみ"},
        ],
    )
    r = retrieve.retrieve(
        "q", {"food": ["飲み物のみ"], "outlets_required": True}, conn,
        similarity=fake_similarity(["S006"]),
    )  # fmt: skip
    assert r.spots == [] and [s["spot_id"] for s in r.classrooms] == ["S006"]
    # 空欄の建物は「その他」にまとめ、「建物未記入」とは出さない
    assert format_unknown(r) == (
        "図書館の3件・その他の1件は飲食・電源ありが未調査・未記入のため、条件に合うか分かりません。"
    )
    assert note_line(r).startswith("図書館の3件")
    # 建物が空欄なだけで一致させなかったスポットは数えない
    r = retrieve.retrieve(
        "q", {"building": "講義棟"}, conn, similarity=fake_similarity([])
    )
    assert format_unknown(r) == ""


def test_spot_text_joins_location_and_note():
    from src.db.vector_store import spot_text

    assert spot_text({**BASE, "note": "窓際で明るい"}) == "図書館 2階 窓側自習エリア 窓際で明るい"
    assert spot_text({**BASE, "building": None, "floor": None}) == "窓側自習エリア"
