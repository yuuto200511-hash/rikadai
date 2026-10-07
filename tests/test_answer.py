"""おすすめ（最大3件）・立ち作業の席・お知らせの確認・対応外の返事・LLM が書く返事の確認。"""

from datetime import date

import pytest

from src import prompts
from src.db import repository
from src.rag import answer, retrieve
from src.rag.answer import Answer, answer_notices, cards, choose_picks, format_headline
from src.rag.extract import normalize
from tests.test_retrieve import BASE, fake_similarity

TODAY = date(2026, 10, 1)
DEMO = "【デモ用・架空】\n件名: x"


@pytest.fixture
def conn(tmp_path):
    c = repository.connect(tmp_path / "t.db")
    repository.replace_spots(
        c,
        [
            {**BASE, "spot_id": "S001"},
            {**BASE, "spot_id": "S002", "area_name": "スタンドアップデスク", "chair_type": "なし",
             "standing": "可"},
            {**BASE, "spot_id": "S003", "area_name": "壁沿いデスク"},
            {**BASE, "spot_id": "S004", "building": None, "floor": None, "area_name": "閲覧室 ソファ"},
        ],
    )  # fmt: skip
    return c


def _picks(conn, ids, cond=None):
    r = retrieve.retrieve("q", cond or {}, conn, similarity=fake_similarity(["S001", "S002", "S003", "S004"]))
    return r, choose_picks({"picks": [{"spot_id": i, "reason": "静粛性：私語厳禁"} for i in ids]}, r)


def test_up_to_three_picks_and_cards_in_same_order(conn):
    r, picks = _picks(conn, ["S003", "S001", "S004", "S002"])
    assert [p["spot"]["spot_id"] for p in picks] == ["S003", "S001", "S004"]
    ans = Answer(format_headline(picks, r), "normal", {}, r, picks)
    assert [line for line in ans.text.split("\n") if line.startswith("**")] == [
        "**おすすめ1：図書館 2階 壁沿いデスク**  ", "**おすすめ2：図書館 2階 窓側自習エリア**  ",
        "**おすすめ3：閲覧室 ソファ**  ",
    ]  # fmt: skip
    assert ans.spot_ids == ["S003", "S001", "S004"]
    assert [c.split("\n")[0] for _, c in cards(ans)][0] == "#### おすすめ1｜図書館 2階 壁沿いデスク"
    # 候補が少なければその件数だけ（無理に3件そろえない）
    r, picks = _picks(conn, ["S001"])
    assert len(picks) == 1 and len(cards(Answer("", "normal", {}, r, picks))) == 1


def test_standing_seat_is_first_only_when_wanted(conn):
    _, picks = _picks(conn, ["S002", "S001"])
    assert [p["spot"]["spot_id"] for p in picks] == ["S001", "S002"]
    # おすすめが立ち作業の席だけなら、座れる候補を1件目に足す
    _, picks = _picks(conn, ["S002"])
    assert [p["spot"]["spot_id"] for p in picks] == ["S001", "S002"]
    # 立ち作業を望んだときはおすすめ1にしてよい
    _, picks = _picks(conn, ["S002"], {"standing": "可"})
    assert [p["spot"]["spot_id"] for p in picks] == ["S002"]


def _notice(c, name, confirmed=True, body=DEMO, **kw):
    n = {"source_file": name, "raw_body": body, "is_relevant": True, "confirmed": confirmed,
         "impact_type": "閉鎖", "title": name, "valid_from": "2026-09-28", "valid_to": "2026-10-09",
         **kw}  # fmt: skip
    repository.upsert_notice(c, n)


def test_notice_answer_lists_confirmed_active_notices(conn):
    ext = normalize({"supported": True, "intent": "notice"})
    assert answer_notices(ext, conn, TODAY).text == "現在、利用に影響するお知らせはありません。"
    _notice(conn, "lib", target_building="図書館", target_floor=2, target_area="壁沿いデスク")
    _notice(conn, "hours", target_building="図書館", impact_type="時間変更", valid_to=None, body="件名: y")
    _notice(conn, "etsuran", target_area="閲覧室", valid_from="2026-10-01", valid_to="2026-10-02")
    _notice(conn, "unconfirmed", confirmed=False, target_building="21号館")
    _notice(conn, "past", target_building="21号館", valid_from="2026-09-24", valid_to="2026-09-26")
    _notice(conn, "irrelevant", is_relevant=False)
    ans = answer_notices(ext, conn, TODAY)
    assert ans.kind == "notice"
    assert ans.text.split("\n") == [
        "現在、利用に影響するお知らせは次の3件です（10/1時点）。", "",
        "- （デモ用）図書館 2階 壁沿いデスク ／ 閉鎖 ／ 9/28〜10/9",
        "- （デモ用）閲覧室 ／ 閉鎖 ／ 10/1〜10/2",
        "- 図書館 ／ 時間変更 ／ 9/28〜（終了日の記載なし）",
    ]  # fmt: skip
    # 建物・場所を指定されたら絞り込む
    ext = normalize({"supported": True, "intent": "notice", "building": "図書館", "notice_place": "図書館"})
    assert answer_notices(ext, conn, TODAY).text.count("\n- ") == 2
    ext = normalize({"supported": True, "intent": "notice", "notice_place": "21号館"})
    assert answer_notices(ext, conn, TODAY).text == "21号館について、現在、利用に影響するお知らせはありません。"


def test_notice_without_building_matches_by_area(conn):
    _notice(conn, "etsuran", target_area="閲覧室", valid_from="2026-10-01", valid_to="2026-10-02")
    r = retrieve.retrieve("q", {}, conn, today=date(2026, 10, 2), similarity=fake_similarity([]))
    assert [e["spot"]["spot_id"] for e in r.excluded] == ["S004"]
    assert r.excluded[0]["reason"].startswith("（デモ用）お知らせ「etsuran」により 閲覧室 は閉鎖中")
    r = retrieve.retrieve("q", {}, conn, today=date(2026, 10, 3), similarity=fake_similarity([]))
    assert r.excluded == []


def test_unsupported_reply_is_short():
    text = prompts.load("unsupported", reason="空席情報は扱っていません")
    lines = text.strip().split("\n")
    assert 2 <= len(lines) <= 3 and not text.startswith("申し訳")
    assert lines[0].startswith("この質問には対応していません")
    assert "混雑" not in text and "- " not in text  # 機能の一覧は出さない


def _llm_fails(*_a, **_kw):
    raise answer.claude_cli.ClaudeError("x")


def test_answer_question_routes_notice_intent(conn, monkeypatch):
    monkeypatch.setattr(
        answer.extract_mod, "extract",
        lambda q, c, t, **kw: normalize({"supported": True, "intent": "notice", "building": "1号館"}),
    )  # fmt: skip
    monkeypatch.setattr(answer.reply_mod.claude_cli, "ask_json", _llm_fails)
    # お知らせの確認では、建物が未調査かどうかは確かめない（返事を書けなければ定型文）
    ans = answer.answer_question("1号館のお知らせは？", conn, today=TODAY)
    assert ans.kind == "notice" and ans.text == "1号館について、現在、利用に影響するお知らせはありません。"


def test_notice_is_not_narrowed_by_generic_words(conn):
    _notice(conn, "lib", target_building="図書館", target_floor=2, target_area="壁沿いデスク")
    # 「教室」を LLM が講義棟に置き換えても、質問文に「講義棟」と書かれていなければ絞らない
    for raw in ({"building": "講義棟"}, {"notice_place": "教室"}, {"notice_place": "部屋"}):
        ext = normalize({"supported": True, "intent": "notice", **raw})
        ans = answer_notices(ext, conn, TODAY, "お知らせで使えない教室ある？")
        assert [n["title"] for n in ans.notices] == ["lib"]
    ext = normalize({"supported": True, "intent": "notice", "building": "講義棟"})
    assert answer_notices(ext, conn, TODAY, "講義棟のお知らせは？").notices == []


def test_notice_reply_is_written_by_llm_with_details(conn, monkeypatch):
    _notice(conn, "lib", target_building="図書館", target_floor=2, target_area="壁沿いデスク")
    monkeypatch.setattr(
        answer.extract_mod, "extract", lambda q, c, t, **kw: normalize({"supported": True, "intent": "notice"})
    )
    reply = "図書館2階の壁沿いデスクは、10/9まで使えません（デモ用のお知らせ）。"
    monkeypatch.setattr(
        answer.reply_mod.claude_cli, "ask_json", lambda *a, **kw: {"reply": reply, "used_notice_ids": [1]}
    )
    ans = answer.answer_question("お知らせある？", conn, today=TODAY)
    assert ans.text == reply and [n["title"] for n in ans.notices] == ["lib"]
    assert "（デモ用のお知らせ）" in ans.context and "期間: 9/28〜10/9" in ans.context
    # データにない数値を書いたら定型文に戻す
    monkeypatch.setattr(
        answer.reply_mod.claude_cli, "ask_json", lambda *a, **kw: {"reply": "10/31まで使えません。"}
    )
    ans = answer.answer_question("お知らせある？", conn, today=TODAY)
    assert ans.text.startswith("現在、利用に影響するお知らせは次の1件です")


def test_unsupported_reply_uses_related_notices(conn, monkeypatch):
    _notice(conn, "lib", target_building="図書館", target_floor=2, target_area="壁沿いデスク")
    _notice(conn, "etsuran", target_area="閲覧室", valid_from="2026-10-01", valid_to="2026-10-02")
    monkeypatch.setattr(
        answer.extract_mod, "extract",
        lambda q, c, t, **kw: normalize({"supported": False, "unsupported_reason": "開館時間", "building": "図書館"}),
    )  # fmt: skip
    seen = {}

    def fake(prompt, *a, **kw):
        seen["prompt"] = prompt
        return {"reply": "開館時間のデータはありません。", "used_notice_ids": [1, 99]}

    monkeypatch.setattr(answer.reply_mod.claude_cli, "ask_json", fake)
    ans = answer.answer_question("図書館は何時まで？", conn, today=TODAY)
    assert ans.kind == "unsupported" and [n["title"] for n in ans.notices] == ["lib"]
    assert "図書館の調査済みスポット" in seen["prompt"] and "[お知らせID 1]" in seen["prompt"]
    assert "閲覧室" not in seen["prompt"]  # 質問の建物と関係ないお知らせは渡さない


def test_spot_reply_falls_back_when_order_changed(conn):
    r, picks = _picks(conn, ["S001", "S003"])
    raw = {"picks": [{"spot_id": "S001"}, {"spot_id": "S003"}], "reply": "図書館 2階 窓側自習エリアがおすすめです。"}
    assert answer.spot_reply(raw, r, picks, "図書館 2階") == raw["reply"]
    assert answer.spot_reply({**raw, "reply": "座席が99席あります。"}, r, picks, "座席数: 10") is None
    # 立ち作業の席をコードで並べ直したら、返事とカードの順番が合わないので定型文
    r, picks = _picks(conn, ["S002", "S001"])
    raw = {"picks": [{"spot_id": "S002"}, {"spot_id": "S001"}], "reply": "スタンドアップデスクです。"}
    assert answer.spot_reply(raw, r, picks, "") is None
