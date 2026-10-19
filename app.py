"""Streamlit エントリポイント: uv run streamlit run app.py

app.py は利用者向けの画面（チャット・口コミ投稿）。管理画面は pages/ に分け、サイドバーの「管理」にまとめる。
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import streamlit as st

from src.choices import PERIOD_TIMES, REVIEW_RATINGS
from src.db import repository
from src.ingest import error_report as report_ingest
from src.ingest import review as review_ingest
from src.rag.answer import answer_question, cards, notice_body, notice_details, notice_period, side_notes
from src.rag.kitchen_car import menu_line, vendor_line
from src.rag.retrieve import notice_place
from src.rag.when import day_label
from src.summary import NO_INFO, today_summary
from src.ingest.notice import demo_label
from src.ui import conn, spot_label

PHOTO_DIR = repository.ROOT / "data" / "photos"
EXAMPLES = ("電源があって静かなところ", "明日の3限に空いている教室", "グループで話しながら作業したい")

st.set_page_config(page_title="キャンパス・スポットナビ", page_icon="📍", layout="centered")
# スマホの幅で、ボタンの文言を折り返して全文を見せる
st.markdown(
    "<style>div.stButton > button p { white-space: normal; text-align: left; }</style>",
    unsafe_allow_html=True,
)


def show_details(ans) -> None:
    """各スポットの詳細カード（DB の値からのテンプレート）を「詳細を見る」に畳んで表示する。"""
    with st.expander("詳細を見る"):
        for s, card in cards(ans):
            with st.container(border=True):
                st.markdown(card)
                photo = PHOTO_DIR / (s["photo"] or "")
                if s["photo"] and photo.is_file():
                    st.image(str(photo), width="stretch")
        for note in side_notes(ans.retrieval):
            st.caption(note)


def show_notice_details(ans) -> None:
    """返事が触れたお知らせの期間・影響の種類・元のメール本文（DB の値からのテンプレート）。"""
    with st.expander("詳細を見る"):
        for n in ans.notices:
            with st.container(border=True):
                st.markdown(notice_details(n))
                st.text(notice_body(n))


def show_reviews(ans) -> None:
    """口コミの確認で答えた口コミ（DB の値をそのまま表示する）。"""
    with st.expander("口コミを見る"):
        st.caption("口コミは利用者の声で、公式の調査データではありません。")
        for x in ans.reviews:
            with st.container(border=True):
                st.markdown(f"**{spot_label(x['spot'])}**")
                for r in x["reviews"]:
                    rating = f"［{r['rating']}］" if r["rating"] else ""
                    st.markdown(f"{rating}{r['body']}")
                    st.caption(r["posted_at"][:10])


def show_vendor(g: dict) -> None:
    """キッチンカー1店（場所・店名・時間）。メニューは折りたたむ。"""
    st.markdown(f"**{vendor_line(g)}**")
    with st.expander(f"メニュー（{len(g['items'])}品）"):
        st.markdown("\n".join(f"- {menu_line(*it)}" for it in g["items"]))


def show_kitchen_cars(ans) -> None:
    """キッチンカーの確認で答えた日・店とメニュー（出店表の値をそのまま表示する）。"""
    with st.expander("メニューを見る"):
        for day, gs in ans.kitchen_cars:
            st.markdown(f"**{day_label(day)}**")
            for g in gs:
                st.markdown(f"{vendor_line(g)}  \n" + " ／ ".join(menu_line(*it) for it in g["items"]))


def show_today_summary() -> None:
    """今日のまとめ（LLM を使わず DB の値から作る）。データがない項目も隠さない。"""
    today = datetime.now(ZoneInfo("Asia/Tokyo")).date()
    summary = today_summary(conn(), today)
    with st.container(border=True):
        st.markdown(f"##### 今日のまとめ（{day_label(today)}）")
        st.markdown("🚚 **今日のキッチンカー**")
        if not summary.kitchen_cars:
            st.caption(NO_INFO)
        for g in summary.kitchen_cars:
            show_vendor(g)
        st.markdown("🚧 **今日使えない場所**")
        if not summary.closed:
            st.caption(NO_INFO)
        for n in summary.closed:
            st.markdown(f"- {demo_label(n)}{notice_place(n)}（{notice_period(n)}）")


def show_answer_extras(ans, key: str) -> None:
    """カード・お知らせ・口コミ・キッチンカー（詳細を見る）と、聞き返しのときの候補ボタン。"""
    if cards(ans):
        show_details(ans)
    if ans.notices:
        show_notice_details(ans)
    if ans.reviews:
        show_reviews(ans)
    if ans.kitchen_cars:
        show_kitchen_cars(ans)
    for i, suggestion in enumerate(ans.suggestions):
        if st.button(suggestion, key=f"suggest-{key}-{i}", width="stretch"):
            st.session_state["pending_question"] = suggestion
            st.rerun()


def slot_picker() -> tuple[date, int] | None:
    """日付と時限（任意）。時限が選ばれたらその日時を検索条件に使う（日付が空なら今日）。"""
    with st.expander("日付と時限を指定する（任意）"):
        c1, c2 = st.columns(2)
        day = c1.date_input("日付", value=None, format="YYYY/MM/DD", key="slot_day")
        period = c2.selectbox(
            "時限", [None, *PERIOD_TIMES], key="slot_period",
            format_func=lambda p: "指定しない" if p is None
            else f"{p}限（{PERIOD_TIMES[p][0]}〜{PERIOD_TIMES[p][1]}）",
        )  # fmt: skip
        if period is None:
            if day is not None:
                st.caption("時限も選ぶと、その時間に予約がある教室を候補から外します。")
            return None
        day = day or date.today()
        st.caption(f"{day_label(day)} {period}限 で探します（質問文の日時より優先）。")
        return day, period


def past_turns(history: list) -> list:
    """画面の履歴から（質問, 回答）の組を作る（回答の条件を次の質問に引き継ぐため）。"""
    return [
        (q, a)
        for (r1, q, _), (r2, _, a) in zip(history, history[1:], strict=False)
        if r1 == "user" and r2 == "assistant" and a is not None
    ]


def page_chat() -> None:
    st.title("📍 キャンパス・スポットナビ")
    st.caption(
        "野田キャンパスの作業スポットを、公式調査データをもとに案内します。"
        "「飲み物も飲みたい」のように続けて送ると、前の条件に追加して探し直します。"
    )
    show_today_summary()

    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES, strict=True):
        if col.button(example, width="stretch"):
            st.session_state["pending_question"] = example
    slot_choice = slot_picker()

    history = st.session_state.setdefault("history", [])
    if history and st.button("🔄 新しい会話を始める（前の条件をリセット）"):
        history.clear()
        st.rerun()
    for i, (role, content, ans) in enumerate(history):
        with st.chat_message(role):
            st.markdown(content)
            if ans is not None:
                show_answer_extras(ans, str(i))

    question = st.chat_input("例: 静かで電源がある場所は？")
    question = question or st.session_state.pop("pending_question", None)
    if question:
        turns = past_turns(history)
        history.append(("user", question, None))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            # 処理の段階を表示し、回答が出たら消す
            placeholder = st.empty()
            with placeholder.status("質問を読み取っています…") as status:
                ans = answer_question(
                    question, conn(), slot_choice=slot_choice, history=turns,
                    progress=lambda msg: status.update(label=msg),
                )  # fmt: skip
            placeholder.empty()
            st.markdown(ans.text)
            show_answer_extras(ans, str(len(history)))
        history.append(("assistant", ans.text, ans))


def page_review() -> None:
    st.title("口コミ・誤り報告")
    c = conn()
    spots = repository.list_spots(c)

    st.subheader("口コミを投稿")
    st.caption("口コミは「利用者の声」として公式データとは分けて保存・表示されます。")
    with st.form("review", clear_on_submit=True):
        spot = st.selectbox("スポット", spots, format_func=spot_label)
        rating = st.selectbox("静かさ（任意）", [None, *REVIEW_RATINGS], format_func=lambda v: v or "—")
        body = st.text_area("本文", max_chars=review_ingest.MAX_BODY_CHARS)
        if st.form_submit_button("投稿"):
            try:
                review_ingest.post_review(c, spot["spot_id"], body, rating)
                st.success("投稿しました")
            except review_ingest.ReviewError as e:
                st.error(str(e))

    st.subheader("情報の誤りを報告")
    st.caption("コンセントの数・静かさなど、案内の内容が実際と違っていたら教えてください。管理者が確認します。")
    with st.form("error_report", clear_on_submit=True):
        spot = st.selectbox("スポット", spots, format_func=spot_label, key="report_spot")
        body = st.text_area(
            "どこが違っていたか", max_chars=report_ingest.MAX_BODY_CHARS,
            placeholder="例: コンセントは4口ではなく2口だった",
        )  # fmt: skip
        if st.form_submit_button("報告"):
            try:
                report_ingest.post_error_report(c, spot["spot_id"], body)
                st.success("報告しました。ありがとうございます")
            except report_ingest.ErrorReportError as e:
                st.error(str(e))

    st.subheader("最近の口コミ")
    by_id = {s["spot_id"]: s for s in spots}
    reviews = repository.list_reviews(c)[:20]
    if not reviews:
        st.caption("まだ口コミはありません")
    for r in reviews:
        rating = f"［{r['rating']}］" if r["rating"] else ""
        st.markdown(f"**{spot_label(by_id[r['spot_id']])}**  \n{rating}{r['body']}")
        st.caption(r["posted_at"][:10])


st.navigation(
    {
        "": [
            st.Page(page_chat, title="チャット", icon="💬", default=True),
            st.Page(page_review, title="口コミ・誤り報告", icon="✍️"),
        ],
        "管理": [
            st.Page("pages/notices.py", title="お知らせ管理", icon="📨"),
            st.Page("pages/kitchen_cars.py", title="キッチンカー取込", icon="🚚"),
            st.Page("pages/error_reports.py", title="誤り報告の確認", icon="🛠️"),
            st.Page("pages/stats.py", title="調査データ・集計", icon="📊"),
        ],
    }
).run()
