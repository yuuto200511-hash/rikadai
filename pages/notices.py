"""管理: お知らせ（メール）の取込と確認。"""

import streamlit as st

from src.db import repository
from src.ingest import notice as notice_ingest
from src.rag.retrieve import notice_place
from src.ui import conn

st.title("お知らせ管理")
c = conn()
if st.button("data/notices の新着メールを取り込む"):
    with st.spinner("Claude で判定・抽出しています…"):
        done = notice_ingest.ingest(c)
    st.success(f"{len(done)} 件を取り込みました（未確認として保存）")

notices = repository.list_notices(c)
show_all = st.checkbox("無関係と判定したメールも表示", value=False)
for n in notices:
    if not n["is_relevant"] and not show_all:
        continue
    head = "✅" if n["confirmed"] else ("⬜" if n["is_relevant"] else "—")
    title = f"{notice_ingest.demo_label(n)}{n['title'] or n['source_file']}"
    with st.expander(f"{head} {title}"):
        if n["is_relevant"]:
            st.write(
                f"対象: {notice_place(n)} / "
                f"影響: {n['impact_type']} / 期間: {n['valid_from'] or ''}〜{n['valid_to'] or ''}"
            )
            checked = st.checkbox("確認済み（検索に反映）", bool(n["confirmed"]),
                                  key=f"confirm-{n['notice_id']}")  # fmt: skip
            if checked != bool(n["confirmed"]):
                repository.set_notice_confirmed(c, n["notice_id"], checked)
                st.rerun()
        else:
            st.write("施設と無関係と判定")
        st.text(n["raw_body"])
