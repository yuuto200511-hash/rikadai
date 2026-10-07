"""管理: 利用者からの誤り報告の確認。公式データの修正は survey.xlsx 側で行い、再取込する。"""

import streamlit as st

from src.db import repository
from src.ui import admin_spot_label, conn

st.title("誤り報告の確認")
st.caption("報告の内容は検索・回答には使いません。修正が必要なら survey.xlsx を直して再取込してください。")
c = conn()
spots = {s["spot_id"]: s for s in repository.list_spots(c)}
reports = repository.list_error_reports(c)
show_resolved = st.checkbox("対応済みも表示", value=False)
open_count = sum(not r["resolved"] for r in reports)
st.write(f"未対応 {open_count} 件 / 全 {len(reports)} 件")
for r in reports:
    if r["resolved"] and not show_resolved:
        continue
    s = spots.get(r["spot_id"])
    with st.container(border=True):
        st.markdown(f"**{admin_spot_label(s) if s else r['spot_id']}**（{r['posted_at']}）")
        st.write(r["body"])
        done = st.checkbox("対応済み", bool(r["resolved"]), key=f"resolved-{r['report_id']}")
        if done != bool(r["resolved"]):
            repository.set_error_report_resolved(c, r["report_id"], done)
            st.rerun()
