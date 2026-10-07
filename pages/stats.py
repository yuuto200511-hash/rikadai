"""管理: 調査データの再取込と集計（電源確保率はこの画面だけで表示する）。"""

import pandas as pd
import streamlit as st

from src.db import repository, vector_store
from src.ingest import survey
from src.rag.answer import power_level
from src.ui import conn

st.title("調査データ・集計")
c = conn()

if st.button("data/survey.xlsx を再取込してインデックスを再構築"):
    try:
        res = survey.ingest(repository.ROOT / "data" / "survey.xlsx", c)
    except ValueError as e:
        st.error(str(e))
    else:
        n_docs = vector_store.rebuild(c)
        st.success(f"取込 {len(res.accepted)} 件 / 除外 {len(res.rejected)} 件 / 索引 {n_docs} 件")
        if (rv := res.reservations) is not None:
            st.info(f"教室の利用状況: 取込 {len(rv.accepted)} 件 / 除外 {len(rv.rejected)} 件")
            for row, reason in rv.rejected:
                st.error(f"reservations 行{row}: {reason}")
        for w in res.warnings:
            st.warning(w)
        for row, sid, reason in res.rejected:
            st.error(f"行{row} {sid or '(IDなし)'}: {reason}")
        if res.blanks:
            st.write("空欄（NULL として取込）")
            st.dataframe(
                pd.DataFrame(
                    [{"行": r, "spot_id": sid, "空欄の列": ", ".join(cols)}
                     for r, sid, cols in res.blanks]
                ),
                hide_index=True,
            )  # fmt: skip

spots = repository.list_spots(c)
df = pd.DataFrame(spots)
df["building"] = df["building"].fillna("（建物未記入）")
df["電源"] = [power_level(s) for s in spots]

st.subheader("建物ごとの集計")
by_building = df.groupby("building").agg(
    スポット数=("spot_id", "count"), 座席数=("seats", "sum"), コンセント数=("outlets", "sum")
)
by_building["電源確保率"] = (by_building["コンセント数"] / by_building["座席数"]).map(
    lambda v: f"{v:.0%}" if pd.notna(v) and v != float("inf") else "—"
)
st.dataframe(by_building, width="stretch")

st.subheader("電源の段階")
st.dataframe(
    df.groupby(["building", "電源"]).size().unstack(fill_value=0), width="stretch"
)

st.subheader("件数")
reports = repository.list_error_reports(c)
st.write(
    f"口コミ {len(repository.list_reviews(c))} 件 / "
    f"誤り報告 {len(reports)} 件（未対応 {sum(not r['resolved'] for r in reports)} 件） / "
    f"教室の利用状況データ: {', '.join(repository.reservation_dates(c)) or 'なし'}"
)

st.subheader("スポット一覧（電源確保率つき）")
df["電源確保率"] = df["outlet_ratio"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "—")
st.dataframe(df, hide_index=True, width="stretch")
