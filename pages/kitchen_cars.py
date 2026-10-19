"""管理: キッチンカーの出店情報（毎月の CSV）の取込。確認してから保存する。"""

import pandas as pd
import streamlit as st

from src.db import repository
from src.ingest import kitchen_cars as kc_ingest
from src.ui import conn

st.title("キッチンカー取込")
st.caption(
    "列: " + ", ".join(kc_ingest.COLUMNS) + "（1行 = 1日・1店・1メニュー）。"
    "同じ月を取り込み直すと、その月の分を置き換えます。"
)
c = conn()
saved = repository.kitchen_car_months(c)
st.write(
    "登録済み: " + (" / ".join(f"{m}（{n}行）" for m, n in saved.items()) if saved else "なし")
)

upload = st.file_uploader("CSV ファイル", type=["csv"])
if upload is None:
    st.stop()

result = kc_ingest.parse(upload.getvalue())
if result.errors:
    st.error(f"問題のある行が {len(result.errors)} 件あるため、取り込みを止めました。CSV を直してから選び直してください。")
    st.dataframe(
        pd.DataFrame(
            [
                {"行": row or "-", "理由": reason, "行の内容": ",".join(result.raw.get(row, []))}
                for row, reason in result.errors
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.stop()

st.subheader("取り込む内容の確認")
for month, n in result.months.items():
    replace = f"（登録済みの {saved[month]} 行を置き換えます）" if month in saved else "（新規）"
    st.write(f"{month}: {n} 行{replace}")
st.dataframe(pd.DataFrame(result.rows), hide_index=True, width="stretch")
if st.button("この内容で保存", type="primary"):
    done = kc_ingest.save(c, result)
    st.success("保存しました: " + " / ".join(f"{m}（{n}行）" for m, n in done.items()))
