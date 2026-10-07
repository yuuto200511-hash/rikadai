"""口コミ登録。公式データ（spots）とは別テーブルに保存し、ベクトル DB にも登録する。"""

from __future__ import annotations

import sqlite3

from src.choices import REVIEW_RATINGS
from src.db import repository, vector_store

MAX_BODY_CHARS = 500


class ReviewError(ValueError):
    pass


def post_review(
    conn: sqlite3.Connection,
    spot_id: str,
    body: str,
    rating: str | None = None,
    *,
    index: bool = True,
) -> int:
    body = body.strip()
    if not body:
        raise ReviewError("本文を入力してください")
    if len(body) > MAX_BODY_CHARS:
        raise ReviewError(f"本文は{MAX_BODY_CHARS}文字以内にしてください")
    if rating is not None and rating not in REVIEW_RATINGS:
        raise ReviewError(f"評価は {' / '.join(REVIEW_RATINGS)} から選んでください")
    if repository.get_spot(conn, spot_id) is None:
        raise ReviewError(f"スポット {spot_id} は存在しません")
    review_id = repository.add_review(conn, spot_id, body, rating)
    if index:
        vector_store.add_review(review_id, spot_id, body)
    return review_id
