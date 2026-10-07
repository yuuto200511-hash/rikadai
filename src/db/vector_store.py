"""Chroma（ローカル永続化）。スポットの場所と備考をつないだ文、および reviews.body を登録する。"""

from __future__ import annotations

import os
import sqlite3
from functools import lru_cache
from pathlib import Path

from src.db import repository
from src.llm import embedding

CHROMA_PATH = Path(os.environ.get("SPOT_CHROMA_PATH", repository.ROOT / "data" / "chroma"))
COLLECTION = "spot_texts"


@lru_cache(maxsize=1)
def _collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def _upsert(ids: list[str], docs: list[str], metas: list[dict]) -> None:
    if ids:
        _collection().upsert(
            ids=ids, documents=docs, metadatas=metas, embeddings=embedding.embed_passages(docs)
        )


def spot_text(s: dict) -> str:
    """建物・階・エリア名・備考をつないだ文。空欄の項目は省く。"""
    floor = f"{s['floor']}階" if s["floor"] is not None else None
    return " ".join(p for p in (s["building"], floor, s["area_name"], s["note"]) if p)


def rebuild(conn: sqlite3.Connection) -> int:
    """DB の内容からインデックスを作り直す。"""
    col = _collection()
    existing = col.get(include=[])["ids"]
    if existing:
        col.delete(ids=existing)
    ids, docs, metas = [], [], []
    for s in repository.list_spots(conn):
        ids.append(f"spot:{s['spot_id']}")
        docs.append(spot_text(s))
        metas.append({"spot_id": s["spot_id"], "kind": "spot"})
    for r in repository.list_reviews(conn):
        ids.append(f"review:{r['review_id']}")
        docs.append(r["body"])
        metas.append({"spot_id": r["spot_id"], "kind": "review"})
    _upsert(ids, docs, metas)
    return len(ids)


def add_review(review_id: int, spot_id: str, body: str) -> None:
    _upsert([f"review:{review_id}"], [body], [{"spot_id": spot_id, "kind": "review"}])


def similarity_by_spot(query: str, spot_ids: list[str]) -> dict[str, float]:
    """spot ごとに、質問文と spot/review の最大コサイン類似度を返す。"""
    col = _collection()
    if not spot_ids or col.count() == 0:
        return {}
    res = col.query(
        query_embeddings=[embedding.embed_query(query)],
        n_results=col.count(),
        where={"spot_id": {"$in": spot_ids}},
        include=["metadatas", "distances"],
    )
    scores: dict[str, float] = {}
    for meta, dist in zip(res["metadatas"][0], res["distances"][0], strict=True):
        sid = meta["spot_id"]
        scores[sid] = max(scores.get(sid, -1.0), 1.0 - dist)
    return scores


if __name__ == "__main__":
    print(f"indexed {rebuild(repository.connect())} documents")
