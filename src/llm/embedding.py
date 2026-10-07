"""ローカル多言語モデルによる埋め込み（multilingual-e5-small）。

e5 系は検索文に "query: "、検索対象に "passage: " を付ける前提で学習されている。
"""

from __future__ import annotations

from functools import lru_cache

MODEL_NAME = "intfloat/multilingual-e5-small"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed_passages(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vecs = _model().encode([f"passage: {t}" for t in texts], normalize_embeddings=True)
    return vecs.tolist()


def embed_query(text: str) -> list[float]:
    return _model().encode(f"query: {text}", normalize_embeddings=True).tolist()
