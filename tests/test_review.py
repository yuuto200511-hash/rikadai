import pytest

from src.db import repository
from src.ingest import review
from tests.test_retrieve import BASE


@pytest.fixture
def conn(tmp_path):
    c = repository.connect(tmp_path / "t.db")
    repository.replace_spots(c, [{**BASE, "spot_id": "S001"}])
    return c


def test_post_review_stored_separately(conn):
    rid = review.post_review(conn, "S001", " 静かでよかった ", "静かだった", index=False)
    assert repository.list_reviews(conn, "S001")[0]["review_id"] == rid
    # 公式データは変わらない
    assert repository.get_spot(conn, "S001")["quietness"] == "私語厳禁"


@pytest.mark.parametrize(
    "spot_id, body, rating",
    [("S001", "  ", None), ("S999", "よい", None), ("S001", "よい", "最高"), ("S001", "あ" * 501, None)],
)
def test_post_review_rejects(conn, spot_id, body, rating):
    with pytest.raises(review.ReviewError):
        review.post_review(conn, spot_id, body, rating, index=False)
