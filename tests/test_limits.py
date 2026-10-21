import pytest

from src import limits


def test_allows_300_chars_and_records_call():
    calls = []
    assert limits.check("あ" * 300, calls, 100.0) is None
    assert calls == [100.0]


def test_rejects_301_chars_without_counting():
    calls = []
    assert limits.check("あ" * 301, calls, 100.0) == "質問は300文字以内でお願いします"
    assert calls == []


def test_rejects_11th_call_within_a_minute():
    calls = []
    for i in range(10):
        assert limits.check("q", calls, 100.0 + i) is None
    msg = limits.check("q", calls, 110.0)
    # 最初の呼び出し（100秒）から60秒たつまで、あと50秒
    assert msg == limits.too_many(50) and "少し待って" in msg
    assert len(calls) == 10


@pytest.mark.parametrize("now, allowed", [(159.9, False), (160.0, True)])
def test_window_slides(now, allowed):
    calls = [100.0 + i for i in range(10)]
    assert (limits.check("q", calls, now) is None) == allowed
