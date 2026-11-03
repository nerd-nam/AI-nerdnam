import pytest

from app import limiter, llm


@pytest.fixture(autouse=True)
def _no_throttle_backoff(monkeypatch):
    """throttle 재시도 사이에 쉬지 않는다 (테스트 시간만 늘어남)"""
    monkeypatch.setattr(llm, "THROTTLE_BACKOFF", 0)


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    """분당 한도기는 테스트마다 새로, 기다리지 않을 만큼 넉넉히 (한도기 자체는 test_limiter.py)"""
    monkeypatch.setattr(limiter, "_buckets", {m: limiter._Bucket(10**6) for m in limiter._buckets})
