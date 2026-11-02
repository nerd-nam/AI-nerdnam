import pytest

from app import llm


@pytest.fixture(autouse=True)
def _no_throttle_backoff(monkeypatch):
    """throttle 재시도 사이에 쉬지 않는다 (테스트 시간만 늘어남)"""
    monkeypatch.setattr(llm, "THROTTLE_BACKOFF", 0)
