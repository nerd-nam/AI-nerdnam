"""모델별 분당 요청 한도기 (ADR-0009 결정 3): 한도만큼 바로, 그 뒤 일정 속도, HTTP는 마감 안에 자리가 없으면 503"""
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import config, limiter, llm, state
from app.main import app


def test_burst_then_steady_rate():
    b = limiter._Bucket(60)
    assert all(b.reserve(None) == 0 for _ in range(60))  # 분당 한도만큼은 바로
    assert b.reserve(None) == pytest.approx(1, abs=0.05)  # 그 뒤 1초에 1건, 먼저 온 순서
    assert b.reserve(None) == pytest.approx(2, abs=0.05)
    assert b.reserve(0.5) is None and b.reserve(None) == pytest.approx(3, abs=0.05)  # 기다릴 수 없으면 예약하지 않음


def test_http_gets_503_when_no_slot_before_deadline(monkeypatch):
    monkeypatch.setitem(state.models, "llm", "ready")
    monkeypatch.setitem(limiter._buckets, llm.LLM_MODEL, limiter._Bucket(1))
    limiter._buckets[llm.LLM_MODEL].reserve(None)  # 이번 분 자리를 다 씀 → 다음 자리 60초 뒤
    monkeypatch.setitem(config.READ_LIMITS, "/match", config.DEADLINE_MARGIN + 1)
    monkeypatch.setattr(llm.client, "converse", lambda **req: pytest.fail("한도기를 지나쳐 Bedrock을 부름"))
    t = time.monotonic()
    r = TestClient(app).post("/match", json={"requirements": [{"id": 1, "type": "required", "text": "Kafka",
                                                               "candidates": [{"chunk_id": 1, "chunk_type": "skill", "text": "Kafka"}]}]})
    assert r.status_code == 503 and r.json()["usage"]["llm_calls"] == 0 and time.monotonic() - t < 1


def test_without_deadline_waits_instead_of_failing(monkeypatch):
    """큐 경로(마감 없음)는 503 없이 자리가 날 때까지 기다린다"""
    monkeypatch.setitem(limiter._buckets, llm.LLM_MODEL, limiter._Bucket(600))  # 0.1초에 1건
    for _ in range(600):
        limiter._buckets[llm.LLM_MODEL].reserve(None)
    t = time.monotonic()
    try:
        limiter.take_sync(llm.LLM_MODEL)
    except HTTPException:
        pytest.fail("마감이 없는데 503")
    assert 0.05 < time.monotonic() - t < 0.5
