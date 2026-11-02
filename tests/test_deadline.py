"""요청 마감 (기획서 v1.65, AI #47): 읽기 제한보다 DEADLINE_MARGIN초 먼저 503(대기·한도)·504(생성)를 usage와 함께"""
import asyncio
import time

from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

from app import config, embedding, llm, state
from app.main import app

MATCH = {"requirements": [{"id": 1, "type": "required", "text": "Kafka 운영",
                           "candidates": [{"chunk_id": 1, "chunk_type": "skill", "text": "Kafka"}]}]}


def short(monkeypatch, path, seconds):
    """그 경로의 마감을 seconds초 뒤로"""
    monkeypatch.setitem(config.READ_LIMITS, path, config.DEADLINE_MARGIN + seconds)


def test_queue_wait_stops_at_deadline(monkeypatch):
    monkeypatch.setitem(state.models, "llm", "ready")
    monkeypatch.setattr(llm, "_slots", asyncio.Semaphore(0))  # 자리가 안 남
    short(monkeypatch, "/match", 0.3)  # LLM_QUEUE_WAIT(30초)보다 마감이 먼저
    t = time.monotonic()
    r = TestClient(app).post("/match", json=MATCH)
    assert r.status_code == 503 and "usage" in r.json()
    assert time.monotonic() - t < 2


def test_generation_past_deadline_is_504_and_slot_comes_back(monkeypatch):
    monkeypatch.setitem(state.models, "llm", "ready")
    monkeypatch.setattr(llm, "_slots", asyncio.Semaphore(1))
    monkeypatch.setattr(llm.client, "converse", lambda **req: time.sleep(0.8))  # 마감 뒤에 끝나는 호출
    short(monkeypatch, "/match", 0.3)
    with TestClient(app) as c:  # 같은 이벤트 루프를 유지해야 늦게 끝난 호출이 자리를 돌려줌
        t = time.monotonic()
        r = c.post("/match", json=MATCH)
        assert r.status_code == 504 and r.json()["usage"]["llm_calls"] == 1
        assert time.monotonic() - t < 0.8  # 호출이 끝나기 전에 답함
        time.sleep(1)
        assert llm._slots._value == 1  # 늦게 끝난 호출이 자리를 돌려줌


def test_throttle_retry_skipped_when_backoff_passes_deadline(monkeypatch):
    monkeypatch.setitem(state.models, "llm", "ready")
    monkeypatch.setattr(llm, "THROTTLE_BACKOFF", 5.0)  # 쉬면 마감(1초 뒤)을 넘김
    calls = []

    def throttled(**req):
        calls.append(1)
        raise ClientError({"Error": {"Code": "ThrottlingException", "Message": "m"}, "ResponseMetadata": {}}, "Converse")
    monkeypatch.setattr(llm.client, "converse", throttled)
    short(monkeypatch, "/match", 1)
    r = TestClient(app).post("/match", json=MATCH)
    assert r.status_code == 503 and len(calls) == 1


def test_embed_budget_follows_deadline(monkeypatch):
    monkeypatch.setitem(state.models, "embedding", "ready")
    monkeypatch.setattr(embedding, "_one", lambda text: time.sleep(1) or [0.0] * embedding.EMBED_DIM)
    short(monkeypatch, "/embed", 0.2)  # EMBED_BUDGET(110초)보다 마감이 먼저
    t = time.monotonic()
    r = TestClient(app).post("/embed", json={"purpose": "query", "items": [{"text": "a"}]})
    assert r.status_code == 503 and time.monotonic() - t < 0.9
    time.sleep(1)  # 시작한 호출이 끝나 밀린 수가 돌아오게


def test_contract_description_uses_read_limits():
    d = TestClient(app).get("/openapi.json").json()["info"]["description"]
    assert "`/embed` 120초" in d and "`/match` 300초" in d and "10초 먼저" in d
