"""Prometheus 지표 (#44): Bedrock 호출·결과·토큰·시도, 대기열 대기·거절, HTTP 요청. 전역 레지스트리라 전후 차이로 본다"""
import asyncio
import io
import json

from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app import embedding, llm, state
from app.main import app

c = TestClient(app)
LLM, EMB = llm.LLM_MODEL, embedding.EMBED_MODEL_ID
MATCH = {"requirements": [{"id": 1, "type": "required", "text": "Kafka 운영",
                           "candidates": [{"chunk_id": 1, "chunk_type": "skill", "text": "Kafka"}]}]}


def val(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


def converse_ok(**req):
    out = {"results": [{"id": 1, "verdict": "unmet", "evidence_chunk_ids": [], "reason": "없음"}]}
    return {"output": {"message": {"content": [{"toolUse": {"name": "out", "input": out}}]}},
            "usage": {"inputTokens": 100, "outputTokens": 7}, "ResponseMetadata": {"RetryAttempts": 1}}


def test_converse_ok_counts_tokens_and_request(monkeypatch):
    monkeypatch.setitem(state.models, "llm", "ready")
    monkeypatch.setattr(llm.client, "converse", converse_ok)
    op = {"operation": "converse", "model": LLM}
    before = (val("nerdnam_ai_bedrock_calls_total", **op, result="ok"), val("nerdnam_ai_bedrock_attempts_total", **op),
              val("nerdnam_ai_bedrock_tokens_total", model=LLM, direction="input"),
              val("nerdnam_ai_bedrock_tokens_total", model=LLM, direction="output"),
              val("nerdnam_ai_requests_total", route="/match", status="200"),
              val("nerdnam_ai_bedrock_queue_wait_seconds_count", kind="llm"))
    r = c.post("/match", json=MATCH)
    assert r.status_code == 200
    after = (val("nerdnam_ai_bedrock_calls_total", **op, result="ok"), val("nerdnam_ai_bedrock_attempts_total", **op),
             val("nerdnam_ai_bedrock_tokens_total", model=LLM, direction="input"),
             val("nerdnam_ai_bedrock_tokens_total", model=LLM, direction="output"),
             val("nerdnam_ai_requests_total", route="/match", status="200"),
             val("nerdnam_ai_bedrock_queue_wait_seconds_count", kind="llm"))
    assert [a - b for a, b in zip(after, before)] == [1, 2, 100, 7, 1, 1]
    assert r.json()["usage"]["input_tokens"] == 100  # 응답 usage와 같은 값을 센다


def test_throttle_counted(monkeypatch):
    monkeypatch.setitem(state.models, "llm", "ready")

    def throttled(**req):
        raise ClientError({"Error": {"Code": "ThrottlingException", "Message": "m"},
                           "ResponseMetadata": {"RetryAttempts": 0}}, "Converse")
    monkeypatch.setattr(llm.client, "converse", throttled)
    before = val("nerdnam_ai_bedrock_calls_total", operation="converse", model=LLM, result="throttle")
    attempts = val("nerdnam_ai_bedrock_attempts_total", operation="converse", model=LLM)
    r = c.post("/match", json=MATCH)
    assert r.status_code == 503
    assert val("nerdnam_ai_bedrock_calls_total", operation="converse", model=LLM, result="throttle") - before == 1
    assert val("nerdnam_ai_bedrock_attempts_total", operation="converse", model=LLM) - attempts == 1 + llm.LLM_THROTTLE_RETRIES
    assert r.json()["usage"]["llm_calls"] == 1 + llm.LLM_THROTTLE_RETRIES  # 다시 보낸 것도 호출
    assert val("nerdnam_ai_requests_total", route="/match", status="503") >= 1


def test_throttle_retried_then_ok(monkeypatch):
    """한도 초과 1번 뒤 성공 → 200, 호출 결과는 ok, 시도 수 2"""
    monkeypatch.setitem(state.models, "llm", "ready")
    calls = []

    def once_throttled(**req):
        calls.append(1)
        if len(calls) == 1:
            raise ClientError({"Error": {"Code": "ThrottlingException", "Message": "m"},
                               "ResponseMetadata": {"RetryAttempts": 0}}, "Converse")
        return {**converse_ok(), "ResponseMetadata": {"RetryAttempts": 0}}
    monkeypatch.setattr(llm.client, "converse", once_throttled)
    op = {"operation": "converse", "model": LLM}
    before = (val("nerdnam_ai_bedrock_calls_total", **op, result="ok"),
              val("nerdnam_ai_bedrock_calls_total", **op, result="throttle"), val("nerdnam_ai_bedrock_attempts_total", **op))
    r = c.post("/match", json=MATCH)
    assert r.status_code == 200 and r.json()["usage"]["llm_calls"] == 2
    after = (val("nerdnam_ai_bedrock_calls_total", **op, result="ok"),
             val("nerdnam_ai_bedrock_calls_total", **op, result="throttle"), val("nerdnam_ai_bedrock_attempts_total", **op))
    assert [a - b for a, b in zip(after, before)] == [1, 0, 2]


def test_timeout_not_retried(monkeypatch):
    """읽기 시간 초과는 다시 보내지 않는다 (C-04 504: 비용·부하만 두 배)"""
    from botocore.exceptions import ReadTimeoutError
    monkeypatch.setitem(state.models, "llm", "ready")
    calls = []

    def slow(**req):
        calls.append(1)
        raise ReadTimeoutError(endpoint_url="x")
    monkeypatch.setattr(llm.client, "converse", slow)
    assert c.post("/match", json=MATCH).status_code == 504
    assert len(calls) == 1


def test_busy_is_rejected_not_a_call(monkeypatch):
    monkeypatch.setitem(state.models, "llm", "ready")
    monkeypatch.setattr(llm, "_slots", asyncio.Semaphore(0))  # 자리가 안 남
    monkeypatch.setattr(llm, "LLM_QUEUE_WAIT", 0.05)
    calls = sum(val("nerdnam_ai_bedrock_calls_total", operation="converse", model=LLM, result=x)
                for x in ("ok", "throttle", "timeout", "error", "busy"))
    before = val("nerdnam_ai_bedrock_queue_rejected_total", kind="llm")
    assert c.post("/match", json=MATCH).status_code == 503
    assert val("nerdnam_ai_bedrock_queue_rejected_total", kind="llm") - before == 1
    assert sum(val("nerdnam_ai_bedrock_calls_total", operation="converse", model=LLM, result=x)
               for x in ("ok", "throttle", "timeout", "error", "busy")) == calls  # Bedrock에 안 갔으니 호출 아님


def test_embed_counts_each_call(monkeypatch):
    monkeypatch.setitem(state.models, "embedding", "ready")
    monkeypatch.setattr(embedding.client, "invoke_model", lambda **req: {
        "body": io.BytesIO(json.dumps({"embedding": [0.0] * 1024, "inputTextTokenCount": 3}).encode()),
        "ResponseMetadata": {"RetryAttempts": 0}})
    op = {"operation": "invoke_model", "model": EMB}
    before = (val("nerdnam_ai_bedrock_calls_total", **op, result="ok"),
              val("nerdnam_ai_bedrock_tokens_total", model=EMB, direction="input"),
              val("nerdnam_ai_bedrock_queue_wait_seconds_count", kind="embed"),
              val("nerdnam_ai_bedrock_call_duration_seconds_count", **op))
    assert c.post("/embed", json={"purpose": "query", "items": [{"text": "a"}] * 3}).status_code == 200
    after = (val("nerdnam_ai_bedrock_calls_total", **op, result="ok"),
             val("nerdnam_ai_bedrock_tokens_total", model=EMB, direction="input"),
             val("nerdnam_ai_bedrock_queue_wait_seconds_count", kind="embed"),
             val("nerdnam_ai_bedrock_call_duration_seconds_count", **op))
    assert [a - b for a, b in zip(after, before)] == [3, 9, 3, 3]


def test_embed_busy_rejected(monkeypatch):
    monkeypatch.setitem(state.models, "embedding", "ready")
    monkeypatch.setattr(embedding, "_pending", embedding.EMBED_QUEUE)
    before = val("nerdnam_ai_bedrock_queue_rejected_total", kind="embed")
    assert c.post("/embed", json={"purpose": "query", "items": [{"text": "a"}]}).status_code == 503
    assert val("nerdnam_ai_bedrock_queue_rejected_total", kind="embed") - before == 1


def test_metrics_endpoint_and_probes_uncounted():
    for p in ("/livez", "/readyz", "/health"):
        c.get(p)
    r = c.get("/metrics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    assert "nerdnam_ai_bedrock_calls_total" in r.text and "nerdnam_ai_requests_total" in r.text
    for p in ("/livez", "/readyz", "/health", "/metrics"):
        assert f'route="{p}"' not in r.text
    assert "/metrics" not in c.get("/openapi.json").json()["paths"]
