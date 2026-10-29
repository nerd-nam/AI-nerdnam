"""분산 추적 (#42): traceparent 이어 받기, Bedrock span 속성, 스레드 풀 넘어 context 유지, 프로브 제외, JSON 로그"""
import io
import json
import logging

import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import embedding, llm, state, telemetry
from app.main import app

spans = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(spans))
trace.set_tracer_provider(_provider)

c = TestClient(app)
TID = "4bf92f3577b34da6a3ce929d0e0e4736"
HDR = {"traceparent": f"00-{TID}-00f067aa0ba902b7-01"}
SK = {"chunk_id": 1, "chunk_type": "skill", "text": "Kafka"}
MATCH = {"requirements": [{"id": 1, "type": "required", "text": "비밀요건문장", "candidates": [SK]}]}


@pytest.fixture(autouse=True)
def clear():
    spans.clear()


def by_name(name):
    return [s for s in spans.get_finished_spans() if s.name == name]


def converse_ok(**req):
    out = {"results": [{"id": 1, "verdict": "unmet", "evidence_chunk_ids": [], "reason": "없음"}]}
    return {"output": {"message": {"content": [{"toolUse": {"name": "out", "input": out}}]}},
            "usage": {"inputTokens": 100, "outputTokens": 7}, "ResponseMetadata": {"RetryAttempts": 0}}


def test_match_joins_be_trace(monkeypatch):
    monkeypatch.setitem(state.models, "llm", "ready")
    monkeypatch.setattr(llm.client, "converse", converse_ok)
    assert c.post("/match", json=MATCH, headers=HDR).status_code == 200
    call, = by_name("bedrock.converse")
    assert format(call.context.trace_id, "032x") == TID  # BE trace에 붙음
    assert call.kind == trace.SpanKind.CLIENT and call.parent is not None
    assert dict(call.attributes).items() >= {"bedrock.result": "ok", "bedrock.attempts": 1, "gen_ai.usage.input_tokens": 100,
                                         "gen_ai.usage.output_tokens": 7, "gen_ai.request.model": llm.LLM_MODEL}.items()
    assert by_name("bedrock.queue")
    assert not any("비밀요건문장" in str(v) for s in spans.get_finished_spans() for v in s.attributes.values())  # 본문 없음


def test_throttle_marked(monkeypatch):
    monkeypatch.setitem(state.models, "llm", "ready")

    def throttled(**req):
        raise ClientError({"Error": {"Code": "ThrottlingException", "Message": "m"}, "ResponseMetadata": {"RetryAttempts": 0}}, "Converse")
    monkeypatch.setattr(llm.client, "converse", throttled)
    assert c.post("/match", json=MATCH, headers=HDR).status_code == 503
    call, = by_name("bedrock.converse")
    assert call.attributes["bedrock.result"] == "throttle" and call.status.status_code == trace.StatusCode.ERROR


def test_embed_spans_cross_thread_pool(monkeypatch):
    monkeypatch.setitem(state.models, "embedding", "ready")
    monkeypatch.setattr(embedding.client, "invoke_model", lambda **req: {
        "body": io.BytesIO(json.dumps({"embedding": [0.0] * 1024, "inputTextTokenCount": 3}).encode()),
        "ResponseMetadata": {"RetryAttempts": 1}})
    assert c.post("/embed", json={"purpose": "query", "items": [{"text": "a"}] * 3}, headers=HDR).status_code == 200
    calls = by_name("bedrock.invoke_model")
    assert len(calls) == 3 and len(by_name("bedrock.queue")) == 3
    assert {format(s.context.trace_id, "032x") for s in calls} == {TID}  # 스레드 풀을 넘어도 고아가 아님
    assert calls[0].attributes["bedrock.attempts"] == 2 and calls[0].attributes["gen_ai.usage.input_tokens"] == 3


@pytest.mark.parametrize("path", ["/livez", "/readyz", "/health"])
def test_probes_not_traced(path):
    c.get(path, headers=HDR)
    assert spans.get_finished_spans() == ()


def test_json_log_has_trace_id():
    rec = logging.LogRecord("uvicorn.error", logging.INFO, "", 0, "bedrock llm m 0.8s wait=0.0s", None, None)
    f = telemetry.JsonFormatter()
    assert "trace_id" not in json.loads(f.format(rec))
    with telemetry.tracer.start_as_current_span("x") as s:
        d = json.loads(f.format(rec))
    assert d["trace_id"] == format(s.get_span_context().trace_id, "032x") and len(d["span_id"]) == 16
    assert d["msg"] == "bedrock llm m 0.8s wait=0.0s"
