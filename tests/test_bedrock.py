"""Bedrock 전환 (ADR-0007): 출력 스키마 규칙, boto3 오류 → 계약 코드(C-04), /embed 순서·차원"""
import json

import pytest
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
    ReadTimeoutError,
)
from fastapi.testclient import TestClient

from app import embedding, llm, state
from app.main import OUTPUT_MODELS, app

c = TestClient(app)


def walk(x):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from walk(v)


@pytest.mark.parametrize("model", OUTPUT_MODELS, ids=lambda m: m.__name__)
def test_output_schema_fits_bedrock(model):
    # Bedrock 도구 스키마: object마다 additionalProperties false, minItems 0·1만, 길이·범위 제약 금지
    for node in walk(llm.strict_schema(model.model_json_schema())):
        if "properties" in node:
            assert node["additionalProperties"] is False and set(node["required"]) == set(node["properties"])
        bad = {k for k in llm._UNSUPPORTED & node.keys() if not isinstance(node[k], dict)}
        assert not bad, bad
        assert node.get("minItems", 0) in (0, 1) or isinstance(node.get("minItems"), dict)


def test_field_named_like_keyword_kept():
    s = llm.strict_schema({"type": "object", "properties": {"format": {"type": "string"}}})
    assert "format" in s["properties"] and s["required"] == ["format"]


def client_error(code):
    return ClientError({"Error": {"Code": code, "Message": "m"}}, "Converse")


@pytest.mark.parametrize("err,code", [
    (client_error("ThrottlingException"), 503),  # 호출 한도 → GPU_UNAVAILABLE (API가 1회 재시도)
    (client_error("ServiceUnavailableException"), 503),
    (client_error("ModelNotReadyException"), 503),
    (client_error("ModelErrorException"), 503),  # "다시 시도하라"는 일시 오류
    (client_error("ValidationException"), 502),  # 다시 보내도 같다 → AI_ERROR
    (client_error("AccessDeniedException"), 502),
    (client_error("ModelTimeoutException"), 504),
    (ReadTimeoutError(endpoint_url="x"), 504),  # TIMEOUT, 재시도 안 함
    (EndpointConnectionError(endpoint_url="x"), 503),
    (NoCredentialsError(), 502),  # 설정 문제
])
def test_error_mapping(err, code):
    assert llm.http_error(err).status_code == code


def test_embed_keeps_order_and_dim(monkeypatch):
    monkeypatch.setitem(state.models, "embedding", "ready")
    monkeypatch.setattr(embedding, "_one", lambda text: [float(len(text))] + [0.0] * (embedding.EMBED_DIM - 1))
    items = [{"text": "a" * n} for n in (5, 1, 3)]
    r = c.post("/embed", json={"purpose": "query", "items": items}).json()
    assert r["dim"] == 1024 and r["embedding_model"] == "amazon.titan-embed-text-v2:0:1024"
    assert [v[0] for v in r["vectors"]] == [5.0, 1.0, 3.0]


def test_embed_throttled_is_503(monkeypatch):
    monkeypatch.setitem(state.models, "embedding", "ready")

    def throttled(text):
        raise client_error("ThrottlingException")
    monkeypatch.setattr(embedding, "_one", throttled)
    assert c.post("/embed", json={"purpose": "query", "items": [{"text": "a"}]}).status_code == 503


def test_titan_request(monkeypatch):
    # AWS 예제(InvokeModel)와 같은 모양 + 1024차원·정규화를 직접 지정
    import io
    import json
    sent = {}

    def invoke_model(**req):
        sent.update(req)
        return {"body": io.BytesIO(json.dumps({"embedding": [0.1] * 1024, "inputTextTokenCount": 3}).encode())}
    monkeypatch.setattr(embedding.client, "invoke_model", invoke_model)
    assert len(embedding.encode("document", [("본문", "제목")])[0]) == 1024
    assert sent["modelId"] == "amazon.titan-embed-text-v2:0"
    assert json.loads(sent["body"]) == {"inputText": "제목\n본문", "dimensions": 1024, "normalize": True}


def test_usage_summed_per_request(monkeypatch):
    # 요건 11개 → 묶음 2개(10·1)를 동시에 판정 → 두 호출의 토큰이 응답 usage에 합쳐진다. 다음 요청은 0부터
    monkeypatch.setitem(state.models, "llm", "ready")

    def converse(**req):
        ids = [x["id"] for x in json.loads(req["messages"][0]["content"][0]["text"])["requirements"]]
        out = {"results": [{"id": i, "verdict": "unmet", "evidence_chunk_ids": [], "reason": "관련 경험 없음"} for i in ids]}
        return {"output": {"message": {"content": [{"toolUse": {"name": "out", "input": out}}]}},
                "usage": {"inputTokens": 100, "outputTokens": 7}}
    monkeypatch.setattr(llm.client, "converse", converse)
    sk = {"chunk_id": 1, "chunk_type": "skill", "text": "Kafka"}
    req = {"requirements": [{"id": i, "type": "required", "text": "Go", "candidates": [sk]} for i in range(1, 12)]}
    assert c.post("/match", json=req).json()["usage"] == {"input_tokens": 200, "output_tokens": 14, "llm_calls": 2}
    assert c.post("/match", json={"requirements": req["requirements"][:1]}).json()["usage"]["llm_calls"] == 1


def test_error_response_has_usage(monkeypatch):
    # 형식 오류 2번 → 502여도 쓴 토큰은 돌려준다 (BE가 실패 호출 비용도 기록)
    monkeypatch.setitem(state.models, "llm", "ready")
    monkeypatch.setattr(llm.client, "converse", lambda **req: {
        "output": {"message": {"content": [{"toolUse": {"name": "out", "input": {"verdict": "?"}}}]}},
        "usage": {"inputTokens": 50, "outputTokens": 5}})
    sk = {"chunk_id": 1, "chunk_type": "skill", "text": "Kafka"}
    r = c.post("/match", json={"requirements": [{"id": 1, "type": "required", "text": "Go", "candidates": [sk]}]})
    assert r.status_code == 502
    assert r.json() == {"detail": "llm output invalid", "usage": {"input_tokens": 100, "output_tokens": 10, "llm_calls": 2}}


def test_stub_usage_zero():
    r = c.post("/match", json={"requirements": [{"id": 1, "type": "required", "text": "Go", "candidates": []}]}).json()
    assert r["usage"] == {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0}
