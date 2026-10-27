"""Bedrock 전환 (ADR-0007): 출력 스키마 규칙, 오류 → 계약 코드(C-04), /embed 순서·차원"""
import httpx
import pytest
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
    # Bedrock 구조화 출력: object마다 additionalProperties false, minItems 0·1만, 길이·범위 제약 금지
    for node in walk(llm.strict_schema(model.model_json_schema())):
        if "properties" in node:
            assert node["additionalProperties"] is False and set(node["required"]) == set(node["properties"])
        bad = {k for k in llm._UNSUPPORTED & node.keys() if not isinstance(node[k], dict)}
        assert not bad, bad
        assert node.get("minItems", 0) in (0, 1) or isinstance(node.get("minItems"), dict)


def test_field_named_like_keyword_kept():
    s = llm.strict_schema({"type": "object", "properties": {"format": {"type": "string"}}})
    assert "format" in s["properties"] and s["required"] == ["format"]


def status_error(code):
    req = httpx.Request("POST", "https://bedrock")
    return httpx.HTTPStatusError("x", request=req, response=httpx.Response(code, request=req, text="err"))


@pytest.mark.parametrize("err,code", [
    (status_error(429), 503),  # 호출 한도 → GPU_UNAVAILABLE (API가 1회 재시도)
    (status_error(503), 503),
    (status_error(400), 502),  # 다시 보내도 같다 → AI_ERROR
    (status_error(403), 502),
    (httpx.ReadTimeout("slow"), 504),  # TIMEOUT, 재시도 안 함
    (httpx.ConnectError("down"), 503),
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
        raise status_error(429)
    monkeypatch.setattr(embedding, "_one", throttled)
    assert c.post("/embed", json={"purpose": "query", "items": [{"text": "a"}]}).status_code == 503
