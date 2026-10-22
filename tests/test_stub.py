from fastapi.testclient import TestClient

from app.main import app

c = TestClient(app)


def test_health():
    assert c.get("/health").json() == {"status": "ok", "models": "stub"}


def test_restructure():
    raw = "결제 API 개선"
    r = c.post("/restructure", json={"raw_text": raw}).json()
    assert r["episodes"][0]["source_spans"] == [[0, len(raw)]]
    assert r["model"] and r["prompt_version"]


def test_extract_keeps_ids():
    r = c.post("/requirements/extract", json={"lines": [{"id": "r1", "type": "required", "text": "Java 3년"}]})
    assert [line["id"] for line in r.json()["lines"]] == ["r1"]


def test_embed_dim():
    r = c.post("/embed", json={"purpose": "query", "items": [{"text": "Kafka"}, {"text": "Redis"}]}).json()
    assert r["dim"] == 768 and len(r["vectors"]) == 2 and len(r["vectors"][0]) == 768
    assert r["model"] == "stub"  # 가짜 벡터가 실제 모델 이름으로 저장되지 않게


def test_match_evidence_from_candidates():
    req = {"requirements": [{"id": "r1", "text": "Kafka", "candidates": [{"chunk_id": "c1", "text": "..."}]}]}
    res = c.post("/match", json=req).json()["results"][0]
    assert res["evidence_chunk_ids"] == ["c1"]


def test_guide_sse():
    req = {"tab": "intro", "results": [], "evidence": [{"chunk_id": "c1", "text": "..."}]}
    body = c.post("/guide", json=req).text
    assert "event: done" in body and "[c1]" in body


def test_bad_request_is_422():
    assert c.post("/embed", json={"purpose": "query", "items": []}).status_code == 422


def test_embed_prefix():
    from app.embedding import fmt
    assert fmt("query", "Kafka") == "task: search result | query: Kafka"
    assert fmt("document", "본문", "제목") == "title: 제목 | text: 본문"
    assert fmt("document", "본문") == "title: none | text: 본문"

