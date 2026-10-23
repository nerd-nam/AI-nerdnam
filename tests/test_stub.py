from fastapi.testclient import TestClient

from app.main import app

c = TestClient(app)
C1 = {"chunk_id": 501, "chunk_type": "episode", "text": "..."}


def test_health():
    assert c.get("/health").json() == {"status": "ok", "models": {"llm": "stub", "embedding": "stub"}}


def test_restructure():
    raw = "결제 API 개선"
    r = c.post("/restructure", json={"mode": "initial", "source_text": raw}).json()
    assert r["episodes"][0]["source_spans"] == [[0, len(raw)]]
    assert r["model"] and r["prompt_version"]


def test_extract_keeps_ids():
    r = c.post("/requirements/extract", json={"lines": [{"id": 11, "type": "required", "text": "Java 3년"}]})
    assert [line["id"] for line in r.json()["lines"]] == [11]


def test_embed_dim():
    r = c.post("/embed", json={"purpose": "query", "items": [{"text": "Kafka"}, {"text": "Redis"}]}).json()
    assert r["dim"] == 768 and len(r["vectors"]) == 2 and len(r["vectors"][0]) == 768
    assert r["embedding_model"] == "stub"  # 가짜 벡터가 실제 모델 이름으로 저장되지 않게


def test_match_evidence_from_candidates():
    req = {"requirements": [{"id": 11, "type": "required", "text": "Kafka", "candidates": [C1]}]}
    res = c.post("/match", json=req).json()["results"][0]
    assert res["id"] == 11 and res["evidence_chunk_ids"] == [501]


def test_guide_json():
    req = {"tab": "intro", "requirements": [
        {"id": 11, "type": "required", "text": "Kafka", "verdict": "met", "reason": "...", "evidence": [C1]}]}
    r = c.post("/guide", json=req)
    assert r.headers["content-type"] == "application/json"  # SSE 아님 (기획서 v1.23)
    assert r.json()["cited_chunk_ids"] == [501] and r.json()["prompt_version"] == "stub"


def test_guide_tab_values():
    req = {"tab": "motivation", "requirements": [{"id": 11, "type": "required", "text": "Kafka", "verdict": "met", "reason": ""}]}
    assert c.post("/guide", json=req).status_code == 422  # 웹 주소와 같은 motive 만


def test_bad_request_is_422():
    assert c.post("/embed", json={"purpose": "query", "items": []}).status_code == 422


def test_embed_prefix():
    from app.embedding import fmt
    assert fmt("query", "Kafka") == "task: search result | query: Kafka"
    assert fmt("document", "본문", "제목") == "title: 제목 | text: 본문"
    assert fmt("document", "본문") == "title: none | text: 본문"


def test_gap_guide_cites_adjacent_only():
    req = {"requirements": [
        {"id": 13, "type": "required", "text": "iOS 개발", "verdict": "unmet",
         "adjacent": [{"chunk_id": 502, "chunk_type": "project_summary", "text": "React Native 앱"}]},
        {"id": 14, "type": "preferred", "text": "TDD", "verdict": "partial"},
    ], "tasks": ["결제 API 개발"]}
    r = c.post("/gap-guide", json=req).json()
    assert [(x["id"], x["cited_chunk_ids"]) for x in r["guides"]] == [(13, [502]), (14, [])]
    assert r["model"] and r["prompt_version"]


def test_gap_guide_rejects_met():
    req = {"requirements": [{"id": 11, "type": "required", "text": "Java", "verdict": "met"}]}
    assert c.post("/gap-guide", json=req).status_code == 422


def test_split_needs_feedback():
    base = {"mode": "split", "source_text": "결제 API 개선"}
    assert c.post("/restructure", json=base).status_code == 422
    assert c.post("/restructure", json=base | {"feedback": "  "}).status_code == 422
    assert c.post("/restructure", json=base | {"feedback": "사례를 하나로 합쳐 주세요"}).status_code == 200


def test_episode_needs_target():
    assert c.post("/restructure", json={"mode": "episode", "source_text": "결제 API 개선"}).status_code == 422
