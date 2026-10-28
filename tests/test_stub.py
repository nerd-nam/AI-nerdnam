import pytest
from fastapi.testclient import TestClient

from app.main import app

c = TestClient(app)
C1 = {"chunk_id": 501, "chunk_type": "episode", "text": "..."}


def test_health():
    assert c.get("/health").json() == {"status": "ok", "models": {"llm": "stub", "embedding": "stub"}}


@pytest.mark.parametrize("llm,emb,code", [
    ("stub", "stub", 200),
    ("ready", "ready", 200),
    ("loading", "ready", 503),  # 첫 연결 확인 중
    ("error", "error", 503),
    ("ready", "error", 200),  # 한쪽만 쓸 수 있어도 받는다. 임베딩 요청만 503
    ("error", "ready", 200),
])
def test_probes(monkeypatch, llm, emb, code):
    monkeypatch.setattr("app.state.models", {"llm": llm, "embedding": emb})
    assert c.get("/livez").status_code == 200
    r = c.get("/readyz")
    assert r.status_code == code and r.json()["models"] == {"llm": llm, "embedding": emb}
    assert c.get("/health").status_code == 200  # 계약 응답은 그대로
    assert "/readyz" not in app.openapi()["paths"]  # 계약(C-04) 파일 변화 없음


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
    assert r["dim"] == 1024 and len(r["vectors"]) == 2 and len(r["vectors"][0]) == 1024
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


@pytest.mark.parametrize("tab", ["motive", "intro", "portfolio", "career", "image"])
def test_guide_content_shape_per_tab(tab):
    req = {"tab": tab, "requirements": [
        {"id": 11, "type": "required", "text": "Kafka", "verdict": "met", "reason": "...", "evidence": [C1]},
        {"id": 12, "type": "preferred", "text": "Kubernetes", "verdict": "unmet", "reason": "..."}]}
    content = c.post("/guide", json=req).json()["content"]
    assert content["tab"] == tab  # 탭별 모양 구분 (계약의 discriminator)
    if tab == "intro":
        assert [x["requirement_id"] for x in content["coverage"]] == [11, 12]
    if tab == "portfolio":
        pages = content["pages"]
        assert 4 <= len(pages) <= 8 and [p["order"] for p in pages] == list(range(1, len(pages) + 1))
        assert {p["kind"] for p in pages} <= {"cover", "text", "diagram", "chart"}
    if tab != "image":  # 인용은 sources 키, 요청 evidence 안에서만
        assert all(s in (501,) for p in content["outline"] for s in p["sources"])


def test_guide_tab_values():
    req = {"tab": "motivation", "requirements": [{"id": 11, "type": "required", "text": "Kafka", "verdict": "met", "reason": ""}]}
    assert c.post("/guide", json=req).status_code == 422  # 웹 주소와 같은 motive 만


def test_bad_request_is_422():
    assert c.post("/embed", json={"purpose": "query", "items": []}).status_code == 422


def test_embed_text():
    from app.embedding import text_of
    assert text_of("query", "Kafka") == "Kafka"
    assert text_of("query", "Kafka", "제목") == "Kafka"  # 질의에는 제목을 붙이지 않는다
    assert text_of("document", "본문", "제목") == "제목\n본문"
    assert text_of("document", "본문") == "본문"


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
