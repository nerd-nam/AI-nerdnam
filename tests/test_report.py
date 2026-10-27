"""/match·/gap-guide·/guide 실제 모드 검증 로직 (LLM 응답은 가짜로 넣는다)."""
import json

import pytest
from fastapi.testclient import TestClient

from app import llm, state
from app.main import app

c = TestClient(app)
EP = {"chunk_id": 531, "chunk_type": "episode", "text": "결제 승인 응답 평균 1.2초 → 0.3초, Kafka 비동기 분리"}
SK = {"chunk_id": 506, "chunk_type": "skill", "text": "Kafka · infra · 실무 사용"}


@pytest.fixture
def llm_replies(monkeypatch):
    monkeypatch.setitem(state.models, "llm", "ready")
    calls = []

    def set_replies(*replies):
        """응답을 차례로 준다. 응답이 함수면 메시지를 보고 고른다 (요건마다 동시에 부르는 /match용)"""
        async def chat(messages, schema=None, max_tokens=0, temperature=None):
            calls.append((messages, schema))
            r = replies[0] if callable(replies[0]) else replies[len(calls) - 1]
            return json.dumps(r(messages) if callable(r) else r, ensure_ascii=False)
        monkeypatch.setattr(llm, "chat", chat)
        return calls
    return set_replies


def test_strict_schema_requires_defaulted_fields():
    from app.schemas import IntroContent
    s = llm.strict_schema(IntroContent.model_json_schema())
    assert set(s["required"]) == set(s["properties"])  # keywords·avoid·coverage 기본값이 있어도 필수
    assert set(s["$defs"]["Paragraph"]["required"]) == {"title", "detail", "sources"}


def test_match_filters_ids_and_forces_unmet(llm_replies):
    out = {"results": [
        {"id": 14, "verdict": "met", "evidence_chunk_ids": [531, 999, 531], "reason": "Kafka 비동기 분리 경험"},  # 999는 후보 밖
        {"id": 16, "verdict": "partial", "evidence_chunk_ids": [], "reason": "관련 없음"},  # 근거 없는 부분 충족
        {"id": 77, "verdict": "met", "evidence_chunk_ids": [531], "reason": "x"},  # 입력에 없는 요건
    ]}
    calls = llm_replies(out, out)  # 다시 요청해도 같은 답 → 코드가 정리
    req = {"requirements": [
        {"id": 14, "type": "preferred", "text": "Kafka 운영", "candidates": [EP, SK]},
        {"id": 16, "type": "preferred", "text": "Kubernetes 운영", "candidates": [SK]},
        {"id": 17, "type": "preferred", "text": "iOS", "candidates": []},  # 후보 없음 → 모델에 안 보냄
    ]}
    r = c.post("/match", json=req).json()
    assert r["prompt_version"] == "p3_v4"
    got = {x["id"]: (x["verdict"], x["evidence_chunk_ids"]) for x in r["results"]}
    assert got == {14: ("met", [531]), 16: ("unmet", []), 17: ("unmet", [])}
    assert [x["id"] for x in r["results"]] == [14, 16, 17]  # 입력 순서
    assert len(calls) == 2  # 요건 14·16을 한 번에, 문제가 있어 1회 다시 요청
    sent = json.loads(calls[0][0][1]["content"])
    assert [x["chunk_id"] for x in sent["candidates"]] == [531, 506]  # SK는 두 요건이 같이 써도 한 번만
    assert [(x["id"], x["candidate_ids"]) for x in sent["requirements"]] == [(14, [531, 506]), (16, [506])]


def test_match_retries_on_made_up_number(llm_replies):
    calls = llm_replies({"results": [{"id": 13, "verdict": "met", "evidence_chunk_ids": [531], "reason": "응답 75% 단축"}]},
                        {"results": [{"id": 13, "verdict": "met", "evidence_chunk_ids": [531], "reason": "응답 1.2초 → 0.3초"}]})
    req = {"requirements": [{"id": 13, "type": "required", "text": "대용량 트래픽", "candidates": [EP]}]}
    assert c.post("/match", json=req).json()["results"][0]["reason"] == "응답 1.2초 → 0.3초"
    assert len(calls) == 2 and "75" in calls[1][0][-1]["content"]


def test_match_batches_and_fills_missing(llm_replies):
    # 요건 15개 → 10개·5개 두 묶음. 묶음 응답에서 끝내 빠진 요건(15)은 혼자 다시 부른다
    def reply(messages):
        ids = [x["id"] for x in json.loads(messages[1]["content"])["requirements"]]
        keep = ids if ids == [15] else [i for i in ids if i != 15]
        return {"results": [{"id": i, "verdict": "met", "evidence_chunk_ids": [531], "reason": "결제 경험"} for i in keep]}
    calls = llm_replies(reply)
    req = {"requirements": [{"id": i, "type": "required", "text": "결제", "candidates": [EP]} for i in range(1, 16)]}
    r = c.post("/match", json=req).json()
    assert [x["verdict"] for x in r["results"]] == ["met"] * 15
    sizes = [len(json.loads(m[1]["content"])["requirements"]) for m, _ in calls]
    assert sorted(sizes) == [1, 5, 5, 10]  # 10·5 묶음, 5개 묶음은 15가 빠져 1회 다시 요청, 그 뒤 15 혼자


def test_match_missing_after_fallback_is_502(llm_replies):
    llm_replies(lambda messages: {"results": []})  # 몇 번을 다시 불러도 빈 결과
    req = {"requirements": [{"id": 1, "type": "required", "text": "결제", "candidates": [EP]}]}
    assert c.post("/match", json=req).status_code == 502


def test_gap_guide_filters_citations(llm_replies):
    out = {"guides": [
        {"id": 16, "advice": "Kafka 운영 경험을 인접 경험으로", "avoid": "운영해 본 것처럼 쓰기", "cited_chunk_ids": [506, 531]},
        {"id": 99, "advice": "x", "avoid": "y", "cited_chunk_ids": []},  # 입력에 없는 요건
    ]}
    llm_replies(out, out)  # 531 때문에 1회 다시 요청, 같은 답이면 코드가 정리
    req = {"requirements": [{"id": 16, "type": "preferred", "text": "Kubernetes 운영", "verdict": "unmet", "adjacent": [SK]}]}
    r = c.post("/gap-guide", json=req).json()
    assert r["prompt_version"] == "p6_v2"
    assert [(g["id"], g["cited_chunk_ids"]) for g in r["guides"]] == [(16, [506])]  # 531은 adjacent 밖


def _guide_req(tab):
    return {"tab": tab, "company": "OO페이", "position": "백엔드", "requirements": [
        {"id": 14, "type": "preferred", "text": "Kafka 운영", "verdict": "met", "reason": "비동기 분리", "evidence": [EP]},
        {"id": 16, "type": "preferred", "text": "Kubernetes 운영", "verdict": "unmet", "reason": "없음", "evidence": []},
    ]}


def _writing(**extra):
    return {"message": "결제 성능 개선 경험", "keywords": ["Kafka"], "avoid": ["과장"],
            "outline": [{"title": "문단 1", "detail": "평균 1.2초 → 0.3초", "sources": [531, 777]}], **extra}


def test_guide_portfolio_cleans_sources_and_pages(llm_replies):
    pages = [{"order": o, "title": f"{o}쪽", "hint": "내용", "kind": k, "sources": s}
             for o, k, s in ((3, "text", [531]), (1, "cover", []), (5, "chart", [888]), (4, "diagram", [531]))]
    llm_replies(_writing(tab="portfolio", pages=pages), _writing(tab="portfolio", pages=pages))
    r = c.post("/guide", json=_guide_req("portfolio")).json()
    assert r["prompt_version"] == "p4_v2"
    assert [p["order"] for p in r["content"]["pages"]] == [1, 2, 3, 4]  # 쪽 번호 다시 매김
    assert r["content"]["pages"][0]["kind"] == "cover"
    assert r["content"]["outline"][0]["sources"] == [531] and r["cited_chunk_ids"] == [531]  # 777·888은 근거 밖


def test_guide_intro_needs_coverage_for_all(llm_replies):
    one = _writing(tab="intro", coverage=[{"requirement_id": 14, "covered": True, "where": "문단 1"}])
    both = _writing(tab="intro", coverage=[{"requirement_id": 14, "covered": True, "where": "문단 1"},
                                           {"requirement_id": 16, "covered": False, "where": "경험 없음"}])
    calls = llm_replies(one, both)
    r = c.post("/guide", json=_guide_req("intro")).json()
    assert len(calls) == 2 and [x["requirement_id"] for x in r["content"]["coverage"]] == [14, 16]


def test_guide_image_uses_p5(llm_replies):
    llm_replies({"tab": "image", "prompts": [{"title": "커버", "purpose": "커버", "page": 1, "ratio": "16:9",
                                              "prompt": "isometric server pipeline, no text", "negative": "text"}],
                 "tool_visuals": [{"name": "구조도", "page": 3, "kind": "diagram"}]})
    r = c.post("/guide", json=_guide_req("image")).json()
    assert r["prompt_version"] == "p5_v2" and r["cited_chunk_ids"] == []


def test_guide_payload_dedupes_evidence(llm_replies):
    calls = llm_replies(_writing(tab="motive"), _writing(tab="motive"))  # 777(근거 밖) 때문에 1회 다시 요청
    req = _guide_req("motive")
    req["requirements"][1]["evidence"] = [EP]  # 같은 청크가 두 요건에
    c.post("/guide", json=req)
    sent = json.loads(calls[0][0][1]["content"])
    assert [x["chunk_id"] for x in sent["evidence"]] == [531]  # 청크는 한 번만
    assert [r["evidence_ids"] for r in sent["requirements"]] == [[531], [531]] and "evidence" not in sent["requirements"][0]


def test_guide_trims_evidence_when_too_long(llm_replies, monkeypatch):
    from app.routers import guide as g
    calls = llm_replies(_writing(tab="motive"), _writing(tab="motive"))  # 777(근거 밖) 때문에 1회 다시 요청
    long = [{"chunk_id": 600 + i, "chunk_type": "episode", "text": "긴 사례 " * 300} for i in range(5)]
    req = _guide_req("motive")
    req["requirements"][0]["evidence"] = long
    monkeypatch.setattr(llm, "LLM_MAX_MODEL_LEN", 8192)
    c.post("/guide", json=req)
    sent = json.loads(calls[0][0][1]["content"])
    assert len(sent["requirements"][0]["evidence_ids"]) < 5  # 한도를 넘어 앞에서부터 줄임
    monkeypatch.setattr(llm, "LLM_MAX_MODEL_LEN", 1000)  # 1개로도 안 들어가면 422 (API가 요건을 줄여야 함)
    assert c.post("/guide", json=req).status_code == 422


def test_llm_http_error_is_502(monkeypatch):
    from botocore.exceptions import ClientError
    monkeypatch.setitem(state.models, "llm", "ready")

    async def chat(*a, **k):
        raise ClientError({"Error": {"Code": "ValidationException", "Message": "Input is too long"}}, "Converse")
    monkeypatch.setattr(llm, "chat", chat)
    r = c.post("/match", json={"requirements": [{"id": 1, "type": "required", "text": "x", "candidates": [EP]}]})
    assert r.status_code == 502 and "ValidationException" in r.json()["detail"]  # 500이면 API가 같은 요청을 다시 보낸다
