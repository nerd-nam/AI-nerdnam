import json

import pytest
from fastapi.testclient import TestClient

from app import llm, state
from app.main import app

c = TestClient(app)

# SPEC fixtures/projects.json p_102 원문 앞부분
RAW = ("평균 응답이 1.2초까지 늘었고 결제 이탈이 생겼다. "
       "가맹점 조회는 Redis 캐시로 바꾸고 후처리는 Kafka로 비동기 분리했다. 그 결과 평균 응답이 0.3초로 줄었다.")
Q1 = "평균 응답이 1.2초까지 늘었고 결제 이탈이 생겼다."
Q2 = "그 결과 평균 응답이 0.3초로 줄었다."


def draft(result, quotes, skills=("Redis", "Kafka")):
    return json.dumps({"episodes": [{
        "title": "결제 승인 응답 지연 개선", "role": "", "skills": list(skills),
        "problem": "평균 응답 1.2초, 결제 이탈", "solution": "Redis 캐시, Kafka 비동기 분리", "result": result,
        "metrics": [], "quotes": quotes, "questions": []}]}, ensure_ascii=False)


@pytest.fixture
def llm_replies(monkeypatch):
    monkeypatch.setitem(state.models, "llm", "ready")
    calls = []

    def set_replies(*replies):
        async def chat(messages, schema=None, max_tokens=0):
            calls.append(messages)
            return replies[len(calls) - 1]
        monkeypatch.setattr(llm, "chat", chat)
        return calls
    return set_replies


def test_spans_from_quotes_and_skills_filtered(llm_replies):
    llm_replies(draft("평균 응답 1.2초 → 0.3초", [Q2, Q1], skills=("Redis", "Kafka", "Kubernetes")))
    r = c.post("/restructure", json={"raw_text": RAW, "skills": ["Redis"]}).json()
    ep = r["episodes"][0]
    assert r["prompt_version"] == "p1_v1"
    assert ep["source_spans"] == [[RAW.find(Q1), RAW.find(Q1) + len(Q1)], [RAW.find(Q2), RAW.find(Q2) + len(Q2)]]
    assert ep["skills"] == ["Redis", "Kafka"]  # Kubernetes는 입력·원문에 없어 제거
    assert ep["questions"] == []


def test_made_up_number_retried_then_asked(llm_replies):
    calls = llm_replies(draft("응답 75% 단축", [Q2]), draft("응답 75% 단축", [Q2]))
    ep = c.post("/restructure", json={"raw_text": RAW}).json()["episodes"][0]
    assert len(calls) == 2 and "75" in calls[1][-1]["content"]  # 1회 다시 요청하며 문제를 알려 줌
    assert any("'75'" in q for q in ep["questions"])  # 그래도 남으면 확인 필요


def test_number_from_answer_is_allowed(llm_replies):
    calls = llm_replies(draft("피크 TPS 약 2배", [Q2]))
    req = {"raw_text": RAW, "answers": [{"question": "피크 처리량은?", "answer": "피크 TPS 약 2배"}]}
    assert c.post("/restructure", json=req).json()["episodes"][0]["questions"] == []
    assert len(calls) == 1


def test_episode_mode_keeps_target_id(llm_replies):
    llm_replies(draft("평균 응답 0.3초", [Q2]))
    req = {"mode": "episode", "raw_text": RAW, "target_episode_id": "e_201", "feedback": "제목을 짧게"}
    assert c.post("/restructure", json=req).json()["episodes"][0]["episode_id"] == "e_201"


def test_503_while_loading(monkeypatch):
    monkeypatch.setitem(state.models, "llm", "loading")
    assert c.post("/restructure", json={"raw_text": RAW}).status_code == 503
