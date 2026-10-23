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
TARGET = {"title": "결제 승인 응답 지연 개선", "problem": "평균 응답 1.2초", "solution": "Redis 캐시", "result": "평균 응답 0.3초"}


def draft(result, quotes, skills=("Redis", "Kafka")):
    return json.dumps({"episodes": [{
        "title": "결제 승인 응답 지연 개선", "role": "", "skills": list(skills),
        "problem": "평균 응답 1.2초, 결제 이탈", "solution": "Redis 캐시, Kafka 비동기 분리", "result": result,
        "metrics": [], "quotes": quotes, "clarifications": []}]}, ensure_ascii=False)


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
    r = c.post("/restructure", json={"mode": "initial", "source_text": RAW, "skills": ["Redis"]}).json()
    ep = r["episodes"][0]
    assert r["prompt_version"] == "p1_v2"
    assert ep["source_spans"] == [[RAW.find(Q1), RAW.find(Q1) + len(Q1)], [RAW.find(Q2), RAW.find(Q2) + len(Q2)]]
    assert ep["skills"] == ["Redis", "Kafka"]  # Kubernetes는 입력·원문에 없어 제거
    assert ep["clarifications"] == []


def test_made_up_number_retried_then_asked(llm_replies):
    calls = llm_replies(draft("응답 75% 단축", [Q2]), draft("응답 75% 단축", [Q2]))
    ep = c.post("/restructure", json={"mode": "initial", "source_text": RAW}).json()["episodes"][0]
    assert len(calls) == 2 and "75" in calls[1][-1]["content"]  # 1회 다시 요청하며 문제를 알려 줌
    q = ep["clarifications"][0]  # 그래도 남으면 확인 필요
    assert "'75'" in q["question"] and q["field"] == "result" and q["key"] == "q1"


def test_number_from_answer_is_allowed(llm_replies):
    calls = llm_replies(draft("피크 TPS 약 2배", [Q2]))
    req = {"mode": "episode", "source_text": RAW, "target_episode": TARGET,
           "answer": {"question": "피크 처리량은?", "answer": "피크 TPS 약 2배"}}
    assert c.post("/restructure", json=req).json()["episodes"][0]["clarifications"] == []
    assert len(calls) == 1


def test_episode_mode_returns_one(llm_replies):
    llm_replies(json.dumps({"episodes": [json.loads(draft("평균 응답 0.3초", [Q2]))["episodes"][0]] * 2}, ensure_ascii=False),
                draft("평균 응답 0.3초", [Q2]))
    req = {"mode": "episode", "source_text": RAW, "target_episode": TARGET, "feedback": "제목을 짧게"}
    assert len(c.post("/restructure", json=req).json()["episodes"]) == 1


def test_503_while_loading(monkeypatch):
    monkeypatch.setitem(state.models, "llm", "loading")
    assert c.post("/restructure", json={"mode": "initial", "source_text": RAW}).status_code == 503


def test_llm_read_timeout_is_504(monkeypatch):
    # 읽기 시간 초과를 503(GPU 꺼짐)으로 보내면 BE가 재시도해 GPU 부하가 두 배가 된다 (기획서 5.5)
    import httpx
    monkeypatch.setitem(state.models, "llm", "ready")

    async def chat(*a, **k):
        raise httpx.ReadTimeout("slow")
    monkeypatch.setattr(llm, "chat", chat)
    assert c.post("/restructure", json={"mode": "initial", "source_text": RAW}).status_code == 504


def test_spaced_quote_and_metric_source(llm_replies):
    # 모델이 "0.3 초"처럼 띄어 써도 위치를 찾고, 원문에 있는 수치는 출처를 raw_text로 고친다
    out = json.loads(draft("평균 응답 0.3초", ["그 결과 평균 응답이 0.3 초로 줄었다."]))
    out["episodes"][0]["metrics"] = [{"name": "평균 응답", "before": "1.2 초", "after": "0.3 초", "source": "answer"}]
    calls = llm_replies(json.dumps(out, ensure_ascii=False))
    ep = c.post("/restructure", json={"mode": "initial", "source_text": RAW}).json()["episodes"][0]
    assert len(calls) == 1  # 공백 차이로 다시 요청하지 않음
    assert ep["source_spans"] == [[RAW.find(Q2), RAW.find(Q2) + len(Q2)]]
    assert ep["metrics"][0]["source"] == "raw_text"
