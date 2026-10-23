import json

import pytest
from fastapi.testclient import TestClient

from app import llm, state
from app.main import app
from app.routers.requirements import min_years

c = TestClient(app)

# SPEC fixtures/jd.json 요건 6줄
LINES = [
    {"id": 11, "type": "required", "text": "Java/Spring 기반 개발 경력 3년 이상"},
    {"id": 12, "type": "required", "text": "RDBMS 설계 및 쿼리 튜닝 경험"},
    {"id": 13, "type": "required", "text": "대용량 트래픽 처리 경험"},
    {"id": 14, "type": "preferred", "text": "Kafka 등 메시지 큐 운영 경험"},
]


@pytest.fixture
def llm_replies(monkeypatch):
    monkeypatch.setitem(state.models, "llm", "ready")
    calls = []

    def set_replies(*replies):
        async def chat(messages, schema=None, max_tokens=0):
            calls.append(messages)
            return json.dumps(replies[len(calls) - 1], ensure_ascii=False)
        monkeypatch.setattr(llm, "chat", chat)
        return calls
    return set_replies


def test_min_years():
    assert min_years("Java/Spring 기반 개발 경력 3년 이상") == 3
    assert min_years("백엔드 경력 3~5년") == 3
    assert min_years("5년+ 경력") == 5
    assert min_years("3+ years of experience") == 3
    assert min_years("경력 3년차 이상") == 3
    assert min_years("경력 5년 미만") is None  # 최소가 아니라 최대
    assert min_years("대용량 트래픽 처리 경험") is None


def test_extract_skills_and_years(llm_replies):
    llm_replies({"lines": [
        {"id": 11, "skills_raw": ["Java", "Spring", "Spring Boot"]},  # Spring Boot는 줄에 없음 → 버림
        {"id": 12, "skills_raw": ["RDBMS"]},
        {"id": 13, "skills_raw": []},
        {"id": 14, "skills_raw": ["kafka"]},  # 대소문자 차이는 줄에 있는 것으로 봄
    ]})
    r = c.post("/requirements/extract", json={"lines": LINES}).json()
    assert r["prompt_version"] == "p2_v1"
    assert [(x["id"], x["skills_raw"], x["min_years"]) for x in r["lines"]] == [
        (11, ["Java", "Spring"], 3), (12, ["RDBMS"], None), (13, [], None), (14, ["kafka"], None)]


def test_missing_line_retried_then_filled(llm_replies):
    short = {"lines": [{"id": 11, "skills_raw": ["Java"]}]}
    calls = llm_replies(short, short)
    r = c.post("/requirements/extract", json={"lines": LINES[:2]}).json()
    assert len(calls) == 2 and "id" in calls[1][-1]["content"]  # 빠진 줄을 알려 1회 다시 요청
    assert [x["id"] for x in r["lines"]] == [11, 12] and r["lines"][1]["skills_raw"] == []  # 그래도 빠지면 빈 목록


def test_503_while_loading(monkeypatch):
    monkeypatch.setitem(state.models, "llm", "loading")
    assert c.post("/requirements/extract", json={"lines": LINES}).status_code == 503
