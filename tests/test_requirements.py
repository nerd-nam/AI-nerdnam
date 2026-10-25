import json

import pytest
from fastapi.testclient import TestClient

from app import llm, state
from app.main import app
from app.routers.requirements import PROMPT_ABBR, P2Abbr, P2AbbrLine, abbr_candidates, clean_expansions, min_years

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
    """기술명 호출과 약어 호출에 줄 답을 따로 정한다. 호출 기록도 따로 돌려준다."""
    monkeypatch.setitem(state.models, "llm", "ready")
    calls = {"skills": [], "abbr": []}

    def set_replies(skills=(), abbr=()):
        replies = {"skills": list(skills), "abbr": list(abbr)}

        async def chat(messages, schema=None, max_tokens=0, temperature=None):
            assert temperature == 0  # P2는 온도 0
            kind = "abbr" if messages[0]["content"] == PROMPT_ABBR else "skills"
            calls[kind].append(messages)
            return json.dumps(replies[kind][len(calls[kind]) - 1], ensure_ascii=False)
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


def test_extract_skills_expansions_and_years(llm_replies):
    calls = llm_replies(
        skills=[{"lines": [
            {"id": 11, "skills_raw": ["Java", "Spring", "Spring Boot"]},  # Spring Boot는 줄에 없음 → 버림
            {"id": 12, "skills_raw": ["RDBMS"]},
            {"id": 13, "skills_raw": []},
            {"id": 14, "skills_raw": ["kafka"]},  # 대소문자 차이는 줄에 있는 것으로 봄
        ]}],
        abbr=[{"lines": [{"id": 12, "abbreviations": [
            {"abbr": "RDBMS", "full": ["관계형 데이터베이스"], "is_name": False, "ambiguous": False},
            {"abbr": "MSA", "full": ["마이크로서비스 아키텍처"], "is_name": False, "ambiguous": False},  # 줄에 없는 약어 → 버림
        ]}]}],
    )
    r = c.post("/requirements/extract", json={"lines": LINES}).json()
    assert r["prompt_version"] == "p2_v2"
    assert [(x["id"], x["skills_raw"], x["expansions"], x["min_years"]) for x in r["lines"]] == [
        (11, ["Java", "Spring"], [], 3), (12, ["RDBMS"], ["관계형 데이터베이스"], None),
        (13, [], [], None), (14, ["kafka"], [], None)]
    # 약어 호출에는 후보가 있는 줄만 간다
    assert json.loads(calls["abbr"][0][-1]["content"]) == {
        "lines": [{"id": 12, "text": LINES[1]["text"], "abbr_candidates": ["RDBMS"]}]}


def test_no_candidates_no_abbr_call(llm_replies):
    calls = llm_replies(skills=[{"lines": [{"id": 13, "skills_raw": []}]}])
    r = c.post("/requirements/extract", json={"lines": [LINES[2]]}).json()
    assert calls["abbr"] == [] and r["lines"][0]["expansions"] == []


def test_clean_expansions():
    text = "RDBMS(관계형 DB) 설계 및 쿼리 튜닝 경험"

    def line(*full, abbr="RDBMS", is_name=False, ambiguous=False):
        return P2AbbrLine(id=1, abbreviations=[P2Abbr(abbr=abbr, full=list(full), is_name=is_name, ambiguous=ambiguous)])
    assert clean_expansions(line("관계형 DB", " 관계형  db "), text) == []  # 줄에 이미 있는 말 (공백·대소문자 무시)
    assert clean_expansions(line("관계형 데이터베이스", "관계형데이터베이스", "", "  "), text) == ["관계형 데이터베이스"]  # 중복·빈 값
    assert clean_expansions(line("a1", "b2", "c3", "d4"), text) == ["a1", "b2", "c3"]  # 최대 3개
    assert clean_expansions(line("마이크로서비스", abbr="MSA"), text) == []  # 줄에 없는 약어
    assert clean_expansions(None, text) == []
    assert clean_expansions(line("Relational DB", ambiguous=True), text) == []  # 여러 뜻 → 버림
    assert clean_expansions(line("Relational DB", is_name=True), text) == []  # 이름 → 버림
    assert clean_expansions(line("Kubernetes", abbr="k8s", is_name=True), "k8s 운영 경험") == ["Kubernetes"]  # 숫자 줄임말은 이름 아님


def test_missing_candidate_retried(llm_replies):
    """후보를 건너뛰면 알려서 1회 다시 요청한다"""
    skipped = {"lines": [{"id": 12, "abbreviations": []}]}
    filled = {"lines": [{"id": 12, "abbreviations": [
        {"abbr": "RDBMS", "full": ["관계형 데이터베이스"], "is_name": False, "ambiguous": False}]}]}
    calls = llm_replies(skills=[{"lines": [{"id": 12, "skills_raw": ["RDBMS"]}]}], abbr=[skipped, filled])
    r = c.post("/requirements/extract", json={"lines": [LINES[1]]}).json()
    assert len(calls["abbr"]) == 2 and "12:RDBMS" in calls["abbr"][1][-1]["content"]
    assert r["lines"][0]["expansions"] == ["관계형 데이터베이스"]


def test_abbr_candidates():
    assert abbr_candidates("CI/CD 파이프라인과 k8s, MSA 환경") == ["CI/CD", "k8s", "MSA"]
    assert abbr_candidates("IaC·i18n 경험, PM 또는 PO") == ["IaC", "i18n", "PM", "PO"]
    assert abbr_candidates("Java/Spring, JavaScript, PostgreSQL, GitHub, Node.js 경험") == []  # 이름은 후보 아님


def test_stub_expansions_empty():
    r = c.post("/requirements/extract", json={"lines": LINES}).json()
    assert r["prompt_version"] == "stub" and all(x["expansions"] == [] for x in r["lines"])


def test_missing_line_retried_then_filled(llm_replies):
    short = {"lines": [{"id": 11, "skills_raw": ["Java"]}]}
    no_abbr = {"lines": []}
    calls = llm_replies(skills=[short, short], abbr=[no_abbr, no_abbr])
    r = c.post("/requirements/extract", json={"lines": LINES[:2]}).json()
    assert len(calls["skills"]) == 2 and "id" in calls["skills"][1][-1]["content"]  # 빠진 줄을 알려 1회 다시 요청
    assert len(calls["abbr"]) == 2  # 약어 호출도 같음
    assert [x["id"] for x in r["lines"]] == [11, 12] and r["lines"][1]["skills_raw"] == [] == r["lines"][1]["expansions"]  # 그래도 빠지면 빈 목록


def test_503_while_loading(monkeypatch):
    monkeypatch.setitem(state.models, "llm", "loading")
    assert c.post("/requirements/extract", json={"lines": LINES}).status_code == 503
