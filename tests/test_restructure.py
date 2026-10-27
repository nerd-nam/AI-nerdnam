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
        async def chat(messages, schema=None, max_tokens=0, temperature=None):
            calls.append(messages)
            return replies[len(calls) - 1]
        monkeypatch.setattr(llm, "chat", chat)
        return calls
    return set_replies


def test_spans_from_quotes_and_skills_filtered(llm_replies):
    llm_replies(draft("평균 응답 1.2초 → 0.3초", [Q2, Q1], skills=("Redis", "Kafka", "Kubernetes")))
    r = c.post("/restructure", json={"mode": "initial", "source_text": RAW, "skills": ["Redis"]}).json()
    ep = r["episodes"][0]
    assert r["prompt_version"] == "p1_v4"
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
    # 읽기 시간 초과를 503으로 보내면 BE가 재시도해 비용·부하가 두 배가 된다 (기획서 5.5)
    from botocore.exceptions import ReadTimeoutError
    monkeypatch.setitem(state.models, "llm", "ready")

    async def chat(*a, **k):
        raise ReadTimeoutError(endpoint_url="https://bedrock")
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


def with_metrics(*metrics, title="결제 승인 응답 지연 개선"):
    out = json.loads(draft("평균 응답 0.3초", [Q2]))
    out["episodes"][0].update(title=title, metrics=[dict(zip(("name", "before", "after", "source"), m)) for m in metrics])
    return json.dumps(out, ensure_ascii=False)


def test_metric_value_kept_without_retry(llm_replies):
    calls = llm_replies(with_metrics(("평균 응답", "1.2초", "0.3초", "raw_text")))
    ep = c.post("/restructure", json={"mode": "initial", "source_text": RAW}).json()["episodes"][0]
    assert len(calls) == 1
    assert ep["metrics"] == [{"name": "평균 응답", "before": "1.2초", "after": "0.3초", "source": "raw_text"}]


def test_json_like_metric_retried_then_dropped(llm_replies):
    # 로컬 테스트(AI #24): 출처 이름을 값 칸에 JSON 모양으로 넣음 — 20은 원문에 있어 수치 검사는 통과했다
    bad = with_metrics(("평균 응답", None, '{ "raw_text": "0.3 초", "answer": null }', "raw_text"),
                       ("평균 응답 (전)", None, "개선됨", "raw_text"))
    calls = llm_replies(bad, bad)
    ep = c.post("/restructure", json={"mode": "initial", "source_text": RAW}).json()["episodes"][0]
    assert len(calls) == 2 and "3.2초" in calls[1][-1]["content"]  # 값 모양을 알려 주며 1회 다시 요청
    assert ep["metrics"] == []  # 그래도 남으면 표에서 뺀다


def test_foreign_han_retried_then_asked(llm_replies):
    bad = with_metrics(title="결제 승인 고可用性 확보")
    calls = llm_replies(bad, bad)
    ep = c.post("/restructure", json={"mode": "initial", "source_text": RAW}).json()["episodes"][0]
    assert len(calls) == 2 and "可用性" in calls[1][-1]["content"]
    assert ep["clarifications"][0]["field"] == "title" and "可用性" in ep["clarifications"][0]["question"]


def test_llm_schema_requires_all_fields():
    # 기본값이 있으면 JSON 스키마에서 생략 가능이 되어 모델이 role·skills·수치 값을 건너뛴다 (로컬 테스트)
    from app.routers.restructure import P1Out
    defs = P1Out.model_json_schema()["$defs"]
    assert set(defs["Draft"]["required"]) == set(defs["Draft"]["properties"])
    assert set(defs["DraftMetric"]["required"]) == {"name", "before", "after", "source"}
    assert defs["DraftMetric"]["properties"]["after"]["type"] == "string"  # null 불가: 값 없는 성과 항목 금지 (AI #24)
    assert set(defs["DraftQuestion"]["required"]) == {"field", "question", "example"}


def test_converse_request(monkeypatch):
    # Converse 요청 모양: system 분리, 이어진 user는 한 메시지로 합침, temperature·topP 직접 지정, 스키마는 강제 도구 호출
    import asyncio
    sent = {}

    def converse(**req):
        sent.update(req)
        out = [{"toolUse": {"name": "out", "input": {"a": "가"}}}] if "toolConfig" in req else [{"reasoningContent": {}}, {"text": "ok"}]
        return {"output": {"message": {"content": out}}}
    monkeypatch.setattr(llm.client, "converse", converse)
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "A"}, {"role": "user", "content": "B"}]
    assert json.loads(asyncio.run(llm.chat(msgs, schema={"type": "object"}, max_tokens=8))) == {"a": "가"}  # 도구 입력이 출력
    assert sent["modelId"] == "global.amazon.nova-2-lite-v1:0" and sent["system"] == [{"text": "S"}]
    assert sent["messages"] == [{"role": "user", "content": [{"text": "A"}, {"text": "B"}]}]
    assert sent["inferenceConfig"] == {"maxTokens": 8, "temperature": 0.7, "topP": 0.8}
    tc = sent["toolConfig"]
    assert tc["toolChoice"] == {"tool": {"name": "out"}} and tc["tools"][0]["toolSpec"]["inputSchema"] == {"json": {"type": "object"}}
    sent.clear()
    assert asyncio.run(llm.chat([{"role": "user", "content": "p"}], temperature=0)) == "ok"  # text 블록만
    assert sent["inferenceConfig"]["temperature"] == 0 and "system" not in sent and "toolConfig" not in sent


# AI #25: 수정 요청(피드백)에 지금 사례 본문을 그대로 돌려주면 성공으로 넘기지 않는다
Q_NEW = {"field": "result", "question": "성과 수치가 있나요?", "example": None}


def copy_of_target(*clarifications):
    return json.dumps({"episodes": [{**TARGET, "role": "", "skills": [], "metrics": [], "quotes": [Q2],
                                     "clarifications": list(clarifications)}]}, ensure_ascii=False)


def revise(feedback="제목을 짧게", **extra):
    return c.post("/restructure", json={"mode": "episode", "source_text": RAW, "target_episode": TARGET,
                                        "feedback": feedback, **extra})


def test_feedback_ignored_retried_then_502(llm_replies):
    calls = llm_replies(copy_of_target(), copy_of_target())
    assert revise().status_code == 502  # API는 AI_ERROR
    assert calls[0][-1]["content"].startswith("고칠 방향: 제목을 짧게")  # 피드백을 따로 한 번 더 준다
    assert len(calls) == 2 and "그대로 돌려줬다" in calls[1][-1]["content"]


def test_question_only_change_is_retried(llm_replies):
    calls = llm_replies(copy_of_target(Q_NEW), draft("평균 응답 0.3초", [Q2]))
    assert revise().status_code == 200
    assert len(calls) == 2  # 확인 질문만 붙고 본문이 같으면 반영한 것으로 보지 않는다


def test_same_body_with_new_question_after_retry_ok(llm_replies):
    # 피드백이 원문에 없는 사실을 요구한 경우: 본문은 그대로 두고 확인 질문으로 묻는 것이 맞다
    llm_replies(copy_of_target(), copy_of_target(Q_NEW))
    r = revise("성과를 50% 개선으로 써 줘")
    assert r.status_code == 200 and r.json()["episodes"][0]["clarifications"][0]["question"] == Q_NEW["question"]


def test_changed_body_not_retried(llm_replies):
    calls = llm_replies(draft("평균 응답 0.3초", [Q2]))
    assert revise().status_code == 200 and len(calls) == 1


def test_answer_without_feedback_not_checked(llm_replies):
    calls = llm_replies(copy_of_target())
    r = c.post("/restructure", json={"mode": "episode", "source_text": RAW, "target_episode": TARGET,
                                     "answer": {"question": "역할은?", "answer": "백엔드"}})
    assert r.status_code == 200 and len(calls) == 1
