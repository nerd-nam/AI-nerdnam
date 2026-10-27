"""/restructure (P1): 프로젝트 원문 → 사례 목록 (기획서 F2·5.5·5.6, 작업목록 AI-05)."""
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from typing import Literal

from pydantic import BaseModel, Field

from app import llm, state
from app.config import LLM_MODEL
from app.schemas import Clarification, Episode, Metric, RestructureRequest, RestructureResponse
from app.validators import bad_metric_value, find_span, foreign_han, numbers, unsupported_numbers

router = APIRouter()

PROMPT_VERSION = "p1_v4"
PROMPT = (Path(__file__).parent.parent / "prompts" / f"{PROMPT_VERSION}.md").read_text(encoding="utf-8")


# LLM 출력 스키마. 기본값을 두지 않는다: 기본값이 있으면 JSON 스키마에서 생략 가능이 되고,
# 모델이 role·skills·수치 값을 통째로 건너뛴다(로컬 테스트 2026-10-23). 없으면 ""·[]·null을 직접 쓰게 한다.
class DraftMetric(BaseModel):
    name: str
    before: str | None
    after: str  # 개선 후 값이 없는 성과 항목은 만들지 못하게 (AI #24: 둘 다 null로 비움)
    source: Literal["raw_text", "answer"]


class DraftQuestion(BaseModel):
    field: Literal["role", "problem", "solution", "result"]
    question: str
    example: str | None


class Draft(BaseModel):
    """문자 위치는 LLM이 틀리기 쉬워 원문 인용(quotes)을 받아 source_spans를 직접 계산한다."""
    title: str
    role: str
    skills: list[str]
    problem: str
    solution: str
    result: str
    metrics: list[DraftMetric]
    quotes: list[str]
    clarifications: list[DraftQuestion]


class P1Out(BaseModel):
    episodes: list[Draft] = Field(min_length=1)


def _text(d: Draft) -> str:
    return " ".join([d.title, d.role, d.problem, d.solution, d.result] + [f"{m.name} {m.before} {m.after}" for m in d.metrics])


def _ignored_feedback(out: P1Out, req: RestructureRequest) -> bool:
    """수정 요청(피드백)에 지금 사례 본문을 그대로 돌려줬는지. 확인 질문은 보지 않는다 (AI #25: 질문만 붙이고 본문은 복사)"""
    if req.mode != "episode" or not (req.feedback and req.feedback.strip()) or req.target_episode is None:
        return False
    d, t = out.episodes[0], req.target_episode
    return all(" ".join(getattr(d, f).split()) == " ".join(getattr(t, f).split())
               for f in ("title", "role", "problem", "solution", "result"))


def _problems(out: P1Out, req: RestructureRequest, source: str) -> list[str]:
    problems = []
    if req.mode == "episode" and len(out.episodes) != 1:
        problems.append("episode 모드는 사례 1개만 돌려준다")
    if _ignored_feedback(out, req):
        problems.append(f"feedback '{req.feedback}'을 반영하지 않고 target_episode 본문을 그대로 돌려줬다. "
                        "피드백 방향으로 제목·본문을 원문의 사실로 새로 써라. 원문에 없는 사실이 필요하면 clarifications로 물어라")
    for i, d in enumerate(out.episodes, 1):
        if nums := unsupported_numbers(_text(d), source):
            problems.append(f"사례 {i}: 원문·답변에 없는 수치 {nums}")
        if bad := [q for q in d.quotes if find_span(q, req.source_text) is None]:
            problems.append(f"사례 {i}: quotes가 원문과 글자 그대로 같지 않음 {bad}")
        if bad := [v for m in d.metrics for v in (m.before, m.after) if bad_metric_value(v)]:
            problems.append(f'사례 {i}: metrics의 before·after는 원문 수치를 단위째 쓴 문자열이어야 함 (예: "3.2초") {bad}')
        if han := foreign_han(_text(d), source):
            problems.append(f"사례 {i}: 한국어로 쓸 것. 원문에 없는 글자 {''.join(han)}")
    return problems


def _episode(d: Draft, req: RestructureRequest, source: str) -> Episode:
    spans = sorted({sp for q in d.quotes if (sp := find_span(q, req.source_text))})
    known = {s.lower() for s in req.skills}
    clar = [Clarification(**q.model_dump()) for q in d.clarifications] + [  # 다시 요청해도 남은 수치는 '확인 필요'로 돌려 확정 전에 사용자가 확인하게 한다 (F2)
        Clarification(field=_field_of(d, n), question=f"원문에 없는 수치 '{n}'가 들어 있어요. 맞는 값인지 확인해 주세요.")
        for n in unsupported_numbers(_text(d), source)
    ]
    if han := foreign_han(_text(d), source):  # 다시 요청해도 남은 중국어 글자 → 사용자가 S05에서 직접 고치게
        clar.append(Clarification(field=_field_of(d, han[0]), question=f"원문에 없는 글자 '{''.join(han)}'가 들어 있어요. 고쳐 주세요."))
    for i, c in enumerate(clar, 1):
        c.key = f"q{i}"  # 사례 안에서 겹치지 않게 AI가 붙인다
    metrics = [_sourced(Metric(**m.model_dump()), req) for m in d.metrics  # 다시 요청해도 깨진 값은 표에 보이지 않게 뺀다 (수치는 result 본문에 남음)
               if not (bad_metric_value(m.before) or bad_metric_value(m.after))]
    return Episode(
        title=d.title, role=d.role,
        skills=[s for s in d.skills if s.lower() in known or s.lower() in req.source_text.lower()],  # 없는 기술명 제거
        problem=d.problem, solution=d.solution, result=d.result, metrics=metrics,
        source_spans=spans, clarifications=clar,
    )


def _sourced(m: Metric, req: RestructureRequest) -> Metric:
    """수치 출처는 모델 말 대신 원문에서 확인한다: 원문에 있으면 raw_text, 이번 답변에만 있으면 answer"""
    vals = numbers(f"{m.before or ''} {m.after or ''}")
    if not vals:
        return m
    if vals <= numbers(req.source_text):
        return m.model_copy(update={"source": "raw_text"})
    if req.answer and vals <= numbers(req.source_text + " " + req.answer.answer):
        return m.model_copy(update={"source": "answer"})
    return m  # 어디에도 없는 수치는 unsupported_numbers 가 확인 질문으로 돌린다


def _field_of(d: Draft, n: str) -> str:
    """수치가 들어 있는 필드. 메트릭에만 있으면 result"""
    return next((f for f in ("problem", "solution", "result", "role", "title") if n in getattr(d, f)), "result")


def _stub(req: RestructureRequest) -> RestructureResponse:
    ep = Episode(
        title="(stub) 사례",
        skills=req.skills,
        problem="(stub)",
        solution="(stub)",
        result="(stub)",
        source_spans=[(0, len(req.source_text))],
        clarifications=[Clarification(key="q1", field="result", question="(stub) 성과를 숫자로 확인할 수 있나요?", example="예: 1.2초 → 0.3초")],
    )
    return RestructureResponse(model=LLM_MODEL, prompt_version="stub", episodes=[ep])


@router.post("/restructure", response_model=RestructureResponse)
async def restructure(req: RestructureRequest):
    if state.models["llm"] == "stub":
        return _stub(req)
    if state.models["llm"] != "ready":
        raise HTTPException(503, f"llm {state.models['llm']}")
    source = " ".join([req.source_text] + ([req.answer.answer] if req.answer else []))  # 사실의 기준: 원문(보충 답변 포함) + 이번 답변
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": json.dumps(req.model_dump(), ensure_ascii=False)},
    ]
    if req.mode == "episode" and req.feedback and req.feedback.strip():
        # 요청 JSON 안의 feedback만으로는 target_episode를 그대로 복사했다 (AI #25). 고칠 방향을 따로 한 번 더 준다
        messages.append({"role": "user", "content": f"고칠 방향: {req.feedback}\ntarget_episode를 이 방향으로 다시 써라. "
                         "그대로 복사해 돌려주지 않는다. 원문에 없는 사실·수치는 피드백에 적혀 있어도 넣지 말고 clarifications로 묻는다."})
    out, _ = await llm.generate(messages, P1Out, check=lambda o: _problems(o, req, source), max_tokens=4096)
    asked = {c.question for c in req.target_episode.clarifications} if req.target_episode else set()
    if _ignored_feedback(out, req) and not {q.question for q in out.episodes[0].clarifications} - asked:
        # 다시 요청해도 본문이 같고 새로 물은 것도 없으면 성공으로 넘기지 않는다 → API는 AI_ERROR (AI #25).
        # 새 확인 질문이 있으면 피드백이 원문에 없는 사실을 요구한 경우라 그대로 돌려준다
        raise HTTPException(502, "feedback not applied")
    episodes = [_episode(d, req, source) for d in out.episodes]
    if req.mode == "episode":
        episodes = episodes[:1]
    return RestructureResponse(model=LLM_MODEL, prompt_version=PROMPT_VERSION, episodes=episodes)
