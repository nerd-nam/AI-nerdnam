"""/restructure (P1): 프로젝트 원문 → 사례 목록 (기획서 F2·5.5·5.6, 작업목록 AI-05)."""
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from typing import Literal

from pydantic import BaseModel, Field

from app import llm, state
from app.config import LLM_MODEL
from app.schemas import Clarification, Episode, Metric, RestructureRequest, RestructureResponse
from app.validators import find_span, numbers, unsupported_numbers

router = APIRouter()

PROMPT_VERSION = "p1_v2"
PROMPT = (Path(__file__).parent.parent / "prompts" / f"{PROMPT_VERSION}.md").read_text(encoding="utf-8")


# LLM 출력 스키마. 기본값을 두지 않는다: 기본값이 있으면 JSON 스키마에서 생략 가능이 되고,
# 모델이 role·skills·수치 값을 통째로 건너뛴다(로컬 테스트 2026-10-23). 없으면 ""·[]·null을 직접 쓰게 한다.
class DraftMetric(BaseModel):
    name: str
    before: str | None
    after: str | None
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


def _problems(out: P1Out, req: RestructureRequest, source: str) -> list[str]:
    problems = []
    if req.mode == "episode" and len(out.episodes) != 1:
        problems.append("episode 모드는 사례 1개만 돌려준다")
    for i, d in enumerate(out.episodes, 1):
        if nums := unsupported_numbers(_text(d), source):
            problems.append(f"사례 {i}: 원문·답변에 없는 수치 {nums}")
        if bad := [q for q in d.quotes if find_span(q, req.source_text) is None]:
            problems.append(f"사례 {i}: quotes가 원문과 글자 그대로 같지 않음 {bad}")
    return problems


def _episode(d: Draft, req: RestructureRequest, source: str) -> Episode:
    spans = sorted({sp for q in d.quotes if (sp := find_span(q, req.source_text))})
    known = {s.lower() for s in req.skills}
    clar = [Clarification(**q.model_dump()) for q in d.clarifications] + [  # 다시 요청해도 남은 수치는 '확인 필요'로 돌려 확정 전에 사용자가 확인하게 한다 (F2)
        Clarification(field=_field_of(d, n), question=f"원문에 없는 수치 '{n}'가 들어 있어요. 맞는 값인지 확인해 주세요.")
        for n in unsupported_numbers(_text(d), source)
    ]
    for i, c in enumerate(clar, 1):
        c.key = f"q{i}"  # 사례 안에서 겹치지 않게 AI가 붙인다
    return Episode(
        title=d.title, role=d.role,
        skills=[s for s in d.skills if s.lower() in known or s.lower() in req.source_text.lower()],  # 없는 기술명 제거
        problem=d.problem, solution=d.solution, result=d.result, metrics=[_sourced(Metric(**m.model_dump()), req) for m in d.metrics],
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
    out, _ = await llm.generate(messages, P1Out, check=lambda o: _problems(o, req, source), max_tokens=4096)
    episodes = [_episode(d, req, source) for d in out.episodes]
    if req.mode == "episode":
        episodes = episodes[:1]
    return RestructureResponse(model=LLM_MODEL, prompt_version=PROMPT_VERSION, episodes=episodes)
