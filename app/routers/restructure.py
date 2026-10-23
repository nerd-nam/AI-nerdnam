"""/restructure (P1): 프로젝트 원문 → 사례 목록 (기획서 F2·5.5·5.6, 작업목록 AI-05)."""
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import llm, state
from app.config import LLM_MODEL
from app.schemas import Episode, Metric, RestructureRequest, RestructureResponse
from app.validators import unsupported_numbers

router = APIRouter()

PROMPT_VERSION = "p1_v1"
PROMPT = (Path(__file__).parent.parent / "prompts" / f"{PROMPT_VERSION}.md").read_text(encoding="utf-8")


class Draft(BaseModel):
    """LLM 출력. 문자 위치는 LLM이 틀리기 쉬워 원문 인용(quotes)을 받아 source_spans를 직접 계산한다."""
    title: str
    role: str = ""
    skills: list[str] = []
    problem: str
    solution: str
    result: str
    metrics: list[Metric] = []
    quotes: list[str] = []
    questions: list[str] = []


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
        if bad := [q for q in d.quotes if q not in req.raw_text]:
            problems.append(f"사례 {i}: quotes가 원문과 글자 그대로 같지 않음 {bad}")
    return problems


def _episode(d: Draft, req: RestructureRequest, source: str) -> Episode:
    spans = sorted((s, s + len(q)) for q in d.quotes if (s := req.raw_text.find(q)) >= 0)
    known = {s.lower() for s in req.skills}
    questions = d.questions + [  # 다시 요청해도 남은 수치는 '확인 필요'로 돌려 확정 전에 사용자가 확인하게 한다 (F2)
        f"원문에 없는 수치 '{n}'가 들어 있어요. 맞는 값인지 확인해 주세요." for n in unsupported_numbers(_text(d), source)
    ]
    return Episode(
        episode_id=req.target_episode_id if req.mode == "episode" else None,
        title=d.title, role=d.role,
        skills=[s for s in d.skills if s.lower() in known or s.lower() in req.raw_text.lower()],  # 없는 기술명 제거
        problem=d.problem, solution=d.solution, result=d.result, metrics=d.metrics,
        source_spans=spans, questions=questions,
    )


def _stub(req: RestructureRequest) -> RestructureResponse:
    ep = Episode(
        episode_id=req.target_episode_id,
        title="(stub) 사례",
        skills=req.skills,
        problem="(stub)",
        solution="(stub)",
        result="(stub)",
        source_spans=[(0, len(req.raw_text))],
        questions=["(stub) 성과를 숫자로 확인할 수 있나요?"],
    )
    return RestructureResponse(model=LLM_MODEL, prompt_version="stub", episodes=[ep])


@router.post("/restructure", response_model=RestructureResponse)
async def restructure(req: RestructureRequest):
    if state.models["llm"] == "stub":
        return _stub(req)
    if state.models["llm"] != "ready":
        raise HTTPException(503, f"llm {state.models['llm']}")
    source = " ".join([req.raw_text] + [a.answer for a in req.answers])  # 사실의 기준: 원문 + 확인 질문 답변
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": json.dumps(req.model_dump(), ensure_ascii=False)},
    ]
    out, _ = await llm.generate(messages, P1Out, check=lambda o: _problems(o, req, source), max_tokens=4096)
    episodes = [_episode(d, req, source) for d in out.episodes]
    if req.mode == "episode":
        episodes = episodes[:1]
    return RestructureResponse(model=LLM_MODEL, prompt_version=PROMPT_VERSION, episodes=episodes)
