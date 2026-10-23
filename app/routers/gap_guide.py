"""/gap-guide (P6, ADR-0003): 미충족·부분 충족 요건 + 인접 경험 → 요건별 가이드·하지 말 것·인용 (작업목록 AI-14).
요건을 한 번에 묶어 부른다(요건당 인접 경험 최대 5개라 짧다). 빠진 요건은 돌려주지 않는다 → API가 기본 문구로 채운다."""
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import llm, state
from app.config import LLM_MODEL
from app.schemas import GapGuide, GapGuideRequest, GapGuideResponse
from app.validators import unsupported_numbers

router = APIRouter()

PROMPT_VERSION = "p6_v2"
PROMPT = (Path(__file__).parent.parent / "prompts" / f"{PROMPT_VERSION}.md").read_text(encoding="utf-8")


class P6Out(BaseModel):
    guides: list[GapGuide]


def _source(req: GapGuideRequest) -> str:
    return " ".join([r.text for r in req.requirements] + [c.text for r in req.requirements for c in r.adjacent] + req.tasks)


def _problems(o: P6Out, req: GapGuideRequest) -> list[str]:
    adj = {r.id: {c.chunk_id for c in r.adjacent} for r in req.requirements}
    problems = []
    if missing := sorted(set(adj) - {g.id for g in o.guides}):
        problems.append(f"요건 {missing}의 가이드가 없음. 입력의 모든 요건을 쓴다")
    for g in o.guides:
        if bad := [i for i in g.cited_chunk_ids if i not in adj.get(g.id, set())]:
            problems.append(f"요건 {g.id}: cited_chunk_ids {bad}가 그 요건의 adjacent에 없음")
        if nums := unsupported_numbers(f"{g.advice} {g.avoid}", _source(req)):
            problems.append(f"요건 {g.id}: 입력에 없는 수치 {nums}")
    return problems


def _stub(req: GapGuideRequest) -> GapGuideResponse:
    guides = [GapGuide(id=r.id, advice="(stub) 경험한 범위를 정확히 밝혀 쓰세요.", avoid="(stub) 경험 범위를 부풀리기",
                       cited_chunk_ids=[c.chunk_id for c in r.adjacent[:1]]) for r in req.requirements]
    return GapGuideResponse(model=LLM_MODEL, prompt_version="stub", guides=guides)


@router.post("/gap-guide", response_model=GapGuideResponse)
async def gap_guide(req: GapGuideRequest):
    if state.models["llm"] == "stub":
        return _stub(req)
    if state.models["llm"] != "ready":
        raise HTTPException(503, f"llm {state.models['llm']}")
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": json.dumps(req.model_dump(), ensure_ascii=False)},
    ]
    o, _ = await llm.generate(messages, P6Out, check=lambda x: _problems(x, req), max_tokens=1536)
    adj = {r.id: {c.chunk_id for c in r.adjacent} for r in req.requirements}
    seen, guides = set(), []
    for g in o.guides:
        if g.id in adj and g.id not in seen:  # 입력에 있는 요건만, 한 번씩
            seen.add(g.id)
            guides.append(g.model_copy(update={"cited_chunk_ids": list(dict.fromkeys(i for i in g.cited_chunk_ids if i in adj[g.id]))}))
    return GapGuideResponse(model=LLM_MODEL, prompt_version=PROMPT_VERSION, guides=guides)
