"""/match (P3): 요건별 후보 청크(최대 20) → 판정·근거 청크(최대 5)·이유 (기획서 F5·5.6, 작업목록 AI-07).
요건마다 따로 부른다(동시에). 후보 20개 × 요건 여러 개를 한 번에 보내면 문맥 길이(8,192)를 넘을 수 있고,
요건 하나씩 보는 편이 4B 모델의 판정도 안정적이다."""
import asyncio
import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import llm, state
from app.config import LLM_MODEL
from app.schemas import MatchRequest, MatchRequirement, MatchResponse, MatchResult
from app.validators import unsupported_numbers

router = APIRouter()

PROMPT_VERSION = "p3_v2"
PROMPT = (Path(__file__).parent.parent / "prompts" / f"{PROMPT_VERSION}.md").read_text(encoding="utf-8")
MAX_EVIDENCE = 5


class P3Out(BaseModel):
    verdict: Literal["met", "partial", "unmet"]
    evidence_chunk_ids: list[int]
    reason: str


def _source(r: MatchRequirement) -> str:
    return " ".join([r.text] + [c.text for c in r.candidates])


def _problems(o: P3Out, r: MatchRequirement) -> list[str]:
    allowed = {c.chunk_id for c in r.candidates}
    problems = []
    if bad := [i for i in o.evidence_chunk_ids if i not in allowed]:
        problems.append(f"evidence_chunk_ids에 후보에 없는 id {bad}. candidates의 chunk_id만 쓴다")
    if o.verdict != "unmet" and not [i for i in o.evidence_chunk_ids if i in allowed]:
        problems.append(f"verdict={o.verdict}인데 근거가 없음. 근거가 없으면 unmet")
    if nums := unsupported_numbers(o.reason, _source(r)):
        problems.append(f"reason에 후보에 없는 수치 {nums}")
    return problems


async def _judge(r: MatchRequirement) -> MatchResult:
    if not r.candidates:  # 후보가 없으면 모델을 부르지 않는다
        return MatchResult(id=r.id, verdict="unmet", evidence_chunk_ids=[], reason="근거가 될 경험 후보가 없어요.")
    user = {"requirement": r.model_dump(exclude={"candidates"}), "candidates": [c.model_dump() for c in r.candidates]}
    messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]
    o, _ = await llm.generate(messages, P3Out, check=lambda x: _problems(x, r), max_tokens=512)
    allowed = {c.chunk_id for c in r.candidates}
    ids = list(dict.fromkeys(i for i in o.evidence_chunk_ids if i in allowed))[:MAX_EVIDENCE]  # 후보 안, 중복 없이, 5개까지
    verdict = o.verdict if ids else "unmet"  # 근거 없는 충족·부분 충족은 미충족 (BE도 같은 규칙)
    return MatchResult(id=r.id, verdict=verdict, evidence_chunk_ids=ids if verdict != "unmet" else [], reason=o.reason)


def _stub(req: MatchRequest) -> MatchResponse:
    results = [
        MatchResult(id=r.id, verdict="partial" if r.candidates else "unmet",
                    evidence_chunk_ids=[c.chunk_id for c in r.candidates[:1]], reason="(stub)")
        for r in req.requirements
    ]
    return MatchResponse(model=LLM_MODEL, prompt_version="stub", results=results)


@router.post("/match", response_model=MatchResponse)
async def match(req: MatchRequest):
    if state.models["llm"] == "stub":
        return _stub(req)
    if state.models["llm"] != "ready":
        raise HTTPException(503, f"llm {state.models['llm']}")
    results = await asyncio.gather(*(_judge(r) for r in req.requirements))
    return MatchResponse(model=LLM_MODEL, prompt_version=PROMPT_VERSION, results=list(results))
