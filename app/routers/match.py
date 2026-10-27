"""/match (P3): 요건별 후보 청크(최대 20) → 판정·근거 청크(최대 5)·이유 (기획서 F5·5.6, 작업목록 AI-07).
요건을 BATCH개씩 묶어 한 번에 부른다(묶음끼리는 동시에). 요건마다 후보 20개를 받지만 사용자 청크가 수십 개라
요건들의 후보가 거의 겹친다. 요건마다 따로 부르면 같은 청크·프롬프트를 요건 수만큼 보내(입력 9~10배) 호출 한도에 걸렸다.
묶음 안에서는 후보를 한 번만 보내고, 요건마다 볼 수 있는 후보 id(candidate_ids)를 붙여 근거 범위를 지킨다."""
import asyncio
import json
import re
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import llm, state
from app.config import LLM_MODEL
from app.schemas import MatchRequest, MatchRequirement, MatchResponse, MatchResult
from app.validators import unsupported_numbers

router = APIRouter()

PROMPT_VERSION = "p3_v4"
# 요건 묶음 크기. 크면 입력이 더 줄지만 출력이 길어져 읽기 제한(30초)에 가까워지고, 형식 오류 때 다시 요청할 양도 커진다
BATCH = 10
PROMPT = (Path(__file__).parent.parent / "prompts" / f"{PROMPT_VERSION}.md").read_text(encoding="utf-8")
MAX_EVIDENCE = 5
# 판정은 답이 하나라 온도 0 (P2와 같은 이유). 기본 설정이면 같은 입력에서 실행마다 일치율이 81~87%로 흔들려 프롬프트를 비교할 수 없었고, 온도 0에서 87% → 91% (evals/match_eval, Mac bf16)
TEMPERATURE = 0.0
# reason은 S08에 그대로 보이는데 4B가 "청크 32에서"처럼 내부 id를 자주 쓴다(70줄 중 12~14줄).
# 프롬프트로 막으면 판정 일치율이 떨어지고(91% → 86%) 재요청으로도 안 고쳐져, 응답 정리에서 "관련 경험"으로 바꾼다
# ponytail: "청크 25(…), 31(…)"처럼 번호마다 설명이 붙으면 첫 번호만 바뀐다. 자주 보이면 규칙을 넓힌다
_CHUNK_REF = re.compile(r"(?:후보\s*)?(?:청크|chunk(?:_id)?)\s*[(（]?\s*(?:chunk_id\s*[:：]\s*)?\d+(?:\s*(?:,|과|와|및)\s*\d+)*\s*[)）]?\s*(?=\S)", re.I)


def clean_reason(reason: str) -> str:
    return _CHUNK_REF.sub("관련 경험", reason)


class P3Item(BaseModel):
    id: int  # 요건 id
    verdict: Literal["met", "partial", "unmet"]
    evidence_chunk_ids: list[int]
    reason: str


class P3Out(BaseModel):
    results: list[P3Item]


def _source(r: MatchRequirement) -> str:
    return " ".join([r.text] + [c.text for c in r.candidates])


def _item_problems(x: P3Item, r: MatchRequirement) -> list[str]:
    allowed = {c.chunk_id for c in r.candidates}
    problems = []
    if bad := [i for i in x.evidence_chunk_ids if i not in allowed]:
        problems.append(f"evidence_chunk_ids에 그 요건의 candidate_ids에 없는 id {bad}")
    if x.verdict != "unmet" and not [i for i in x.evidence_chunk_ids if i in allowed]:
        problems.append(f"verdict={x.verdict}인데 근거가 없음. 근거가 없으면 unmet")
    if nums := unsupported_numbers(clean_reason(x.reason), _source(r)):
        problems.append(f"reason에 그 요건의 후보에 없는 수치 {nums}")
    return problems


def _problems(o: P3Out, reqs: list[MatchRequirement]) -> list[str]:
    by = {r.id: r for r in reqs}
    problems = []
    if missing := sorted(set(by) - {x.id for x in o.results}):
        problems.append(f"요건 {missing}의 결과가 없음. 입력의 모든 요건에 결과를 하나씩 쓴다")
    for x in o.results:
        if x.id in by:
            problems += [f"요건 {x.id}: {p}" for p in _item_problems(x, by[x.id])]
    return problems


# ADR-0006 B10: 기술 목록에 학습 중뿐이면 partial. 모델(Nova 2 Lite)이 자주 unmet으로 내려(평가 70줄 중 묶음 8줄, 요건별 5줄)
# 판정 뒤 코드로 올린다. 기술 청크 글은 "기술명 · 카테고리 · 숙련도 · 사용처"(기획서 4.4). 숙련도는 이름 또는 기준 문장
_LEARNING = ("학습 중", "공부했거나 개인 프로젝트에서 써 봤")


def _learning_skill(r: MatchRequirement):
    """요건의 기술이 후보 기술 목록에 학습 중으로 있으면 (청크, 기술명)"""
    skills = {s.lower() for s in r.skills}
    for c in r.candidates:
        if c.chunk_type == "skill" and any(m in c.text for m in _LEARNING):
            name = c.text.split("·")[0].strip()
            if name.lower() in skills or re.search(rf"(?<![a-z0-9]){re.escape(name.lower())}(?![a-z0-9])", r.text.lower()):
                return c, name
    return None


def _result(x: P3Item, r: MatchRequirement) -> MatchResult:
    allowed = {c.chunk_id for c in r.candidates}
    ids = list(dict.fromkeys(i for i in x.evidence_chunk_ids if i in allowed))[:MAX_EVIDENCE]  # 그 요건 후보 안, 중복 없이, 5개까지
    verdict = x.verdict if ids else "unmet"  # 근거 없는 충족·부분 충족은 미충족 (BE도 같은 규칙)
    if verdict == "unmet" and (hit := _learning_skill(r)):
        return MatchResult(id=r.id, verdict="partial", evidence_chunk_ids=[hit[0].chunk_id],
                           reason=f"기술 목록에 {hit[1]}(학습 중)만 있고 실무에서 쓴 경험은 보이지 않아요. 경험이나 프로젝트로 보완해야 해요.")
    return MatchResult(id=r.id, verdict=verdict, evidence_chunk_ids=ids if verdict != "unmet" else [], reason=clean_reason(x.reason))


async def _judge(reqs: list[MatchRequirement]) -> dict[int, MatchResult]:
    """요건 묶음 1회 호출. 후보는 chunk_id로 중복을 없애 한 번만 보낸다. 빠진 요건은 돌려주지 않는다(호출한 쪽이 처리)"""
    pool = {c.chunk_id: c for r in reqs for c in r.candidates}
    user = {"candidates": [c.model_dump() for c in pool.values()],
            "requirements": [{**r.model_dump(exclude={"candidates"}), "candidate_ids": [c.chunk_id for c in r.candidates]}
                             for r in reqs]}
    messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]
    o, _ = await llm.generate(messages, P3Out, check=lambda x: _problems(x, reqs), max_tokens=160 * len(reqs) + 256,
                              temperature=TEMPERATURE)
    by = {r.id: r for r in reqs}
    out: dict[int, MatchResult] = {}
    for x in o.results:
        if x.id in by and x.id not in out:  # 입력에 있는 요건만, 한 번씩
            out[x.id] = _result(x, by[x.id])
    return out


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
    judged = [r for r in req.requirements if r.candidates]  # 후보가 없으면 모델을 부르지 않는다
    done: dict[int, MatchResult] = {}
    for d in await asyncio.gather(*(_judge(judged[i:i + BATCH]) for i in range(0, len(judged), BATCH))):
        done.update(d)
    if missing := [r for r in judged if r.id not in done]:  # 다시 요청해도 묶음 응답에서 빠진 요건은 하나씩 (드묾)
        for d in await asyncio.gather(*(_judge([r]) for r in missing)):
            done.update(d)
    if any(r.id not in done for r in judged):
        raise HTTPException(502, "llm output invalid")  # API는 AI_ERROR
    results = [done.get(r.id) or MatchResult(id=r.id, verdict="unmet", evidence_chunk_ids=[], reason="근거가 될 경험 후보가 없어요.")
               for r in req.requirements]
    return MatchResponse(model=LLM_MODEL, prompt_version=PROMPT_VERSION, results=results)
