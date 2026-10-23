"""/requirements/extract (P2): 요건 줄 → 줄별 기술명 원문·최소 연차 (기획서 F4·5.5·5.6, 작업목록 AI-06).
기술명은 LLM이, 연차는 정규식이 뽑는다. 연차는 표현이 정해져 있어 규칙이 더 정확하고 숫자를 지어낼 수 없다."""
import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import llm, state
from app.config import LLM_MODEL
from app.schemas import ExtractedLine, ExtractRequest, ExtractResponse

router = APIRouter()

PROMPT_VERSION = "p2_v1"
PROMPT = (Path(__file__).parent.parent / "prompts" / f"{PROMPT_VERSION}.md").read_text(encoding="utf-8")

# "3년 이상", "3~5년", "5년+", "3년차", "3+ years". "5년 미만"·"이하"는 최소 연차가 아니다
_YEARS = re.compile(r"(\d{1,2})\s*(?:[~\-]\s*\d{1,2}\s*)?(?:\+\s*)?(?:년|years?)(?!\s*(?:미만|이하))", re.I)


def min_years(text: str) -> int | None:
    # ponytail: 숫자 + "년"만 본다. "삼 년" 같은 한글 수사는 못 잡음, 공고에서 나오면 추가
    found = [int(n) for n in _YEARS.findall(text)]
    return min(found) if found else None


class P2Line(BaseModel):  # LLM 출력. 기본값을 두지 않는다 (생략 가능이 되면 모델이 건너뜀)
    id: int
    skills_raw: list[str]


class P2Out(BaseModel):
    lines: list[P2Line]


def _norm(s: str) -> str:
    return "".join(s.split()).lower()


def _in_text(skill: str, text: str) -> bool:
    return bool(_norm(skill)) and _norm(skill) in _norm(text)


def _problems(out: P2Out, req: ExtractRequest) -> list[str]:
    want, got = [x.id for x in req.lines], [x.id for x in out.lines]
    return [] if sorted(got) == sorted(want) else [f"lines의 id가 입력과 다름: 입력 {want}, 출력 {got}"]


@router.post("/requirements/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest):
    if state.models["llm"] == "stub":
        lines = [ExtractedLine(id=x.id, skills_raw=[], min_years=min_years(x.text)) for x in req.lines]
        return ExtractResponse(model=LLM_MODEL, prompt_version="stub", lines=lines)
    if state.models["llm"] != "ready":
        raise HTTPException(503, f"llm {state.models['llm']}")
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": json.dumps(req.model_dump(), ensure_ascii=False)},
    ]
    out, _ = await llm.generate(messages, P2Out, check=lambda o: _problems(o, req), max_tokens=1024)
    by_id = {x.id: x.skills_raw for x in out.lines}
    lines = [
        ExtractedLine(
            id=x.id,
            skills_raw=list(dict.fromkeys(s.strip() for s in by_id.get(x.id, []) if _in_text(s, x.text))),  # 줄에 없는 이름은 버림
            min_years=min_years(x.text),
        )
        for x in req.lines  # 입력 순서, 빠진 줄은 빈 목록
    ]
    return ExtractResponse(model=LLM_MODEL, prompt_version=PROMPT_VERSION, lines=lines)
