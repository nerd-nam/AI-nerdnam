"""/requirements/extract (P2): 요건 줄 → 줄별 기술명 원문·약어 확장어·최소 연차 (기획서 F4·5.4·5.5·5.6, 작업목록 AI-06·AI-16).
기술명·확장어는 LLM이, 연차는 정규식이 뽑는다. 연차는 표현이 정해져 있어 규칙이 더 정확하고 숫자를 지어낼 수 없다.

기술명과 약어 확장어는 LLM을 따로(동시에) 부른다. 4B에게 한 번에 시키면 기술명을 빠뜨리고(9개 중 8 → 4~6)
약어도 건너뛴다(GGUF Q8_0 확인 2026-10-25, ADR-0001 "요건 약어 확장 실험"). 두 프롬프트를 합쳐 버전 하나(p2_v2)로 본다.

약어 호출은 풀네임을 비워 둘 수 없게 하고(`full` 최소 1개), 이름·여러 뜻인지는 표시만 받아 코드가 버린다.
비워 둘 수 있으면 약어 줄이 1~2개뿐인 요청(실제 공고 대부분)에서 4B가 거의 다 비웠다(한 줄씩 21/28 → 27/28, Mac bf16 2026-10-26)."""
import asyncio
import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import llm, state
from app.config import LLM_MODEL
from app.schemas import ExtractedLine, ExtractRequest, ExtractResponse

router = APIRouter()

PROMPT_VERSION = "p2_v2"
_DIR = Path(__file__).parent.parent / "prompts"
PROMPT = (_DIR / f"{PROMPT_VERSION}.md").read_text(encoding="utf-8")            # 기술명
PROMPT_ABBR = (_DIR / f"{PROMPT_VERSION}_abbr.md").read_text(encoding="utf-8")  # 약어 확장어
MAX_EXPANSIONS = 3  # 줄마다 (기획서 5.5)
# 추출은 답이 하나라 온도 0. 0.7이면 같은 입력에 묶음 하나를 통째로 비우는 등 실행마다 24~26/28로 흔들렸다 (evals/extract_eval)
TEMPERATURE = 0.0

# "3년 이상", "3~5년", "5년+", "3년차", "3+ years". "5년 미만"·"이하"는 최소 연차가 아니다
_YEARS = re.compile(r"(\d{1,2})\s*(?:[~\-]\s*\d{1,2}\s*)?(?:\+\s*)?(?:년|years?)(?!\s*(?:미만|이하))", re.I)
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:/[A-Za-z0-9]+)*")
_NUMERONYM = re.compile(r"[a-z]\d+[a-z]")  # k8s, i18n


def min_years(text: str) -> int | None:
    # ponytail: 숫자 + "년"만 본다. "삼 년" 같은 한글 수사는 못 잡음, 공고에서 나오면 추가
    found = [int(n) for n in _YEARS.findall(text)]
    return min(found) if found else None


def abbr_candidates(text: str) -> list[str]:
    """줄의 약어 후보: 대문자가 절반 이상인 짧은 영문 토큰(MSA, CI/CD, IaC)이나 k8s 같은 줄임말.
    후보 없이 "약어를 찾아 풀어라"만 시키면 4B가 거의 다 비워 둔다(GGUF Q8_0 확인 2026-10-25). 후보를 주고 풀네임만 채우게 한다.
    MySQL·iOS 같은 이름도 후보가 되지만 P2가 is_name으로 표시해 버린다."""
    out = []
    for tok in _TOKEN.findall(text):
        letters = tok.replace("/", "")
        caps = sum(c.isupper() for c in letters)
        if _NUMERONYM.fullmatch(tok) or (caps >= 2 and len(letters) <= 8 and caps * 2 >= len(letters)):
            out.append(tok)
    return list(dict.fromkeys(out))


# LLM 출력. 기본값을 두지 않는다 (생략 가능이 되면 모델이 건너뜀)
class P2Line(BaseModel):
    id: int
    skills_raw: list[str]


class P2Out(BaseModel):
    lines: list[P2Line]


class P2Abbr(BaseModel):
    abbr: str                            # 줄에 쓰인 약어
    full: list[str] = Field(min_length=1)  # 풀네임·같은 뜻의 다른 표기. 비울 수 없다 (비울 수 있으면 4B가 다 비움)
    is_name: bool                        # 약어가 아니라 제품 이름 그 자체(MySQL, iOS) → 버림
    ambiguous: bool                      # 뜻이 여러 개이고 문맥으로 정해지지 않음(PM, SM) → 버림


def _keep(a: P2Abbr) -> bool:
    # k8s·i18n 같은 숫자 줄임말은 이름일 수 없다. 모델이 k8s를 "Kubernetes라는 이름"으로 보고 is_name을 켜기도 한다
    return not a.ambiguous and (not a.is_name or bool(_NUMERONYM.fullmatch(a.abbr)))


class P2AbbrLine(BaseModel):
    id: int
    abbreviations: list[P2Abbr]


class P2AbbrOut(BaseModel):
    lines: list[P2AbbrLine]


def _norm(s: str) -> str:
    return "".join(s.split()).lower()


def _in_text(word: str, text: str) -> bool:
    return bool(_norm(word)) and _norm(word) in _norm(text)


def clean_expansions(line: P2AbbrLine | None, text: str) -> list[str]:
    """이름·여러 뜻이 아닌, 줄에 실제로 있는 약어의 확장어만 모으고(줄에 없는 약어는 지어낸 것), 줄에 이미 있는 말·빈 값·중복(공백·대소문자 무시)을
    버리고 최대 3개 (기획서 5.5 AI 쪽 응답 정리)"""
    out, seen = [], set()
    for s in (f.strip() for a in (line.abbreviations if line else []) if _keep(a) and _in_text(a.abbr, text) for f in a.full):
        if not s or _in_text(s, text) or _norm(s) in seen:
            continue
        seen.add(_norm(s))
        out.append(s)
    return out[:MAX_EXPANSIONS]


def _id_problems(want: list[int], got: list[int]) -> list[str]:
    return [] if sorted(got) == sorted(want) else [f"lines의 id가 입력과 다름: 입력 {want}, 출력 {got}"]


async def _skills(req: ExtractRequest) -> dict[int, P2Line]:
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": json.dumps(req.model_dump(), ensure_ascii=False)},
    ]
    want = [x.id for x in req.lines]
    out, _ = await llm.generate(messages, P2Out, check=lambda o: _id_problems(want, [x.id for x in o.lines]),
                                max_tokens=1024, temperature=TEMPERATURE)
    return {x.id: x for x in out.lines}


async def _abbreviations(req: ExtractRequest) -> dict[int, P2AbbrLine]:
    todo = [{"id": x.id, "text": x.text, "abbr_candidates": c} for x in req.lines if (c := abbr_candidates(x.text))]
    if not todo:  # 약어 후보가 없으면 부르지 않는다
        return {}
    messages = [
        {"role": "system", "content": PROMPT_ABBR},
        {"role": "user", "content": json.dumps({"lines": todo}, ensure_ascii=False)},
    ]
    want = [t["id"] for t in todo]

    def check(o: P2AbbrOut) -> list[str]:
        got = {(x.id, a.abbr) for x in o.lines for a in x.abbreviations}
        missing = [f"{t['id']}:{c}" for t in todo for c in t["abbr_candidates"] if (t["id"], c) not in got]
        return _id_problems(want, [x.id for x in o.lines]) + (
            [f"abbr_candidates {missing}를 abbreviations에 적지 않음. 후보마다 하나씩 적는다"] if missing else [])
    out, _ = await llm.generate(messages, P2AbbrOut, check=check, max_tokens=1024, temperature=TEMPERATURE)
    return {x.id: x for x in out.lines}


@router.post("/requirements/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest):
    if state.models["llm"] == "stub":
        lines = [ExtractedLine(id=x.id, skills_raw=[], expansions=[], min_years=min_years(x.text)) for x in req.lines]
        return ExtractResponse(model=LLM_MODEL, prompt_version="stub", lines=lines)
    if state.models["llm"] != "ready":
        raise HTTPException(503, f"llm {state.models['llm']}")
    skills, abbrs = await asyncio.gather(_skills(req), _abbreviations(req))
    lines = [
        ExtractedLine(
            id=x.id,
            skills_raw=list(dict.fromkeys(
                s.strip() for s in (skills[x.id].skills_raw if x.id in skills else []) if _in_text(s, x.text))),  # 줄에 없는 이름은 버림
            expansions=clean_expansions(abbrs.get(x.id), x.text),
            min_years=min_years(x.text),
        )
        for x in req.lines  # 입력 순서, 빠진 줄은 빈 목록
    ]
    return ExtractResponse(model=LLM_MODEL, prompt_version=PROMPT_VERSION, lines=lines)
