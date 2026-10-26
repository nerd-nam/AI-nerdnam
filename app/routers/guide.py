"""/guide (P4·P5): 판정·근거·갭 가이드·공고 → 탭 하나의 작성 방향 (기획서 F5·F6·5.6, 작업목록 AI-08~10).
탭마다 1회, 그 탭의 content 스키마로 생성한다. 이미지 탭은 P5, 나머지는 P4."""
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from app import llm, state
from app.config import LLM_MODEL
from app.schemas import (CareerContent, Coverage, GuideRequest, GuideResponse, ImageContent, ImagePrompt,
                         IntroContent, MotiveContent, Page, PageKind, Paragraph, PortfolioContent, Rewrite, ToolVisual)
from app.validators import unsupported_numbers

router = APIRouter()

_PROMPTS = {v: (Path(__file__).parent.parent / "prompts" / f"{v}.md").read_text(encoding="utf-8") for v in ("p4_v2", "p5_v2")}
_MODEL = {"motive": MotiveContent, "intro": IntroContent, "portfolio": PortfolioContent, "career": CareerContent,
          "image": ImageContent}
_NO_NUMBER_CHECK = {"tab", "kind", "ratio", "prompt", "negative"}  # 값이 정해진 칸, 영어 이미지 프롬프트


MAX_TOKENS = 3072


def _payload(req: GuideRequest, per_req: int | None = None) -> dict:
    """모델에 보내는 입력. 같은 근거 청크가 여러 요건에 반복되므로(fixtures: 17번 중 서로 다른 청크 9개)
    청크는 `evidence`에 한 번씩, 요건에는 `evidence_ids`만 둔다. per_req가 있으면 요건별 근거를 앞에서 그만큼만"""
    pool, reqs = {}, []
    for r in req.requirements:
        ev = r.evidence[:per_req] if per_req else r.evidence
        for c in ev:
            pool.setdefault(c.chunk_id, c.model_dump())
        reqs.append(r.model_dump(exclude={"evidence"}) | {"evidence_ids": [c.chunk_id for c in ev]})
    head = req.model_dump(exclude={"requirements"})
    return head | {"requirements": reqs, "evidence": list(pool.values())}


def _version(tab: str) -> str:
    return "p5_v2" if tab == "image" else "p4_v2"


def _walk(x, key=None):
    """content 안의 (키, 값)을 모두 돈다"""
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _walk(v, k)
    elif isinstance(x, list):
        for v in x:
            yield from _walk(v, key)
    else:
        yield key, x


def _sources(content: dict) -> list[int]:
    return [v for k, v in _walk(content) if k == "sources"]


def _texts(content: dict) -> str:
    return " ".join(v for k, v in _walk(content) if isinstance(v, str) and k not in _NO_NUMBER_CHECK)


def _source_text(req: GuideRequest) -> str:
    parts = [req.company, req.position, req.company_intro, *req.tasks]
    for r in req.requirements:
        parts += [r.text, r.reason, *(c.text for c in r.evidence)]
        if r.gap_guide:
            parts += [r.gap_guide.advice, r.gap_guide.avoid]
    return " ".join(parts)


def _problems(o, req: GuideRequest) -> list[str]:
    content = o.model_dump()
    allowed = {c.chunk_id for r in req.requirements for c in r.evidence}
    ids = {r.id for r in req.requirements}
    problems = []
    if bad := sorted({i for i in _sources(content) if i not in allowed}):
        problems.append(f"sources에 evidence에 없는 chunk_id {bad}")
    if req.tab != "image" and (nums := unsupported_numbers(_texts(content), _source_text(req))):
        problems.append(f"근거·공고에 없는 수치 {nums}")
    if req.tab == "intro" and (missing := sorted(ids - {c["requirement_id"] for c in content["coverage"]})):
        problems.append(f"coverage에 요건 {missing}이 없음. 모든 요건을 쓴다")
    return problems


def _clean(o, req: GuideRequest):
    """검증 뒤 정리: 근거 밖 sources·요건 id 버림, 쪽 번호 다시 매김"""
    allowed = {c.chunk_id for r in req.requirements for c in r.evidence}
    ids = {r.id for r in req.requirements}
    d = o.model_dump()

    def keep(x):
        if isinstance(x, dict):
            if "sources" in x:
                x["sources"] = list(dict.fromkeys(i for i in x["sources"] if i in allowed))
            for v in x.values():
                keep(v)
        elif isinstance(x, list):
            for v in x:
                keep(v)
    keep(d)
    if req.tab == "intro":
        d["coverage"] = [c for c in d["coverage"] if c["requirement_id"] in ids]
    if req.tab == "career":
        for rw in d["rewrites"]:
            if rw["requirement_id"] not in ids:
                rw["requirement_id"] = None
    if req.tab == "portfolio":
        for n, page in enumerate(sorted(d["pages"], key=lambda x: x["order"]), 1):
            page["order"] = n
        d["pages"].sort(key=lambda x: x["order"])
    return type(o).model_validate(d)


def _stub_content(req: GuideRequest, cited: list[int]):
    """탭별 content 모양 그대로의 가짜 응답. FE·BE가 GPU 없이 개발할 때 쓴다 (AI 이슈 #9)"""
    common: dict[str, Any] = dict(
        tab=req.tab,
        message=f"(stub) {req.tab} 핵심 메시지",
        outline=[Paragraph(title="(stub) 문단 1", detail="(stub) 근거 경험으로 쓰기", sources=cited)],
        keywords=["(stub) 키워드"],
        avoid=["(stub) 피할 표현"],
    )
    reqs = req.requirements
    match req.tab:
        case "motive":
            return MotiveContent(**common)
        case "intro":
            return IntroContent(**common, coverage=[
                Coverage(requirement_id=r.id, covered=r.verdict != "unmet", where="(stub) 문단 1") for r in reqs])
        case "portfolio":
            kinds: list[PageKind] = ["cover", "text", "diagram", "chart"]
            return PortfolioContent(**common, pages=[
                Page(order=i, title=f"(stub) {i}쪽", hint="(stub) 내용", kind=k, sources=cited if k != "cover" else [])
                for i, k in enumerate(kinds, 1)])
        case "career":
            return CareerContent(**common, rewrites=[
                Rewrite(before="(stub) 지금 표현", after=f"(stub) {reqs[0].text}에 맞춘 표현", requirement_id=reqs[0].id)])
        case "image":
            return ImageContent(tab="image",
                prompts=[ImagePrompt(title="(stub) 커버 이미지", purpose="포트폴리오 커버", page=1, ratio="16:9",
                                     prompt="(stub) minimal flat tech illustration, no text", negative="text, logos")],
                tool_visuals=[ToolVisual(name="(stub) 구조 도식", page=3, kind="diagram")])


@router.post("/guide", response_model=GuideResponse)
async def guide(req: GuideRequest):
    if state.models["llm"] == "stub":
        cited = [c.chunk_id for r in req.requirements for c in r.evidence][:1]
        return GuideResponse(model=LLM_MODEL, prompt_version="stub", content=_stub_content(req, cited),
                             cited_chunk_ids=cited if req.tab != "image" else [])
    if state.models["llm"] != "ready":
        raise HTTPException(503, f"llm {state.models['llm']}")
    version = _version(req.tab)
    for per_req in (None, 3, 1):  # 문맥 한도를 넘으면 요건별 근거를 앞에서 3개, 1개로 줄인다
        messages = [
            {"role": "system", "content": _PROMPTS[version]},
            {"role": "user", "content": json.dumps(_payload(req, per_req), ensure_ascii=False)},
        ]
        if llm.fits(messages, MAX_TOKENS):
            break
    else:
        raise HTTPException(422, "요건·근거가 많아 문맥 한도를 넘음")  # API 쪽에서 요건 수를 줄여야 함, 재시도 안 함
    o, _ = await llm.generate(messages, _MODEL[req.tab], check=lambda x: _problems(x, req), max_tokens=MAX_TOKENS)
    content = _clean(o, req)
    cited = list(dict.fromkeys(_sources(content.model_dump())))
    return GuideResponse(model=LLM_MODEL, prompt_version=version, content=content, cited_chunk_ids=cited)
