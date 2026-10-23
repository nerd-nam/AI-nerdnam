from fastapi import APIRouter

from app.config import LLM_MODEL
from app.schemas import (CareerContent, Coverage, GuideRequest, GuideResponse, ImageContent, ImagePrompt,
                         IntroContent, MotiveContent, Page, Paragraph, PortfolioContent, Rewrite, ToolVisual)

router = APIRouter()


def _stub_content(req: GuideRequest, cited: list[int]):
    """탭별 content 모양 그대로의 가짜 응답. FE·BE가 GPU 없이 개발할 때 쓴다 (AI 이슈 #9)"""
    common = dict(
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
            kinds = ["cover", "text", "diagram", "chart"]
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
def guide(req: GuideRequest):
    # TODO(AI-08~10): P4·P5 프롬프트. 지금은 탭별 모양의 가짜 응답, 첫 근거를 인용
    cited = [c.chunk_id for r in req.requirements for c in r.evidence][:1]
    content = _stub_content(req, cited)
    return GuideResponse(model=LLM_MODEL, prompt_version="stub", content=content,
                         cited_chunk_ids=cited if req.tab != "image" else [])
