from fastapi import APIRouter

from app.config import LLM_MODEL
from app.schemas import GapGuideRequest, GapGuideResponse, GapGuideResult

router = APIRouter()


@router.post("/gap-guide", response_model=GapGuideResponse)
def gap_guide(req: GapGuideRequest):
    # TODO(AI-14): P6 프롬프트로 생성 + 공통 검증기(AI-04). 지금은 인접 경험 첫 번째를 인용하는 가짜 응답
    results = [
        GapGuideResult(
            requirement_id=r.id,
            guide="(stub) 경험한 범위를 정확히 밝혀 쓰세요.",
            avoid="(stub) 경험 범위를 부풀리기",
            evidence_chunk_ids=[c.chunk_id for c in r.adjacent[:1]],
        )
        for r in req.requirements
    ]
    return GapGuideResponse(model=LLM_MODEL, prompt_version="stub", results=results)
