from fastapi import APIRouter

from app.config import LLM_MODEL
from app.schemas import MatchRequest, MatchResponse, MatchResult

router = APIRouter()


@router.post("/match", response_model=MatchResponse)
def match(req: MatchRequest):
    # TODO(AI-07): 리랭킹(20 → 5) + P3 판정. 지금은 후보 첫 번째를 근거로 '부분 충족'
    results = [
        MatchResult(
            id=r.id,
            verdict="partial" if r.candidates else "unmet",
            evidence_chunk_ids=[c.chunk_id for c in r.candidates[:1]],
            reason="(stub)",
        )
        for r in req.requirements
    ]
    return MatchResponse(model=LLM_MODEL, prompt_version="stub", results=results)
