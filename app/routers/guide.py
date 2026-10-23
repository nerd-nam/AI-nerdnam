from fastapi import APIRouter

from app.config import LLM_MODEL
from app.schemas import GuideRequest, GuideResponse

router = APIRouter()


@router.post("/guide", response_model=GuideResponse)
def guide(req: GuideRequest):
    # TODO(AI-08~10): P4·P5 프롬프트. 지금은 첫 근거를 인용하는 가짜 응답
    cited = [c.chunk_id for r in req.requirements for c in r.evidence][:1]
    content = {"text": f"(stub) {req.tab} 방향성", "sources": cited}
    return GuideResponse(model=LLM_MODEL, prompt_version="stub", content=content, cited_chunk_ids=cited)
