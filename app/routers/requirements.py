from fastapi import APIRouter

from app.config import LLM_MODEL
from app.schemas import ExtractedLine, ExtractRequest, ExtractResponse

router = APIRouter()


@router.post("/requirements/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    # TODO(AI-06): P2 프롬프트로 추출. 지금은 빈 결과
    lines = [ExtractedLine(id=line.id, skills_raw=[]) for line in req.lines]
    return ExtractResponse(model=LLM_MODEL, prompt_version="stub", lines=lines)
