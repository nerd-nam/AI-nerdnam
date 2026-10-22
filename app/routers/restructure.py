from fastapi import APIRouter

from app.config import LLM_MODEL
from app.schemas import Episode, RestructureRequest, RestructureResponse

router = APIRouter()


@router.post("/restructure", response_model=RestructureResponse)
def restructure(req: RestructureRequest):
    # TODO(AI-05): P1 프롬프트로 LLM 호출. 지금은 원문 전체를 사례 1개로 돌려주는 가짜 응답
    ep = Episode(
        episode_id=req.target_episode_id,
        title="(stub) 사례",
        skills=req.skills,
        problem="(stub)",
        solution="(stub)",
        result="(stub)",
        source_spans=[(0, len(req.raw_text))],
        questions=["(stub) 성과를 숫자로 확인할 수 있나요?"],
    )
    return RestructureResponse(model=LLM_MODEL, prompt_version="stub", episodes=[ep])
