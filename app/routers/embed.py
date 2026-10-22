from fastapi import APIRouter

from app.config import EMBED_DIM
from app.schemas import EmbedRequest, EmbedResponse

router = APIRouter()


def with_prefix(purpose: str, text: str, title: str | None) -> str:
    # embeddinggemma 접두어 규칙 (ADR-0001). 접두어는 AI 서버만 붙인다
    if purpose == "query":
        return f"task: search result | query: {text}"
    return f"title: {title or 'none'} | text: {text}"


@router.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    texts = [with_prefix(req.purpose, i.text, i.title) for i in req.items]
    # TODO(AI-03): 임베딩 모델 호출 후 model=EMBED_MODEL 로 돌려준다.
    # 지금은 가짜 벡터라 model="stub" — BE는 이 값이면 벡터를 저장하지 않는다
    unit = [1.0] + [0.0] * (EMBED_DIM - 1)
    return EmbedResponse(model="stub", dim=EMBED_DIM, vectors=[unit for _ in texts])
