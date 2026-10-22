from fastapi import APIRouter, HTTPException

from app import embedding, state
from app.config import EMBED_DIM, EMBED_MODEL
from app.schemas import EmbedRequest, EmbedResponse

router = APIRouter()


@router.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):  # 동기 함수: 계산을 스레드풀에서 돌려 이벤트 루프를 막지 않음
    if state.models == "stub":
        # 가짜 벡터라 embedding_model="stub" — BE는 이 값이면 벡터를 저장하지 않는다
        unit = [1.0] + [0.0] * (EMBED_DIM - 1)
        return EmbedResponse(embedding_model="stub", dim=EMBED_DIM, vectors=[unit for _ in req.items])
    if state.embedder is None:
        raise HTTPException(503, f"models {state.models}")  # 로딩 중·실패. 오류 코드 규칙은 D-7에서 확정
    vectors = embedding.encode(state.embedder, req.purpose, [(i.text, i.title) for i in req.items])
    return EmbedResponse(embedding_model=EMBED_MODEL, dim=EMBED_DIM, vectors=vectors)
