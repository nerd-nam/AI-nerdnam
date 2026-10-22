"""임베딩 모델 로딩과 계산. /embed 와 evals 가 같이 쓴다 (접두어·로딩 규칙을 한 곳에).

텍스트만 임베딩한다. 이미지·음성 인코더는 올리지 않는다.
"""
from app.config import EMBED_DTYPE, EMBED_REPO, EMBED_REVISION


def fmt(purpose: str, text: str, title: str | None = None) -> str:
    # embeddinggemma 접두어 규칙 (ADR-0001). 접두어는 AI 서버만 붙인다.
    # 접두어가 붙어 있으면 sentence-transformers가 문자열을 이미지 URL·경로로 오인하지 않는다
    if purpose == "query":
        return f"task: search result | query: {text}"
    return f"title: {title or 'none'} | text: {text}"


def load(full: bool = False, dtype: str = EMBED_DTYPE, device: str | None = None):
    """기본: 텍스트 인코더만 (가중치 FP32 2.77GB → 1.01GB). full=True는 비교용."""
    import torch
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        EMBED_REPO,
        revision=EMBED_REVISION,
        device=device,
        model_kwargs={"torch_dtype": getattr(torch, dtype)},
        config_kwargs={} if full else {"vision_config": None, "audio_config": None},
    )


def encode(model, purpose: str, items: list[tuple[str, str | None]]) -> list[list[float]]:
    texts = [fmt(purpose, text, title) for text, title in items]
    return model.encode(texts, normalize_embeddings=True).tolist()
