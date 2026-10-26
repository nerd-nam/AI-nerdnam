"""임베딩 모델 로딩과 계산. /embed 와 evals 가 같이 쓴다 (접두어·로딩 규칙을 한 곳에).

텍스트만 임베딩한다. 이미지·음성 인코더는 올리지 않는다.
"""
import threading
from contextlib import nullcontext

from app.config import EMBED_DTYPE, EMBED_REPO, EMBED_REVISION

_mps_lock = threading.Lock()


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
    # MPS(Mac GPU)는 여러 스레드가 같은 모델을 동시에 쓰면 프로세스가 죽는다(SIGSEGV, 이슈 #22).
    # /embed 는 요청마다 다른 스레드에서 돌므로 MPS일 때만 한 번에 하나씩 계산한다. CUDA·CPU는 그대로
    with _mps_lock if model.device.type == "mps" else nullcontext():
        return model.encode(texts, normalize_embeddings=True).tolist()
