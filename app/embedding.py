"""Titan Text Embeddings V2 호출 (Bedrock InvokeModel). /embed 와 evals 가 같이 쓴다.

Titan은 한 번에 문장 하나만 받으므로 항목마다 병렬로 부르고 순서대로 돌려준다.
질의·문서 접두어 규칙이 없어 문서는 제목을 본문 앞에 붙이기만 한다.
"""
from concurrent.futures import ThreadPoolExecutor

import httpx

from app.config import (
    BEDROCK_API_KEY,
    BEDROCK_URL,
    EMBED_DIM,
    EMBED_MODEL_ID,
    LLM_TIMEOUT,
)

client = httpx.Client(base_url=BEDROCK_URL, headers={"Authorization": f"Bearer {BEDROCK_API_KEY}"},
                      timeout=httpx.Timeout(min(LLM_TIMEOUT, 10), connect=2))
# ponytail: 동시 8건 고정. 호출 한도(분당 요청 수)에 걸리면 줄이고, 처리량이 모자라면 한도 상향 요청
_pool = ThreadPoolExecutor(8)


def text_of(purpose: str, text: str, title: str | None = None) -> str:
    return f"{title}\n{text}" if purpose == "document" and title else text


def _one(text: str) -> list[float]:
    r = client.post(f"/model/{EMBED_MODEL_ID}/invoke", json={"inputText": text, "dimensions": EMBED_DIM, "normalize": True})
    r.raise_for_status()
    return r.json()["embedding"]


def encode(purpose: str, items: list[tuple[str, str | None]]) -> list[list[float]]:
    return list(_pool.map(_one, [text_of(purpose, text, title) for text, title in items]))
