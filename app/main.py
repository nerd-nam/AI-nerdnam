import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import embedding, llm, state
from app.config import AI_MODE
from app.routers import embed, guide, match, requirements, restructure

log = logging.getLogger("uvicorn.error")


async def warmup():
    """임베딩 로딩 → 임베딩·LLM 예열 요청 → ready. 켠 직후 첫 요청이 느린 문제(Kaggle 20~24초) 대응."""
    try:
        model = await asyncio.to_thread(embedding.load)
        await asyncio.to_thread(embedding.encode, model, "query", [("warmup", None)])
        state.embedder = model
        await llm.chat([{"role": "user", "content": "ping"}], max_tokens=8)
        state.models = "ready"
        log.info("models ready")
    except Exception:
        state.models = "error"
        log.exception("model warmup failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 로딩은 백그라운드로: 그동안 /health 는 models=loading 으로 응답한다
    task = asyncio.create_task(warmup()) if AI_MODE == "real" else None
    yield
    if task:
        task.cancel()
    await llm.client.aclose()


app = FastAPI(title="nerdnam-ai", version="0.1.0", lifespan=lifespan)

for r in (restructure, requirements, embed, match, guide):
    app.include_router(r.router)


@app.get("/health")
def health():
    # BE는 models == "ready" 일 때만 '임베딩 대기' 재처리를 시작한다
    return {"status": "ok", "models": state.models}
