import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import embedding, llm, state
from app.config import AI_MODE
from app.routers import embed, guide, match, requirements, restructure

log = logging.getLogger("uvicorn.error")


async def warmup():
    """모델별로 로딩·예열 요청 후 ready. 켠 직후 첫 요청이 느린 문제(Kaggle 20~24초) 대응.
    한쪽이 실패해도 다른 쪽은 쓸 수 있게 상태를 따로 기록한다."""
    try:
        model = await asyncio.to_thread(embedding.load)
        await asyncio.to_thread(embedding.encode, model, "query", [("warmup", None)])
        state.embedder = model
        state.models["embedding"] = "ready"
    except Exception:
        state.models["embedding"] = "error"
        log.exception("embedding warmup failed")
    try:
        await llm.chat([{"role": "user", "content": "ping"}], max_tokens=8)
        state.models["llm"] = "ready"
    except Exception:
        state.models["llm"] = "error"
        log.exception("llm warmup failed")
    log.info("models %s", state.models)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 로딩은 백그라운드로: 그동안 /health 는 각 모델을 loading 으로 응답한다
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
    # 예: {"status": "ok", "models": {"llm": "ready", "embedding": "ready"}}
    return {"status": "ok", "models": state.models}
