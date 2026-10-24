import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from app import embedding, llm, state
from app.config import AI_MODE
from app.routers import embed, gap_guide, guide, match, requirements, restructure
from app.schemas import HealthResponse

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


DESCRIPTION = """API ↔ AI 계약 (C-04). 출처는 AI 레포 `app/schemas.py`, 규칙은 SPEC 기획서 5.5 "AI 서버 공통 규칙".

- 모든 생성 응답에 `model`·`prompt_version`. `prompt_version`이 `"stub"`이면 가짜 응답(저장 안 함), `/embed`는 `embedding_model`이 `"stub"`.
- 실패: 연결 실패·연결 시간 초과·503 → `GPU_UNAVAILABLE`(1회 재시도), 읽기 시간 초과 → `TIMEOUT`(재시도 안 함), AI 서버의 504 → `TIMEOUT`(재시도 안 함), 422 → `AI_ERROR`(재시도 안 함), 그 밖 5xx → `AI_ERROR`(1회 재시도).
- 타임아웃: 연결 2초. 읽기 `/health` 2초 · `/embed` 10초 · `/requirements/extract` 20초 · `/restructure` 30초 · `/match` 30초 · `/gap-guide` 20초 · `/guide` 120초.
- `/guide`는 탭마다 1회 호출, JSON 응답(SSE 아님).
"""

app = FastAPI(title="nerdnam-ai", version="0.2.1", description=DESCRIPTION, lifespan=lifespan)

# 503: 모델이 꺼짐·로딩 중 → API는 GPU_UNAVAILABLE. 502: LLM 출력 형식이 두 번 틀림 → AI_ERROR (기획서 5.5)
_503: dict[int | str, dict[str, Any]] = {503: {"description": "모델 준비 안 됨 (GPU_UNAVAILABLE)"}}
_502: dict[int | str, dict[str, Any]] = {502: {"description": "LLM 출력 형식 오류 (AI_ERROR)"}, 504: {"description": "LLM 생성 시간 초과 (TIMEOUT, 재시도 안 함)"}}
app.include_router(embed.router, responses=_503)
for r in (restructure, requirements, match, gap_guide, guide):
    app.include_router(r.router, responses=_503 | _502)


@app.get("/health", response_model=HealthResponse)
def health():
    # 예: {"status": "ok", "models": {"llm": "ready", "embedding": "ready"}}
    return {"status": "ok", "models": state.models}
