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


# Bedrock은 새 JSON 스키마를 처음 받을 때 문법을 만드느라 최대 90초 걸리고 24시간 캐시한다.
# BE 읽기 제한(20~30초)보다 길어 사용자 요청 전에 출력 스키마마다 한 번씩 미리 보낸다
OUTPUT_MODELS = [restructure.P1Out, requirements.P2Out, requirements.P2AbbrOut, match.P3Out, gap_guide.P6Out,
                 *guide._MODEL.values()]
REWARM_SECONDS = 12 * 3600  # ponytail: 고정 주기 재예열 (캐시 24시간). 만료 시점이 드러나면 그때 맞춘다


async def _prime(model):
    msg = [{"role": "user", "content": "예열 요청. 짧은 예시 JSON 하나."}]
    await llm.chat(msg, schema=llm.strict_schema(model.model_json_schema()), max_tokens=16)


async def warmup():
    """Bedrock 연결·키를 확인하고 스키마를 예열한 뒤 ready. 한쪽이 실패해도 다른 쪽은 쓸 수 있게 상태를 따로 기록한다.
    이미 ready인 쪽은 재예열이 한 번 실패해도 내리지 않는다 (일시 장애는 요청마다 503으로 처리됨)"""
    for name, job in (("embedding", lambda: asyncio.to_thread(embedding.encode, "query", [("warmup", None)])),
                      ("llm", lambda: asyncio.gather(*(_prime(m) for m in OUTPUT_MODELS)))):
        try:
            await job()
            state.models[name] = "ready"
        except Exception:
            if state.models[name] != "ready":
                state.models[name] = "error"
            log.exception("%s warmup failed", name)
    log.info("models %s", state.models)


async def keep_warm():
    while True:
        await warmup()
        ok = all(v == "ready" for v in state.models.values())
        await asyncio.sleep(REWARM_SECONDS if ok else 60)  # 실패하면 1분 뒤 다시


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 예열은 백그라운드로: 그동안 /health 는 각 모델을 loading 으로 응답한다
    task = asyncio.create_task(keep_warm()) if AI_MODE == "real" else None
    yield
    if task:
        task.cancel()
    await llm.client.aclose()
    embedding.client.close()


DESCRIPTION = """API ↔ AI 계약 (C-04). 출처는 AI 레포 `app/schemas.py`, 규칙은 SPEC 기획서 5.5 "AI 서버 공통 규칙".

- 모든 생성 응답에 `model`·`prompt_version`. `prompt_version`이 `"stub"`이면 가짜 응답(저장 안 함), `/embed`는 `embedding_model`이 `"stub"`.
- 실패: 연결 실패·연결 시간 초과·503 → `GPU_UNAVAILABLE`(1회 재시도), 읽기 시간 초과 → `TIMEOUT`(재시도 안 함), AI 서버의 504 → `TIMEOUT`(재시도 안 함), 422 → `AI_ERROR`(재시도 안 함), 그 밖 5xx → `AI_ERROR`(1회 재시도).
- 타임아웃: 연결 2초. 읽기 `/health` 2초 · `/embed` 10초 · `/requirements/extract` 20초 · `/restructure` 30초 · `/match` 30초 · `/gap-guide` 20초 · `/guide` 120초.
- `/guide`는 탭마다 1회 호출, JSON 응답(SSE 아님).
"""

app = FastAPI(title="nerdnam-ai", version="0.3.0", description=DESCRIPTION, lifespan=lifespan)

# 503: 예열 중·Bedrock 호출 한도·장애 → API는 GPU_UNAVAILABLE. 502: LLM 출력 형식이 두 번 틀림 → AI_ERROR (기획서 5.5)
_503: dict[int | str, dict[str, Any]] = {503: {"description": "모델 준비 안 됨·Bedrock 사용 불가 (GPU_UNAVAILABLE)"}}
_502: dict[int | str, dict[str, Any]] = {502: {"description": "LLM 출력 형식 오류 (AI_ERROR)"}, 504: {"description": "LLM 생성 시간 초과 (TIMEOUT, 재시도 안 함)"}}
app.include_router(embed.router, responses=_503)
for r in (restructure, requirements, match, gap_guide, guide):
    app.include_router(r.router, responses=_503 | _502)


@app.get("/health", response_model=HealthResponse)
def health():
    # 예: {"status": "ok", "models": {"llm": "ready", "embedding": "ready"}}
    return {"status": "ok", "models": state.models}
