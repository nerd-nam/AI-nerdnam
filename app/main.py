import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app import embedding, llm, state, telemetry
from app.config import AI_MODE
from app.routers import embed, gap_guide, guide, match, requirements, restructure
from app.schemas import ErrorResponse, HealthResponse

log = logging.getLogger("uvicorn.error")


# LLM 출력 스키마 전부 (tests/test_bedrock.py 가 Bedrock 스키마 규칙을 확인)
OUTPUT_MODELS = [restructure.P1Out, requirements.P2Out, requirements.P2AbbrOut, match.P3Out, gap_guide.P6Out,
                 *guide._MODEL.values()]


async def warmup():
    """Bedrock 연결·키·모델 권한을 짧은 호출로 확인한 뒤 ready. 한쪽이 실패해도 다른 쪽은 쓸 수 있게 상태를 따로 기록한다.
    이미 ready인 쪽은 다시 확인하다 실패해도 내리지 않는다 (일시 장애는 요청마다 503으로 처리됨)"""
    for name, job in (("embedding", lambda: asyncio.to_thread(embedding.encode, "query", [("warmup", None)])),
                      ("llm", lambda: llm.chat([{"role": "user", "content": "ping"}], max_tokens=8))):
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
        if all(v == "ready" for v in state.models.values()):
            return
        await asyncio.sleep(60)  # 실패하면 1분 뒤 다시


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 연결 확인은 백그라운드로: 그동안 /health 는 각 모델을 loading 으로 응답한다
    task = asyncio.create_task(keep_warm()) if AI_MODE == "real" else None
    yield
    if task:
        task.cancel()


DESCRIPTION = """API ↔ AI 계약 (C-04). 출처는 AI 레포 `app/schemas.py`, 규칙은 SPEC 기획서 5.5 "AI 서버 공통 규칙".

- 모든 생성 응답에 `model`·`prompt_version`·`usage`(요청 1건의 LLM 입력·출력 토큰과 호출 수 합계). AI 서버가 낸 오류(502·503·504, `/guide` 문맥 초과 422)도 `{detail, usage}`로 실패 전까지 쓴 토큰을 담는다. 요청 형식 검사 422는 LLM을 부르기 전이라 `usage`가 없다(`detail`이 목록). `prompt_version`이 `"stub"`이면 가짜 응답(저장 안 함), `/embed`는 `embedding_model`이 `"stub"`.
- 실패: 연결 실패·연결 시간 초과·503 → `GPU_UNAVAILABLE`(1회 재시도), 읽기 시간 초과 → `TIMEOUT`(재시도 안 함), AI 서버의 504 → `TIMEOUT`(재시도 안 함), 422 → `AI_ERROR`(재시도 안 함), 그 밖 5xx → `AI_ERROR`(1회 재시도).
- 타임아웃: 연결 2초. 읽기 `/health` 2초 · `/embed` 10초 · `/requirements/extract` 20초 · `/restructure` 30초 · `/match` 30초 · `/gap-guide` 20초 · `/guide` 120초.
- `/guide`는 탭마다 1회 호출, JSON 응답(SSE 아님).
"""

app = FastAPI(title="nerdnam-ai", version="0.4.0", description=DESCRIPTION, lifespan=lifespan)
telemetry.setup(app)

# 503: 연결 확인 중·Bedrock 호출 한도·장애 → API는 GPU_UNAVAILABLE. 502: LLM 출력 형식이 두 번 틀림 → AI_ERROR (기획서 5.5)
_503: dict[int | str, dict[str, Any]] = {503: {"model": ErrorResponse, "description": "모델 준비 안 됨·Bedrock 사용 불가 (GPU_UNAVAILABLE)"}}
_502: dict[int | str, dict[str, Any]] = {502: {"model": ErrorResponse, "description": "LLM 출력 형식 오류 (AI_ERROR)"},
                                         504: {"model": ErrorResponse, "description": "LLM 생성 시간 초과 (TIMEOUT, 재시도 안 함)"}}
app.include_router(embed.router, responses=_503)
for r in (restructure, requirements, match, gap_guide, guide):
    app.include_router(r.router, responses=_503 | _502)


@app.middleware("http")
async def track_usage(request: Request, call_next):
    state.usage.set({"input_tokens": 0, "output_tokens": 0, "llm_calls": 0})
    return await call_next(request)


@app.exception_handler(HTTPException)
async def error_with_usage(request: Request, e: HTTPException):
    return JSONResponse({"detail": e.detail, "usage": state.current_usage()}, status_code=e.status_code, headers=e.headers)


@app.get("/health", response_model=HealthResponse)
def health():
    # 예: {"status": "ok", "models": {"llm": "ready", "embedding": "ready"}}
    return {"status": "ok", "models": state.models}


# k8s 프로브용 (계약 C-04 밖이라 OpenAPI에 넣지 않음). 저장된 상태만 읽고 Bedrock은 부르지 않는다(유료·호출 한도)
@app.get("/livez", include_in_schema=False)
def livez():
    return {"status": "ok"}


@app.get("/readyz", include_in_schema=False)
def readyz():
    # 첫 연결 확인이 끝났고(loading 없음) 한쪽이라도 쓸 수 있으면 트래픽을 받는다. 한쪽만 error면 그 기능만 503
    # (warmup이 모델별로 상태를 두는 이유. 둘 다 요구하면 Pod가 빠져 BE가 /health로 임베딩 ready도 못 본다)
    ok = "loading" not in state.models.values() and any(v in ("ready", "stub") for v in state.models.values())
    return JSONResponse({"status": "ok" if ok else "not ready", "models": state.models}, status_code=200 if ok else 503)
