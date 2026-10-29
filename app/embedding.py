"""Titan Text Embeddings V2 호출 (Bedrock InvokeModel, boto3). /embed 와 evals 가 같이 쓴다.

Titan은 한 번에 문장 하나만 받으므로 항목마다 병렬로 부르고 순서대로 돌려준다.
질의·문서 접두어 규칙이 없어 문서는 제목을 본문 앞에 붙이기만 한다.
"""
import contextvars
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config
from fastapi import HTTPException
from opentelemetry import trace

from app import llm
from app.config import AWS_REGION, EMBED_DIM, EMBED_MODEL_ID
from app.telemetry import tracer

# 항목마다 따로 부르므로 한꺼번에 많이 보내면 호출 한도(Throttling)에 걸린다. 지금 계정은 약 67건을 몰아 받은 뒤
# 초당 약 1건(실측 2026-10-27). adaptive 재시도가 한도에 맞춰 속도를 낮춘다. 서버는 /embed 읽기 제한(10초) 때문에 4번까지,
# 평가 스크립트는 EMBED_MAX_ATTEMPTS를 늘려 끝까지 돌린다
client = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=Config(
    read_timeout=10, connect_timeout=2, max_pool_connections=8,
    retries={"mode": "adaptive", "total_max_attempts": int(os.getenv("EMBED_MAX_ATTEMPTS", "4"))}))
# ponytail: 동시 8건 고정. 호출 한도(분당 요청 수)에 걸리면 줄이고, 처리량이 모자라면 한도 상향 요청
# 이 풀이 서버 전체 임베딩 동시 호출 상한이다 (#37, replica 1)
_pool = ThreadPoolExecutor(8)
EMBED_QUEUE = int(os.getenv("EMBED_QUEUE", "64"))  # 이미 밀린 문장 수가 이만큼이면 새 요청은 바로 503 (요청 하나는 크기와 상관없이 받음)
BUDGET = float(os.getenv("EMBED_BUDGET", "8"))  # /embed 1건이 재시도까지 쓰는 시간 상한(초). BE 읽기 제한 10초 안
_pending = 0
_lock = threading.Lock()
log = logging.getLogger("uvicorn.error")


def text_of(purpose: str, text: str, title: str | None = None) -> str:
    return f"{title}\n{text}" if purpose == "document" and title else text


def _one(text: str) -> list[float]:
    body = json.dumps({"inputText": text, "dimensions": EMBED_DIM, "normalize": True}, ensure_ascii=False)
    r = client.invoke_model(modelId=EMBED_MODEL_ID, body=body, contentType="application/json", accept="application/json")
    if n := r.get("ResponseMetadata", {}).get("RetryAttempts", 0):
        log.warning("bedrock embed retries=%d", n)  # 대개 호출 한도(Throttling)로 다시 보낸 것
    out = json.loads(r["body"].read())
    trace.get_current_span().set_attributes({"bedrock.result": "ok", "bedrock.attempts": n + 1,
                                             "gen_ai.usage.input_tokens": out.get("inputTextTokenCount", 0)})
    return out["embedding"]


def _traced(text: str, submitted_ns: int) -> list[float]:
    """풀에서 기다린 시간(bedrock.queue)과 호출(bedrock.invoke_model)을 span으로. 문장 하나 = span 하나"""
    tracer.start_span("bedrock.queue", start_time=submitted_ns, attributes={"gen_ai.request.model": EMBED_MODEL_ID}).end()
    with llm._span("bedrock.invoke_model", EMBED_MODEL_ID) as s:
        try:
            return _one(text)
        except Exception as e:
            llm.mark_failed(s, e)
            raise


def _done(_):
    global _pending
    with _lock:
        _pending -= 1


def encode(purpose: str, items: list[tuple[str, str | None]], budget: float | None = None) -> list[list[float]]:
    """budget(초)을 넘기면 아직 시작 안 한 호출을 취소하고 503. 평가 스크립트는 budget 없이 끝까지 기다린다"""
    global _pending
    with _lock:
        if _pending >= EMBED_QUEUE:
            log.warning("bedrock embed busy pending=%d", _pending)
            raise HTTPException(503, "bedrock busy")
        _pending += len(items)
    t = time.monotonic()
    # 스레드 풀은 contextvars를 넘겨주지 않아 span이 요청 trace에서 떨어진다 → 호출마다 복사해서 넘긴다
    # (Context 하나를 여러 스레드가 동시에 run하면 오류라 호출마다 새로 복사)
    futures = [_pool.submit(contextvars.copy_context().run, _traced, text_of(purpose, text, title), time.time_ns())
               for text, title in items]
    for f in futures:
        f.add_done_callback(_done)
    try:
        out = [f.result(None if budget is None else max(0.0, t + budget - time.monotonic())) for f in futures]
    except TimeoutError:
        log.warning("bedrock embed n=%d %.1fs over budget", len(items), time.monotonic() - t)
        raise HTTPException(503, "bedrock busy")
    except Exception as e:
        log.warning("bedrock embed n=%d %.1fs %s", len(items), time.monotonic() - t,
                    getattr(e, "response", {}).get("Error", {}).get("Code") or type(e).__name__)
        raise
    finally:
        for f in futures:
            f.cancel()
    log.info("bedrock embed n=%d %.1fs ok", len(items), time.monotonic() - t)
    return out
