"""모델별 분당 요청 한도기 (SPEC ADR-0009 결정 3). Bedrock 호출마다 자리를 받고 부른다.

계정 한도(Nova 2 Lite 20회/분, Titan 임베딩 60회/분, Service Quotas 11-03)를 넘겨 보내면 throttle → 재시도로 한도를 더 쓴다
(nerd-nam/INFRA-nerdnam#19: 시도 수가 호출 수의 2~3.7배). 한도기는 보내기 전에 기다려 throttle 자체를 막는다.
토큰 한도(Nova 8M·Titan 30만/분)는 닿지 않아 세지 않는다.

토큰 버킷: 비어 있으면 분당 한도만큼 바로 보내고(실측: Titan 약 67건을 몰아 받은 뒤 초당 약 1건), 그 뒤로는 일정 속도.
HTTP·큐 경로가 같은 버킷을 쓴다. HTTP는 요청 마감 안에 자리가 없으면 바로 503, 큐 경로는 마감이 없어 기다린다.
"""
import asyncio
import os
import threading
import time

from fastapi import HTTPException

from app import metrics, state
from app.config import EMBED_MODEL_ID, LLM_MODEL
from app.telemetry import tracer


class _Bucket:
    def __init__(self, rpm: int):
        self.rate, self.cap, self.tokens, self.t = rpm / 60, rpm, float(rpm), time.monotonic()
        self.lock = threading.Lock()  # 임베딩은 스레드 풀에서 부른다

    def reserve(self, max_wait: float | None) -> float | None:
        """자리를 예약하고 기다릴 초를 돌려준다(먼저 온 순서). max_wait보다 오래 기다려야 하면 예약하지 않고 None"""
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.cap, self.tokens + (now - self.t) * self.rate)
            self.t = now
            wait = max(0.0, (1 - self.tokens) / self.rate)
            if max_wait is not None and wait > max_wait:
                return None
            self.tokens -= 1  # 음수 = 앞에 예약된 호출 수
            return wait


# ponytail: 프로세스 메모리 버킷. ai Pod 1개라 곧 계정 전체 한도. 2대 이상이면 Redis 공유 버킷으로 옮긴다
_buckets = {LLM_MODEL: _Bucket(int(os.getenv("LLM_RPM", "20"))),
            EMBED_MODEL_ID: _Bucket(int(os.getenv("EMBED_RPM", "60")))}
_KIND = {LLM_MODEL: "llm", EMBED_MODEL_ID: "embed"}


def _reserve(model: str) -> float:
    wait = _buckets[model].reserve(state.left())  # 요청 마감까지만. 큐 경로·연결 확인은 마감이 없다
    if wait is None:
        metrics.REJECTED.labels(_KIND[model]).inc()
        raise HTTPException(503, "bedrock busy")  # API는 GPU_UNAVAILABLE. 큐 경로에는 생기지 않음
    metrics.STAGE_SECONDS.labels(model, "limiter").observe(wait)
    return wait


def _span(model: str, wait: float):
    return tracer.start_as_current_span("limiter.wait", attributes={"gen_ai.request.model": model, "limiter.wait_s": wait})


async def take(model: str) -> None:
    if wait := _reserve(model):
        with _span(model, wait), metrics.INFLIGHT.labels(model, "limiter").track_inprogress():
            await asyncio.sleep(wait)


def take_sync(model: str) -> None:
    """스레드 풀(임베딩)용"""
    if wait := _reserve(model):
        with _span(model, wait), metrics.INFLIGHT.labels(model, "limiter").track_inprogress():
            time.sleep(wait)
