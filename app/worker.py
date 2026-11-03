"""Redis 큐 작업자 (SPEC ADR-0009, AI #53). AI_QUEUE_ENABLED=true 일 때만 시작한다.

BE가 ai:req:{llm|embed|embed-bg}에 넣은 단계를 꺼내 지금 HTTP 라우터로 처리하고(같은 프로세스 안, httpx ASGI),
결과를 메시지의 reply_to 스트림(BE Pod별 ai:res:{Pod})에 넣는다. 라우터를 그대로 쓰므로 응답 본문·실패 usage·trace가
HTTP 경로와 같다. 큐 경로에는 요청 마감이 없고, 잠시 뒤 다시 하면 되는 실패(503·504)는 결과로 보내지 않고
기다렸다 다시 한다 → BE가 받는 status는 200·422·502뿐이다(큐 대기는 실패가 아니다).

같은 id는 한 번만 실행한다: Bedrock을 부르기 전에 ai:run:{id}를 SET NX로 선점하고, 처리하는 동안 주기적으로
XCLAIM JUSTID로 idle을 초기화해 살아 있는 작업이 회수되지 않게 한다. 작업자가 죽으면 선점 키가 RUN_TTL 뒤 풀리고
CLAIM_IDLE 뒤 다른 작업자가 XAUTOCLAIM으로 가져간다. 처리한 요청은 XACK + XDEL로 지운다(AI #52).
"""
import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime

import httpx
import redis.asyncio as redis
from redis.exceptions import ResponseError

ENABLED = os.getenv("AI_QUEUE_ENABLED", "false") == "true"
HEADER = "x-nerdnam-queue"  # 작업자가 라우터를 부를 때 붙인다 → 요청 마감 없음, HTTP 요청 지표에서 뺌 (main.py)
GROUP = "ai"
STREAMS = ("ai:req:llm", "ai:req:embed", "ai:req:embed-bg")
# 작업자마다 꺼내는 스트림(앞이 먼저). 수가 곧 스트림별 동시 처리 상한. 문서 임베딩 작업자는 사용자가 기다리는 embed를 먼저 본다
WORKERS = [("ai:req:llm",)] * 8 + [("ai:req:embed",)] * 2 + [("ai:req:embed", "ai:req:embed-bg")]
BLOCK_MS = 5000  # 새 메시지를 기다리는 시간. 그사이 회수할 것이 있는지 다시 본다
CLAIM_IDLE_MS = 60_000  # 이만큼 손대지 않은 pending은 죽은 작업자 몫으로 보고 가져간다
HEARTBEAT = 20  # 처리 중 idle 초기화·선점 연장 주기(초). CLAIM_IDLE·RUN_TTL보다 충분히 짧게
RUN_TTL = 60  # 처리 중 선점 키 만료(초). 작업자가 죽으면 이 뒤에 다시 실행할 수 있다
# 끝난 id를 기억하는 시간(초). BE가 결과를 기다리는 시간(wait-timeout 20분)보다 넉넉히 길게: 그 안의 중복은 건너뜀
DONE_TTL = 1800
RES_TTL = 3600  # 결과 스트림 키 만료(초). BE Pod가 읽지 못하고 내려가도 키가 남지 않게 쓸 때마다 다시 건다
# enqueued_at이 이보다 오래면 BE가 이미 포기(TIMEOUT)했으므로 건너뜀. wait-timeout이 바뀌어도 살아 있는 작업을 버리지 않게 넉넉히
MAX_AGE = float(os.getenv("AI_QUEUE_MAX_AGE", "1800"))
RETRY_BASE = 5.0  # 503·504 뒤 다시 하기까지 쉬는 시간(초). 2배씩, 최대 60초
log = logging.getLogger("uvicorn.error")


def _age(f: dict) -> float:
    try:
        return time.time() - datetime.fromisoformat(f["enqueued_at"]).timestamp()
    except (KeyError, ValueError):
        return 0.0


async def _groups(r: redis.Redis) -> None:
    """읽는 쪽이 소비자 그룹을 만든다(SPEC #101). 이미 있으면 무시. 큐 Redis가 재시작되면 다시 만든다"""
    for s in STREAMS:
        try:
            await r.xgroup_create(s, GROUP, id="0", mkstream=True)
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise


async def _next(r: redis.Redis, streams: tuple[str, ...], consumer: str):
    """(스트림, 메시지 id, 필드) 하나. 앞 스트림부터 회수할 것 → 새 것 순. 없으면 마지막 스트림에서 BLOCK_MS 기다림"""
    for s in streams:
        _, claimed, *_ = await r.xautoclaim(s, GROUP, consumer, CLAIM_IDLE_MS, count=1)
        if claimed:
            return s, *claimed[0]
        if got := await r.xreadgroup(GROUP, consumer, {s: ">"}, count=1):
            return s, *got[0][1][0]
    if got := await r.xreadgroup(GROUP, consumer, {streams[-1]: ">"}, count=1, block=BLOCK_MS):
        return streams[-1], *got[0][1][0]
    return None


async def _heartbeat(r: redis.Redis, stream: str, consumer: str, msg_id: str, run: str) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT)
        try:
            await r.xclaim(stream, GROUP, consumer, 0, [msg_id], justid=True)
            await r.set(run, consumer, xx=True, ex=RUN_TTL)
        except Exception:
            log.warning("queue heartbeat failed id=%s", msg_id, exc_info=True)


async def _call(http: httpx.AsyncClient, f: dict) -> dict | None:
    """라우터를 불러 결과 메시지 필드를 만든다. 503·504는 쉬었다 다시. BE가 포기할 만큼 오래되면 None"""
    usage, seen, n = {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0}, False, 0
    headers = {"content-type": "application/json", HEADER: "1"}
    if f.get("traceparent"):
        headers["traceparent"] = f["traceparent"]  # BE trace에 이어 붙인다 (FastAPI 계측이 읽음)
    while True:
        res = await http.post("/" + f["op"], content=f.get("body", ""), headers=headers)
        try:
            data = res.json()
        except ValueError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("usage"), dict):
            seen = True  # 다시 한 것까지 실패 전까지 쓴 토큰을 합친다
            for k in usage:
                usage[k] += data["usage"].get(k, 0)
        if res.status_code not in (503, 504):
            break
        if _age(f) > MAX_AGE:
            return None
        delay = min(RETRY_BASE * 2 ** n, 60) + random.random()
        n += 1
        log.warning("queue retry id=%s status=%d in %.0fs", f.get("id"), res.status_code, delay)
        await asyncio.sleep(delay)
    code = res.status_code
    out = {"id": f["id"], "status": "200" if code < 300 else "422" if code < 500 else "502"}
    if code < 300:
        out["body"] = res.text
    else:
        detail = data.get("detail") if isinstance(data, dict) else res.text
        out["detail"] = detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)
    if seen:
        out["usage"] = json.dumps(usage)
    return out


async def _done(r: redis.Redis, stream: str, msg_id: str) -> None:
    async with r.pipeline(transaction=True) as p:
        await p.xack(stream, GROUP, msg_id).xdel(stream, msg_id).execute()


async def _handle(r: redis.Redis, http: httpx.AsyncClient, stream: str, msg_id: str, f: dict | None, consumer: str) -> None:
    # 어떻게 끝나든 메시지는 XACK + XDEL 한다. 남으면 BE 접수 거절 기준(XLEN)과 큐 Redis 메모리가 샌다
    if not f or not f.get("id") or not f.get("reply_to"):
        log.error("queue drop malformed message %s %s", stream, msg_id)  # 답할 곳이 없다
        return await _done(r, stream, msg_id)
    if not f.get("op"):  # 답할 곳은 있으니 422로 바로 끝낸다(BE가 20분 기다리지 않게). 모르는 op는 라우터 404 → 422
        log.error("queue malformed id=%s", f["id"])
        async with r.pipeline(transaction=True) as p:
            p.xadd(f["reply_to"], {"id": f["id"], "status": "422", "detail": "op missing"}).expire(f["reply_to"], RES_TTL)
            return await p.xack(stream, GROUP, msg_id).xdel(stream, msg_id).execute()
    if (age := _age(f)) > MAX_AGE:  # 결과를 보내지 않는다: BE는 이미 TIMEOUT으로 끝냈다
        log.warning("queue skip id=%s age=%.0fs", f["id"], age)
        return await _done(r, stream, msg_id)
    run = f"ai:run:{f['id']}"
    if not await r.set(run, consumer, nx=True, ex=RUN_TTL):
        log.info("queue skip duplicate id=%s", f["id"])  # 실행 중이거나 끝남. 원래 실행이 결과를 보낸다
        return await _done(r, stream, msg_id)
    beat = asyncio.create_task(_heartbeat(r, stream, consumer, msg_id, run))
    try:
        out = await _call(http, f)
    finally:
        beat.cancel()
    if out is None:
        log.warning("queue give up id=%s age=%.0fs", f["id"], _age(f))
        return await _done(r, stream, msg_id)
    async with r.pipeline(transaction=True) as p:
        p.xadd(f["reply_to"], out).expire(f["reply_to"], RES_TTL).set(run, "done", ex=DONE_TTL)
        await p.xack(stream, GROUP, msg_id).xdel(stream, msg_id).execute()
    log.info("queue done id=%s op=%s status=%s wait=%.1fs", f["id"], f["op"], out["status"], _age(f))


async def _worker(r: redis.Redis, http: httpx.AsyncClient, streams: tuple[str, ...], consumer: str) -> None:
    ready = False
    while True:
        try:
            if not ready:
                await _groups(r)
                ready = True
            if got := await _next(r, streams, consumer):
                await _handle(r, http, *got, consumer)
        except asyncio.CancelledError:
            raise
        except Exception:
            # 큐 Redis 끊김 등. /readyz는 그대로 두어 HTTP 경로는 계속 받는다(AI #52). 잠시 뒤 다시 접속
            log.warning("queue worker %s error", consumer, exc_info=True)
            ready = False
            await asyncio.sleep(5)


def start(app) -> list[asyncio.Task]:
    r = redis.Redis(host=os.getenv("AI_QUEUE_REDIS_HOST", "localhost"), port=int(os.getenv("AI_QUEUE_REDIS_PORT", "6379")),
                    username=os.getenv("AI_QUEUE_REDIS_USERNAME"), password=os.getenv("AI_QUEUE_REDIS_PASSWORD"),
                    socket_timeout=BLOCK_MS / 1000 + 5,  # 기본 5초는 BLOCK과 같아 빈 큐에서 읽기 시간 초과가 난다
                    decode_responses=True)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                             base_url="http://queue", timeout=None)
    pod = os.getenv("HOSTNAME", "ai")
    return [asyncio.create_task(_worker(r, http, streams, f"{pod}-{i}")) for i, streams in enumerate(WORKERS)]
