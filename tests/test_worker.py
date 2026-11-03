"""큐 작업자 (ADR-0009, AI #53): reply_to로 결과, 같은 id 한 번만, 503은 기다렸다 다시, 처리한 요청은 지움"""
import asyncio
import json
import time
from datetime import UTC, datetime

import fakeredis
import httpx
from botocore.exceptions import ReadTimeoutError

from app import llm, state, worker
from app.main import app

MATCH = {"requirements": [{"id": 1, "type": "required", "text": "Kafka 운영",
                           "candidates": [{"chunk_id": 1, "chunk_type": "skill", "text": "Kafka"}]}]}


def msg(id="match:1", body=MATCH, op="match", age=0.0):
    enq = datetime.fromtimestamp(time.time() - age, UTC).isoformat().replace("+00:00", "Z")
    return {"id": id, "op": op, "body": json.dumps(body), "enqueued_at": enq, "reply_to": "ai:res:be-1",
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"}


async def run(*messages, before=None, held=None):
    """메시지를 넣고 작업자 한 개로 하나씩 처리한 뒤 (결과 목록, Redis)"""
    r = fakeredis.FakeAsyncRedis(decode_responses=True)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://queue")
    await worker._groups(r)
    for m in messages:
        await r.xadd("ai:req:llm", m)
    if held:
        await r.set(held, "ai-9")  # 다른(죽은) 작업자가 선점 중
    if before:
        before()
    while got := await worker._next(r, ("ai:req:llm",), "ai-0"):
        await worker._handle(r, http, *got, "ai-0")
    return [f for _, f in await r.xrange("ai:res:be-1")], r


def test_result_goes_to_reply_to_and_request_is_deleted(monkeypatch):
    monkeypatch.setattr(worker, "BLOCK_MS", 1)

    async def go():
        res, r = await run(msg())
        assert len(res) == 1 and res[0]["id"] == "match:1" and res[0]["status"] == "200"
        assert "results" in json.loads(res[0]["body"]) and json.loads(res[0]["usage"])["llm_calls"] == 0  # stub
        assert await r.xlen("ai:req:llm") == 0  # XACK + XDEL
        assert await r.ttl("ai:res:be-1") > 0 and await r.get("ai:run:match:1") == "done"
    asyncio.run(go())


def test_same_id_runs_once_and_old_message_is_skipped(monkeypatch):
    monkeypatch.setattr(worker, "BLOCK_MS", 1)

    async def go():
        res, r = await run(msg(), msg(), msg(id="match:2", age=worker.MAX_AGE + 1))
        assert [f["id"] for f in res] == ["match:1"]
        assert await r.xlen("ai:req:llm") == 0
    asyncio.run(go())


def test_validation_error_is_422_with_detail(monkeypatch):
    monkeypatch.setattr(worker, "BLOCK_MS", 1)

    async def go():
        res, _ = await run(msg(body={}))
        assert res[0]["status"] == "422" and "requirements" in res[0]["detail"] and "usage" not in res[0]
    asyncio.run(go())


def test_malformed_messages_end_with_422_or_drop_and_are_deleted(monkeypatch):
    monkeypatch.setattr(worker, "BLOCK_MS", 1)
    no_op, no_reply, no_time = msg(id="bad:1"), msg(id="bad:3"), msg(id="bad:4")
    del no_op["op"], no_reply["reply_to"], no_time["enqueued_at"]

    async def go():
        res, r = await run(no_op, msg(id="bad:2", op="nope"), no_reply, no_time)
        assert [(f["id"], f["status"]) for f in res] == [("bad:1", "422"), ("bad:2", "422"), ("bad:4", "422")]
        assert await r.xlen("ai:req:llm") == 0
    asyncio.run(go())


def test_503_waits_and_retries_instead_of_failing(monkeypatch):
    monkeypatch.setattr(worker, "BLOCK_MS", 1)
    monkeypatch.setattr(worker, "RETRY_BASE", 0.01)
    monkeypatch.setitem(state.models, "llm", "loading")  # 연결 확인 중 → 503

    async def go():
        loop = asyncio.get_running_loop()
        res, _ = await run(msg(), before=lambda: loop.call_later(0.2, state.models.__setitem__, "llm", "stub"))
        assert [f["status"] for f in res] == ["200"]
    asyncio.run(go())


def test_held_by_other_worker_stays_pending(monkeypatch):
    """선점이 남아 있으면(실행 중이거나 죽은 작업자) 지우지 않는다 → 선점이 풀린 뒤 회수해 실행"""
    monkeypatch.setattr(worker, "BLOCK_MS", 1)

    async def go():
        res, r = await run(msg(), held="ai:run:match:1")
        assert res == [] and await r.xlen("ai:req:llm") == 1
    asyncio.run(go())


def test_permanent_errors_end_as_502_without_retry(monkeypatch):
    monkeypatch.setattr(worker, "BLOCK_MS", 1)
    monkeypatch.setattr(worker, "RETRY_BASE", 10)  # 다시 하면 테스트가 멈춘다
    calls = []

    def slow(**req):
        calls.append(1)
        raise ReadTimeoutError(endpoint_url="bedrock")
    monkeypatch.setattr(llm.client, "converse", slow)

    async def go():
        monkeypatch.setitem(state.models, "llm", "error")  # 연결 확인 실패(키·권한) → 503 llm error
        res, _ = await run(msg(id="a:1"))
        monkeypatch.setitem(state.models, "llm", "ready")  # Bedrock 생성 시간 초과 → 504
        res2, _ = await run(msg(id="a:2"))
        assert [f["status"] for f in res + res2] == ["502", "502"] and len(calls) == 1
    asyncio.run(go())


def test_backlog_metrics_split_lag_and_pending(monkeypatch):
    """queue_lag = 아직 안 꺼낸 것, queue_pending = 꺼냈지만 끝나지 않은 것 (AI #52 4번)"""
    from app import metrics
    from app.config import LLM_MODEL

    async def go():
        r = fakeredis.FakeAsyncRedis(decode_responses=True)
        await worker._groups(r)
        for i in range(3):
            await r.xadd("ai:req:llm", msg(id=f"m:{i}"))
        await r.xreadgroup(worker.GROUP, "ai-0", {"ai:req:llm": ">"}, count=1)  # 1건 처리 중
        t = asyncio.create_task(worker._watch(r))
        await asyncio.sleep(0.05)
        t.cancel()
        get = lambda g: g.labels("llm", LLM_MODEL)._value.get()  # noqa: E731
        assert (get(metrics.QUEUE_LAG), get(metrics.QUEUE_PENDING), metrics.QUEUE_UP._value.get()) == (2, 1, 1)
    asyncio.run(go())
