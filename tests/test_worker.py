"""큐 작업자 (ADR-0009, AI #53): reply_to로 결과, 같은 id 한 번만, 503은 기다렸다 다시, 처리한 요청은 지움"""
import asyncio
import json
import time
from datetime import UTC, datetime

import fakeredis
import httpx

from app import state, worker
from app.main import app

MATCH = {"requirements": [{"id": 1, "type": "required", "text": "Kafka 운영",
                           "candidates": [{"chunk_id": 1, "chunk_type": "skill", "text": "Kafka"}]}]}


def msg(id="match:1", body=MATCH, op="match", age=0.0):
    enq = datetime.fromtimestamp(time.time() - age, UTC).isoformat().replace("+00:00", "Z")
    return {"id": id, "op": op, "body": json.dumps(body), "enqueued_at": enq, "reply_to": "ai:res:be-1",
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"}


async def run(*messages, before=None):
    """메시지를 넣고 작업자 한 개로 하나씩 처리한 뒤 (결과 목록, Redis)"""
    r = fakeredis.FakeAsyncRedis(decode_responses=True)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://queue")
    await worker._groups(r)
    for m in messages:
        await r.xadd("ai:req:llm", m)
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
    no_op, no_reply = msg(id="bad:1"), msg(id="bad:3")
    del no_op["op"], no_reply["reply_to"]

    async def go():
        res, r = await run(no_op, msg(id="bad:2", op="nope"), no_reply)
        assert [(f["id"], f["status"]) for f in res] == [("bad:1", "422"), ("bad:2", "422")]  # 답할 곳이 없으면 버림
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
