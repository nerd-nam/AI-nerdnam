import asyncio

import pytest
from fastapi import HTTPException
from pydantic import BaseModel

from app import llm
from app.validators import numbers, unknown_ids, unsupported_numbers

RAW = "평균 응답이 1.2초까지 늘었다. 팀 4명 중 리드. 평균 응답이 0.3초로 줄었다. 처리량 1,200건"


def test_numbers():
    assert numbers(RAW) == {1.2, 4, 0.3, 1200}


def test_unsupported_numbers():
    assert unsupported_numbers("1.2s → 0.3s, 1200건, 팀 4명", RAW) == []
    assert unsupported_numbers("응답 75% 단축, 피크 TPS 2배", RAW) == ["2", "75"]


def test_unknown_ids():
    assert unknown_ids(["c1", "c9"], {"c1", "c2"}) == ["c9"]


class Out(BaseModel):
    n: int


def fake_chat(replies):
    calls = []

    async def chat(messages, schema=None, max_tokens=0):
        calls.append(messages)
        return replies[len(calls) - 1]
    return chat, calls


def test_generate_retries_once_on_bad_json(monkeypatch):
    chat, calls = fake_chat(['{"n": "x"}', '{"n": 3}'])
    monkeypatch.setattr(llm, "chat", chat)
    obj, problems = asyncio.run(llm.generate([{"role": "user", "content": "q"}], Out))
    assert obj.n == 3 and problems == [] and len(calls) == 2
    assert "JSON 형식 오류" in calls[1][-1]["content"]  # 다시 요청할 때 문제를 알려 준다


def test_generate_returns_remaining_problems(monkeypatch):
    chat, calls = fake_chat(['{"n": 1}', '{"n": 2}'])
    monkeypatch.setattr(llm, "chat", chat)
    obj, problems = asyncio.run(llm.generate([], Out, check=lambda o: ["too small"] if o.n < 5 else []))
    assert obj.n == 2 and problems == ["too small"] and len(calls) == 2


def test_generate_502_when_invalid_twice(monkeypatch):
    chat, _ = fake_chat(["not json", "still not"])
    monkeypatch.setattr(llm, "chat", chat)
    with pytest.raises(HTTPException) as e:
        asyncio.run(llm.generate([], Out))
    assert e.value.status_code == 502
