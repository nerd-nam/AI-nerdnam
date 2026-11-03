"""Bedrock Converse API(boto3) 호출과 JSON 출력 강제 (기획서 5.2, 작업목록 AI-04, ADR-0007).

LLM(Converse)·임베딩(InvokeModel, app/embedding.py) 모두 boto3로 bedrock-runtime을 부른다.
인증은 boto3 기본 순서: 환경변수 AWS_BEARER_TOKEN_BEDROCK(Bedrock API 키) → IAM 자격 증명(운영 Pod 역할 등)."""
import asyncio
import json
import logging
import os
import random
import time

import boto3
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)
from fastapi import HTTPException
from opentelemetry.trace import Span, SpanKind, StatusCode
from pydantic import BaseModel, ValidationError

from app import metrics, state
from app.config import AWS_REGION, LLM_MAX_MODEL_LEN, LLM_MODEL, LLM_SAMPLING, LLM_TIMEOUT
from app.telemetry import tracer

# SDK 재시도는 끈다(1회). SDK는 읽기 시간 초과·5xx도 다시 보내 한 호출이 LLM_TIMEOUT × 시도 수로 늘어나고,
# 시간 초과는 다시 보내지 않는다는 규칙(C-04 504)과도 어긋난다. 호출 한도 초과(throttle)만 _converse()가 다시 보낸다.
# LLM_MAX_ATTEMPTS는 평가 스크립트처럼 몰아서 부를 때만 올린다
client = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=Config(
    read_timeout=LLM_TIMEOUT, connect_timeout=2, max_pool_connections=32,
    retries={"mode": "adaptive", "total_max_attempts": int(os.getenv("LLM_MAX_ATTEMPTS", "1"))}))

# 서버 전체 LLM 동시 호출 상한 (#37). 운영은 replica 1이라 프로세스 상한이 곧 전체 상한. 모든 LLM 호출이 chat()을 지난다
# ponytail: 동시 8건 고정. 계정 한도(Nova 분당 20)는 올릴 수 없어(SPEC ADR-0008), 몰림은 BE 큐 대기로 나타난다
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "8"))
LLM_QUEUE = int(os.getenv("LLM_QUEUE", "32"))  # 자리를 기다리는 호출 수 상한. 넘치면 바로 503
# 호출 1번이 자리를 기다리는 시간 상한(초). 요청 마감(읽기 제한 − 10초, 기획서 v1.65)이 더 가까우면 그때까지만.
# 한 호출이 마감을 다 쓰지 않게 해 형식 오류 재생성 몫을 남긴다. 짧은 몰림은 실패 대신 기다린다
LLM_QUEUE_WAIT = float(os.getenv("LLM_QUEUE_WAIT", "30"))
LLM_THROTTLE_RETRIES = int(os.getenv("LLM_THROTTLE_RETRIES", "2"))  # 호출 한도 초과 때 다시 보내는 횟수
THROTTLE_BACKOFF = 2.0  # 첫 쉼(초). 다시 보낼 때마다 두 배(2·4초) + 흔들기 0~1초
_slots = asyncio.Semaphore(LLM_CONCURRENCY)
_waiting = 0

ERRORS = (ClientError, BotoCoreError)
# 잠시 뒤 다시 하면 되는 실패 → 503 (API는 GPU_UNAVAILABLE, 1회 재시도)
_RETRYABLE = {"ThrottlingException", "ServiceUnavailableException", "InternalServerException",
              "ModelNotReadyException", "ServiceQuotaExceededException", "ModelErrorException"}
_UNREACHABLE = (EndpointConnectionError, ConnectTimeoutError, ConnectionClosedError)
log = logging.getLogger("uvicorn.error")


def http_error(e: Exception) -> HTTPException:
    """Bedrock(boto3) 호출 실패를 계약 C-04 응답 코드로 (기획서 5.5). LLM·임베딩이 같이 쓴다"""
    if isinstance(e, ReadTimeoutError):
        return HTTPException(504, "bedrock timeout")  # 읽기 제한 안에 안 끝남 → API는 TIMEOUT, 재시도 안 함
    if isinstance(e, ClientError):
        code = e.response.get("Error", {}).get("Code", "")
        if code == "ModelTimeoutException":
            return HTTPException(504, "bedrock model timeout")
        if code in _RETRYABLE:
            return HTTPException(503, f"bedrock unavailable {code}")
        # ValidationException(입력이 한도를 넘음 등)·AccessDeniedException(키·권한·모델 계약) 등. 다시 보내도 같다 → API는 AI_ERROR
        return HTTPException(502, f"bedrock error {code}: {e.response.get('Error', {}).get('Message', '')[:200]}")
    if isinstance(e, _UNREACHABLE):
        return HTTPException(503, "bedrock unreachable")
    return HTTPException(502, f"bedrock error {type(e).__name__}")  # 자격 증명 없음 등 설정 문제


def _code(e: Exception) -> str:
    return e.response.get("Error", {}).get("Code", "") if isinstance(e, ClientError) else type(e).__name__


def result_of(e: Exception) -> str:
    """실패 결과 분류. span 속성과 지표(app/metrics.py)가 같이 쓴다"""
    if isinstance(e, HTTPException):
        return "busy"  # 우리 대기열(#37)에서 막힘
    if isinstance(e, ReadTimeoutError) or _code(e) == "ModelTimeoutException":
        return "timeout"
    if _code(e) in ("ThrottlingException", "ServiceQuotaExceededException"):
        return "throttle"
    return "error"


def attempts_of(e: Exception) -> int:
    """실패한 호출의 실제 시도 횟수(SDK 재시도 + throttle 재시도). 응답이 없는 실패(연결·읽기 시간 초과)는 SDK 몫을 1로 센다"""
    sdk = e.response.get("ResponseMetadata", {}).get("RetryAttempts", 0) + 1 if isinstance(e, ClientError) else 1
    return sdk + getattr(e, "throttle_retries", 0)


def _converse(req: dict) -> tuple[dict, int]:
    """Converse 호출. 호출 한도 초과(throttle)만 2·4초(+흔들기) 쉬고 다시 보낸다. 돌려준 int = throttle 재시도 횟수.
    자리(_slots)를 쥔 채 쉬므로 한도에 걸린 동안은 동시 호출도 저절로 줄어든다. 쉬면 요청 마감을 넘길 때는 바로 503"""
    for n in range(LLM_THROTTLE_RETRIES + 1):
        delay = THROTTLE_BACKOFF * 2 ** n + random.random() * min(THROTTLE_BACKOFF, 1)
        try:
            return client.converse(**req), n
        except ClientError as e:
            left = state.left()
            if n == LLM_THROTTLE_RETRIES or result_of(e) != "throttle" or (left is not None and left <= delay):
                setattr(e, "throttle_retries", n)  # noqa: B010 — 타입 검사기용. 읽는 쪽은 attempts_of의 getattr
                raise
            log.warning("bedrock llm throttle retry=%d", n + 1)
        time.sleep(delay)
    raise AssertionError("unreachable")  # 마지막 회차는 return 아니면 raise로 끝난다. 타입 검사기용


def mark_failed(span: Span, e: Exception) -> None:
    """실패한 Bedrock 호출 span에 결과 분류(throttle·timeout·busy·error)·오류 코드·시도 횟수. 오류 메시지는 넣지 않는다"""
    code = _code(e)
    span.set_attributes({"bedrock.result": result_of(e), "error.type": code})
    if isinstance(e, ClientError):
        span.set_attribute("bedrock.attempts", attempts_of(e))
    span.set_status(StatusCode.ERROR, code)


def _span(name: str, model: str, kind: SpanKind = SpanKind.CLIENT):
    # 예외는 mark_failed가 직접 기록한다(메시지에 입력 일부가 섞일 수 있어 record_exception 끔)
    return tracer.start_as_current_span(name, kind=kind, record_exception=False, set_status_on_exception=False,
                                        attributes={"gen_ai.system": "aws.bedrock", "gen_ai.request.model": model})


def _request(messages: list[dict], schema: dict | None, max_tokens: int, temperature: float | None) -> dict:
    """OpenAI 모양 messages → Converse 요청. system은 따로 빼고, 같은 역할이 이어지면 한 메시지로 합친다
    (Converse는 user·assistant가 번갈아 와야 한다. /restructure는 user를 두 번 보냄)"""
    msgs: list[dict] = []
    for m in messages:
        if m["role"] == "system":
            continue
        if msgs and msgs[-1]["role"] == m["role"]:
            msgs[-1]["content"].append({"text": m["content"]})
        else:
            msgs.append({"role": m["role"], "content": [{"text": m["content"]}]})
    req = {
        "modelId": LLM_MODEL,
        "messages": msgs,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": LLM_SAMPLING["temperature"] if temperature is None else temperature,  # 추출처럼 답이 하나인 작업은 0 (P2)
            "topP": LLM_SAMPLING["top_p"],
        },
    }
    if system := [{"text": m["content"]} for m in messages if m["role"] == "system"]:
        req["system"] = system
    if schema:
        # 출력 강제는 도구 1개(out)를 반드시 부르게 하는 방식. Nova는 outputConfig(json_schema)를 지원하지 않는다
        req["toolConfig"] = {"tools": [{"toolSpec": {"name": "out", "description": "결과를 이 형식으로 돌려준다",
                                                     "inputSchema": {"json": schema}}}],
                             "toolChoice": {"tool": {"name": "out"}}}
    return req


async def _acquire() -> float:
    """동시 호출 자리를 얻고 기다린 초를 돌려준다. 대기열이 차 있거나 오래 기다리면 503 (API는 GPU_UNAVAILABLE, 1회 재시도)"""
    global _waiting
    if _waiting >= LLM_QUEUE:
        log.warning("bedrock llm busy queue=%d", _waiting)
        metrics.REJECTED.labels("llm").inc()
        raise HTTPException(503, "bedrock busy")
    t = time.monotonic()
    _waiting += 1
    try:
        await asyncio.wait_for(_slots.acquire(), state.left(LLM_QUEUE_WAIT))  # 요청 마감이 더 가까우면 그때까지만
    except TimeoutError:
        log.warning("bedrock llm busy wait=%.1fs", time.monotonic() - t)
        metrics.REJECTED.labels("llm").inc()
        raise HTTPException(503, "bedrock busy")
    finally:
        _waiting -= 1
    wait = time.monotonic() - t
    metrics.QUEUE_WAIT.labels("llm").observe(wait)
    return wait


def _release(call: asyncio.Future) -> None:
    _slots.release()
    if not call.cancelled():
        call.exception()  # 마감 뒤에 끝난 호출의 오류는 이미 504로 답했으므로 읽어만 둔다(경고 로그 방지)


async def chat(messages: list[dict], schema: dict | None = None, max_tokens: int = 2048, temperature: float | None = None) -> str:
    with _span("bedrock.queue", LLM_MODEL, SpanKind.INTERNAL) as q:  # 대기열(#37) 대기 시간이 trace에 보이게
        try:
            wait = await _acquire()
        except HTTPException as e:
            mark_failed(q, e)
            raise
    t, used = time.monotonic(), state.usage.get()
    if used is not None:
        used["llm_calls"] += 1
    with _span("bedrock.converse", LLM_MODEL) as s:
        # 생성은 마감까지만 기다린다(넘으면 504). 스레드는 멈출 수 없으므로 자리는 호출이 실제로 끝날 때 돌려준다
        call = asyncio.ensure_future(asyncio.to_thread(_converse, _request(messages, schema, max_tokens, temperature)))
        call.add_done_callback(_release)
        try:
            r, throttled = await asyncio.wait_for(asyncio.shield(call), state.left())
        except TimeoutError:
            mark_failed(s, TimeoutError())
            s.set_attribute("bedrock.result", "timeout")
            metrics.bedrock("converse", LLM_MODEL, "timeout", time.monotonic() - t, 1)
            log.warning("bedrock llm %s %.1fs wait=%.1fs deadline", LLM_MODEL, time.monotonic() - t, wait)
            raise HTTPException(504, "llm deadline")  # API는 TIMEOUT(재시도 안 함)
        except ERRORS as e:
            mark_failed(s, e)
            if used is not None:
                used["llm_calls"] += getattr(e, "throttle_retries", 0)  # 한도 초과로 다시 보낸 것도 호출(C-04 usage)
            metrics.bedrock("converse", LLM_MODEL, result_of(e), time.monotonic() - t, attempts_of(e))
            log.warning("bedrock llm %s %.1fs wait=%.1fs %s", LLM_MODEL, time.monotonic() - t, wait, _code(e))
            raise
        # 호출마다 한 줄 (본문·키는 남기지 않음). 읽기 시간 초과로 끊긴 호출의 토큰은 알 수 없어 합계에 빠진다
        u, retries = r.get("usage", {}), r.get("ResponseMetadata", {}).get("RetryAttempts", 0) + throttled
        metrics.bedrock("converse", LLM_MODEL, "ok", time.monotonic() - t, retries + 1,
                        u.get("inputTokens", 0), u.get("outputTokens", 0))
        if used is not None:
            used["llm_calls"] += throttled
            used["input_tokens"] += u.get("inputTokens", 0)
            used["output_tokens"] += u.get("outputTokens", 0)
        s.set_attributes({"bedrock.result": "ok", "bedrock.attempts": retries + 1,
                          "gen_ai.usage.input_tokens": u.get("inputTokens", 0), "gen_ai.usage.output_tokens": u.get("outputTokens", 0)})
        log.info("bedrock llm %s %.1fs wait=%.1fs retries=%s ok in=%s out=%s", LLM_MODEL, time.monotonic() - t, wait,
                 retries, u.get("inputTokens"), u.get("outputTokens"))
    blocks = r["output"]["message"]["content"]
    if schema:  # 도구 입력이 곧 출력 JSON. 도구 호출이 없으면(길이 초과 등) "" → 검증 실패로 1회 재시도
        return next((json.dumps(b["toolUse"]["input"], ensure_ascii=False) for b in blocks if "toolUse" in b), "")
    # 추론 블록(reasoningContent)이 먼저 올 수 있어 text 블록을 고른다
    return next(b["text"] for b in blocks if "text" in b)


def fits(messages: list[dict], max_tokens: int) -> bool:
    """입력+출력이 문맥 한도 안에 드는지 대략 확인. 한국어 JSON은 토큰 하나에 약 2.1자(1.97~2.68, 로컬 실측)라
    1.8자로 넉넉하게 잡는다(Qwen 기준). 스키마·채팅 틀 몫으로 300토큰을 더 뺀다."""
    # ponytail: 글자 수 어림. 한도 근처에서 빗나가면 Bedrock CountTokens 호출로 바꾼다
    chars = sum(len(m["content"]) for m in messages)
    return chars / 1.8 + max_tokens + 300 <= LLM_MAX_MODEL_LEN


# Bedrock 도구 스키마에서 빼는 키워드. 빼도 pydantic 검증 → 1회 재시도가 그대로 잡는다
_UNSUPPORTED = {"minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "maxItems", "pattern", "format"}


def strict_schema(schema: dict) -> dict:
    """모든 필드를 필수로 바꾼 JSON 스키마. 기본값이 있는 필드는 생략 가능이 되어 모델이 통째로 건너뛴다
    (로컬 테스트 2026-10-23: role·skills 4번 중 4번 빠짐). 값이 없으면 ""·[]·null을 직접 쓰게 한다.
    Bedrock 스키마 규칙도 맞춘다: object마다 additionalProperties false, minItems는 0·1만, 길이·범위 제약 없음."""
    def walk(x):
        if isinstance(x, dict):
            if "properties" in x:
                x["required"] = list(x["properties"])
                x["additionalProperties"] = False
            for k in _UNSUPPORTED & x.keys():
                if not isinstance(x[k], dict):  # dict면 키워드가 아니라 그 이름의 필드
                    del x[k]
            if isinstance(x.get("minItems"), int) and x["minItems"] > 1:
                x["minItems"] = 1
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(schema)
    return schema


async def generate[T: BaseModel](messages: list[dict], out: type[T], check=None, max_tokens: int = 2048,
                                 temperature: float | None = None) -> tuple[T, list[str]]:
    """스키마로 출력을 강제하고 검증한다. 형식 오류나 check()가 돌려준 문제가 있으면 1회 다시 요청한다.
    두 번째도 형식이 틀리면 502, 내용 문제만 남으면 (결과, 남은 문제)를 돌려줘 호출한 쪽이 처리한다."""
    obj, problems = None, []
    for _ in range(2):
        try:
            raw = await chat(messages, schema=strict_schema(out.model_json_schema()), max_tokens=max_tokens, temperature=temperature)
        except ERRORS as e:
            raise http_error(e)
        try:
            obj = out.model_validate_json(raw)
            problems = check(obj) if check else []
        except ValidationError as e:
            obj, problems = None, [f"JSON 형식 오류: {e.errors()[0]['msg']} ({e.errors()[0]['loc']})"]
        if obj is not None and not problems:
            return obj, []
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "아래 문제를 고쳐 같은 JSON 형식으로 다시 답하세요.\n- " + "\n- ".join(problems)},
        ]
    if obj is None:
        raise HTTPException(502, "llm output invalid")  # API는 AI_ERROR
    return obj, problems
