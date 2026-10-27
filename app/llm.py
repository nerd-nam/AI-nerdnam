"""Bedrock(OpenAI 호환 Chat Completions) 호출과 JSON 출력 강제 (기획서 5.2, 작업목록 AI-04, ADR-0007)."""
import httpx
from fastapi import HTTPException
from pydantic import BaseModel, ValidationError

from app.config import (
    BEDROCK_API_KEY,
    BEDROCK_URL,
    LLM_MAX_MODEL_LEN,
    LLM_MODEL,
    LLM_SAMPLING,
    LLM_TIMEOUT,
)

client = httpx.AsyncClient(base_url=BEDROCK_URL, headers={"Authorization": f"Bearer {BEDROCK_API_KEY}"},
                           timeout=httpx.Timeout(LLM_TIMEOUT, connect=2))

# 호출 한도 초과(429)·Bedrock 장애(5xx)는 잠시 뒤 다시 하면 되는 실패 → 503 (API는 GPU_UNAVAILABLE, 1회 재시도)
_RETRYABLE = {429, 500, 502, 503, 504}


def http_error(e: httpx.HTTPError) -> HTTPException:
    """Bedrock 호출 실패를 계약 C-04 응답 코드로 (기획서 5.5). LLM·임베딩이 같이 쓴다"""
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code in _RETRYABLE:
            return HTTPException(503, f"bedrock unavailable {code}")
        # 400 입력이 한도를 넘음·403 키 오류 등. 다시 보내도 같다 → API는 AI_ERROR
        return HTTPException(502, f"bedrock error {code}: {e.response.text[:200]}")
    if isinstance(e, httpx.ReadTimeout):
        return HTTPException(504, "bedrock timeout")  # 읽기 제한 안에 안 끝남 → API는 TIMEOUT, 재시도 안 함
    return HTTPException(503, "bedrock unreachable")  # 연결 실패


async def chat(messages: list[dict], schema: dict | None = None, max_tokens: int = 2048, temperature: float | None = None) -> str:
    body = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        **LLM_SAMPLING,
    }
    if temperature is not None:  # 추출처럼 답이 하나인 작업은 0 (P2)
        body["temperature"] = temperature
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
    r = await client.post("/openai/v1/chat/completions", json=body)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def fits(messages: list[dict], max_tokens: int) -> bool:
    """입력+출력이 문맥 한도 안에 드는지 대략 확인. 한국어 JSON은 토큰 하나에 약 2.1자(1.97~2.68, 로컬 실측)라
    1.8자로 넉넉하게 잡는다(Qwen 기준, DeepSeek도 비슷). 스키마·채팅 틀 몫으로 300토큰을 더 뺀다."""
    # ponytail: 글자 수 어림. 한도 근처에서 빗나가면 Bedrock CountTokens 호출로 바꾼다
    chars = sum(len(m["content"]) for m in messages)
    return chars / 1.8 + max_tokens + 300 <= LLM_MAX_MODEL_LEN


# Bedrock 구조화 출력이 받지 않는 키워드. 빼도 pydantic 검증 → 1회 재시도가 그대로 잡는다
_UNSUPPORTED = {"minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "maxItems", "pattern", "format"}


def strict_schema(schema: dict) -> dict:
    """모든 필드를 필수로 바꾼 JSON 스키마. 기본값이 있는 필드는 생략 가능이 되어 모델이 통째로 건너뛴다
    (로컬 테스트 2026-10-23: role·skills 4번 중 4번 빠짐). 값이 없으면 ""·[]·null을 직접 쓰게 한다.
    Bedrock 규칙도 맞춘다: object마다 additionalProperties false, minItems는 0·1만, 길이·범위 제약 없음."""
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
        except httpx.HTTPError as e:
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
