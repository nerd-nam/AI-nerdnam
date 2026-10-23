"""vLLM(OpenAI 호환 API) 호출과 JSON 출력 강제 (기획서 5.2, 작업목록 AI-04)."""
import httpx
from fastapi import HTTPException
from pydantic import BaseModel, ValidationError

from app.config import LLM_MODEL, LLM_SAMPLING, LLM_TIMEOUT, VLLM_URL

client = httpx.AsyncClient(base_url=VLLM_URL, timeout=httpx.Timeout(LLM_TIMEOUT, connect=2))


async def chat(messages: list[dict], schema: dict | None = None, max_tokens: int = 2048) -> str:
    body = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},  # thinking 끔 (Kaggle에서 확인)
        **LLM_SAMPLING,
    }
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
    r = await client.post("/v1/chat/completions", json=body)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


async def generate[T: BaseModel](messages: list[dict], out: type[T], check=None, max_tokens: int = 2048) -> tuple[T, list[str]]:
    """스키마로 출력을 강제하고 검증한다. 형식 오류나 check()가 돌려준 문제가 있으면 1회 다시 요청한다.
    두 번째도 형식이 틀리면 502, 내용 문제만 남으면 (결과, 남은 문제)를 돌려줘 호출한 쪽이 처리한다."""
    obj, problems = None, []
    for _ in range(2):
        try:
            raw = await chat(messages, schema=out.model_json_schema(), max_tokens=max_tokens)
        except httpx.ReadTimeout:
            raise HTTPException(504, "llm timeout")  # 생성이 LLM_TIMEOUT 안에 안 끝남 → API는 TIMEOUT, 재시도 안 함 (기획서 5.5)
        except httpx.TransportError:
            raise HTTPException(503, "llm unavailable")  # vLLM이 꺼짐·로딩 중 → API는 GPU_UNAVAILABLE (기획서 5.5)
        try:
            obj = out.model_validate_json(raw)
            problems = check(obj) if check else []
        except ValidationError as e:
            obj, problems = None, [f"JSON 형식 오류: {e.errors()[0]['msg']} ({e.errors()[0]['loc']})"]
        if not problems:
            return obj, []
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "아래 문제를 고쳐 같은 JSON 형식으로 다시 답하세요.\n- " + "\n- ".join(problems)},
        ]
    if obj is None:
        raise HTTPException(502, "llm output invalid")  # API는 AI_ERROR
    return obj, problems
