"""vLLM(OpenAI 호환 API) 호출."""
import httpx

from app.config import LLM_MODEL, VLLM_URL

client = httpx.AsyncClient(base_url=VLLM_URL, timeout=httpx.Timeout(60, connect=2))


async def chat(messages: list[dict], schema: dict | None = None, max_tokens: int = 2048) -> str:
    body = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},  # thinking 끔 (Kaggle에서 확인)
    }
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
    r = await client.post("/v1/chat/completions", json=body)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
