"""로컬(Mac) 전용 LLM 서버: vLLM 대신 transformers로 원본 Qwen/Qwen3.5-4B(bf16)를 띄운다.

app/llm.py가 쓰는 /v1/chat/completions만 흉내 낸다. 운영(GPU 서버)에서는 쓰지 않는다.
실행: uv run --extra model python -m evals.local_llm   (포트 8001, 첫 실행은 모델 9.3GB 다운로드)
안 돌아가면 README "로컬 (real, Mac)"의 GGUF 방법(llama-server)으로 바꾼다.
"""
import json
import os
import threading

import torch
import uvicorn
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = os.getenv("LOCAL_LLM_REPO", "Qwen/Qwen3.5-4B")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16).to(DEVICE).eval()  # 텍스트 모델만 (Qwen3_5ForCausalLM)
lock = threading.Lock()  # ponytail: 한 번에 요청 1개. 로컬 로직 확인용이라 동시 처리는 하지 않음
app = FastAPI()


def with_schema(messages: list[dict], response_format: dict | None) -> list[dict]:
    """vLLM처럼 생성을 강제하지 못해 JSON 스키마를 시스템 프롬프트로 알린다. 형식이 틀리면 AI 서버 검증기가 1회 재요청."""
    if not response_format or response_format.get("type") != "json_schema":
        return messages
    rule = ("다음 JSON 스키마를 따르는 JSON 객체 하나만 출력한다. 설명이나 코드 블록 없이.\n"
            + json.dumps(response_format["json_schema"]["schema"], ensure_ascii=False))
    if messages and messages[0]["role"] == "system":
        return [{"role": "system", "content": messages[0]["content"] + "\n\n" + rule}] + messages[1:]
    return [{"role": "system", "content": rule}] + messages


@app.get("/health")
def health():
    return {"status": "ok", "model": REPO, "device": DEVICE}


@app.post("/v1/chat/completions")
def chat(body: dict):
    messages = with_schema(body["messages"], body.get("response_format"))
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt",
                                  **(body.get("chat_template_kwargs") or {})).to(DEVICE)  # enable_thinking=False 그대로 전달
    with lock, torch.no_grad():
        out = model.generate(**enc, max_new_tokens=body.get("max_tokens", 2048), do_sample=False)
    text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return {"model": REPO, "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8001")))
