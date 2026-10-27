from contextvars import ContextVar

from app.config import AI_MODE

# /health 의 models 값 (모델별): stub(가짜 응답) / loading(연결 확인 중) / ready / error
# BE는 embedding 이 ready 이면 '임베딩 대기' 재처리를 시작한다 (llm 은 보지 않음, 기획서 v1.29)
_initial = "stub" if AI_MODE == "stub" else "loading"
models = {"llm": _initial, "embedding": _initial}

# 요청 1건 안의 LLM 사용량 합계 (main.py 미들웨어가 요청마다 새로 만든다). /match처럼 동시에 부른 호출도 같은 dict에 더한다
usage: ContextVar[dict | None] = ContextVar("usage", default=None)


def current_usage() -> dict:
    return dict(usage.get() or {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0})
