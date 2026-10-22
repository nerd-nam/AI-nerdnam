from app.config import AI_MODE

# /health 의 models 값 (모델별): stub(가짜 응답) / loading(로딩·예열 중) / ready / error
# BE는 모든 값이 ready 일 때만 '임베딩 대기' 재처리를 시작한다
_initial = "stub" if AI_MODE == "stub" else "loading"
models = {"llm": _initial, "embedding": _initial}
embedder = None  # real 모드에서 시작할 때 로딩한 임베딩 모델
