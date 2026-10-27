from app.config import AI_MODE

# /health 의 models 값 (모델별): stub(가짜 응답) / loading(연결 확인 중) / ready / error
# BE는 embedding 이 ready 이면 '임베딩 대기' 재처리를 시작한다 (llm 은 보지 않음, 기획서 v1.29)
_initial = "stub" if AI_MODE == "stub" else "loading"
models = {"llm": _initial, "embedding": _initial}
