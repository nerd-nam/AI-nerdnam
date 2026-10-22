from app.config import AI_MODE

# /health 의 models 값: stub(가짜 응답) / loading(모델 로딩·예열 중) / ready / error
models = "stub" if AI_MODE == "stub" else "loading"
embedder = None  # real 모드에서 시작할 때 로딩한 임베딩 모델
