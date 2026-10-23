import os

# stub: 가짜 응답 (GPU 없이 개발할 때, 기본값) / real: vLLM·임베딩 모델 사용 (GPU 서버)
AI_MODE = os.getenv("AI_MODE", "stub")

# LLM (vLLM). 응답에 담는 모델 이름이기도 하다 (기획서 5.5)
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3.5-4B")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8001")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))  # 읽기 제한(초). 로컬 Mac은 느려서 늘린다 (.env.local.example)
# 생성 설정. 서버마다 기본값이 달라(llama-server temperature 0.8, vLLM 1.0) 직접 보낸다.
# Qwen3.5 권장(생각 모드 끔, 일반 작업). temperature·presence_penalty는 평가 세트로 비교할 값이라 환경변수로 바꿀 수 있다
LLM_SAMPLING = {
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": float(os.getenv("LLM_PRESENCE_PENALTY", "1.5")),
}

# 임베딩. revision을 고정한다: 바꾸면 저장된 벡터를 전부 다시 계산해야 함 (ADR-0001, 2026-10-22 평가·검증한 버전)
EMBED_REPO = os.getenv("EMBED_REPO", "google/embeddinggemma-2")
EMBED_REVISION = os.getenv("EMBED_REVISION") or "914f7f89142e33e77833254d9c9b90c3cef7303b"
EMBED_DTYPE = os.getenv("EMBED_DTYPE", "float32")  # T4: float32 (fp16은 NaN 위험) / L4: bfloat16
EMBED_DIM = 768
EMBED_MODEL = f"{EMBED_REPO}@{EMBED_REVISION}:{EMBED_DIM}"  # 응답·청크에 기록하는 이름
