import os

# 응답에 담는 모델 이름 (기획서 5.5: 모든 응답에 모델·프롬프트 버전 포함)
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3.5-9B")
EMBED_MODEL = os.getenv("EMBED_MODEL", "google/embeddinggemma-2@unpinned:768")
EMBED_DIM = 768
