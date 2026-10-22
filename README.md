# nerdnam-ai

API 서버만 호출하는 내부 AI 서버 (FastAPI, 상태 없음). 기획서 5.5의 엔드포인트 6개와 `/health`.

| 모델 | 용도 | 서빙 |
| --- | --- | --- |
| `Qwen/Qwen3.5-4B` | LLM (P1~P5) | vLLM 컨테이너 |
| `google/embeddinggemma-2` (텍스트 인코더만, 768차원) | `/embed` | AI 서버 컨테이너 |

## 실행 모드 (`AI_MODE`)

| 모드 | 언제 | 동작 |
| --- | --- | --- |
| `stub` (기본) | 로컬 개발, BE 연동 개발 | 모든 엔드포인트가 가짜 응답. `/health` → `models: {"llm": "stub", "embedding": "stub"}` |
| `real` | GPU 서버 | 시작하면 모델별로 로딩 → 예열 → `ready`. 그 전엔 `loading`, 실패하면 `error`. BE는 모두 `ready`일 때만 재처리 |

### 로컬 (stub)

```bash
uv sync
uv run pytest
uv run uvicorn app.main:app --reload --port 8000   # http://localhost:8000/docs
```

### GPU 서버 (real)

```bash
docker compose up -d --build        # vllm 이 healthy가 된 뒤 ai 가 시작됨
curl localhost:8000/health          # models 의 llm·embedding 이 모두 ready 가 될 때까지 대기
```

- 기본값은 T4(g4dn.xlarge). L4(g6)는 `LLM_DTYPE=auto EMBED_DTYPE=bfloat16 docker compose up -d`
- 호스트에 NVIDIA 드라이버와 NVIDIA Container Toolkit 필요
- AI 서버(8000)는 보안 그룹에서 API 서버만 허용. vLLM(8001)은 서버 자신에서만 접근

## 평가

```bash
# 임베딩 검색 품질 (Mac·Kaggle·GPU 서버 어디서나)
uv run --extra model python -m evals.embed_eval
uv run --extra model python -m evals.embed_eval --compare-full   # 텍스트 전용 로딩 검증

# 동시 요청 부하 테스트 (GPU 서버, compose 실행 후)
uv run python -m evals.load_test
```

## 구조

| 경로 | 내용 |
| --- | --- |
| `app/schemas.py` | 요청·응답 모델. 계약 C-04 초안의 출처 (`/openapi.json`) |
| `app/routers/` | restructure, requirements, embed, match, gap_guide, guide |
| `app/config.py` | 환경변수 (`AI_MODE`, `LLM_MODEL`, `VLLM_URL`, `EMBED_*`) |
| `app/embedding.py` | 임베딩 로딩·접두어 규칙 (서버와 evals가 같이 씀) |
| `app/llm.py` | vLLM 호출 (thinking 끔, JSON 스키마 강제) |
| `docker-compose.yml`, `Dockerfile` | GPU 서버 실행 설정 (ADR-0001) |
| `evals/` | 임베딩 평가, 부하 테스트 |
