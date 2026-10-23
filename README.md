# nerdnam-ai

API 서버만 호출하는 내부 AI 서버 (FastAPI, 상태 없음). 기획서 5.5의 엔드포인트 6개와 `/health`.

| 모델 | 용도 | 서빙 |
| --- | --- | --- |
| `Qwen/Qwen3.5-4B` | LLM (P1~P5) | vLLM 컨테이너 |
| `google/embeddinggemma-2` (텍스트 인코더만, 768차원) | `/embed` | AI 서버 컨테이너 |

## 실행 모드 (`AI_MODE`)

| 모드 | 언제 | 동작 |
| --- | --- | --- |
| `stub` (기본) | 로컬 개발, BE 연동 개발 (GPU 없이) | 모든 엔드포인트가 가짜 응답. `/health` → `models: {"llm": "stub", "embedding": "stub"}` |
| `real` | GPU 서버, 로컬 Mac(llama-server) | 시작하면 모델별로 로딩 → 예열 → `ready`. 그 전엔 `loading`, 실패하면 `error`. BE는 모두 `ready`일 때만 재처리 |

### 로컬 (stub)

```bash
uv sync
uv run pytest
uv run uvicorn app.main:app --reload --port 8000   # http://localhost:8000/docs
```

### 로컬 (real, Mac)

Mac에서는 vLLM 대신 llama.cpp의 `llama-server`(OpenAI 호환)로 LLM을 띄운다. `app/llm.py`가 보내는 JSON 스키마 강제(`response_format`)와 thinking 끄기(`chat_template_kwargs`)를 지원해 코드 변경이 없다. AI 서버는 Docker 밖에서 실행한다(Docker에서는 Mac GPU를 쓸 수 없음).

```bash
# 1) LLM 서버 — 모델 파일은 bartowski/Qwen_Qwen3.5-4B-GGUF
llama-server -hf bartowski/Qwen_Qwen3.5-4B-GGUF:bf16 --port 8001 --ctx-size 8192 --jinja --reasoning-budget 0
#   bf16 = 원본 정밀도 (8.7GB). 메모리가 부족하면 :Q8_0 (4.6GB, 거의 같은 품질)

# 2) AI 서버
cp .env.local.example .env
uv run --env-file .env uvicorn app.main:app --port 8000
curl localhost:8000/health          # llm·embedding 이 모두 ready 가 될 때까지 대기
```

| 모델 파일 | 크기 | 언제 |
| --- | --- | --- |
| `bf16` | 8.7GB | AI 서버만 띄워 프롬프트·검증 로직을 볼 때. BE·DB Docker(약 8GB)까지 함께면 16GB Mac에서 부족 |
| `Q8_0` | 4.6GB | BE·DB까지 함께 띄워 전체 흐름을 볼 때 |

- 로컬은 **로직 확인용**이다. 서버(llama.cpp ↔ vLLM)와 정밀도(Mac은 bf16, T4는 fp16)가 달라 출력이 GPU 서버와 같지 않다. 품질(G1)·응답 시간·부하는 GPU 서버에서 잰다.
- `LLM_MODEL`을 실제 모델 이름과 다르게 두어(`.env.local.example`) 로컬 결과가 GPU 서버 결과로 저장되지 않게 한다.
- BE가 Docker 안에 있으면 AI 주소는 `http://host.docker.internal:8000`.

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
