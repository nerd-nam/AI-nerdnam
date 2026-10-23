# nerdnam-ai

API 서버만 호출하는 내부 AI 서버 (FastAPI, 상태 없음). 기획서 5.5의 엔드포인트 6개와 `/health`.

**따르는 계약:** SPEC `contracts/api-ai.openapi.yaml` 태그 `contracts-v0.2.1` (출처는 이 레포 `app/schemas.py`).

| 모델 | 용도 | 서빙 |
| --- | --- | --- |
| `Qwen/Qwen3.5-4B` | LLM (P1~P5) | vLLM 컨테이너 |
| `google/embeddinggemma-2` (텍스트 인코더만, 768차원) | `/embed` | AI 서버 컨테이너 |

## 실행 모드 (`AI_MODE`)

| 모드 | 언제 | 동작 |
| --- | --- | --- |
| `stub` (기본) | 로컬 개발, BE 연동 개발 (GPU 없이) | 모든 엔드포인트가 가짜 응답. `/health` → `models: {"llm": "stub", "embedding": "stub"}` |
| `real` | GPU 서버, 로컬 Mac(원본 모델 또는 GGUF) | 시작하면 모델별로 로딩 → 예열 → `ready`. 그 전엔 `loading`, 실패하면 `error`. BE는 모두 `ready`일 때만 재처리 |

### 로컬 (stub)

```bash
uv sync
uv run pytest
uv run uvicorn app.main:app --reload --port 8000   # http://localhost:8000/docs
```

### 로컬 (real, Mac)

Mac에서는 vLLM을 쓸 수 없어 vLLM 자리(`VLLM_URL`, 포트 8001)에 로컬 LLM 서버를 띄운다. AI 서버 코드는 같고, AI 서버는 Docker 밖에서 실행한다(Docker에서는 Mac GPU를 쓸 수 없음).

**1) 원본 모델 `Qwen/Qwen3.5-4B` (bf16, 먼저 시도)** — `evals/local_llm.py` (transformers, MPS)

```bash
uv run --extra model python -m evals.local_llm      # :8001, 첫 실행은 모델 9.3GB 다운로드
curl localhost:8001/health
```

- GPU 서버와 같은 가중치다. 생성을 JSON 스키마로 강제하지 못해 스키마를 프롬프트로 알리고, 형식이 틀리면 AI 서버 검증기가 1회 다시 요청한다.
- 선형 어텐션을 빠른 커널 없이 순수 torch로 돌려 느리다. `.env`의 `LLM_TIMEOUT`(기본 600초)으로 끊기지 않게 한다.
- 모델을 못 불러오거나 너무 느리면 2)로 바꾼다.

**2) GGUF (1이 안 될 때)** — llama.cpp `llama-server`, 파일은 `bartowski/Qwen_Qwen3.5-4B-GGUF`

```bash
llama-server -hf bartowski/Qwen_Qwen3.5-4B-GGUF:bf16 --port 8001 --ctx-size 8192 --jinja --reasoning-budget 0
#   메모리가 부족하면 :Q8_0 (4.6GB). JSON 스키마 강제·thinking 끄기를 지원한다
```

**AI 서버 (1·2 공통)**

```bash
cp .env.local.example .env       # 2)를 쓰면 LLM_MODEL을 local/Qwen3.5-4B-gguf 로
uv run --env-file .env uvicorn app.main:app --port 8000
curl localhost:8000/health        # llm·embedding 이 모두 ready 가 될 때까지 대기
```

- 메모리(16GB Mac): 모델 bf16 약 9GB + 임베딩 약 2GB. BE·DB Docker(약 8GB)까지 함께면 부족하다. 전체 흐름은 Docker 메모리를 줄이거나 GGUF `Q8_0`로 본다.
- 로컬은 **로직 확인용**이다. 실행기·정밀도(Mac bf16, T4 fp16)·JSON 강제 방식이 달라 출력이 GPU 서버와 같지 않다. 품질(G1)·응답 시간·부하는 GPU 서버에서 잰다.
- `LLM_MODEL`을 실제 모델 이름과 다르게 두어 로컬 결과가 GPU 서버 결과로 저장되지 않게 한다.
- BE가 Docker 안에 있으면 AI 주소는 `http://host.docker.internal:8000`. 로컬에서는 BE의 AI 읽기 제한도 늘려야 한다.

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
| `app/schemas.py` | 요청·응답 모델. 계약 C-04의 출처 (`/openapi.json`). 바꾸면 `uv run python -m scripts.export_contract ../SPEC-nerdnam/contracts/api-ai.openapi.yaml`로 SPEC에 내보낸다 |
| `app/routers/` | restructure, requirements, embed, match, gap_guide, guide |
| `app/config.py` | 환경변수 (`AI_MODE`, `LLM_MODEL`, `VLLM_URL`, `EMBED_*`) |
| `app/embedding.py` | 임베딩 로딩·접두어 규칙 (서버와 evals가 같이 씀) |
| `app/llm.py` | vLLM 호출 (thinking 끔, JSON 스키마 강제) |
| `evals/local_llm.py` | 로컬 Mac 전용 LLM 서버 (원본 Qwen3.5-4B, vLLM 대신) |
| `docker-compose.yml`, `Dockerfile` | GPU 서버 실행 설정 (ADR-0001) |
| `evals/` | 임베딩 평가, 부하 테스트 |
