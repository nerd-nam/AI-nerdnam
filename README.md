# nerdnam-ai

nerdnam(AI 맞춤형 지원서 코치)의 **AI 서버**입니다. 프로젝트 원문을 사례로 나누고, 공고 요건과 경험을 비교하고, 지원서 작성 방향을 만듭니다.

- **API 서버(BE)만 호출합니다.** 웹(FE)이나 사용자는 직접 부르지 않습니다.
- **상태를 저장하지 않습니다.** DB가 없고, 사용자 정보도 받지 않습니다. 결과 저장은 API 서버가 합니다.
- FastAPI, Python 3.12, 패키지 관리는 [uv](https://docs.astral.sh/uv/)를 씁니다.

> 기획·계약·예시 데이터는 공용 문서 레포 `nerdnam-docs`(로컬 폴더 `SPEC-nerdnam`)에 있습니다.
> 이 서버는 계약 `contracts/api-ai.openapi.yaml`의 태그 **`contracts-v0.3.0`** 을 따릅니다.

## 목차

1. [빠르게 시작하기](#빠르게-시작하기)
2. [엔드포인트](#엔드포인트)
3. [실행 모드](#실행-모드-ai_mode)
4. [실제 모델로 실행하기](#실제-모델로-실행하기)
5. [테스트와 평가](#테스트와-평가)
6. [폴더 구조](#폴더-구조)
7. [계약(API 모양)을 바꿀 때](#계약api-모양을-바꿀-때)
8. [환경변수](#환경변수)
9. [관련 문서](#관련-문서)

## 빠르게 시작하기

GPU 없이 가짜 응답(stub)으로 바로 띄울 수 있습니다. BE 연동 개발은 이것으로 충분합니다.

```bash
uv sync                                              # 의존성 설치
uv run pytest                                        # 테스트
uv run uvicorn app.main:app --reload --port 8000     # 서버 실행
```

- API 문서(Swagger): http://localhost:8000/docs
- 상태 확인: `curl localhost:8000/health` → `{"status":"ok","models":{"llm":"stub","embedding":"stub"}}`

## 엔드포인트

| 엔드포인트 | 하는 일 | 프롬프트 | 쓰는 화면 |
| --- | --- | --- | --- |
| `GET /health` | 모델이 준비됐는지 알려 줌 | — | — |
| `POST /restructure` | 프로젝트 원문을 사례(에피소드) 목록으로 나눔. `mode`: `initial` 처음 나누기 / `split` 다시 나누기 / `episode` 사례 하나 고치기 | P1 | S05 |
| `POST /requirements/extract` | 공고 요건 줄마다 기술명(`skills_raw`), 약어 확장어(`expansions`, 예: MSA → 마이크로서비스 아키텍처), 최소 연차(`min_years`)를 뽑음 | P2 | S07 |
| `POST /embed` | 텍스트를 768차원 벡터로 바꿈. `purpose`: `document` 청크 저장 / `query` 요건 검색 | — | — |
| `POST /match` | 요건마다 경험 후보(최대 20개)를 보고 판정(`met`/`partial`/`unmet`)·근거(최대 5개)·이유를 줌 | P3 | S08 |
| `POST /gap-guide` | 미충족·부분 충족 요건을 지원서에서 정직하게 다루는 가이드 | P6 | S08 |
| `POST /guide` | 탭 하나의 작성 방향. 탭: `motive` `intro` `portfolio` `career`(P4), `image`(P5) | P4·P5 | S09 |

### 모든 엔드포인트의 공통 규칙

- 생성 응답에는 `model`·`prompt_version`이, `/embed` 응답에는 `embedding_model`이 들어 있습니다. 값이 `"stub"`이면 가짜 응답이니 API 서버는 저장하지 않습니다.
- 오류 응답

  | 상태 코드 | 뜻 | API 서버 처리 |
  | --- | --- | --- |
  | 503 | 모델이 꺼졌거나 로딩 중 | `GPU_UNAVAILABLE`, 1회 재시도 |
  | 502 | LLM 출력 형식이 두 번 연속 틀림 | `AI_ERROR`, 1회 재시도 |
  | 504 | LLM 생성 시간 초과 | `TIMEOUT`, 재시도 안 함 |
  | 422 | 요청 형식이 틀림 | `AI_ERROR`, 재시도 안 함 |

- 타임아웃(API 서버 기준): 연결 2초, 읽기는 `/health` 2초 · `/embed` 10초 · `/requirements/extract` 20초 · `/gap-guide` 20초 · `/restructure` 30초 · `/match` 30초 · `/guide` 120초.

자세한 요청·응답 필드는 서버를 띄운 뒤 `/docs`에서 보거나, SPEC의 계약 파일을 보세요. 규칙의 원문은 기획서 5.5입니다.

## 실행 모드 (`AI_MODE`)

| 모드 | 언제 쓰나 | 동작 |
| --- | --- | --- |
| `stub` (기본값) | 로컬 개발, BE 연동 개발 | GPU 없이 형식만 같은 가짜 응답을 돌려줌 |
| `real` | GPU 서버, 로컬 Mac에서 실제 모델 확인 | 시작하면 모델을 로딩·예열함 |

`real` 모드의 `/health` 응답 `models` 값은 아래와 같습니다. LLM과 임베딩은 따로 표시됩니다.

| 값 | 뜻 |
| --- | --- |
| `loading` | 로딩·예열 중 (켠 직후 몇 분) |
| `ready` | 사용 가능 |
| `error` | 로딩 실패 (AI 서버를 다시 시작해야 함) |

API 서버는 `embedding`이 `ready`일 때 "임베딩 대기" 청크를 다시 처리합니다. `llm`이 준비되지 않았는데 생성 요청이 오면 AI 서버가 503을 돌려줍니다.

### 사용하는 모델

| 모델 | 용도 | 어디서 돌아가나 |
| --- | --- | --- |
| `Qwen/Qwen3.5-4B` | LLM (P1~P6) | vLLM 컨테이너 (AI 서버와 따로) |
| `google/embeddinggemma-2` (텍스트 인코더만, 768차원) | `/embed` | AI 서버 안 |

임베딩 모델의 버전(revision)과 차원은 **고정**되어 있습니다(`app/config.py`). 바꾸면 저장된 벡터를 전부 다시 계산해야 하므로 ADR-0001 합의 없이 바꾸지 마세요.

## 실제 모델로 실행하기

### A. 로컬 Mac

Mac에서는 vLLM을 쓸 수 없어서, vLLM 자리(포트 8001)에 로컬 LLM 서버를 띄웁니다. AI 서버 코드는 같습니다. Docker에서는 Mac GPU를 쓸 수 없으므로 AI 서버는 Docker 밖에서 실행합니다.

**1단계: LLM 서버 띄우기 (둘 중 하나)**

① 원본 모델 (먼저 시도) — GPU 서버와 같은 가중치입니다.

```bash
uv run --extra model python -m evals.local_llm     # :8001, 첫 실행은 모델 9.3GB 다운로드
curl localhost:8001/health
```

- JSON 형식을 강제하지 못해 형식을 프롬프트로 알려 줍니다. 형식이 틀리면 AI 서버가 1회 다시 요청합니다.
- 느립니다. `.env`의 `LLM_TIMEOUT`(600초)으로 끊기지 않게 해 둡니다.
- 모델을 못 불러오거나 너무 느리면 ②를 쓰세요.

② GGUF (①이 안 될 때) — llama.cpp `llama-server`

```bash
llama-server -hf bartowski/Qwen_Qwen3.5-4B-GGUF:bf16 --port 8001 --ctx-size 8192 --jinja --reasoning-budget 0
# 메모리가 부족하면 :bf16 대신 :Q8_0 (4.6GB)
```

**2단계: AI 서버 띄우기**

```bash
cp .env.local.example .env       # ②를 쓰면 .env의 LLM_MODEL을 local/Qwen3.5-4B-gguf 로 바꾸기
uv run --env-file .env uvicorn app.main:app --port 8000
curl localhost:8000/health       # llm·embedding이 모두 ready가 될 때까지 기다리기
```

**3단계 (선택): 흐름 확인** — SPEC의 예시 데이터로 BE 호출 순서를 흉내 냅니다.

```bash
uv run python evals/smoke_restructure.py ../SPEC-nerdnam/fixtures/projects.json   # 사례 나누기
uv run python evals/smoke_report.py ../SPEC-nerdnam/fixtures                      # embed → match → gap-guide → guide
```

**4단계 (선택): 요건 약어 확장어 확인 (AI-16)** — 28줄을 P2에 보내고, 나온 확장어로 검색 순위를 비교합니다.

```bash
uv run python -m evals.extract_eval                                           # → evals/out/extract.json
uv run --extra model python -m evals.embed_eval --p2 evals/out/extract.json   # 원문 / 사람 확장어 / P2 확장어 순위
```

- ① 원본 모델이면 `.env`의 `LLM_MODEL`을 그대로, ② GGUF면 `local/Qwen3.5-4B-gguf`로 둡니다. 결과 JSON의 `model` 값으로 어떤 모델로 쟀는지 구분합니다.
- P2는 온도 0입니다. ①은 항상 같은 답을 고르므로(`do_sample=False`) 결과가 같습니다.

**주의**

- 메모리(16GB Mac 기준): LLM bf16 약 9GB + 임베딩 약 2GB. BE·DB Docker(약 8GB)까지 같이 띄우면 부족합니다. 그럴 땐 Docker 메모리를 줄이거나 GGUF `Q8_0`를 쓰세요.
- 로컬은 **로직 확인용**입니다. 실행 방식·정밀도·JSON 강제 방식이 달라서 결과가 GPU 서버와 같지 않습니다. 품질·응답 시간·부하는 GPU 서버에서 잽니다.
- `LLM_MODEL`을 실제 모델 이름과 다르게 두어, 로컬 결과가 GPU 서버 결과로 저장되지 않게 합니다(`.env.local.example`에 이미 그렇게 되어 있음).
- BE가 Docker 안에 있으면 AI 서버 주소는 `http://host.docker.internal:8000`입니다. 로컬은 느리므로 BE의 AI 읽기 타임아웃도 늘려야 합니다.

### B. GPU 서버 한 대 (docker compose)

GPU 서버 한 대에서 시험할 때 씁니다. 운영 배포는 아래 C를 보세요.

```bash
docker compose up -d --build     # vllm이 healthy가 된 뒤 ai가 시작됨
curl localhost:8000/health       # llm·embedding이 모두 ready가 될 때까지 기다리기
```

- 기본값은 T4(g4dn.xlarge)입니다. L4(g6)는 `LLM_DTYPE=auto EMBED_DTYPE=bfloat16 docker compose up -d`
- 호스트에 NVIDIA 드라이버와 NVIDIA Container Toolkit이 필요합니다.
- **시작 순서가 중요합니다.** vLLM이 GPU 메모리의 80%를 먼저 잡아야 해서, 임베딩(AI 서버)이 먼저 올라가면 vLLM이 시작에 실패합니다. compose가 이 순서를 지켜 줍니다.
- 첫 실행은 모델 다운로드와 컴파일 때문에 최대 15분 걸립니다.
- AI 서버(8000)는 보안 그룹에서 API 서버만 허용합니다. vLLM(8001)은 서버 자신에서만 접근할 수 있습니다.

### C. 운영 (EKS)

운영은 EKS 클러스터에서 합니다(ADR-0005). 배포 설정은 이 레포가 아니라 **`nerdnam-infra` 레포의 `apps/ai/`** 에 있고, 이 레포는 이미지를 ECR에 올리는 데까지 맡습니다.

- vLLM과 AI 서버는 한 Pod 안의 두 컨테이너로 돌고, GPU 한 장을 나눠 씁니다.
- 클러스터 안 주소: `http://ai.nerdnam.svc:8000` (라벨 `app: api`인 Pod만 접근 가능)
- 켜기·끄기: `kubectl scale deploy/ai -n nerdnam --replicas=1` / `--replicas=0`

## 테스트와 평가

```bash
uv run pytest                                                     # 단위 테스트 (stub, GPU 불필요)

uv run --extra model python -m evals.embed_eval                   # 임베딩 검색 품질 (Mac·Kaggle·GPU 어디서나)
uv run --extra model python -m evals.embed_eval --compare-full    # 텍스트 인코더만 불러와도 결과가 같은지 확인

uv run python -m evals.extract_eval                               # P2 약어 확장어 확인 (real 모드 서버를 띄운 뒤)
uv run --extra model python -m evals.embed_eval --p2 evals/out/extract.json   # P2 확장어로 검색 순위 비교

uv run python -m evals.load_test                                  # 동시 요청 부하 테스트 (GPU 서버, compose 실행 후)
```

`--extra model`은 torch·sentence-transformers 같은 무거운 패키지를 함께 설치합니다. stub 개발에는 필요 없습니다.

## 폴더 구조

```text
AI-nerdnam/
├── app/                      서버 본체
│   ├── main.py               앱 시작, 모델 로딩·예열, /health, 라우터 등록
│   ├── config.py             환경변수 읽기
│   ├── schemas.py            요청·응답 모델 = API↔AI 계약의 출처
│   ├── llm.py                vLLM 호출 (thinking 끔, JSON 형식 강제)
│   ├── embedding.py          임베딩 모델 로딩, 접두어 규칙 (서버와 evals가 같이 씀)
│   ├── validators.py         LLM 출력 검증 (형식, 원문에 없는 수치, 인용 ID)
│   ├── state.py              모델 상태 (loading/ready/error)
│   ├── routers/              엔드포인트별 코드
│   │   ├── restructure.py    /restructure
│   │   ├── requirements.py   /requirements/extract
│   │   ├── embed.py          /embed
│   │   ├── match.py          /match
│   │   ├── gap_guide.py      /gap-guide
│   │   └── guide.py          /guide
│   └── prompts/              프롬프트 (p1_v2.md = P1 버전 2)
├── evals/                    평가·확인 도구
│   ├── embed_eval.py         임베딩 검색 품질 (약어 확장어 비교 포함)
│   ├── extract_eval.py       P2 약어 확장어 확인 (extract_cases.json)
│   ├── load_test.py          부하 테스트
│   ├── local_llm.py          Mac용 로컬 LLM 서버 (vLLM 대신)
│   ├── smoke_restructure.py  /restructure 흐름 확인
│   └── smoke_report.py       리포트 흐름 확인
├── tests/                    pytest
├── scripts/export_contract.py   schemas.py → SPEC 계약 파일 내보내기
├── Dockerfile, docker-compose.yml   GPU 서버 한 대 실행 설정
├── .env.local.example        로컬 Mac real 모드 설정 예시
└── pyproject.toml, uv.lock
```

### 프롬프트를 바꿀 때

- 파일 이름이 버전입니다(`p3_v2.md`). 내용을 크게 바꾸면 새 버전 파일을 만듭니다. 이 버전이 응답의 `prompt_version`으로 나가 API 서버에 결과와 함께 저장됩니다.
- P2는 LLM을 두 번(동시에) 부릅니다. 기술명 `p2_v2.md`와 약어 확장어 `p2_v2_abbr.md`가 한 버전(`p2_v2`)입니다. 둘 중 하나를 바꿔도 버전을 같이 올립니다.
- 바꾼 뒤에는 평가 세트로 품질을 다시 확인합니다(작업목록 AI-11).

## 계약(API 모양)을 바꿀 때

API↔AI 계약(`contracts/api-ai.openapi.yaml`)의 출처는 이 레포의 `app/schemas.py`입니다. **SPEC의 계약 파일은 손으로 고치지 않습니다.**

1. `app/schemas.py`를 고칩니다.
2. 계약 파일을 다시 내보냅니다.
   ```bash
   uv run python -m scripts.export_contract ../SPEC-nerdnam/contracts/api-ai.openapi.yaml
   ```
3. SPEC 레포에 PR을 올립니다. `contracts/` 변경은 세 담당(FE·BE·AI)이 모두 리뷰합니다. 합의되면 `contracts-vX.Y.Z` 태그를 달고, 이 README 맨 위의 태그도 바꿉니다.

BE 레포에는 계약과 구현을 대조하는 테스트(`ApiAiContractTest`)가 있어서, 둘이 어긋나면 CI에서 잡힙니다.

## 환경변수

| 이름 | 기본값 | 설명 |
| --- | --- | --- |
| `AI_MODE` | `stub` | `stub` 가짜 응답 / `real` 실제 모델 |
| `LLM_MODEL` | `Qwen/Qwen3.5-4B` | vLLM에 요청할 모델 이름이자 응답의 `model` 값 |
| `VLLM_URL` | `http://localhost:8001` | LLM 서버 주소 |
| `LLM_TIMEOUT` | `60` | LLM 읽기 제한(초). 로컬 Mac은 600 |
| `LLM_MAX_MODEL_LEN` | `8192` | 입력+출력 토큰 한도. vLLM `--max-model-len`과 같게 |
| `LLM_TEMPERATURE` | `0.7` | 생성 온도. P2(`/requirements/extract`)는 추출이라 이 값과 상관없이 0 |
| `LLM_PRESENCE_PENALTY` | `1.5` | 반복 억제 |
| `EMBED_REPO` | `google/embeddinggemma-2` | 임베딩 모델 (바꾸지 않음) |
| `EMBED_REVISION` | `914f7f8…` | 임베딩 모델 버전 (고정, 바꾸지 않음) |
| `EMBED_DTYPE` | `float32` | T4는 `float32`(fp16은 NaN 위험), L4는 `bfloat16` |
| `HF_TOKEN` | (없음) | Hugging Face 토큰. compose가 컨테이너에 넘겨 줌 |

## 관련 문서

모두 `SPEC-nerdnam`(nerdnam-docs) 레포에 있습니다.

| 문서 | 내용 |
| --- | --- |
| `기획서.md` 5.5 | API↔AI 엔드포인트, 공통 규칙(오류·타임아웃·stub) |
| `기획서.md` 5.6 | 프롬프트 설계 |
| `작업목록.md` AI-01~16 | AI 담당 작업 목록 |
| `contracts/api-ai.openapi.yaml` | API↔AI 계약 |
| `fixtures/` | 세 레포가 같이 쓰는 예시 데이터 (평가·smoke 스크립트 입력) |
| `adr/0001-gpu-model-serving.md` | 모델·GPU 선택, 임베딩 고정 규칙, 시작 순서 |
| `adr/0003-report-diagnosis-gap-guide.md` | `/gap-guide`를 만든 이유 |
| `adr/0005-eks-serving.md` | EKS 운영 구성 |
