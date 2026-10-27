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

Bedrock 키 없이 가짜 응답(stub)으로 바로 띄울 수 있습니다. BE 연동 개발은 이것으로 충분합니다.

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
| `POST /embed` | 텍스트를 1024차원 벡터로 바꿈. `purpose`: `document` 청크 저장 / `query` 요건 검색 | — | — |
| `POST /match` | 요건마다 경험 후보(최대 20개)를 보고 판정(`met`/`partial`/`unmet`)·근거(최대 5개)·이유를 줌 | P3 | S08 |
| `POST /gap-guide` | 미충족·부분 충족 요건을 지원서에서 정직하게 다루는 가이드 | P6 | S08 |
| `POST /guide` | 탭 하나의 작성 방향. 탭: `motive` `intro` `portfolio` `career`(P4), `image`(P5) | P4·P5 | S09 |

### 모든 엔드포인트의 공통 규칙

- 생성 응답에는 `model`·`prompt_version`이, `/embed` 응답에는 `embedding_model`이 들어 있습니다. 값이 `"stub"`이면 가짜 응답이니 API 서버는 저장하지 않습니다.
- 오류 응답

  | 상태 코드 | 뜻 | API 서버 처리 |
  | --- | --- | --- |
  | 503 | 연결 확인 중, Bedrock 호출 한도 초과·일시 장애·연결 실패 | `GPU_UNAVAILABLE`, 1회 재시도 |
  | 502 | LLM 출력 형식이 두 번 연속 틀림, Bedrock이 요청을 거부(400·403 등) | `AI_ERROR`, 1회 재시도 |
  | 504 | LLM 생성 시간 초과 | `TIMEOUT`, 재시도 안 함 |
  | 422 | 요청 형식이 틀림 | `AI_ERROR`, 재시도 안 함 |

- 타임아웃(API 서버 기준): 연결 2초, 읽기는 `/health` 2초 · `/embed` 10초 · `/requirements/extract` 20초 · `/gap-guide` 20초 · `/restructure` 30초 · `/match` 30초 · `/guide` 120초.

자세한 요청·응답 필드는 서버를 띄운 뒤 `/docs`에서 보거나, SPEC의 계약 파일을 보세요. 규칙의 원문은 기획서 5.5입니다.

## 실행 모드 (`AI_MODE`)

| 모드 | 언제 쓰나 | 동작 |
| --- | --- | --- |
| `stub` (기본값) | 로컬 개발, BE 연동 개발 | Bedrock 호출 없이 형식만 같은 가짜 응답을 돌려줌 |
| `real` | 운영, 로컬에서 실제 모델 확인 | AWS Bedrock을 호출함. 시작하면 연결 확인 |

`real` 모드의 `/health` 응답 `models` 값은 아래와 같습니다. LLM과 임베딩은 따로 표시됩니다.

| 값 | 뜻 |
| --- | --- |
| `loading` | 연결 확인 중 (켠 직후 몇 초) |
| `ready` | 사용 가능 |
| `error` | 연결 확인 실패 (키·권한·리전 확인. 1분마다 다시 시도) |

API 서버는 `embedding`이 `ready`일 때 "임베딩 대기" 청크를 다시 처리합니다. `llm`이 준비되지 않았는데 생성 요청이 오면 AI 서버가 503을 돌려줍니다.

### 사용하는 모델

| 모델 | 용도 | 어디서 돌아가나 |
| --- | --- | --- |
| `global.amazon.nova-2-lite-v1:0` (Amazon Nova 2 Lite) | LLM (P1~P6) | AWS Bedrock 서울 리전, Global 교차 리전 추론 프로필 (boto3 Converse) |
| `amazon.titan-embed-text-v2:0` (1024차원) | `/embed` | AWS Bedrock 서울 리전, bedrock-runtime(boto3 InvokeModel) |

- 임베딩은 서울 리전 안에서만 처리합니다(In-Region). LLM(Nova 2 Lite)은 서울에서 Global 교차 리전 프로필로만 제공돼 **요청이 국외에서 처리될 수 있습니다.**
- 임베딩 모델과 차원은 **고정**입니다(`app/config.py`). 바꾸면 저장된 벡터를 전부 다시 계산해야 하므로 ADR 합의 없이 바꾸지 마세요.
- LLM을 바꿀 때는 `LLM_MODEL`만 바꾸고 평가를 다시 돌립니다.
- Bedrock은 새 JSON 스키마를 처음 받을 때 준비에 최대 90초가 걸리고 24시간 캐시합니다. 그래서 AI 서버가 시작할 때와 12시간마다 출력 스키마를 미리 한 번씩 보냅니다(`app/main.py`).

## 실제 모델로 실행하기

Bedrock API 키가 필요합니다(Bedrock 콘솔 → API keys). 키는 `.env`에만 두고 커밋하거나 공유하지 않습니다.

```bash
cp .env.local.example .env       # AWS_BEARER_TOKEN_BEDROCK 채우기
uv run --env-file .env uvicorn app.main:app --port 8000
curl localhost:8000/health       # llm·embedding이 모두 ready가 될 때까지 기다리기
```

Docker로 띄울 때:

```bash
AWS_BEARER_TOKEN_BEDROCK=... docker compose up -d --build
```

흐름 확인(선택) — SPEC의 예시 데이터로 BE 호출 순서를 흉내 냅니다.

```bash
uv run python evals/smoke_restructure.py ../SPEC-nerdnam/fixtures/projects.json   # 사례 나누기
uv run python evals/smoke_report.py ../SPEC-nerdnam/fixtures                      # embed → match → gap-guide → guide
```

- BE가 Docker 안에 있으면 AI 서버 주소는 `http://host.docker.internal:8000`입니다.
- 호출할 때마다 Bedrock 요금이 나갑니다(토큰 단위). 평가를 반복해서 돌릴 때 참고하세요.

운영은 EKS에서 GPU 없이 일반 Pod로 돌립니다. 배포 설정은 `nerdnam-infra` 레포의 `apps/ai/`에 있고, 키는 Secret으로 넣습니다.

## 테스트와 평가

```bash
uv run pytest                                                     # 단위 테스트 (stub, Bedrock 호출 없음)

# 아래는 Bedrock을 실제로 호출합니다 (--env-file .env 로 키 전달)
uv run --env-file .env --extra eval python -m evals.embed_eval    # 임베딩 검색 품질
uv run --env-file .env --extra eval python -m evals.embed_eval --p2 evals/out/extract.json   # P2 확장어로 검색 순위 비교
uv run --env-file .env --extra eval python -m evals.search_eval   # 검색 후보 구성 비교 (D-12): 지금 방식 / 종류별 상위 N / RRF
uv run --env-file .env python -m evals.revise_eval                # 사례 수정 요청 반영 확인 (AI #25, 서버 없이)

uv run python -m evals.extract_eval                               # P2 약어 확장어 확인 (real 모드 서버를 띄운 뒤)
uv run python -m evals.match_eval                                 # 요건 판정 평가 (ADR-0006, real 모드 서버를 띄운 뒤)
```

`--extra eval`은 평가 스크립트가 쓰는 numpy를 설치합니다. 서버에는 필요 없습니다.

## 폴더 구조

```text
AI-nerdnam/
├── app/                      서버 본체
│   ├── main.py               앱 시작, 연결 확인, /health, 라우터 등록
│   ├── config.py             환경변수 읽기
│   ├── schemas.py            요청·응답 모델 = API↔AI 계약의 출처
│   ├── llm.py                Bedrock LLM 호출(boto3 Converse), JSON 형식 강제, 오류 → 상태 코드
│   ├── embedding.py          Bedrock Titan 임베딩 호출(boto3) (서버와 evals가 같이 씀)
│   ├── validators.py         LLM 출력 검증 (형식, 원문에 없는 수치, 인용 ID)
│   ├── state.py              모델 상태 (loading/ready/error)
│   ├── routers/              엔드포인트별 코드
│   │   ├── restructure.py    /restructure
│   │   ├── requirements.py   /requirements/extract
│   │   ├── embed.py          /embed
│   │   ├── match.py          /match
│   │   ├── gap_guide.py      /gap-guide
│   │   └── guide.py          /guide
│   └── prompts/              프롬프트 (p1_v3.md = P1 버전 3)
├── evals/                    평가·확인 도구
│   ├── embed_eval.py         임베딩 검색 품질 (약어 확장어 비교 포함)
│   ├── extract_eval.py       P2 약어 확장어 확인 (extract_cases.json)
│   ├── search_eval.py        검색 후보 구성 비교 D-12 (search_cases.json, 사용자 4명)
│   ├── match_eval.py         요건 판정 평가 ADR-0006 (match_cases.json, 요건 70줄)
│   ├── revise_eval.py        사례 수정 요청 반영 확인 (revise_cases.json)
│   ├── smoke_restructure.py  /restructure 흐름 확인
│   └── smoke_report.py       리포트 흐름 확인
├── tests/                    pytest
├── scripts/export_contract.py   schemas.py → SPEC 계약 파일 내보내기
├── Dockerfile, docker-compose.yml   AI 서버 실행 설정
├── .env.local.example        real 모드 설정 예시 (Bedrock 키)
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
| `AWS_REGION` | `ap-northeast-2` | Bedrock 리전 (서울) |
| `AWS_BEARER_TOKEN_BEDROCK` | (없음) | Bedrock API 키. LLM·임베딩 모두 이 키를 씀. 커밋·공유 금지 |
| `LLM_MODEL` | `global.amazon.nova-2-lite-v1:0` | Bedrock 모델(추론 프로필) ID이자 응답의 `model` 값 |
| `LLM_TIMEOUT` | `60` | LLM 읽기 제한(초) |
| `LLM_MAX_ATTEMPTS` | `1` | LLM 호출 시도 횟수. 운영은 1(재시도는 BE가 함). 평가 스크립트로 몰아서 부를 때만 올린다 |
| `LLM_MAX_MODEL_LEN` | `128000` | 입력+출력 토큰 한도 (모델 문맥 길이) |
| `LLM_TEMPERATURE` | `0.7` | 생성 온도. P2(`/requirements/extract`)는 추출이라 이 값과 상관없이 0 |
| `EMBED_MODEL_ID` | `amazon.titan-embed-text-v2:0` | 임베딩 모델 (바꾸지 않음, 바꾸면 전체 재임베딩) |

## 관련 문서

모두 `SPEC-nerdnam`(nerdnam-docs) 레포에 있습니다.

| 문서 | 내용 |
| --- | --- |
| `기획서.md` 5.5 | API↔AI 엔드포인트, 공통 규칙(오류·타임아웃·stub) |
| `기획서.md` 5.6 | 프롬프트 설계 |
| `작업목록.md` AI-01~16 | AI 담당 작업 목록 |
| `contracts/api-ai.openapi.yaml` | API↔AI 계약 |
| `fixtures/` | 세 레포가 같이 쓰는 예시 데이터 (평가·smoke 스크립트 입력) |
| `adr/0007-bedrock-serving.md` | Bedrock 전환 (모델·리전·차원) |
| `adr/0001-gpu-model-serving.md` | (0007로 대체) 이전 GPU 자체 서빙 구성 |
| `adr/0003-report-diagnosis-gap-guide.md` | `/gap-guide`를 만든 이유 |
| `adr/0005-eks-serving.md` | (0007로 대체) 이전 EKS GPU 운영 구성 |
