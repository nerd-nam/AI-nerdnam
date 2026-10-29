# AI 서버 (FastAPI). LLM·임베딩은 AWS Bedrock 호출 (ADR-0007)
FROM python:3.12-slim

# .pyc는 빌드 때 만들어 두고 실행 중엔 쓰지 않는다 (읽기 전용 root)
ENV PYTHONDONTWRITEBYTECODE=1 UV_COMPILE_BYTECODE=1
RUN pip install --no-cache-dir uv==0.13.5
WORKDIR /app

# 의존성 먼저 (코드만 바뀌면 이 층은 캐시 재사용)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
# 비root. uid는 인프라 runAsUser(65532)와 같게. 파일은 root 소유 그대로 둔다(앱은 /tmp 말고 쓰지 않음)
RUN groupadd --gid 65532 app && useradd --uid 65532 --gid 65532 --no-create-home --shell /usr/sbin/nologin app
USER 65532:65532
EXPOSE 8000
# 종료 신호(SIGTERM) 뒤 진행 중 요청을 최대 300초 기다림 (BE 읽기 제한 300초, 인프라 종료 유예 330초 안)
# 로그는 JSON 한 줄씩 stdout (trace_id·span_id 포함, #42). 로컬 uv run은 텍스트 그대로
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "300", "--log-config", "app/log.json"]
