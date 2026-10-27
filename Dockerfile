# AI 서버 (FastAPI). LLM·임베딩은 AWS Bedrock 호출 (ADR-0007)
FROM python:3.12-slim

RUN pip install --no-cache-dir uv==0.13.5
WORKDIR /app

# 의존성 먼저 (코드만 바뀌면 이 층은 캐시 재사용)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
EXPOSE 8000
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
