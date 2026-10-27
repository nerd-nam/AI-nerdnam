# AI 서버 (FastAPI + 임베딩). vLLM은 docker-compose.yml 의 별도 컨테이너
FROM python:3.12-slim

RUN pip install --no-cache-dir uv==0.13.5
WORKDIR /app

# 의존성 먼저 (코드만 바뀌면 이 층은 캐시 재사용)
COPY pyproject.toml uv.lock ./
# 로컬 stub 이미지는 EXTRAS="" 로 모델 의존성(torch 등 수 GB)을 빼고 만든다 (docker-compose.local.yml)
ARG EXTRAS="--extra model"
RUN uv sync --frozen --no-dev ${EXTRAS} --no-install-project

COPY app ./app
EXPOSE 8000
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
