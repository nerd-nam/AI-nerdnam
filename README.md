# nerdnam-ai

API 서버만 호출하는 내부 AI 서버 (FastAPI, 상태 없음). 기획서 5.5의 엔드포인트 5개와 `/health`.
지금은 모든 엔드포인트가 **가짜 응답**을 돌려줍니다. BE는 GPU 없이 이 서버로 연동 개발을 할 수 있습니다.

```bash
uv sync
uv run uvicorn app.main:app --reload --port 8000   # 문서: http://localhost:8000/docs
uv run pytest
```

| 경로 | 내용 |
| --- | --- |
| `app/schemas.py` | 요청·응답 모델. 계약 C-04 초안의 출처 (`/openapi.json`으로 내보냄) |
| `app/routers/` | restructure, requirements, embed, match, guide |
| `app/config.py` | 응답에 담는 모델 이름 (`LLM_MODEL`, `EMBED_MODEL` 환경변수) |
