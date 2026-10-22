from fastapi import FastAPI

from app.routers import embed, guide, match, requirements, restructure

app = FastAPI(title="nerdnam-ai", version="0.1.0")

for r in (restructure, requirements, embed, match, guide):
    app.include_router(r.router)


@app.get("/health")
def health():
    # TODO(AI-02): vLLM·임베딩 모델 준비 여부를 확인해 models="ready"로.
    # 지금은 모델이 없으므로 "stub" — BE는 이 값으로 임베딩 대기 재처리를 시작하지 않는다
    return {"status": "ok", "models": "stub"}
