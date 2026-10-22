import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.config import LLM_MODEL
from app.schemas import GuideRequest

router = APIRouter()


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/guide")
def guide(req: GuideRequest):
    # TODO(AI-08~10): P4·P5 스트리밍. 지금은 delta 2개 + done 1개 가짜 이벤트
    cite = req.evidence[0].chunk_id if req.evidence else None

    def events():
        yield sse("delta", {"text": f"(stub) {req.tab} 방향성 "})
        yield sse("delta", {"text": f"[{cite}]" if cite else ""})
        yield sse("done", {"tab": req.tab, "model": LLM_MODEL, "prompt_version": "stub"})

    return StreamingResponse(events(), media_type="text/event-stream")
