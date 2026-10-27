"""사례 수정 요청 반영 확인 (AI #25). revise_cases.json의 피드백마다 /restructure(mode=episode)를 부르고
본문이 바뀌었는지, 원문에 없는 수치가 생겼는지, LLM을 몇 번 불렀는지 본다.

    uv run --env-file .env python -m evals.revise_eval     # Bedrock 직접 (AWS_BEARER_TOKEN_BEDROCK)

AI 서버를 띄우지 않고 라우터 함수를 같은 프로세스에서 부른다.
판정: ok / 그대로(change인데 피드백이 가리키는 fields가 하나도 안 바뀜) / 수치(원문에 없는 수치) / 502(서버가 반영 못 함으로 실패).
"""
import asyncio
import json
import sys
import time
from pathlib import Path

from fastapi import HTTPException  # noqa: E402

from app import llm, state  # noqa: E402
from app.routers import restructure as r  # noqa: E402
from app.schemas import RestructureRequest  # noqa: E402
from app.validators import unsupported_numbers  # noqa: E402

CASES = Path(__file__).with_name("revise_cases.json")
BODY = ("title", "role", "problem", "solution", "result")


def body(e: dict) -> tuple[str, ...]:
    return tuple(" ".join(e.get(f, "").split()) for f in BODY)


async def run(case: dict, p: dict) -> dict:
    calls = 0
    chat = llm.chat

    async def counted(*a, **k):
        nonlocal calls
        calls += 1
        return await chat(*a, **k)
    llm.chat = counted
    req = RestructureRequest(mode="episode", title=p["title"], source_text=p["source_text"], skills=p["skills"],
                             target_episode=p["target"], feedback=case["feedback"])
    t = time.time()
    try:
        ep = (await r.restructure(req)).episodes[0].model_dump()
        status = 200
    except HTTPException as e:
        ep, status = None, e.status_code
    finally:
        llm.chat = chat
    row = {**case, "status": status, "calls": calls, "secs": round(time.time() - t, 1), "episode": ep}
    if ep:
        row["changed"] = [f for f, a, b in zip(BODY, body(ep), body(p["target"])) if a != b]
        row["numbers"] = sorted(unsupported_numbers(" ".join(body(ep)), p["source_text"]))
    if status != 200:
        row["verdict"] = str(status)
    elif case["expect"] == "change" and not set(case["fields"]) & set(row["changed"]):
        row["verdict"] = "그대로"  # 피드백이 가리키는 필드가 하나도 안 바뀜
    else:
        row["verdict"] = "수치" if row["numbers"] else "ok"
    return row


async def main(out_path: Path):
    data = json.loads(CASES.read_text(encoding="utf-8"))
    state.models["llm"] = "ready"
    rows = [await run(c, data["projects"][c["project"]]) for c in data["cases"]]
    print(f"prompt={r.PROMPT_VERSION}  ok {sum(x['verdict'] == 'ok' for x in rows)}/{len(rows)}  "
          f"다시 요청 {sum(x['calls'] > 1 for x in rows)}건  평균 {sum(x['secs'] for x in rows) / len(rows):.1f}s")
    for x in rows:
        ep = x["episode"] or {}
        print(f"[{x['verdict']}] {x['project']} '{x['feedback']}'  {x['secs']}s 호출 {x['calls']}  바뀜 {x.get('changed')}")
        if ep:
            print(f"     제목: {ep['title']} | 해결: {ep['solution']}")
            for c in ep["clarifications"]:
                print(f"     ? [{c['field']}] {c['question']}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1] if len(sys.argv) > 1 else "evals/out/revise.json")))
