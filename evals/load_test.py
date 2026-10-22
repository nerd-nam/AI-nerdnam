"""동시 요청 부하 테스트 (GPU 서버에서 실행, docker compose up 이후).

재구성과 같은 크기의 LLM 요청(vLLM 직접)과 /embed 요청(AI 서버)을 동시에 N개씩 보내
응답 시간·실패 수·VRAM 최댓값을 잰다. --max-num-seqs 16 기준으로 어디까지 버티는지 확인.

    python -m evals.load_test                      # 동시 1, 4, 8, 16
    python -m evals.load_test --levels 16 24 --rounds 2
"""
import argparse
import asyncio
import json
import statistics
import subprocess
import time

import httpx

from app import llm

RAW = (  # 기획서 7장 예시 원문 (Kaggle 테스트와 같은 입력)
    "OO커머스 결제플랫폼팀에서 백엔드 리드(팀 4명)로 결제 승인 API 성능 개선을 맡았다. "
    "프로모션 기간 결제 승인 응답이 평균 1.2초까지 느려져 이탈이 발생했다. "
    "가맹점 조회를 Redis 캐시로 전환하고 승인 후처리를 Kafka 비동기로 분리해 "
    "평균 응답을 0.3초로 줄였고 피크 TPS를 약 2배 처리하게 됐다. "
    "또 프로모션 트래픽을 미리 확인할 방법이 없어 문제를 운영 중에야 발견했기 때문에, "
    "실제 트래픽 패턴의 부하 테스트 절차와 장애 대응 매뉴얼을 만들어 팀에 공유했고 "
    "다음 프로모션 피크를 문제없이 넘겼다."
)
PROMPT = (
    "다음 프로젝트 원문을 서로 다른 문제 해결 사례로 나누고, 사례마다 [문제]-[해결]-[결과]로 정리하라. "
    f"원문에 없는 사실·수치는 쓰지 말고 빠진 정보는 questions에 질문으로 남겨라.\n\n원문:\n{RAW}"
)
SCHEMA = {
    "type": "object",
    "required": ["episodes"],
    "properties": {"episodes": {"type": "array", "items": {
        "type": "object",
        "required": ["title", "problem", "solution", "result", "questions"],
        "properties": {k: {"type": "string"} for k in ("title", "problem", "solution", "result")}
        | {"questions": {"type": "array", "items": {"type": "string"}}},
    }}},
}
EMBED_BODY = {"purpose": "document", "items": [{"title": "결제 승인 응답 지연 개선", "text": RAW}] * 16}


async def call_llm(_):
    json.loads(await llm.chat([{"role": "user", "content": PROMPT}], schema=SCHEMA))


async def call_embed(ai: httpx.AsyncClient):
    (await ai.post("/embed", json=EMBED_BODY)).raise_for_status()


async def timed(fn, ai):
    t = time.perf_counter()
    await fn(ai)
    return time.perf_counter() - t


def vram_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None  # nvidia-smi 없음 (GPU 서버가 아님)


async def watch_vram(peak: list):
    while True:
        used = await asyncio.to_thread(vram_used_mib)
        if used:
            peak[0] = max(peak[0], used)
        await asyncio.sleep(0.5)


async def level(n: int, rounds: int, ai: httpx.AsyncClient):
    """LLM n개 + /embed n개를 동시에 보내기를 rounds 번."""
    lat = {"llm": [], "embed": []}
    errors = []
    peak = [0]
    watcher = asyncio.create_task(watch_vram(peak))
    t0 = time.perf_counter()
    for _ in range(rounds):
        jobs = [("llm", call_llm)] * n + [("embed", call_embed)] * n
        results = await asyncio.gather(*(timed(fn, ai) for _, fn in jobs), return_exceptions=True)
        for (kind, _), r in zip(jobs, results):
            if isinstance(r, Exception):
                errors.append(f"{kind}: {type(r).__name__} {str(r)[:80]}")
            else:
                lat[kind].append(r)
    wall = time.perf_counter() - t0
    watcher.cancel()

    def stat(xs):
        if not xs:
            return "-"
        p95 = sorted(xs)[max(0, round(len(xs) * 0.95) - 1)]
        return f"p50 {statistics.median(xs):5.1f}s  p95 {p95:5.1f}s  max {max(xs):5.1f}s"

    print(f"\n동시 {n:>2}  ({rounds}회, {wall:.0f}s, VRAM 최대 {peak[0] or '-'} MiB, 실패 {len(errors)})")
    print(f"  LLM   {stat(lat['llm'])}")
    print(f"  embed {stat(lat['embed'])}")
    for e in errors[:5]:
        print(f"  ! {e}")


async def main(levels, rounds, ai_url):
    async with httpx.AsyncClient(base_url=ai_url, timeout=60) as ai:
        health = (await ai.get("/health")).json()
        if health.get("models") != "ready":
            raise SystemExit(f"AI 서버가 준비되지 않음: {health}")
        await call_llm(ai)  # 예열
        for n in levels:
            await level(n, rounds, ai)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--levels", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--ai-url", default="http://localhost:8000")
    a = p.parse_args()
    asyncio.run(main(a.levels, a.rounds, a.ai_url))
