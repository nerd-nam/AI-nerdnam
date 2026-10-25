"""P2 약어 확장어 확인 (AI-16). 띄워 둔 AI 서버(real 모드)의 /requirements/extract에 extract_cases.json을 보내고
확장어가 맞는지 줄별로 본다. 결과 JSON은 embed_eval --p2 입력이 된다.

    AI_URL=http://localhost:8000 python -m evals.extract_eval [evals/out/extract.json]

판정: ok / 검토(allowed에 없는 확장어: 관련어·틀린 뜻일 수 있음) / 빠짐(need인데 비어 있음).
기술명(skills_raw)은 기대값과 비교해 빠진 것·더한 것을 센다 (확장어 프롬프트를 바꾸며 기술명이 나빠지지 않았는지 확인).
"검토"는 틀렸다는 뜻이 아니라 사람이 봐야 한다는 뜻이다. allowed에 넣을지 프롬프트를 고칠지 정한다.

묶음(8줄)과 한 줄씩 두 번 잰다. 실제 공고는 약어 줄이 1~2개뿐이라, 묶음에서만 맞고 한 줄씩은 비우는 경우를 잡기 위해서다
(p2_v2 첫 버전이 묶음 26/28, 한 줄씩 21/28이었다). 결과 JSON은 묶음 결과다.
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

AI = os.getenv("AI_URL", "http://localhost:8000")
CASES = Path(__file__).with_name("extract_cases.json")
BATCH = 8  # 공고 한 건의 요건 줄 수 정도


def norm(s: str) -> str:
    return "".join(s.split()).lower()


def verdict(case: dict, got: list[str]) -> str:
    allowed = {norm(a) for a in case["allowed"]}
    if any(norm(e) not in allowed for e in got):
        return "검토"
    return "빠짐" if case["need"] and not got else "ok"


def post(body: dict) -> tuple[dict, float]:
    t = time.time()
    req = urllib.request.Request(AI + "/requirements/extract", json.dumps(body, ensure_ascii=False).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900)), time.time() - t


if __name__ == "__main__":
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "evals/out/extract.json")
    cases = json.loads(CASES.read_text(encoding="utf-8"))["lines"]
    rows, meta, secs = [], {}, []
    for i in range(0, len(cases), BATCH):
        part = cases[i:i + BATCH]
        r, s = post({"lines": [{"id": i + j + 1, "type": "required", "text": c["text"]} for j, c in enumerate(part)]})
        secs.append(s)
        meta = {"model": r["model"], "prompt_version": r["prompt_version"]}
        for c, x in zip(part, r["lines"]):
            rows.append({**c, "skills_raw": x["skills_raw"], "expansions": x["expansions"], "verdict": verdict(c, x["expansions"])})

    print(f"{meta}  요청 {len(secs)}번, 평균 {sum(secs) / len(secs):.1f}s, 최대 {max(secs):.1f}s")
    for g in dict.fromkeys(r["group"] for r in rows):
        part = [r for r in rows if r["group"] == g]
        print(f"\n[{g}] ok {sum(r['verdict'] == 'ok' for r in part)}/{len(part)}")
        for r in part:
            mark = "  " if r["verdict"] == "ok" else f"{r['verdict']} "
            print(f"  {mark}{r['text']}  → {r['expansions']}")
    print(f"\n전체 ok {sum(r['verdict'] == 'ok' for r in rows)}/{len(rows)}")
    want = sum(len(r["skills"]) for r in rows)
    missed = [(r["text"], s) for r in rows for s in r["skills"] if s not in r["skills_raw"]]
    extra = [(r["text"], s) for r in rows for s in r["skills_raw"] if s not in r["skills"]]
    print(f"기술명 {want - len(missed)}/{want}  빠짐 {missed}  추가 {extra}")

    solo = 0
    print("\n[한 줄씩]")
    for c in cases:
        r, _ = post({"lines": [{"id": 1, "type": "required", "text": c["text"]}]})
        e = r["lines"][0]["expansions"]
        v = verdict(c, e)
        solo += v == "ok"
        if v != "ok":
            print(f"  {v} {c['text']}  → {e}")
    print(f"한 줄씩 ok {solo}/{len(cases)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({**meta, "lines": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {out_path}  (다음: python -m evals.embed_eval --p2 {out_path})")
