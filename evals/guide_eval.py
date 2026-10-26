"""작성 방향 근거 정확도 평가 (AI-11, 기획서 9.1 "근거 정확도 98%"). 띄워 둔 AI 서버(real 모드)의 /guide를 부른다.

    AI_URL=http://localhost:8000 python -m evals.guide_eval [evals/out/guide.json]   # 생성 + 자동 검사
    python -m evals.guide_eval --score evals/out/guide.json                          # 검수(ok) 붙인 뒤 점수

입력은 match_cases.json의 사람 정답 판정·정답 근거다(판정 오류가 섞이지 않게). 텍스트 탭 4개(motive·intro·portfolio·career).
경험 언급 = sources가 붙는 문단(outline)·쪽(pages)과 바꿔 쓰기(rewrites) 한 줄. 줄마다 검수자가 `ok`(true/false)를 붙인다.
- 근거 정확도: 검수한 언급 중 ok 비율 (KPI 98%)
- 자동 검사 (검수 전에 보는 표시): 인용한 근거에 없는 수치, 인용 없는 언급, rewrites의 before가 근거 문장에 없음
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from app.validators import find_span, unsupported_numbers

AI = os.getenv("AI_URL", "http://localhost:8000")
CASES = Path(__file__).with_name("match_cases.json")
DOCS = Path(__file__).with_name("search_cases.json")
TABS = ("motive", "intro", "portfolio", "career")


def mentions(content: dict, texts: dict[int, str]) -> list[dict]:
    """content → 경험 언급 줄. texts: chunk_id → 원문"""
    rows = []
    for kind, key, fields in (("outline", "outline", ("title", "detail")), ("page", "pages", ("title", "hint"))):
        for x in content.get(key, []):
            text = " · ".join(x[f] for f in fields)
            src = " ".join(texts[i] for i in x["sources"])
            rows.append({"kind": kind, "text": text, "sources": x["sources"],
                         "auto": [p for p, bad in (("인용 없음", not x["sources"] and x.get("kind") != "cover"),
                                                   (f"근거에 없는 수치 {unsupported_numbers(text, src)}",
                                                    x["sources"] and unsupported_numbers(text, src))) if bad]})
    for x in content.get("rewrites", []):
        rows.append({"kind": "rewrite", "text": f"{x['before']} → {x['after']}", "sources": [],
                     "auto": [] if any(find_span(x["before"], t) for t in texts.values()) else ["before가 근거 문장에 없음"]})
    return rows


def score(rows: list[dict]) -> dict:
    done = [r for r in rows if r.get("ok") is not None]
    return {"mentions": len(rows), "reviewed": len(done), "accuracy": sum(r["ok"] for r in done) / len(done) if done else None,
            "auto_flagged": sum(bool(r["auto"]) for r in rows)}


def post(body: dict) -> dict:
    req = urllib.request.Request(AI + "/guide", json.dumps(body, ensure_ascii=False).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


def run(out: Path):
    cases, docs = json.loads(CASES.read_text(encoding="utf-8")), json.loads(DOCS.read_text(encoding="utf-8"))
    rows, meta = [], {}
    for u, du in zip(cases["users"], docs["users"]):
        by_id = {x["id"]: x for x in du["docs"]}
        num = {cid: n for n, cid in enumerate(by_id, 1)}  # 계약의 chunk_id는 정수
        texts = {num[c]: by_id[c]["text"] for c in by_id}
        reqs = [{"id": i, "type": r["type"], "text": r["text"], "verdict": r["verdict"], "reason": r["note"],
                 "evidence": [{"chunk_id": num[c], "chunk_type": by_id[c]["type"], "text": by_id[c]["text"]} for c in r["evidence"]]}
                for i, r in enumerate(u["requirements"], 1)]
        for tab in TABS:
            t = time.time()
            try:
                res = post({"tab": tab, "position": u["user"].split(" ", 1)[1], "requirements": reqs})
            except urllib.error.HTTPError as e:
                rows.append({"user": u["user"][:2], "tab": tab, "kind": "error", "text": e.read().decode()[:200],
                             "sources": [], "auto": [f"error {e.code}"], "ok": False})
                print(f"{u['user'][:2]} {tab}: error {e.code}")
                continue
            meta = {"model": res["model"], "prompt_version": res["prompt_version"]}
            got = mentions(res["content"], texts)
            for r in got:
                rows.append({"user": u["user"][:2], "tab": tab, **r,
                             "source_texts": [texts[i] for i in r["sources"]], "ok": None})
            print(f"{u['user'][:2]} {tab}: 언급 {len(got)}개 {time.time() - t:.0f}초")
    s = score(rows)
    print(f"\n{meta}  언급 {s['mentions']}개  자동 검사 표시 {s['auto_flagged']}개")
    for r in rows:
        if r["auto"]:
            print(f"  ! {r['user']} {r['tab']} {r['kind']} {r['auto']}  {r['text'][:80]}")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({**meta, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {out}  (줄마다 ok를 붙인 뒤 --score)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", nargs="?", type=Path, default=Path("evals/out/guide.json"))
    p.add_argument("--score", action="store_true")
    a = p.parse_args()
    if a.score:
        s = score(json.loads(a.out.read_text(encoding="utf-8"))["rows"])
        print(f"근거 정확도 {s['accuracy']:.1%} (검수 {s['reviewed']}/{s['mentions']})  자동 검사 표시 {s['auto_flagged']}" if s["accuracy"] is not None
              else f"검수한 줄 없음 (언급 {s['mentions']})")
    else:
        run(a.out)
