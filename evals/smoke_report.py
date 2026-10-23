"""로컬 real 모드 리포트 흐름 확인: SPEC fixtures로 BE 순서를 흉내 내 /embed → /match → /gap-guide → /guide(5탭).
사용: AI_URL=http://localhost:8000 python evals/smoke_report.py ../SPEC-nerdnam/fixtures [결과.json]
후보는 요건 임베딩과 청크 임베딩의 코사인 순서(BE 하이브리드 검색의 벡터 부분). 판정은 fixtures match.json과 비교한다."""
import json, os, sys, time, urllib.request

AI = os.getenv("AI_URL", "http://localhost:8000")
D = sys.argv[1]
chunks = json.load(open(f"{D}/chunks.json"))["chunks"]
jd = json.load(open(f"{D}/jd.json"))
gold = {r["requirement_id"]: r for r in json.load(open(f"{D}/match.json"))["match_results"]}
by_id = {c["chunk_id"]: c for c in chunks}
ctx = jd.get("context", {})
TASKS = ctx.get("duties", [])
INTRO = " ".join(x for x in (ctx.get("intro", ""), ctx.get("talent", "")) if x)
out = {}


def post(path, body):
    t = time.time()
    req = urllib.request.Request(AI + path, json.dumps(body, ensure_ascii=False).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    print(f"\n--- {path} {time.time() - t:.1f}s prompt={r.get('prompt_version', '-')}")
    return r


def cand(c):
    return {"chunk_id": c["chunk_id"], "chunk_type": c["chunk_type"], "text": c["text"]}


# 1) 후보: 벡터 순위
dv = post("/embed", {"purpose": "document", "items": [{"text": c["text"], "title": c["title"]} for c in chunks]})["vectors"]
reqs = jd["requirements"]
qv = post("/embed", {"purpose": "query", "items": [{"text": r["text"]} for r in reqs]})["vectors"]
ranked = {r["id"]: [c for _, c in sorted(((sum(a * b for a, b in zip(q, d)), c) for d, c in zip(dv, chunks)),
                                         key=lambda x: -x[0])] for r, q in zip(reqs, qv)}

# 2) 판정
m = post("/match", {"requirements": [
    {"id": r["id"], "type": r["type"], "text": r["text"], "skills": r["skills"], "min_years": r["min_years"],
     "candidates": [cand(c) for c in ranked[r["id"]][:20]]} for r in reqs]})
out["match"] = m
ok = 0
for x in m["results"]:
    g = gold[x["id"]]
    same = x["verdict"] == g["verdict"]
    ok += same
    text = next(r["text"] for r in reqs if r["id"] == x["id"])
    print(f"  {'O' if same else 'X'} {x['id']} {text[:20]:<22} {x['verdict']:<8} 근거 {x['evidence_chunk_ids']}  "
          f"(fixtures {g['verdict']} {g['evidence_chunk_ids']})\n      이유: {x['reason']}")
print(f"  판정 일치 {ok}/{len(reqs)}")

# 3) 갭 대응 가이드: 부분 충족·미충족, 인접 경험 = 근거로 뽑히지 못한 상위 후보 5개
res = {x["id"]: x for x in m["results"]}
gaps = [r for r in reqs if res[r["id"]]["verdict"] != "met"]
guides = {}
if gaps:
    gg = post("/gap-guide", {"tasks": TASKS, "requirements": [
        {"id": r["id"], "type": r["type"], "text": r["text"], "verdict": res[r["id"]]["verdict"],
         "adjacent": [cand(c) for c in ranked[r["id"]] if c["chunk_id"] not in res[r["id"]]["evidence_chunk_ids"]][:5]}
        for r in gaps]})
    out["gap_guide"] = gg
    for g in gg["guides"]:
        guides[g["id"]] = g
        print(f"  {g['id']} 가이드: {g['advice']}\n      하지 말 것: {g['avoid']}  인용 {g['cited_chunk_ids']}")

# 4) 작성 방향 5탭
greqs = [{"id": r["id"], "type": r["type"], "text": r["text"], "verdict": res[r["id"]]["verdict"],
          "reason": res[r["id"]]["reason"], "evidence": [cand(by_id[i]) for i in res[r["id"]]["evidence_chunk_ids"]],
          "gap_guide": {"advice": guides[r["id"]]["advice"], "avoid": guides[r["id"]]["avoid"]} if r["id"] in guides else None}
         for r in reqs]
out["guide"] = {}
for tab in ("motive", "intro", "portfolio", "career", "image"):
    g = post("/guide", {"tab": tab, "company": jd.get("company", ""), "position": jd.get("position", ""),
                        "company_intro": INTRO, "tasks": TASKS, "requirements": greqs})
    out["guide"][tab] = g
    print(json.dumps(g["content"], ensure_ascii=False, indent=1)[:1500], "\n  cited:", g["cited_chunk_ids"])

if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False, indent=2)
