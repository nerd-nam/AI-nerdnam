"""요건 판정 평가 (AI-11, ADR-0006). 띄워 둔 AI 서버(real 모드)의 /match에 match_cases.json을 보내 사람 정답과 비교한다.

    python -m evals.match_eval --build                      # 후보 20개를 D-12 방식별로 미리 뽑아 저장 (--extra model, 서버 불필요)
    AI_URL=http://localhost:8000 python -m evals.match_eval [--candidates per_type|current] [evals/out/match.json]

후보를 미리 뽑아 두는 이유: Mac 16GB에서 LLM·AI 서버와 임베딩 모델을 한꺼번에 올리기 어렵다.

지표 (ADR-0006 "사람 검수 기준"):
- 판정 일치율: 3단계가 정확히 같은 비율 (KPI 매칭 정확도 85%)
- 과대 판정률: 정답보다 높게 판정한 비율. 일치율이 같으면 이게 낮은 쪽이 낫다
- 큰 오류: met ↔ unmet
- 근거 정밀도: met·partial로 고른 근거 중 정답 근거 목록에 있는 비율
- 이유 수치 오류: reason에 후보에 없는 수치 (자동으로 잡을 수 있는 사실 오류만)
- 검색 누락: 정답 근거가 후보에 하나도 없는 met·partial 요건. 판정 일치율에서 빼고 따로 센다
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from app.validators import unsupported_numbers

AI = os.getenv("AI_URL", "http://localhost:8000")
CASES = Path(__file__).with_name("match_cases.json")
DOCS = Path(__file__).with_name("search_cases.json")
ORDER = {"unmet": 0, "partial": 1, "met": 2}


def build(device: str):
    import numpy as np

    from app import embedding
    from evals.embed_eval import query_text
    from evals.search_eval import current, per_type

    cases, docs = json.loads(CASES.read_text(encoding="utf-8")), json.loads(DOCS.read_text(encoding="utf-8"))
    model = embedding.load(device=device)
    for u, du in zip(cases["users"], docs["users"]):
        d = du["docs"]
        dv = np.array(embedding.encode(model, "document", [(x["text"], x["title"]) for x in d]))
        qv = np.array(embedding.encode(model, "query", [(query_text(r["text"], r["expansions"]), None) for r in u["requirements"]]))
        types = [x["type"] for x in d]
        for r, sims in zip(u["requirements"], qv @ dv.T):
            kw = np.array([len(set(r["skills"]) & set(x["skills"])) for x in d], dtype=float)
            r["candidates"] = {"per_type": [d[i]["id"] for i in per_type(sims, kw, types, 20)],
                               "current": [d[i]["id"] for i in current(sims, kw, 20)]}
    CASES.write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"저장: {CASES}")


def post(body: dict) -> tuple[dict, float]:
    t = time.time()
    req = urllib.request.Request(AI + "/match", json.dumps(body, ensure_ascii=False).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600)), time.time() - t


def score(rows: list[dict]) -> dict:
    """rows: gold, got, gold_evidence(후보 안), evidence, reason_numbers"""
    judged = [r for r in rows if not r["search_miss"]]
    n = len(judged)
    sel = [i for r in judged if r["got"] != "unmet" for i in r["evidence"]]
    sel_ok = sum(i in r["gold_evidence"] for r in judged if r["got"] != "unmet" for i in r["evidence"])
    return {
        "n": n,
        "agree": sum(r["got"] == r["gold"] for r in judged) / n,
        "over": sum(ORDER.get(r["got"], -1) > ORDER[r["gold"]] for r in judged) / n,
        "under": sum(-1 < ORDER.get(r["got"], -1) < ORDER[r["gold"]] for r in judged) / n,
        "big": sum({r["got"], r["gold"]} == {"met", "unmet"} for r in judged),
        "errors": sum(r["got"] not in ORDER for r in judged),
        "evidence_precision": sel_ok / len(sel) if sel else None,
        "reason_numbers": sum(bool(r["reason_numbers"]) for r in rows),
        "search_miss": sum(r["search_miss"] for r in rows),
    }


def run(which: str, out: Path):
    cases, docs = json.loads(CASES.read_text(encoding="utf-8")), json.loads(DOCS.read_text(encoding="utf-8"))
    rows, meta, total = [], None, 0.0
    for u, du in zip(cases["users"], docs["users"]):
        by_id = {x["id"]: x for x in du["docs"]}
        num = {cid: n for n, cid in enumerate(by_id, 1)}  # 계약의 chunk_id는 정수
        back = {n: cid for cid, n in num.items()}
        reqs = [{"id": i, "type": r["type"], "text": r["text"], "skills": r["skills"],
                 "candidates": [{"chunk_id": num[c], "chunk_type": by_id[c]["type"], "text": by_id[c]["text"]}
                                for c in r["candidates"][which]]}
                for i, r in enumerate(u["requirements"], 1)]
        # 요건 하나씩 보낸다. 묶어 보내면 한 요건의 출력 오류(502)로 나머지 결과까지 잃는다
        t0 = time.time()
        for req, r in zip(reqs, u["requirements"]):
            try:
                res, _ = post({"requirements": [req]})
                meta = {"model": res["model"], "prompt_version": res["prompt_version"]}
                g = res["results"][0]
            except urllib.error.HTTPError as e:
                g = {"verdict": f"error {e.code}", "evidence_chunk_ids": [], "reason": e.read().decode()[:200]}
            cand = set(r["candidates"][which])
            gold_ev = set(r["evidence"]) & cand
            source = " ".join([r["text"]] + [by_id[c]["text"] for c in cand])
            rows.append({"user": u["user"][:2], "text": r["text"], "rule": r["rule"], "gold": r["verdict"],
                         "got": g["verdict"], "evidence": [back[e] for e in g["evidence_chunk_ids"]],
                         "gold_evidence": sorted(gold_ev), "reason": g["reason"],
                         "reason_numbers": unsupported_numbers(g["reason"], source),
                         "search_miss": r["verdict"] != "unmet" and not gold_ev})
        total += time.time() - t0
        print(f"{u['user']}: 요건 {len(reqs)}개 {time.time() - t0:.0f}초")
    s = score(rows)
    print(f"\n{meta}  후보 {which}  전체 {total:.0f}초")
    print(f"판정 일치율 {s['agree']:.0%} ({s['n']}줄)  과대 {s['over']:.0%}  과소 {s['under']:.0%}  큰 오류 {s['big']}  오류 응답 {s['errors']}"
          f"  근거 정밀도 {s['evidence_precision']:.0%}  이유 수치 오류 {s['reason_numbers']}  검색 누락 {s['search_miss']}")
    conf = Counter((r["gold"], r["got"]) for r in rows if not r["search_miss"])
    print("정답 \\ 판정   met  partial  unmet")
    for gv in ("met", "partial", "unmet"):
        print(f"  {gv:<10} " + "  ".join(f"{conf[(gv, v)]:>5}" for v in ("met", "partial", "unmet")))
    for r in rows:
        if r["got"] != r["gold"] and not r["search_miss"]:
            print(f"  X {r['user']} [{r['rule']}] 정답 {r['gold']} → {r['got']}  {r['text']}  | {r['reason']}")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({**meta, "candidates": which, "score": s, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", nargs="?", type=Path, default=Path("evals/out/match.json"))
    p.add_argument("--build", action="store_true")
    p.add_argument("--candidates", default="per_type", choices=["per_type", "current"])
    p.add_argument("--device", default=None)
    a = p.parse_args()
    if a.build:
        from evals.embed_eval import default_device
        build(a.device or default_device())
        sys.exit()
    run(a.candidates, a.out)
