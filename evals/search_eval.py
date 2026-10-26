"""검색 후보 구성 비교 (D-12, AI-11). 요건마다 /match로 넘길 후보 20개를 세 방식으로 뽑아 정답 청크가 얼마나 들어가는지 잰다.

- current: BE `ChunkSearch`와 같은 방식. 벡터 상위 20 ∪ 기술명이 겹치는 청크, 점수 = 0.7 × 코사인(벡터 상위 20 안일 때) + 0.3 × 겹침
- per_type: 같은 점수로 청크 종류별 상위 N(기술 5 · 경력 3 · 프로젝트 4 · 사례 8)을 뽑고, 모자라면 나머지에서 채움
- rrf: 벡터 순위와 기술명 겹침 순위의 순위 합산(RRF, k=60)

    python -m evals.search_eval                # 후보 20개 (BE 기본)
    python -m evals.search_eval --limit 10     # 후보를 줄였을 때 차이가 더 잘 보인다
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from app import embedding
from evals.embed_eval import default_device, query_text

CASES = Path(__file__).with_name("search_cases.json")
TYPES = ["skill", "career", "project_summary", "episode"]
QUOTA = {"skill": 5, "career": 3, "project_summary": 4, "episode": 8}  # limit 20 기준, 다른 limit이면 비율대로
VW, KW, RRF_K = 0.7, 0.3, 60


def current(cos: np.ndarray, kw: np.ndarray, limit: int) -> list[int]:
    vec = set(np.argsort(-cos)[:limit].tolist())
    # ponytail: BE SQL의 kw는 ORDER BY 없이 LIMIT이라 겹치는 청크가 limit보다 많으면 어느 것이 남을지 정해지지 않는다. 여기선 전부 넣는다
    pool = vec | set(np.flatnonzero(kw > 0).tolist())
    score = {i: VW * cos[i] * (i in vec) + KW * (kw[i] > 0) for i in pool}
    return sorted(pool, key=lambda i: (-score[i], i))[:limit]


def per_type(cos: np.ndarray, kw: np.ndarray, types: list[str], limit: int) -> list[int]:
    score = VW * cos + KW * (kw > 0)
    order = np.argsort(-score).tolist()
    picked = []
    for t, n in QUOTA.items():
        picked += [i for i in order if types[i] == t][: round(n * limit / 20)]
    picked += [i for i in order if i not in picked][: max(0, limit - len(picked))]
    return sorted(picked[:limit], key=lambda i: -score[i])


def rrf(cos: np.ndarray, kw: np.ndarray, limit: int) -> list[int]:
    vrank = {i: r for r, i in enumerate(np.argsort(-cos).tolist(), 1)}
    hits = sorted(np.flatnonzero(kw > 0).tolist(), key=lambda i: (-kw[i], -cos[i]))
    krank = {i: r for r, i in enumerate(hits, 1)}
    score = {i: 1 / (RRF_K + vrank[i]) + (1 / (RRF_K + krank[i]) if i in krank else 0) for i in vrank}
    return sorted(score, key=lambda i: -score[i])[:limit]


def run(cases: dict, device: str, limit: int, expand: bool):
    model = embedding.load(device=device)
    stats = {m: {"gold": Counter(), "found": Counter(), "all": 0, "n": 0, "mix": Counter(), "miss": []}
             for m in ("current", "per_type", "rrf")}
    for u in cases["users"]:
        docs = u["docs"]
        dv = np.array(embedding.encode(model, "document", [(d["text"], d["title"]) for d in docs]))
        texts = [query_text(q["text"], q["expansions"] if expand else []) for q in u["queries"]]
        qv = np.array(embedding.encode(model, "query", [(t, None) for t in texts]))
        types = [d["type"] for d in docs]
        ids = [d["id"] for d in docs]
        for q, sims in zip(u["queries"], qv @ dv.T):
            want = set(q["skills"])
            kw = np.array([len(want & set(d["skills"])) for d in docs], dtype=float)
            gold = set(q["relevant"])
            picks = {"current": current(sims, kw, limit), "per_type": per_type(sims, kw, types, limit),
                     "rrf": rrf(sims, kw, limit)}
            for m, idx in picks.items():
                s = stats[m]
                got = {ids[i] for i in idx}
                s["mix"].update(types[i] for i in idx)
                s["n"] += 1
                if not gold:
                    continue
                for g in gold:
                    t = types[ids.index(g)]
                    s["gold"][t] += 1
                    s["found"][t] += g in got
                s["all"] += gold <= got
                if lost := sorted(gold - got):
                    s["miss"].append(f"{u['user'][:2]} {q['text']} → 빠짐 {lost}")
    del model
    return stats


def report(stats: dict, limit: int):
    n_gold_q = None
    print(f"\n=== 후보 {limit}개")
    head = "방식       재현율   정답 모두 포함   " + "  ".join(f"{t[:7]:>7}" for t in TYPES) + "   후보 구성(평균, 기술/경력/프로젝트/사례)"
    print(head)
    for m, s in stats.items():
        g, f = sum(s["gold"].values()), sum(s["found"].values())
        n_gold_q = n_gold_q or sum(1 for x in s["miss"]) + s["all"]
        per = "  ".join(f"{s['found'][t] / s['gold'][t]:>7.2f}" if s["gold"][t] else f"{'-':>7}" for t in TYPES)
        mix = "/".join(f"{s['mix'][t] / s['n']:.1f}" for t in TYPES)
        print(f"{m:<9} {f / g:>7.2f}   {s['all']:>3}/{n_gold_q:<3}          {per}   {mix}")
    for m, s in stats.items():
        if s["miss"]:
            print(f"\n[{m}] 정답이 빠진 요건 {len(s['miss'])}개")
            for line in s["miss"]:
                print("  " + line)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=default_device())
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--no-expand", action="store_true", help="약어 확장어 없이 원문 질의만")
    args = p.parse_args()
    report(run(json.loads(CASES.read_text(encoding="utf-8")), args.device, args.limit, not args.no_expand), args.limit)
