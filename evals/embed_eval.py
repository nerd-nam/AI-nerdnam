"""임베딩 검색 품질 확인 (AI-03). 서버와 같은 호출 코드(app/embedding.py, Bedrock Titan V2)를 쓴다.

한글 JD 요건(질의)으로 영문 기술명이 섞인 경험 청크(문서)를 얼마나 잘 찾는지 잰다.
AWS_BEARER_TOKEN_BEDROCK 필요 (.env.local.example).

    python -m evals.embed_eval
    python -m evals.embed_eval --p2 evals/out/extract.json   # P2가 만든 확장어로도 비교 (evals.extract_eval 출력)

질의 확장(AI-16): 원문 / 사람이 쓴 확장어(embed_cases.json) / P2 확장어로 질의를 바꿔 같은 문서에서 순위를 비교한다.
embeddinggemma-2(768) 기준 결과: hit@1 0.82 · MRR 0.86 (2026-10-26)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from app import embedding

CASES = Path(__file__).with_name("embed_cases.json")


def query_text(text: str, expansions: list[str]) -> str:
    """API가 만드는 요건 검색 질의 `원문 (확장어1, 확장어2)` (기획서 5.4)"""
    return f"{text} ({', '.join(expansions)})" if expansions else text


def first_ranks(sims: np.ndarray, doc_ids: list[str], queries: list[dict]) -> list[tuple[int | None, float]]:
    """질의별 (정답의 첫 순위, 1위 점수). 정답 없는 질의는 순위 None"""
    out = []
    for q, row in zip(queries, sims):
        order = np.argsort(-row)
        rank = next(i for i, j in enumerate(order) if doc_ids[j] in q["relevant"]) + 1 if q["relevant"] else None
        out.append((rank, float(row[order[0]])))
    return out


def compare_expansions(dv: np.ndarray, ids: list[str], queries: list[dict], variants: dict[str, dict]):
    """확장어별 질의로 다시 검색해 원문 질의와 비교. 확장어가 붙은 질의만 줄별로 보여 준다."""
    base = first_ranks(np.array(embedding.encode("query", [(q["text"], None) for q in queries])) @ dv.T, ids, queries)
    for name, exp in variants.items():
        texts = [query_text(q["text"], exp.get(q["text"], [])) for q in queries]
        sims = np.array(embedding.encode("query", [(t, None) for t in texts])) @ dv.T
        scores, _, _ = rank_metrics(sims, ids, queries)
        print(f"\n--- 질의 확장: {name}  " + "  ".join(f"{k} {v:.2f}" for k, v in scores.items()))
        for q, t, (r0, s0), (r1, s1) in zip(queries, texts, base, first_ranks(sims, ids, queries)):
            if t == q["text"]:
                continue
            what = f"정답 {r0}위 → {r1}위" if q["relevant"] else f"[정답 없음] 1위 점수 {s0:.3f} → {s1:.3f}"
            print(f"  {what}  {t}")


def rank_metrics(sims: np.ndarray, doc_ids: list[str], queries: list[dict]):
    """질의별 정답의 첫 순위로 hit@1, hit@3, MRR. 정답 없는 질의는 1위 점수만 모은다."""
    hit1 = hit3 = rr = 0.0
    misses, neg_top = [], []
    for q, row in zip(queries, sims):
        order = np.argsort(-row)
        if not q["relevant"]:
            neg_top.append((q["text"], float(row[order[0]]), doc_ids[order[0]]))
            continue
        rank = next(i for i, j in enumerate(order) if doc_ids[j] in q["relevant"]) + 1
        hit1 += rank == 1
        hit3 += rank <= 3
        rr += 1 / rank
        if rank > 1:
            misses.append((q["text"], rank, [(doc_ids[j], round(float(row[j]), 3)) for j in order[:3]]))
    n = sum(bool(q["relevant"]) for q in queries)
    return {"hit@1": hit1 / n, "hit@3": hit3 / n, "MRR": rr / n}, misses, neg_top


def run(cases: dict, variants: dict[str, dict] | None = None):
    docs, queries = cases["docs"], cases["queries"]
    t = time.time()
    dv = np.array(embedding.encode("document", [(d["text"], d["title"]) for d in docs]))
    qv = np.array(embedding.encode("query", [(q["text"], None) for q in queries]))
    enc_s = time.time() - t

    sims = qv @ dv.T
    ids = [d["id"] for d in docs]
    scores, misses, neg_top = rank_metrics(sims, ids, queries)
    # 정답이 있는 질의들의 "정답 최고 점수" 중 최솟값. 정답 없는 질의의 1위가 이보다 높으면 점수만으로 거르기 어렵다
    floor = min(max(row[ids.index(r)] for r in q["relevant"]) for q, row in zip(queries, sims) if q["relevant"])

    print(f"\n=== {embedding.EMBED_MODEL_ID}")
    print(f"dim {dv.shape[1]} | encode {len(docs) + len(queries)}개 {enc_s:.2f}s")
    print("  ".join(f"{k} {v:.2f}" for k, v in scores.items()))
    for text, rank, top3 in misses:
        print(f"  X {rank}위  {text}  → {top3}")
    print(f"  정답 최고 점수의 최솟값 {floor:.3f}")
    for text, score, doc in neg_top:
        print(f"  [정답 없음] {text}  1위 {doc} {score:.3f}  {'오탐 위험' if score >= floor else 'ok'}")

    if variants:
        compare_expansions(dv, ids, queries, variants)



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--p2", type=Path, help="evals.extract_eval 출력 JSON. P2가 만든 확장어로도 비교")
    args = p.parse_args()

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    variants = {"사람이 쓴 확장어": {q["text"]: q.get("expansions", []) for q in cases["queries"]}}
    if args.p2:
        p2 = json.loads(args.p2.read_text(encoding="utf-8"))
        variants[f"P2 확장어 ({p2['prompt_version']}, {p2['model']})"] = {x["text"]: x["expansions"] for x in p2["lines"]}
    run(cases, variants)
