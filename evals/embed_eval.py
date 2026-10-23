"""embeddinggemma-2 검색 품질 확인 (AI-03). 서버와 같은 로딩·접두어 코드(app/embedding.py)를 쓴다.

한글 JD 요건(질의)으로 영문 기술명이 섞인 경험 청크(문서)를 얼마나 잘 찾는지 잰다.

    python -m evals.embed_eval                  # 텍스트 인코더만 (서버와 같은 방식)
    python -m evals.embed_eval --compare-full   # 전체 모델과 벡터·메모리 비교 (텍스트 전용 로딩 검증)
    python -m evals.embed_eval --dtype bfloat16 # L4

T4는 bfloat16 미지원, float16은 NaN 위험 → T4·CPU·Mac은 float32(기본값).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from app import embedding

CASES = Path(__file__).with_name("embed_cases.json")


def default_device():
    if torch.cuda.is_available():
        return "cuda:0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


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


def run(cases: dict, full: bool, device: str, dtype: str):
    docs, queries = cases["docs"], cases["queries"]
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t = time.time()
    model = embedding.load(full=full, dtype=dtype, device=device)
    load_s = time.time() - t
    weights_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30

    t = time.time()
    dv = np.array(embedding.encode(model, "document", [(d["text"], d["title"]) for d in docs]))
    qv = np.array(embedding.encode(model, "query", [(q["text"], None) for q in queries]))
    enc_s = time.time() - t

    sims = qv @ dv.T
    ids = [d["id"] for d in docs]
    scores, misses, neg_top = rank_metrics(sims, ids, queries)
    # 정답이 있는 질의들의 "정답 최고 점수" 중 최솟값. 정답 없는 질의의 1위가 이보다 높으면 점수만으로 거르기 어렵다
    floor = min(max(row[ids.index(r)] for r in q["relevant"]) for q, row in zip(queries, sims) if q["relevant"])

    print(f"\n=== {'전체 모델' if full else '텍스트 인코더만 (서버 방식)'}  ({device}, {dtype})")
    print(f"dim {dv.shape[1]} | NaN {bool(np.isnan(dv).any() or np.isnan(qv).any())} | 가중치 {weights_gb:.2f}GB"
          f" | load {load_s:.1f}s | encode {len(docs) + len(queries)}개 {enc_s:.2f}s"
          + (f" | peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f}GB" if cuda else ""))
    print("  ".join(f"{k} {v:.2f}" for k, v in scores.items()))
    for text, rank, top3 in misses:
        print(f"  X {rank}위  {text}  → {top3}")
    print(f"  정답 최고 점수의 최솟값 {floor:.3f}")
    for text, score, doc in neg_top:
        print(f"  [정답 없음] {text}  1위 {doc} {score:.3f}  {'오탐 위험' if score >= floor else 'ok'}")

    del model
    return np.vstack([dv, qv])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=default_device())
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--compare-full", action="store_true", help="전체 모델도 올려서 텍스트 벡터가 같은지 비교")
    args = p.parse_args()

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    vecs = run(cases, False, args.device, args.dtype)
    if args.compare_full:
        full_vecs = run(cases, True, args.device, args.dtype)
        cos = (vecs * full_vecs).sum(axis=1)  # 둘 다 정규화된 벡터
        print(f"\n=== 텍스트 전용 vs 전체: 코사인 최소 {cos.min():.6f} (1.0이면 같은 벡터)")
