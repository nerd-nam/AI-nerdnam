"""evals/match_eval의 지표 계산 (ADR-0006)."""
from evals.match_eval import score


def row(gold, got, evidence=(), gold_evidence=(), search_miss=False, reason_numbers=()):
    return {"gold": gold, "got": got, "evidence": list(evidence), "gold_evidence": list(gold_evidence),
            "search_miss": search_miss, "reason_numbers": list(reason_numbers)}


def test_score():
    s = score([
        row("met", "met", ["e1", "sk"], ["e1"]),        # 일치, 근거 1/2
        row("partial", "met", ["e2"], ["e2"]),          # 과대
        row("unmet", "met", ["e3"], []),                # 과대 + 큰 오류
        row("met", "partial", ["e4"], ["e4"], reason_numbers=["9"]),  # 과소
        row("partial", "error 502"),                    # 오류 응답: 불일치, 과대·과소 아님
        row("met", "unmet", search_miss=True),          # 검색 누락: 일치율에서 뺌
    ])
    assert s["n"] == 5 and s["agree"] == 1 / 5
    assert s["over"] == 2 / 5 and s["under"] == 1 / 5
    assert s["big"] == 1 and s["errors"] == 1 and s["search_miss"] == 1
    assert s["evidence_precision"] == 3 / 5  # 고른 근거 5개 중 정답 3개
    assert s["reason_numbers"] == 1


def test_clean_reason():
    from app.routers.match import clean_reason
    assert clean_reason("후보 청크 33 에서 인덱스를 재설계했다") == "관련 경험에서 인덱스를 재설계했다"
    assert clean_reason("경험이 청크 32, 31, 26 에 명시되어 있다") == "경험이 관련 경험에 명시되어 있다"
    assert clean_reason("청크 (chunk_id: 24) 에서 확인") == "관련 경험에서 확인"
    assert clean_reason("청크 단위 처리로 바꿨다") == "청크 단위 처리로 바꿨다"  # 원문에 있는 말은 그대로
