"""evals/guide_eval의 경험 언급 추출·점수 (기획서 9.1 근거 정확도)."""
from evals.guide_eval import mentions, score


def test_mentions():
    texts = {1: "정산 배치를 40분에서 8분으로 줄였다", 2: "Kafka 비동기 처리"}
    content = {
        "outline": [{"title": "정산", "detail": "40분 → 8분 단축", "sources": [1]},
                    {"title": "결제", "detail": "응답 30% 개선", "sources": [2]},  # 근거에 없는 수치
                    {"title": "회사", "detail": "공고 업무와 연결", "sources": []}],  # 인용 없음
        "pages": [{"title": "표지", "hint": "제목", "kind": "cover", "sources": []}],  # 표지는 인용 없어도 됨
        "rewrites": [{"before": "Kafka 비동기 처리", "after": "메시지 큐 운영"},
                     {"before": "MSA 전환 주도", "after": "MSA 설계"}],  # before가 근거에 없음
    }
    rows = mentions(content, texts)
    assert [r["auto"] for r in rows] == [[], ["근거에 없는 수치 ['30']"], ["인용 없음"], [], [], ["before가 근거 문장에 없음"]]


def test_score():
    s = score([{"auto": [], "ok": True}, {"auto": ["x"], "ok": False}, {"auto": [], "ok": None}])
    assert s == {"mentions": 3, "reviewed": 2, "accuracy": 0.5, "auto_flagged": 1}
