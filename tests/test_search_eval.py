"""evals/search_eval의 후보 선택 방식 (D-12). 모델 없이 점수 배열로 확인한다."""
import pytest

np = pytest.importorskip("numpy")  # numpy는 --extra model에만 있다
from evals.search_eval import current, per_type, rrf  # noqa: E402

# 기술 청크 4개가 사례 청크 2개보다 코사인이 높고, 사례 5번만 기술명이 겹친다
COS = np.array([0.80, 0.79, 0.78, 0.77, 0.70, 0.60])
KW = np.array([0, 0, 0, 0, 0, 1.0])
TYPES = ["skill"] * 4 + ["episode"] * 2


def test_current_keyword_only_chunk_never_enters():
    # 벡터 상위 limit 밖의 청크는 겹침 점수 0.3만 받아 0.7 × 코사인을 못 넘는다 (D-12 지금 방식)
    assert current(COS, KW, 4) == [0, 1, 2, 3]
    assert 5 in current(COS, KW, 6)


def test_per_type_keeps_episodes():
    picked = per_type(COS, KW, TYPES, 4)  # limit 4 → 사례 몫 round(8 × 4/20) = 2
    assert {4, 5} <= set(picked) and len(picked) == 4


def test_rrf_lifts_keyword_hit():
    assert 5 in rrf(COS, KW, 4)
