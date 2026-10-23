"""생성 결과 검증: 원문에 없는 수치, 입력에 없는 인용 ID (기획서 5.4 6단계·8장, 작업목록 AI-04)."""
import re

_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def numbers(text: str) -> set[float]:
    # "1.2초" → 1.2, "1,200" → 1200, "05" → 5
    # ponytail: 아라비아 숫자만 본다. "두 배" 같은 한글 수사는 못 잡음, 놓치는 사례가 평가에서 나오면 추가
    return {float(n.replace(",", "")) for n in _NUM.findall(text)}


def unsupported_numbers(text: str, source: str) -> list[str]:
    """text에 있지만 source(원문 + 확인 질문 답변)에 없는 수치."""
    allowed = numbers(source)
    return sorted({n for n in _NUM.findall(text) if float(n.replace(",", "")) not in allowed})


def unknown_ids(ids: list[str], allowed: set[str]) -> list[str]:
    """입력에 없는 인용 청크 ID."""
    return [i for i in ids if i not in allowed]
