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


_NOT_VALUE = re.compile(r'[{}"]|raw_text|answer')
_HAN = re.compile(r"[一-鿿]")


def bad_metric_value(v: str | None) -> bool:
    """성과 값은 "3.2초"처럼 수치가 든 짧은 문자열. 숫자가 없거나 JSON 조각·출처 이름이 섞이면 잘못 (AI #24)."""
    return v is not None and (not _NUM.search(v) or bool(_NOT_VALUE.search(v)))


def foreign_han(text: str, source: str) -> list[str]:
    """text에 있지만 source에 없는 한자. 4B 모델이 "고可用性"처럼 중국어를 섞는다 (AI #24)."""
    allowed = set(_HAN.findall(source))
    return list(dict.fromkeys(ch for ch in _HAN.findall(text) if ch not in allowed))  # 나온 순서 그대로 (메시지에 '可用性'으로 보이게)


def unknown_ids(ids: list[str], allowed: set[str]) -> list[str]:
    """입력에 없는 인용 청크 ID."""
    return [i for i in ids if i not in allowed]


def find_span(quote: str, text: str) -> tuple[int, int] | None:
    """quote가 text에 나오는 위치 [시작, 끝). 공백 차이는 무시한다.
    모델이 원문을 베껴도 "1.2초" → "1.2 초", "Kafka로" → "Kafka 로"처럼 띄어 쓰는 경우가 있다 (로컬 테스트)."""
    q = "".join(quote.split())
    if not q:
        return None
    pos = [i for i, ch in enumerate(text) if not ch.isspace()]  # 공백을 뺀 글자 → 원문 위치
    s = "".join(text[i] for i in pos).find(q)
    return (pos[s], pos[s + len(q) - 1] + 1) if s >= 0 else None
