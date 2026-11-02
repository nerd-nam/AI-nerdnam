"""Prometheus 지표 (#44). GET /metrics 로 낸다.

Bedrock 호출은 trace span 속성(#43)을 정하는 같은 자리에서 센다 → trace와 숫자가 맞는다.
라벨은 유한한 값만 쓴다(모델 id·경로 템플릿·결과 분류). 회원·작업 id·프롬프트는 라벨에 넣지 않는다.
"""
from prometheus_client import Counter, Histogram, disable_created_metrics

disable_created_metrics()  # 지표마다 붙는 *_created 시계열은 쓰지 않는다 (시계열 수만 두 배)
# ponytail: 단일 프로세스 레지스트리. uvicorn 워커를 2개 이상으로 늘리면 multiprocess 모드(PROMETHEUS_MULTIPROC_DIR)로 바꾼다
_OP = ["operation", "model"]
CALLS = Counter("nerdnam_ai_bedrock_calls", "Bedrock 실제 호출 수 (대기열 거절은 queue_rejected)", [*_OP, "result"])
CALL_SECONDS = Histogram("nerdnam_ai_bedrock_call_duration_seconds", "Bedrock 호출 시간 (SDK 재시도 포함)", _OP,
                         buckets=(0.5, 1, 3, 10, 30, 60))
ATTEMPTS = Counter("nerdnam_ai_bedrock_attempts", "실제 시도 수 (SDK 재시도 + throttle 재시도)", _OP)
TOKENS = Counter("nerdnam_ai_bedrock_tokens", "Bedrock 토큰 (응답 usage 기준)", ["model", "direction"])
QUEUE_WAIT = Histogram("nerdnam_ai_bedrock_queue_wait_seconds", "동시 호출 자리를 기다린 시간", ["kind"],
                       buckets=(0.01, 0.1, 0.5, 1, 2, 5, 10))
REJECTED = Counter("nerdnam_ai_bedrock_queue_rejected", "대기열이 차거나 시간 상한을 넘어 503으로 거절", ["kind"])
REQUESTS = Counter("nerdnam_ai_requests", "HTTP 요청 수", ["route", "status"])
REQUEST_SECONDS = Histogram("nerdnam_ai_request_duration_seconds", "HTTP 요청 처리 시간", ["route"],
                            buckets=(0.1, 0.5, 1, 3, 10, 20, 30, 60, 120))


def bedrock(operation: str, model: str, result: str, seconds: float, attempts: int,
            input_tokens: int = 0, output_tokens: int = 0) -> None:
    CALLS.labels(operation, model, result).inc()
    CALL_SECONDS.labels(operation, model).observe(seconds)
    ATTEMPTS.labels(operation, model).inc(attempts)
    if input_tokens:
        TOKENS.labels(model, "input").inc(input_tokens)
    if output_tokens:
        TOKENS.labels(model, "output").inc(output_tokens)
