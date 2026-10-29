"""분산 추적(OpenTelemetry)과 JSON 로그 (#42).

설정은 표준 OTEL_* 환경변수를 SDK가 직접 읽는다(서비스 이름·리소스·샘플러·수집 주소). 코드에 주소를 두지 않는다.
OTEL_EXPORTER_OTLP_ENDPOINT가 없으면(로컬·테스트) span을 내보내지 않는다. 프롬프트·응답 본문·키는 span과 로그에 넣지 않는다.
"""
import json
import logging
import os

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

tracer = trace.get_tracer("nerdnam.ai")


def setup(app: FastAPI) -> None:
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") and os.getenv("OTEL_TRACES_EXPORTER", "otlp") == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider()  # OTEL_SERVICE_NAME·OTEL_RESOURCE_ATTRIBUTES·OTEL_TRACES_SAMPLER(_ARG)를 읽음
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))  # http/protobuf, 주소는 환경변수
        trace.set_tracer_provider(provider)
    # BE가 보낸 traceparent를 이어 받는다. 프로브는 주기마다 들어오므로 빼고, ASGI receive·send 하위 span(요청마다 4개)도 뺀다
    FastAPIInstrumentor.instrument_app(app, excluded_urls="livez,readyz,health", exclude_spans=["receive", "send"])


class JsonFormatter(logging.Formatter):
    """로그 한 줄 = JSON 하나. span 안에서 찍힌 줄에는 trace_id·span_id (Grafana가 Tempo로 연결)"""

    def format(self, record: logging.LogRecord) -> str:
        d = {"time": self.formatTime(record), "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:
            d["trace_id"], d["span_id"] = format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d, ensure_ascii=False)
