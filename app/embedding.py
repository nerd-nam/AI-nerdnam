"""Titan Text Embeddings V2 호출 (Bedrock InvokeModel, boto3). /embed 와 evals 가 같이 쓴다.

Titan은 한 번에 문장 하나만 받으므로 항목마다 병렬로 부르고 순서대로 돌려준다.
질의·문서 접두어 규칙이 없어 문서는 제목을 본문 앞에 붙이기만 한다.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

from app.config import AWS_REGION, EMBED_DIM, EMBED_MODEL_ID

# 항목마다 따로 부르므로 한꺼번에 많이 보내면 호출 한도(Throttling)에 걸린다. 지금 계정은 약 67건을 몰아 받은 뒤
# 초당 약 1건(실측 2026-10-27). adaptive 재시도가 한도에 맞춰 속도를 낮춘다. 서버는 /embed 읽기 제한(10초) 때문에 4번까지,
# 평가 스크립트는 EMBED_MAX_ATTEMPTS를 늘려 끝까지 돌린다
client = boto3.client("bedrock-runtime", region_name=AWS_REGION, config=Config(
    read_timeout=10, connect_timeout=2, max_pool_connections=8,
    retries={"mode": "adaptive", "total_max_attempts": int(os.getenv("EMBED_MAX_ATTEMPTS", "4"))}))
# ponytail: 동시 8건 고정. 호출 한도(분당 요청 수)에 걸리면 줄이고, 처리량이 모자라면 한도 상향 요청
_pool = ThreadPoolExecutor(8)


def text_of(purpose: str, text: str, title: str | None = None) -> str:
    return f"{title}\n{text}" if purpose == "document" and title else text


def _one(text: str) -> list[float]:
    body = json.dumps({"inputText": text, "dimensions": EMBED_DIM, "normalize": True}, ensure_ascii=False)
    r = client.invoke_model(modelId=EMBED_MODEL_ID, body=body, contentType="application/json", accept="application/json")
    return json.loads(r["body"].read())["embedding"]


def encode(purpose: str, items: list[tuple[str, str | None]]) -> list[list[float]]:
    return list(_pool.map(_one, [text_of(purpose, text, title) for text, title in items]))
