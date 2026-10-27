import os

# stub: 가짜 응답 (키 없이 개발할 때, 기본값) / real: AWS Bedrock 호출 (ADR-0007)
AI_MODE = os.getenv("AI_MODE", "stub")

# Bedrock 서울 리전. 임베딩은 In-Region, LLM은 Global 교차 리전 프로필. 키 값은 로그·응답에 남기지 않는다
# 호출은 boto3(bedrock-runtime). 인증: AWS_BEARER_TOKEN_BEDROCK(Bedrock API 키, boto3가 직접 읽음) 또는 IAM 자격 증명
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")

# LLM. 응답에 담는 모델 이름이기도 하다 (기획서 5.5).
# Nova 2 Lite는 서울에서 Global 교차 리전 추론 프로필로만 제공된다 (요청이 국외에서 처리될 수 있음, ADR-0007)
LLM_MODEL = os.getenv("LLM_MODEL", "global.amazon.nova-2-lite-v1:0")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))  # 읽기 제한(초)
LLM_MAX_MODEL_LEN = int(os.getenv("LLM_MAX_MODEL_LEN", "128000"))  # 입력+출력 토큰 한도. 모델 문맥(1M)보다 작게 128K로 둔다
# 생성 설정. temperature는 평가 세트로 비교할 값이라 환경변수로 바꿀 수 있다
LLM_SAMPLING = {
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
    "top_p": 0.8,
}

# 임베딩. 모델·차원을 바꾸면 저장된 벡터를 전부 다시 계산해야 함 (BE는 embedding_model 값으로 재임베딩 대상을 찾는다)
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
EMBED_DIM = 1024
EMBED_MODEL = f"{EMBED_MODEL_ID}:{EMBED_DIM}"  # 응답·청크에 기록하는 이름
