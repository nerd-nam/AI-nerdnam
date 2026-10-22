"""API ↔ AI 요청·응답 모델 (기획서 4.3, 5.5 기준). 계약 C-04 초안의 출처."""
from typing import Literal

from pydantic import BaseModel, Field


class Meta(BaseModel):
    model: str
    prompt_version: str | None = None  # /embed 는 프롬프트 없음


# ── /restructure (P1) ─────────────────────────────
class Metric(BaseModel):
    name: str
    before: str | None = None
    after: str | None = None
    source: Literal["raw_text", "answer"] = "raw_text"


class Episode(BaseModel):
    episode_id: str | None = None  # 새 사례는 BE가 ID를 붙인다
    title: str
    role: str = ""
    skills: list[str] = []
    problem: str
    solution: str
    result: str
    metrics: list[Metric] = []
    source_spans: list[tuple[int, int]] = []
    questions: list[str] = []  # 확인 질문


class QA(BaseModel):
    question: str
    answer: str


class RestructureRequest(BaseModel):
    mode: Literal["initial", "split", "episode"] = "initial"
    raw_text: str = Field(min_length=1)
    skills: list[str] = []
    episodes: list[Episode] = []  # split·episode 모드의 기존 사례
    target_episode_id: str | None = None  # episode 모드에서 고칠 사례
    answers: list[QA] = []
    feedback: str | None = None


class RestructureResponse(Meta):
    episodes: list[Episode]


# ── /requirements/extract (P2) ────────────────────
class RequirementLine(BaseModel):
    id: str
    type: Literal["required", "preferred"]
    text: str


class ExtractRequest(BaseModel):
    lines: list[RequirementLine] = Field(min_length=1)


class ExtractedLine(BaseModel):
    id: str
    skills_raw: list[str]
    min_years: int | None = None


class ExtractResponse(Meta):
    lines: list[ExtractedLine]


# ── /embed ────────────────────────────────────────
class EmbedItem(BaseModel):
    text: str = Field(min_length=1)
    title: str | None = None  # document 용도에서만 사용


class EmbedRequest(BaseModel):
    purpose: Literal["document", "query"]
    items: list[EmbedItem] = Field(min_length=1)


class EmbedResponse(BaseModel):
    embedding_model: str  # 청크마다 저장해 재임베딩 대상을 찾는 데 씀 (기획서 4.4·5.5). 가짜 응답이면 "stub"
    dim: int
    vectors: list[list[float]]


# ── /match (P3) ───────────────────────────────────
class Chunk(BaseModel):
    chunk_id: str
    text: str


class MatchRequirement(BaseModel):
    id: str
    text: str
    skills: list[str] = []
    candidates: list[Chunk]  # 요건별 후보 (최대 20)


class MatchRequest(BaseModel):
    requirements: list[MatchRequirement] = Field(min_length=1)


Verdict = Literal["met", "partial", "unmet"]  # D-10 임시값


class MatchResult(BaseModel):
    requirement_id: str
    verdict: Verdict
    evidence_chunk_ids: list[str]  # 리랭킹 상위 5개 이내
    reason: str


class MatchResponse(Meta):
    results: list[MatchResult]


# ── /guide (P4·P5, SSE) ───────────────────────────
GuideTab = Literal["motive", "intro", "portfolio", "career", "image"]  # FE routes.ts 와 동일


class GuideRequest(BaseModel):
    tab: GuideTab
    results: list[MatchResult]
    evidence: list[Chunk]
    jd_context: str = ""
    feedback: str | None = None
