"""API ↔ AI 요청·응답 모델 = 계약 C-04의 출처 (/openapi.json → SPEC contracts/api-ai.openapi.yaml).
기획서 5.5가 정한 것(/embed items, mode·tab 값, /guide JSON)은 기획서를, 나머지 이름·구조는 BE infra.ai 타입을 따른다.
ID(요건·청크)는 BE DB의 숫자 ID다."""
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ReqType = Literal["required", "preferred"]
ChunkType = Literal["skill", "career", "project_summary", "episode"]  # 기획서 4.4
Verdict = Literal["met", "partial", "unmet"]  # D-10 임시값


class Meta(BaseModel):
    model: str
    prompt_version: str  # 가짜 응답이면 "stub"


class Candidate(BaseModel):
    """청크. 사용자 정보 없이 id와 텍스트만 (기획서 5.5)"""
    chunk_id: int
    chunk_type: ChunkType
    text: str


# ── /health ───────────────────────────────────────
ModelState = Literal["stub", "loading", "ready", "error"]


class HealthModels(BaseModel):
    llm: ModelState
    embedding: ModelState


class HealthResponse(BaseModel):
    status: Literal["ok"]
    models: HealthModels  # 모두 ready 일 때만 준비된 것으로 본다


# ── /restructure (P1) ─────────────────────────────
class Metric(BaseModel):
    name: str
    before: str | None = None
    after: str | None = None
    source: Literal["raw_text", "answer"] = "raw_text"


class Clarification(BaseModel):
    """확인 질문. key가 없거나 겹치면 API가 q1, q2…를 붙인다"""
    key: str | None = None
    field: str | None = None  # 질문 대상 필드 (예: result)
    question: str
    example: str | None = None


class Episode(BaseModel):
    """사례 (기획서 4.3 에피소드 모양)"""
    title: str
    role: str = ""
    problem: str
    solution: str
    result: str
    skills: list[str] = []
    metrics: list[Metric] = []
    source_spans: list[tuple[int, int]] = []  # source_text 안의 문자 위치
    clarifications: list[Clarification] = []


class Answer(BaseModel):
    question: str
    answer: str


class RestructureRequest(BaseModel):
    """mode=split 이면 feedback(공백 아님) 필수, mode=episode 면 target_episode 필수. 어기면 422"""
    mode: Literal["initial", "split", "episode"]  # initial = 처음 나누기, split = 다시 나누기, episode = 사례 하나 고치기
    title: str = ""  # 프로젝트 제목
    source_text: str = Field(min_length=1)  # 원문 + 보충 답변. source_spans 의 기준
    skills: list[str] = []  # 프로젝트 사용 기술 (표준 이름)
    previous_episodes: list[Episode] = []  # 재등록 중이면 이전 확정 사례 (맥락)
    current_episodes: list[Episode] = []  # split: 지금 검토 중인 사례 (피드백의 대상)
    target_episode: Episode | None = None  # episode: 고칠 사례
    feedback: str | None = None
    answer: Answer | None = None  # episode: 확인 질문 답변

    @model_validator(mode="after")
    def mode_rules(self):
        # 다시 나누기는 피드백 없이 누를 수 없다 (기획서 F2). 원문을 고친 뒤 새로 나누기는 initial
        if self.mode == "split" and not (self.feedback and self.feedback.strip()):
            raise ValueError("mode=split 은 feedback 이 필요함")
        if self.mode == "episode" and self.target_episode is None:
            raise ValueError("mode=episode 는 target_episode 가 필요함")
        return self


class RestructureResponse(Meta):
    episodes: list[Episode]  # episode 모드면 1개


# ── /requirements/extract (P2) ────────────────────
class RequirementLine(BaseModel):
    id: int
    type: ReqType
    text: str


class ExtractRequest(BaseModel):
    lines: list[RequirementLine] = Field(min_length=1)


class ExtractedLine(BaseModel):
    """문구는 돌려주지 않는다(바꾸지 않으므로). 기술명은 원문 그대로, 정규화는 API가 한다"""
    id: int
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
    vectors: list[list[float]]  # items 순서


# ── /match (P3) ───────────────────────────────────
class MatchRequirement(BaseModel):
    id: int
    type: ReqType
    text: str
    skills: list[str] = []
    min_years: int | None = None
    candidates: list[Candidate] = Field(max_length=20)  # 요건별 후보


class MatchRequest(BaseModel):
    requirements: list[MatchRequirement] = Field(min_length=1)


class MatchResult(BaseModel):
    id: int  # 요건 id
    verdict: Verdict
    evidence_chunk_ids: list[int]  # 그 요건의 후보 안에서만, 최대 5개
    reason: str


class MatchResponse(Meta):
    results: list[MatchResult]


# ── /gap-guide (P6, ADR-0003) ─────────────────────
class GapRequirement(BaseModel):
    id: int
    type: ReqType
    text: str
    verdict: Literal["partial", "unmet"]  # 충족(met)은 보내지 않는다
    adjacent: list[Candidate] = []  # /match 에서 뽑히지 못한 후보 (인접 경험)


class GapGuideRequest(BaseModel):
    requirements: list[GapRequirement] = Field(min_length=1)
    tasks: list[str] = []  # 공고 주요 업무


class GapGuide(BaseModel):
    id: int  # 요건 id
    advice: str  # 가이드 문장
    avoid: str  # 하지 말 것 문장
    cited_chunk_ids: list[int]  # 그 요건의 adjacent 안에서만


class GapGuideResponse(Meta):
    guides: list[GapGuide]


# ── /guide (P4·P5) — 탭마다 1회, JSON 응답 (SSE 아님, 기획서 v1.23) ──
GuideTab = Literal["motive", "intro", "portfolio", "career", "image"]  # 웹 주소 /apply/[id]/guide/[tab] 과 같은 값


class GapAdvice(BaseModel):
    advice: str
    avoid: str


class GuideRequirement(BaseModel):
    id: int
    type: ReqType
    text: str
    verdict: Verdict
    reason: str
    evidence: list[Candidate] = []
    gap_guide: GapAdvice | None = None


class GuideRequest(BaseModel):
    tab: GuideTab
    company: str = ""
    position: str = ""
    company_intro: str = ""
    tasks: list[str] = []
    requirements: list[GuideRequirement] = Field(min_length=1)
    feedback: str | None = None


class GuideResponse(Meta):
    # ponytail: 탭별 content 모양은 AI-08~10(P4·P5 프롬프트)에서 정한다. v0.1은 객체로만 고정, 인용은 sources(청크 id 목록)
    content: dict[str, Any]
    cited_chunk_ids: list[int]
