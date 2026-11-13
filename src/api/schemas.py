"""The backend ↔ AI service contract (v1). The backend regenerates its client models from
/openapi.json, so every request and response is typed here."""

from typing import Literal

from pydantic import BaseModel, Field

from agents.clarification import ClientQuestion
from agents.extraction import (
    CatalogueFeature,
    CustomRequirement,
    Fact,
    FeatureHit,
    RequiredDocument,
    RfpQuestion,
    UnclearPoint,
)
from rules.matching import DEFAULT_MATCH_THRESHOLD

Id = Field(min_length=1, max_length=64)


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int
    calls: int
    tool_calls: int


class Meta(BaseModel):
    """What produced the result, for the backend's trace and usage records."""

    model: str
    prompt_versions: dict[str, str]
    usage: UsageOut
    cached: bool


class Problem(BaseModel):
    item: str
    reason: str


# ---- Analyse (Requirement Extraction Agent) ---------------------------------------


class AnalyseRequest(BaseModel):
    catalogue: list[CatalogueFeature] = Field(min_length=1, max_length=500)


class AnalyseResult(BaseModel):
    summary: str
    features: list[FeatureHit]
    custom_requirements: list[CustomRequirement]
    questions: list[RfpQuestion]
    required_documents: list[RequiredDocument]
    unclear_points: list[UnclearPoint]
    facts: list[Fact]
    rejected: list[Problem]  # dropped after the checks; worth a human look


class AnalyseResponse(AnalyseResult):
    meta: Meta


# ---- Library (indexer + semantic search) -------------------------------------------


class EntryMetaIn(BaseModel):
    client: str = Field(max_length=200)
    title: str = Field(max_length=300)
    industry: str = Field(max_length=60)
    outcome: Literal["won", "lost", "pending"]
    year: int = Field(ge=1990, le=2100)
    summary: str = Field("", max_length=4000)


class IndexResponse(BaseModel):
    entry_id: str
    passages: int
    content_hash: str
    unchanged: bool


class RequirementText(BaseModel):
    key: str = Id
    text: str = Field(min_length=1, max_length=500)


class SimilarRequest(BaseModel):
    requirements: list[RequirementText] = Field(default_factory=list, max_length=100)
    rfp_summary: str | None = Field(None, max_length=4000)
    entry_ids: list[str] | None = Field(None, max_length=1000)  # candidates; None = whole library
    top_k: int = Field(3, ge=1, le=10)
    threshold: float = Field(DEFAULT_MATCH_THRESHOLD, ge=0, le=1)


class PassageMatch(BaseModel):
    entry_id: str
    similarity: float
    passage_id: str
    page: int
    text: str


class RequirementMatches(BaseModel):
    key: str
    matches: list[PassageMatch]


class ProjectSimilarity(BaseModel):
    entry_id: str
    similarity: float


class SimilarResponse(BaseModel):
    requirements: list[RequirementMatches]
    projects: list[ProjectSimilarity]
    embedding_model: str


# ---- Draft answers (Response Drafting Agent + Review Agent) -------------------------


class QuestionIn(BaseModel):
    id: str = Id
    text: str = Field(min_length=1, max_length=2000)
    draftable: bool = True  # the backend sets False for items a person must write


class DraftAnswersRequest(BaseModel):
    questions: list[QuestionIn] = Field(min_length=1, max_length=40)
    selected_entry_ids: list[str] = Field(min_length=1, max_length=10)
    rfp_summary: str = Field("", max_length=4000)
    requirements: list[str] = Field(default_factory=list, max_length=100)


class ClaimOut(BaseModel):
    statement: str
    passage_id: str
    entry_id: str
    page: int
    quote: str


DraftStatus = Literal["drafted", "needs_person", "skipped_pricing", "skipped"]


class AnswerOut(BaseModel):
    id: str
    status: DraftStatus
    answer: str
    claims: list[ClaimOut]
    reason: str


class DraftAnswersResponse(BaseModel):
    answers: list[AnswerOut]
    meta: Meta


# ---- Client queries (Clarification Agent + Review Agent) ----------------------------


class RequirementIn(BaseModel):
    key: str = Id
    label: str = Field(min_length=1, max_length=300)
    covered: bool


class UnclearIn(BaseModel):
    id: str = Id
    text: str = Field(min_length=1, max_length=1000)


class ClientQueriesRequest(BaseModel):
    requirements: list[RequirementIn] = Field(max_length=100)
    unclear_points: list[UnclearIn] = Field(default_factory=list, max_length=50)
    engagement_model: str = Field(max_length=60)
    existing_questions: list[str] = Field(default_factory=list, max_length=100)


class ClientQueriesResponse(BaseModel):
    questions: list[ClientQuestion]
    dropped: list[str]
    meta: Meta
