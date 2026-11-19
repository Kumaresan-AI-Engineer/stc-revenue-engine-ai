"""The backend ↔ AI service contract (v1). The backend regenerates its client models from
/openapi.json, so every request and response is typed here."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.clarification import ClientQuestion
from agents.extraction import CatalogueFeature
from agents.sizing import SizeClass
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


class CoreFeature(BaseModel):
    key: str
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(description="Words copied from the document.")
    page: int


class StandardEvidence(BaseModel):
    key: str
    evidence: str
    page: int


class Unmatched(BaseModel):
    """A requirement no catalogue entry covers: a custom requirement for the backend."""

    label: str
    evidence: str
    page: int


class RfpQuestionOut(BaseModel):
    text: str
    page: int
    evidence: str


class RequiredDocumentOut(BaseModel):
    name: str
    evidence: str
    page: int


class UnclearPointOut(BaseModel):
    topic: str
    question: str
    evidence: str
    page: int


class AnalyseResult(BaseModel):
    """What the Requirement Extraction Agent read from an RFP, every item with its quote and page."""

    summary: str
    core: list[CoreFeature] = Field(description="Catalogue keys of core (non-commodity) features.")
    standard: list[str] = Field(description="Catalogue keys of commodity features.")
    standard_evidence: list[StandardEvidence]
    unmatched: list[Unmatched]
    due_date: str | None
    start_date: str | None
    budget_hint: str | None
    client: str | None
    questions: list[RfpQuestionOut]
    required_documents: list[str]
    documents: list[RequiredDocumentOut] = Field(description="required_documents with their evidence.")
    unclear_points: list[UnclearPointOut]
    rejected: list[Problem]  # dropped after the checks; worth a human look
    ocr: bool = Field(description="Some pages were scans, read with OCR.")
    pages: int


class AnalyseResponse(AnalyseResult):
    meta: Meta


# ---- Asynchronous jobs -------------------------------------------------------------


class JobSubmission(BaseModel):
    """POST /v1/jobs. `input` is validated against the job type's own input model."""

    type: str = Field(min_length=1, max_length=40)
    input: dict[str, Any]
    callback_url: str = Field(min_length=1, max_length=2000)
    idempotency_key: str = Field(min_length=1, max_length=200)
    priority: Literal["interactive", "batch"] = "interactive"


class JobAccepted(BaseModel):
    job_id: str


class JobOut(BaseModel):
    job_id: str
    type: str
    status: Literal["queued", "running", "succeeded", "failed"]
    error: str | None


class RfpAnalysisInput(BaseModel):
    """Input of the rfp_analysis job."""

    file_url: str = Field(min_length=1, max_length=4000, description="Signed link; expires in minutes.")
    filename: str = Field(min_length=1, max_length=300)
    mode: Literal["upload_rfp", "upload_req", "paste"] = "upload_rfp"
    client: str = Field("", max_length=200)
    industry: str = Field("", max_length=100)
    engagement_model: str = Field("", max_length=60)
    catalogue: list[CatalogueFeature] = Field(min_length=1, max_length=500)


class TemplateExtractInput(BaseModel):
    """Input of the template_extract job: a proposal or case study template (DOCX) to read."""

    file_url: str = Field(min_length=1, max_length=4000, description="Signed link; expires in minutes.")
    filename: str = Field(min_length=1, max_length=300)
    kind: Literal["proposal", "case_study"] = "proposal"


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


class CitationOut(BaseModel):
    """Where a drafted answer comes from, for the reviewer to verify (HL-04)."""

    source_id: str = Field(description="The library entry id.")
    title: str = Field(description="Client — title (year).")
    quote: str
    page: int
    passage_id: str


class AnswerOut(BaseModel):
    id: str
    status: DraftStatus
    answer: str
    claims: list[ClaimOut]
    reason: str
    confidence: float = Field(0.0, ge=0, le=1, description="0 unless drafted.")
    no_confident_source: bool = Field(False, description="True when no past proposal answers it.")
    citations: list[CitationOut] = Field(default_factory=list)


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
    existing_assumptions: list[str] = Field(default_factory=list, max_length=100)
    limit: int = Field(12, ge=1, le=12, description="At most this many questions (Suggest with AI: 3).")


class ClientQueriesResponse(BaseModel):
    questions: list[ClientQuestion]
    dropped: list[str]
    meta: Meta


# ---- Final proposal check (Final Proposal Check Agent) -----------------------------


class FeatureRef(BaseModel):
    key: str = Id
    label: str = Field(min_length=1, max_length=300)


class ApprovedResponse(BaseModel):
    id: str = Id
    q: str = Field(min_length=1, max_length=2000)
    response: str = Field("", max_length=8000)


class DraftSnapshotIn(BaseModel):
    """What the draft says, to compare the uploaded file with. `price` only for roles that may see it."""

    price: int | None = Field(None, ge=0)
    weeks: int | None = Field(None, ge=0)
    core_features: list[FeatureRef] = Field(default_factory=list, max_length=200)
    standard_features: list[str] = Field(default_factory=list, max_length=200)
    responses: list[ApprovedResponse] = Field(default_factory=list, max_length=200)
    responses_closed: int = Field(0, ge=0)
    responses_total: int = Field(0, ge=0)
    attachments: list[str] = Field(default_factory=list, max_length=100)


class FinalCheckRequest(BaseModel):
    draft: DraftSnapshotIn


class FindingOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    area: Literal["Scope", "Cost", "Timeline", "Responses", "Attachments"]
    ok: bool
    text: str
    from_: int | None = Field(None, alias="from", description="The draft's figure.")
    to: int | None = Field(None, description="The figure in the file.")


class FinalCheckResponse(BaseModel):
    files: list[str]
    checked: bool
    price: int | None = Field(description="Price in the file; null without can_see_restricted.")
    weeks: int | None
    draft_price: int | None
    draft_weeks: int | None
    findings: list[FindingOut] = Field(description="Exactly Scope, Cost, Timeline, Responses, Attachments.")
    meta: Meta


def _unique(values: list[str], what: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"Each {what} must be unique.")


# ---- Write sections (Section Writer Agent) -------------------------------------------


class SectionToWrite(BaseModel):
    key: str = Id
    title: str = Field(min_length=1, max_length=200)
    guidance: str = Field("", max_length=2000, description="How the template says to write it.")


class ClarificationFact(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field("", max_length=2000)


class ExperienceFact(BaseModel):
    client: str = Field(max_length=200)
    title: str = Field(max_length=300)
    year: int = Field(ge=1990, le=2100)
    outcome: str = Field("", max_length=1000)


class ProposalFacts(BaseModel):
    """Everything the writer may say. A fact left empty is written as "[to be completed]"."""

    client: str = Field("", max_length=200)
    title: str = Field("", max_length=300)
    summary: str = Field("", max_length=4000)
    industry: str = Field("", max_length=100)
    engagement_model: str = Field("", max_length=60)
    delivery_approach: str = Field("", max_length=2000)
    requirements: list[Annotated[str, Field(max_length=500)]] = Field(default_factory=list, max_length=200)
    standard_features: list[Annotated[str, Field(max_length=300)]] = Field(
        default_factory=list, max_length=200
    )
    modules: list[Annotated[str, Field(max_length=300)]] = Field(default_factory=list, max_length=100)
    timeline_weeks: int | None = Field(None, ge=0, le=520)
    team_size: int | None = Field(None, ge=0, le=1000)
    assumptions: list[Annotated[str, Field(max_length=1000)]] = Field(default_factory=list, max_length=100)
    clarifications: list[ClarificationFact] = Field(default_factory=list, max_length=100)
    experience: list[ExperienceFact] = Field(default_factory=list, max_length=20)


class WriteSectionsRequest(BaseModel):
    sections: list[SectionToWrite] = Field(min_length=1, max_length=30)
    facts: ProposalFacts

    @model_validator(mode="after")
    def _keys_are_unique(self) -> "WriteSectionsRequest":
        _unique([s.key for s in self.sections], "section key")
        return self


class WrittenSectionOut(BaseModel):
    key: str
    paragraphs: list[str] = Field(description="1 to 5 plain-text paragraphs, each at most 1200 characters.")


class WriteSectionsResponse(BaseModel):
    sections: list[WrittenSectionOut] = Field(description="One per requested section, in the same order.")
    dropped: list[str] = Field(description="Paragraphs removed by the checks (e.g. a money amount).")
    meta: Meta


# ---- Assume answers (Assumption Agent) -----------------------------------------------


class OpenQuestion(BaseModel):
    id: str = Id
    question: str = Field(min_length=1, max_length=2000)
    category: str = Field("", max_length=40)
    why: str = Field("", max_length=1000)


class AssumeAnswersRequest(BaseModel):
    rfp_summary: str = Field("", max_length=4000)
    engagement_model: str = Field("", max_length=60)
    requirements: list[Annotated[str, Field(max_length=500)]] = Field(default_factory=list, max_length=200)
    questions: list[OpenQuestion] = Field(min_length=1, max_length=60)

    @model_validator(mode="after")
    def _ids_are_unique(self) -> "AssumeAnswersRequest":
        _unique([q.id for q in self.questions], "question id")
        return self


class AssumptionOut(BaseModel):
    id: str
    assumption: str = Field(description="What we estimate on, stated as a fact (at most 400 characters).")
    risk: str = Field(description="What changes if the assumption is wrong (at most 300 characters).")


class AssumeAnswersResponse(BaseModel):
    assumptions: list[AssumptionOut] = Field(description="Exactly one per question, in the same order.")
    meta: Meta


# ---- Check an answer (Answer Check Agent) --------------------------------------------


class CheckAnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    why: str = Field("", max_length=1000)
    category: str = Field("", max_length=40)
    answer: str = Field("", max_length=4000)


class CheckAnswerResponse(BaseModel):
    acceptable: bool
    reason: str = Field(description="One sentence for the user when not acceptable; empty when acceptable.")
    suggestion: str = Field(description="A short example of a good answer.")
    meta: Meta


# ---- Size requirements (Sizing Agent) ------------------------------------------------


class RequirementToSize(BaseModel):
    key: str = Id
    label: str = Field(min_length=1, max_length=300)
    detail: str = Field("", max_length=2000)


class SizeRequirementsRequest(BaseModel):
    industry: str = Field("", max_length=100)
    requirements: list[RequirementToSize] = Field(min_length=1, max_length=80)

    @model_validator(mode="after")
    def _keys_are_unique(self) -> "SizeRequirementsRequest":
        _unique([r.key for r in self.requirements], "requirement key")
        return self


class SizeOut(BaseModel):
    key: str
    size: SizeClass
    days: int = Field(description="Build person-days for the size class (set by code, not the model).")
    rationale: str


class SizeRequirementsResponse(BaseModel):
    sizes: list[SizeOut] = Field(description="Exactly one per requirement, in the same order.")
    meta: Meta
