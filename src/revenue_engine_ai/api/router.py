"""HTTP endpoints (v1). Translate HTTP only; the use-cases live in services/."""

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError

from revenue_engine_ai.api.schemas import (
    AnalyseRequest,
    AnalyseResponse,
    AssumeAnswersRequest,
    AssumeAnswersResponse,
    CheckAnswerRequest,
    CheckAnswerResponse,
    ClientQueriesRequest,
    ClientQueriesResponse,
    DocumentMetaIn,
    DraftAnswersRequest,
    DraftAnswersResponse,
    EntryMetaIn,
    EstimateRequestRequest,
    EstimateRequestResponse,
    FinalCheckRequest,
    FinalCheckResponse,
    IndexResponse,
    JobAccepted,
    JobOut,
    JobSubmission,
    ProviderProbeRequest,
    ProviderProbeResponse,
    OutreachEmailDraftRequest,
    OutreachEmailDraftResponse,
    SearchRequest,
    SearchResponse,
    SimilarRequest,
    SimilarResponse,
    SizeRequirementsRequest,
    SizeRequirementsResponse,
    WriteCaseStudyRequest,
    WriteCaseStudyResponse,
    WriteSectionsRequest,
    WriteSectionsResponse,
)
from revenue_engine_ai.core.config import Settings, get_settings
from revenue_engine_ai.core.errors import NotFound, RuleViolation
from revenue_engine_ai.core.security import ServiceContext, service_context
from revenue_engine_ai.llm.ports import LLM, Embedder
from revenue_engine_ai.llm.providers import ProbeLLMBuilder, get_embedder, get_llm, get_probe_llm_builder
from revenue_engine_ai.services import analyse as analyse_service
from revenue_engine_ai.services import (
    answer_check,
    assumptions,
    case_studies,
    clarification,
    drafting,
    estimate_assistant,
    final_check,
    library,
    outreach_email,
    proposal_writing,
    provider_probe,
    sizing,
)
from revenue_engine_ai.services.jobs import JobRunner, get_job_runner
from revenue_engine_ai.services.vectors import Vectorizer, vectorizer
from revenue_engine_ai.store.ports import LibraryIndex, ResultCache
from revenue_engine_ai.store.postgres import library_index, result_cache

router = APIRouter(prefix="/v1")
# A final proposal is one document, or a technical and a commercial one.
MAX_FINAL_FILES = 2


def _form_json[M: BaseModel](model: type[M], raw: str) -> M:
    """Multipart requests carry their JSON part as a string; validate it like a JSON body."""
    try:
        return model.model_validate_json(raw)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


async def _read(file: UploadFile, settings: Settings) -> bytes:
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise RuleViolation("file_size", f"Files can be at most {settings.max_upload_mb} MB.")
    return data


@router.post(
    "/provider/probe",
    response_model=ProviderProbeResponse,
    tags=["provider"],
    dependencies=[Depends(service_context)],
)
async def probe_provider(
    body: ProviderProbeRequest,
    build: ProbeLLMBuilder = Depends(get_probe_llm_builder),
    settings: Settings = Depends(get_settings),
) -> ProviderProbeResponse:
    """Try a model before it is saved: three tiny requests (a structured answer, its speed, a tool call)
    through the same client the agents use. The settings and key come in the body, used for this call only."""
    llm = build(
        api_mode=body.api_mode,
        base_url=body.base_url,
        model=body.model,
        api_key=body.api_key.get_secret_value() if body.api_key else None,
    )
    return await provider_probe.probe(llm, settings)


@router.post("/rfp/analyse", response_model=AnalyseResponse, tags=["rfp"])
async def analyse(
    file: UploadFile = File(...),
    request: str = Form(..., description="AnalyseRequest as JSON (the feature catalogue)."),
    ctx: ServiceContext = Depends(service_context),
    llm: LLM = Depends(get_llm),
    cache: ResultCache = Depends(result_cache),
    settings: Settings = Depends(get_settings),
) -> AnalyseResponse:
    """Requirement Extraction Agent: requirements, questions, documents, facts and unclear points,
    each with the quote and page it comes from."""
    body = _form_json(AnalyseRequest, request)
    data = await _read(file, settings)
    return await analyse_service.analyse(
        ctx, file.filename or "upload", data, body.catalogue, llm, cache, settings
    )


@router.put("/library/entries/{entry_id}", response_model=IndexResponse, tags=["library"])
async def index_entry(
    entry_id: str,
    meta: str = Form(..., description="EntryMetaIn as JSON."),
    rfp: UploadFile | None = File(None),
    proposal: UploadFile | None = File(None),
    ctx: ServiceContext = Depends(service_context),
    index: LibraryIndex = Depends(library_index),
    vectors: Vectorizer = Depends(vectorizer),
    settings: Settings = Depends(get_settings),
) -> IndexResponse:
    """Library indexer: (re)index one past proposal. Unchanged content is not embedded again."""
    files = {}
    for section, upload in (("rfp", rfp), ("proposal", proposal)):
        if upload is not None:
            files[section] = (upload.filename or section, await _read(upload, settings))
    return await library.index_entry(ctx, entry_id, _form_json(EntryMetaIn, meta), files, index, vectors)


@router.put("/library/documents/{entry_id}", response_model=IndexResponse, tags=["library"])
async def index_document(
    entry_id: str,
    meta: str = Form(..., description="DocumentMetaIn as JSON."),
    file: UploadFile = File(...),
    ctx: ServiceContext = Depends(service_context),
    index: LibraryIndex = Depends(library_index),
    vectors: Vectorizer = Depends(vectorizer),
    settings: Settings = Depends(get_settings),
) -> IndexResponse:
    """Library indexer: (re)index one standalone document (a case study or another file) for report search.
    Unchanged content is not embedded again. Remove it with DELETE /library/entries/{entry_id}."""
    document = (file.filename or "document", await _read(file, settings))
    return await library.index_document(
        ctx, entry_id, _form_json(DocumentMetaIn, meta), document, index, vectors
    )


@router.delete("/library/entries/{entry_id}", status_code=204, tags=["library"])
async def delete_entry(
    entry_id: str,
    ctx: ServiceContext = Depends(service_context),
    index: LibraryIndex = Depends(library_index),
) -> Response:
    """Remove a past proposal from the index."""
    await library.delete_entry(ctx, entry_id, index)
    return Response(status_code=204)


@router.post("/library/similar", response_model=SimilarResponse, tags=["library"])
async def similar(
    body: SimilarRequest,
    ctx: ServiceContext = Depends(service_context),
    index: LibraryIndex = Depends(library_index),
    vectors: Vectorizer = Depends(vectorizer),
    embedder: Embedder = Depends(get_embedder),
) -> SimilarResponse:
    """Semantic search for Step 3: past proposals covering each requirement, and similar projects."""
    return await library.similar(ctx, body, index, vectors, embedder.model)


@router.post("/library/search", response_model=SearchResponse, tags=["library"])
async def search_documents(
    body: SearchRequest,
    ctx: ServiceContext = Depends(service_context),
    index: LibraryIndex = Depends(library_index),
    vectors: Vectorizer = Depends(vectorizer),
    embedder: Embedder = Depends(get_embedder),
) -> SearchResponse:
    """Semantic search over the indexed documents (past bids, case studies, other files) for the report
    builder: filters narrow the entries, then each is ranked by its best passages."""
    return await library.search(ctx, body, index, vectors, embedder.model)


@router.post("/proposals/draft-answers", response_model=DraftAnswersResponse, tags=["proposals"])
async def draft_answers(
    body: DraftAnswersRequest,
    ctx: ServiceContext = Depends(service_context),
    llm: LLM = Depends(get_llm),
    index: LibraryIndex = Depends(library_index),
    vectors: Vectorizer = Depends(vectorizer),
    cache: ResultCache = Depends(result_cache),
    settings: Settings = Depends(get_settings),
) -> DraftAnswersResponse:
    """Response Drafting Agent + Review Agent: grounded drafts, or "needs a person" with the reason."""
    return await drafting.draft_answers(ctx, body, llm, index, vectors, cache, settings)


@router.post("/proposals/client-queries", response_model=ClientQueriesResponse, tags=["proposals"])
async def client_queries(
    body: ClientQueriesRequest,
    ctx: ServiceContext = Depends(service_context),
    llm: LLM = Depends(get_llm),
    cache: ResultCache = Depends(result_cache),
    settings: Settings = Depends(get_settings),
) -> ClientQueriesResponse:
    """Clarification Agent + Review Agent: questions for the client with the assumption to price on."""
    return await clarification.client_queries(ctx, body, llm, cache, settings)


@router.post("/proposals/write-sections", response_model=WriteSectionsResponse, tags=["proposals"])
async def write_sections(
    body: WriteSectionsRequest,
    ctx: ServiceContext = Depends(service_context),
    llm: LLM = Depends(get_llm),
    cache: ResultCache = Depends(result_cache),
    settings: Settings = Depends(get_settings),
) -> WriteSectionsResponse:
    """Section Writer Agent: each section written from the facts only, following its template guidance, as
    plain paragraphs. A missing fact reads "[to be completed]"; a paragraph with money in it is dropped."""
    return await proposal_writing.write_sections(ctx, body, llm, cache, settings)


@router.post("/outreach/draft", response_model=OutreachEmailDraftResponse, tags=["outreach"])
async def draft_outreach_email(
    body: OutreachEmailDraftRequest,
    ctx: ServiceContext = Depends(service_context),
    llm: LLM = Depends(get_llm),
    cache: ResultCache = Depends(result_cache),
    settings: Settings = Depends(get_settings),
) -> OutreachEmailDraftResponse:
    """Outreach Email Writer + Review Agent: a plain-text subject and body grounded in the supplied context."""
    return await outreach_email.draft_email(ctx, body, llm, cache, settings)


@router.post("/proposals/estimate-request", response_model=EstimateRequestResponse, tags=["proposals"])
async def estimate_request(
    body: EstimateRequestRequest,
    ctx: ServiceContext = Depends(service_context),
    llm: LLM = Depends(get_llm),
    cache: ResultCache = Depends(result_cache),
    settings: Settings = Depends(get_settings),
) -> EstimateRequestResponse:
    """Estimate Assistant Agent (Step 6 'Ask AI'): the person's request about the estimate as actions the
    backend applies: leave out or add back scope, change the delivery approach, set the timeline or team, set
    or reset the price. Only actions the estimate allows; no figure is computed here and no price is ever
    sent."""
    return await estimate_assistant.estimate_request(ctx, body, llm, cache, settings)


@router.post("/case-studies/write", response_model=WriteCaseStudyResponse, tags=["case studies"])
async def write_case_study(
    body: WriteCaseStudyRequest,
    ctx: ServiceContext = Depends(service_context),
    llm: LLM = Depends(get_llm),
    cache: ResultCache = Depends(result_cache),
    index: LibraryIndex = Depends(library_index),
    vectors: Vectorizer = Depends(vectorizer),
    settings: Settings = Depends(get_settings),
) -> WriteCaseStudyResponse:
    """Case Study Writer Agent: a case study from a won (customer story) or lost (lessons learned) proposal,
    in the case study template's voice, from its facts and its indexed passages. No money; a metric only when
    its number is in the facts or the proposal."""
    return await case_studies.write_case_study(ctx, body, llm, cache, index, vectors, settings)


@router.post("/proposals/clarifications/assume", response_model=AssumeAnswersResponse, tags=["proposals"])
async def assume_answers(
    body: AssumeAnswersRequest,
    ctx: ServiceContext = Depends(service_context),
    llm: LLM = Depends(get_llm),
    cache: ResultCache = Depends(result_cache),
    settings: Settings = Depends(get_settings),
) -> AssumeAnswersResponse:
    """Assumption Agent: one conservative assumption to estimate on, with its risk, per open question."""
    return await assumptions.assume_answers(ctx, body, llm, cache, settings)


@router.post("/proposals/clarifications/check-answer", response_model=CheckAnswerResponse, tags=["proposals"])
async def check_answer(
    body: CheckAnswerRequest,
    ctx: ServiceContext = Depends(service_context),
    llm: LLM = Depends(get_llm),
    cache: ResultCache = Depends(result_cache),
    settings: Settings = Depends(get_settings),
) -> CheckAnswerResponse:
    """Answer Check Agent: whether a typed answer answers its question, with a reason and an example."""
    return await answer_check.check_answer(ctx, body, llm, cache, settings)


@router.post("/proposals/size-requirements", response_model=SizeRequirementsResponse, tags=["proposals"])
async def size_requirements(
    body: SizeRequirementsRequest,
    ctx: ServiceContext = Depends(service_context),
    llm: LLM = Depends(get_llm),
    cache: ResultCache = Depends(result_cache),
    settings: Settings = Depends(get_settings),
) -> SizeRequirementsResponse:
    """Sizing Agent: a size class (XS-XL) per requirement by industry benchmarks; the person-days per class
    are fixed in code (XS 5, S 10, M 20, L 40, XL 80)."""
    return await sizing.size_requirements(ctx, body, llm, cache, settings)


@router.post("/proposals/final-check", response_model=FinalCheckResponse, tags=["proposals"])
async def check_final_proposal(
    files: list[UploadFile] = File(
        ..., description="The final proposal: one file, or technical + commercial."
    ),
    request: str = Form(..., description="FinalCheckRequest as JSON (the draft snapshot)."),
    ctx: ServiceContext = Depends(service_context),
    llm: LLM = Depends(get_llm),
    cache: ResultCache = Depends(result_cache),
    settings: Settings = Depends(get_settings),
) -> FinalCheckResponse:
    """Final Proposal Check Agent: what the uploaded proposal states against the draft, as five findings
    (Scope, Cost, Timeline, Responses, Attachments). The price is compared only with can_see_restricted."""
    if not 1 <= len(files) <= MAX_FINAL_FILES:
        raise RuleViolation("files", f"Send one or {MAX_FINAL_FILES} files.")
    body = _form_json(FinalCheckRequest, request)
    content = [(f.filename or "proposal", await _read(f, settings)) for f in files]
    return await final_check.final_check(ctx, content, body.draft, llm, cache, settings)


@router.post("/jobs", status_code=202, response_model=JobAccepted, tags=["jobs"])
async def submit_job(
    body: JobSubmission,
    ctx: ServiceContext = Depends(service_context),
    runner: JobRunner = Depends(get_job_runner),
) -> JobAccepted:
    """Accept an asynchronous job (`rfp_analysis`, `template_extract`); its progress and result are POSTed,
    signed, to callback_url. The same idempotency_key in a workspace never starts a second job: it returns
    the first."""
    return JobAccepted(job_id=await runner.submit(ctx, body))


@router.get("/jobs/{job_id}", response_model=JobOut, tags=["jobs"])
async def get_job(
    job_id: str,
    ctx: ServiceContext = Depends(service_context),
    runner: JobRunner = Depends(get_job_runner),
) -> JobOut:
    """A job's status, for when a callback was missed. Only jobs of the token's workspace."""
    job = await runner.store.get(ctx.workspace_id, job_id)
    if job is None:
        raise NotFound("No such job.")
    return JobOut(job_id=job.id, type=job.type, status=job.status, error=job.error)  # type: ignore[arg-type]
