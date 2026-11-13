"""HTTP endpoints (v1). Translate HTTP only; the use-cases live in services/."""

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError

from api.schemas import (
    AnalyseRequest,
    AnalyseResponse,
    ClientQueriesRequest,
    ClientQueriesResponse,
    DraftAnswersRequest,
    DraftAnswersResponse,
    EntryMetaIn,
    IndexResponse,
    SimilarRequest,
    SimilarResponse,
)
from core.config import Settings, get_settings
from core.errors import RuleViolation
from core.security import ServiceContext, service_context
from llm.ports import LLM, Embedder
from llm.providers import get_embedder, get_llm
from services import analyse as analyse_service
from services import clarification, drafting, library
from services.vectors import Vectorizer, vectorizer
from store.ports import LibraryIndex, ResultCache
from store.postgres import library_index, result_cache

router = APIRouter(prefix="/v1")


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
