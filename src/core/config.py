"""Settings from environment variables (RE_AI_ prefix). Infrastructure only: prompts, schemas and
thresholds that shape the output are versioned in code so a change is reviewed and evaluated."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ReasoningEffort = Literal["minimal", "low", "medium", "high"]

# The Qdrant collection is created with this size; another embedding model needs a new collection and a
# re-index, never a silent change.
EMBEDDING_DIMENSIONS = 384
# Local, free embedding models (fastembed) and their vector size.
EMBEDDING_MODEL_DIMENSIONS = {"BAAI/bge-small-en-v1.5": 384, "sentence-transformers/all-MiniLM-L6-v2": 384}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RE_AI_", env_file=".env", extra="ignore")

    env: Literal["local", "test", "staging", "production"] = "local"
    database_url: str = "postgresql+asyncpg://revenue_ai:revenue_ai@localhost:5433/revenue_ai"
    # Vector database (Qdrant). Empty URL: embedded, stored in qdrant_path; set a URL for a Qdrant server.
    qdrant_url: str = ""
    qdrant_api_key: SecretStr = SecretStr("")
    qdrant_path: str = ".qdrant"

    openai_api_key: SecretStr = SecretStr("")
    # Empty: OpenAI itself. Set it to use an OpenAI-compatible gateway.
    openai_base_url: str = ""
    # "responses": OpenAI's Responses API (enforced schemas, reasoning with tools). "chat_completions": for
    # gateways that only serve Chat Completions (e.g. Yantra); answers are validated and corrected instead.
    llm_api: Literal["responses", "chat_completions"] = "responses"
    # Output budget per Chat Completions call, reasoning tokens included.
    max_output_tokens: int = Field(default=32000, ge=1024, le=128000)
    # Pinned snapshot: the same input must keep producing the same kind of output (determinism).
    chat_model: str = "gpt-5.5-2026-04-23"
    # Free and local (fastembed, ONNX on CPU): no key, and documents never leave the service.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: str | None = None  # where the model is downloaded once; default: temp dir
    # Reasoning models take an effort instead of a temperature; leave empty for non-reasoning models.
    extraction_effort: ReasoningEffort | None = "medium"
    drafting_effort: ReasoningEffort | None = "medium"
    clarification_effort: ReasoningEffort | None = "low"
    review_effort: ReasoningEffort | None = "medium"
    # Reading an uploaded final proposal against the draft: a fact-finding task, kept fast (< 30 s).
    final_check_effort: ReasoningEffort | None = "low"
    # Reading a template's sections (and, for a template without headings, its images) once per upload.
    template_analysis_effort: ReasoningEffort | None = "low"
    # Writing the proposal's sections from the facts: the longest output, in the template's voice.
    section_writer_effort: ReasoningEffort | None = "medium"
    # Assumed answers and effort sizes: short, conservative picks.
    assumption_effort: ReasoningEffort | None = "low"
    sizing_effort: ReasoningEffort | None = "low"
    # Checking one typed answer while the user waits: kept as fast as possible.
    answer_check_effort: ReasoningEffort | None = "low"
    # Step 6 'Ask AI' turns a request into estimate actions while the user waits: kept fast.
    estimate_assistant_effort: ReasoningEffort | None = "low"
    # A case study from a past proposal: a story in the template's voice, written once per proposal.
    case_study_writer_effort: ReasoningEffort | None = "medium"
    # Only sent when set, and only accepted by non-reasoning models (e.g. gpt-4.1).
    temperature: float | None = Field(default=None, ge=0, le=2)
    request_timeout_s: float = 120.0
    max_parallel_llm_calls: int = Field(default=8, ge=1, le=64)

    # Shared HS256 secret with the backend (core/service_tokens.py there). Production moves to RS256
    # against the backend's JWKS (ARCHITECTURE §10).
    service_token_secret: SecretStr = SecretStr("")
    service_token_audience: str = "revenue-engine-ai"
    service_token_issuer: str = "revenue-engine-api"
    # Job callbacks are signed with this shared secret (HMAC-SHA256; the backend's RE_AI_CALLBACK_SECRET).
    callback_secret: SecretStr = SecretStr("")
    # A job's file_url must start with one of these (the backend's object storage), so a job can never make
    # this service fetch an arbitrary address.
    file_url_prefixes: list[str] = ["http://localhost:9000/", "http://127.0.0.1:9000/"]
    # Jobs run in this process, at most this many at once; the rest wait their turn.
    max_parallel_jobs: int = Field(default=4, ge=1, le=32)

    max_upload_mb: int = Field(default=25, ge=1, le=100)
    # Scanned PDF pages (no text layer) are read with OCR, at most this many per document.
    ocr_max_pages: int = Field(default=100, ge=0, le=500)


@lru_cache
def get_settings() -> Settings:
    return Settings()
