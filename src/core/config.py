"""Settings from environment variables (RE_AI_ prefix). Infrastructure only: prompts, schemas and
thresholds that shape the output are versioned in code so a change is reviewed and evaluated."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ReasoningEffort = Literal["minimal", "low", "medium", "high"]

# The pgvector column is created with this size by migration 0001; another embedding model needs a
# new migration and a re-index, never a silent change.
EMBEDDING_DIMENSIONS = 384
# Local, free embedding models (fastembed) and their vector size.
EMBEDDING_MODEL_DIMENSIONS = {"BAAI/bge-small-en-v1.5": 384, "sentence-transformers/all-MiniLM-L6-v2": 384}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RE_AI_", env_file=".env", extra="ignore")

    env: Literal["local", "test", "staging", "production"] = "local"
    database_url: str = "postgresql+asyncpg://revenue_ai:revenue_ai@localhost:5433/revenue_ai"

    openai_api_key: SecretStr = SecretStr("")
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
    # Only sent when set, and only accepted by non-reasoning models (e.g. gpt-4.1).
    temperature: float | None = Field(default=None, ge=0, le=2)
    request_timeout_s: float = 120.0
    max_parallel_llm_calls: int = Field(default=8, ge=1, le=64)

    # Shared HS256 secret with the backend (core/service_tokens.py there). Production moves to RS256
    # against the backend's JWKS (ARCHITECTURE §10).
    service_token_secret: SecretStr = SecretStr("")
    service_token_audience: str = "revenue-engine-ai"

    max_upload_mb: int = Field(default=25, ge=1, le=100)


@lru_cache
def get_settings() -> Settings:
    return Settings()
