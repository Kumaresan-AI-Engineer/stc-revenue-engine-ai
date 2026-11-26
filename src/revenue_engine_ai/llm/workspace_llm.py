"""A workspace's own AI client (admin > AI provider in the backend).

The settings (model, endpoint, API type) arrive in the signed service token. The key does not: the backend
writes it to a secret store and this service reads it from there, at a path built from the token's workspace
id, so nobody can ask for another workspace's key. A workspace with no configuration gets an error that says
so: there is no service-level key to fall back on, the AI provider page is the only source of credentials.
"""

import json
import logging
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from openai import AsyncOpenAI

from revenue_engine_ai.core.config import Settings
from revenue_engine_ai.core.errors import ProviderNotConfigured, UpstreamError
from revenue_engine_ai.core.security import AiProvider, ServiceContext
from revenue_engine_ai.llm.chat_completions_provider import ChatCompletionsLLM
from revenue_engine_ai.llm.openai_provider import OpenAILLM
from revenue_engine_ai.llm.ports import LLM

log = logging.getLogger(__name__)

# The backend stores `{"schema_version": 1, "api_key": ...}` (modules/ai_config/service.py there).
SECRET_SCHEMA_VERSION = 1
# A self-hosted server that needs no key still needs something in the client's key field.
NO_KEY_PLACEHOLDER = "not-needed"
API_KEY_AUTH = "api_key"

KEY_MISSING = (
    "This workspace's AI key could not be found. Ask an admin to save it again on the AI provider page."
)
KEY_UNREADABLE = "This workspace's AI key could not be read. Please try again, or tell an admin."
NOT_SUPPORTED = "This workspace's AI provider settings are not supported by the AI service."
NO_SECRET_STORE = (
    "The AI service cannot read workspace AI keys: its secret store is not set up (RE_AI_SECRET_STORE_FILE)."
)

PROVIDERS: dict[str, type[OpenAILLM] | type[ChatCompletionsLLM]] = {
    "responses": OpenAILLM,
    "chat_completions": ChatCompletionsLLM,
}


def secret_path(prefix: str, environment: str, workspace_id: uuid.UUID, provider: str) -> str:
    """Where the backend keeps a workspace's key (the same rule as `parameter_path` there)."""
    parts = [prefix.strip("/"), environment, "workspaces", str(workspace_id), "ai", provider, "credentials"]
    return "/" + "/".join(p for p in parts if p)


def read_api_key(file: str, path: str) -> str | None:
    """The key stored at `path` in the backend's local secret file; None when there is none."""
    try:
        data = json.loads(Path(file).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise UpstreamError(KEY_UNREADABLE) from exc
    entry = data.get(path) if isinstance(data, dict) else None
    value = entry.get("value") if isinstance(entry, dict) else None
    if not isinstance(value, str):
        return None
    try:
        document = json.loads(value)
    except ValueError as exc:
        raise UpstreamError(KEY_UNREADABLE) from exc
    if not isinstance(document, dict) or document.get("schema_version") != SECRET_SCHEMA_VERSION:
        raise UpstreamError(KEY_UNREADABLE)
    key = document.get("api_key")
    return key if isinstance(key, str) and key else None


class WorkspaceLLMs:
    """Hands out the LLM for a call: the workspace's own. One that has not configured it gets an error."""

    def __init__(self, settings: Settings, clock: Callable[[], float] = time.monotonic) -> None:
        self._settings = settings
        self._clock = clock
        # One client per workspace. A saved change carries a new config version in the token, so it is a new
        # entry; the lifetime only bounds how long a replaced key can be served.
        self._cache: dict[tuple[uuid.UUID, AiProvider], tuple[float, LLM]] = {}

    def get(self, ctx: ServiceContext) -> LLM:
        provider = ctx.ai_provider
        if provider is None:
            log.warning("workspace_llm.no_provider", extra={"workspace_id": str(ctx.workspace_id)})
            raise ProviderNotConfigured()
        key = (ctx.workspace_id, provider)
        now = self._clock()
        hit = self._cache.get(key)
        if hit and now - hit[0] < self._settings.workspace_llm_ttl_s:
            return hit[1]
        llm = self._build(ctx.workspace_id, provider)
        self._cache = {k: v for k, v in self._cache.items() if k[0] != ctx.workspace_id}
        self._cache[key] = (now, llm)
        return llm

    def _api_key(self, workspace_id: uuid.UUID, provider: AiProvider) -> str:
        if provider.auth_type != API_KEY_AUTH:
            return NO_KEY_PLACEHOLDER
        s = self._settings
        if not s.secret_store_file:
            raise UpstreamError(NO_SECRET_STORE)
        path = secret_path(s.secret_store_prefix, s.env, workspace_id, provider.provider)
        key = read_api_key(s.secret_store_file, path)
        if key is None:
            log.warning("workspace_llm.key_missing", extra={"workspace_id": str(workspace_id)})
            raise UpstreamError(KEY_MISSING)
        return key

    def _build(self, workspace_id: uuid.UUID, provider: AiProvider) -> LLM:
        _implementation(provider.api_mode)  # refuse an unknown mode before the key is read
        return make_llm(
            self._settings,
            api_mode=provider.api_mode,
            base_url=provider.base_url,
            model=provider.model,
            api_key=self._api_key(workspace_id, provider),
        )


def _implementation(api_mode: str) -> type[OpenAILLM] | type[ChatCompletionsLLM]:
    implementation = PROVIDERS.get(api_mode)
    if implementation is None:
        raise UpstreamError(NOT_SUPPORTED)
    return implementation


def make_llm(
    settings: Settings,
    *,
    api_mode: str,
    base_url: str | None,
    model: str,
    api_key: str,
    max_retries: int = 2,
) -> LLM:
    """The LLM client for one provider's settings. `base_url` None is OpenAI itself, not the gateway this
    service's own settings name."""
    implementation = _implementation(api_mode)
    client = AsyncOpenAI(
        api_key=api_key, base_url=base_url, timeout=settings.request_timeout_s, max_retries=max_retries
    )
    own = settings.model_copy(
        update={"chat_model": model, "llm_api": api_mode, "openai_base_url": base_url or ""}
    )
    return implementation(client, own)
