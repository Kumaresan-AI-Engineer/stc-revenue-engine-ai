import json
import time
import uuid
from pathlib import Path

import jwt
import pytest

from revenue_engine_ai.core.config import Settings
from revenue_engine_ai.core.errors import UpstreamError
from revenue_engine_ai.core.security import AiProvider, ServiceContext, decode_service_token
from revenue_engine_ai.llm.chat_completions_provider import ChatCompletionsLLM
from revenue_engine_ai.llm.openai_provider import OpenAILLM
from revenue_engine_ai.llm.workspace_llm import (
    KEY_MISSING,
    KEY_UNREADABLE,
    NO_KEY_PLACEHOLDER,
    WorkspaceLLMs,
    read_api_key,
    secret_path,
)

from tests.fakes import FakeLLM

WORKSPACE = uuid.uuid4()
OTHER = uuid.uuid4()
SECRET = "unit-test-secret-long-enough-for-hs256"


def _provider(**overrides: object) -> AiProvider:
    fields: dict[str, object] = {
        "provider": "openai",
        "model": "gpt-test",
        "base_url": None,
        "api_mode": "responses",
        "auth_type": "api_key",
        "config_version": 1,
    }
    fields.update(overrides)
    return AiProvider(**fields)  # type: ignore[arg-type]


def _ctx(provider: AiProvider | None, workspace: uuid.UUID = WORKSPACE) -> ServiceContext:
    return ServiceContext(workspace, uuid.uuid4(), False, None, provider)


def _store(
    tmp_path: Path, workspace: uuid.UUID = WORKSPACE, key: str = "sk-workspace", provider: str = "openai"
):
    """The file the backend writes: path -> {value: '{"schema_version":1,"api_key":...}', tags}."""
    path = tmp_path / "credentials.json"
    document = json.dumps({"schema_version": 1, "api_key": key})
    entry = {secret_path("", "test", workspace, provider): {"value": document, "tags": {}}}
    path.write_text(json.dumps(entry), encoding="utf-8")
    return path


def _settings(file: Path | str = "", **overrides: object) -> Settings:
    return Settings(
        env="test",
        secret_store_file=str(file),
        service_token_secret=SECRET,  # type: ignore[arg-type]
        **overrides,  # type: ignore[arg-type]
    )


def _resolver(settings: Settings, clock=time.monotonic) -> tuple[WorkspaceLLMs, FakeLLM]:
    own = FakeLLM()
    return WorkspaceLLMs(settings, lambda: own, clock), own


def test_the_path_is_the_one_the_backend_writes() -> None:
    assert (
        secret_path("", "local", WORKSPACE, "openai")
        == f"/local/workspaces/{WORKSPACE}/ai/openai/credentials"
    )
    assert secret_path("/stc/", "prod", WORKSPACE, "self_hosted") == (
        f"/stc/prod/workspaces/{WORKSPACE}/ai/self_hosted/credentials"
    )


def test_reading_a_key(tmp_path: Path) -> None:
    file = _store(tmp_path)
    assert read_api_key(str(file), secret_path("", "test", WORKSPACE, "openai")) == "sk-workspace"
    assert read_api_key(str(file), secret_path("", "test", OTHER, "openai")) is None
    assert read_api_key(str(tmp_path / "missing.json"), "/x") is None


def test_a_corrupt_file_is_an_error_not_a_missing_key(tmp_path: Path) -> None:
    file = tmp_path / "credentials.json"
    file.write_text("not json", encoding="utf-8")
    with pytest.raises(UpstreamError, match=KEY_UNREADABLE):
        read_api_key(str(file), "/x")


def test_a_file_that_is_not_an_object_has_no_key(tmp_path: Path) -> None:
    file = tmp_path / "credentials.json"
    file.write_text(json.dumps([1]), encoding="utf-8")
    assert read_api_key(str(file), "/x") is None


def test_a_stored_document_of_another_version_is_refused(tmp_path: Path) -> None:
    path = secret_path("", "test", WORKSPACE, "openai")
    file = tmp_path / "credentials.json"
    document = json.dumps({"schema_version": 2, "api_key": "sk"})
    file.write_text(json.dumps({path: {"value": document}}), encoding="utf-8")
    with pytest.raises(UpstreamError, match=KEY_UNREADABLE):
        read_api_key(str(file), path)


def test_a_workspace_without_a_configuration_uses_the_services_own_llm(tmp_path: Path) -> None:
    resolver, own = _resolver(_settings(_store(tmp_path)))
    assert resolver.get(_ctx(None)) is own


def test_a_configured_workspace_gets_its_own_client_key_and_model(tmp_path: Path) -> None:
    resolver, own = _resolver(_settings(_store(tmp_path)))
    llm = resolver.get(_ctx(_provider(model="gpt-mine")))
    assert isinstance(llm, OpenAILLM) and llm is not own
    assert llm.model == "gpt-mine"
    assert llm._client.api_key == "sk-workspace"
    assert "api.openai.com" in str(llm._client.base_url)


def test_the_chat_completions_mode_and_a_custom_endpoint(tmp_path: Path) -> None:
    file = _store(tmp_path, provider="self_hosted")
    resolver, _ = _resolver(_settings(file, openai_base_url="https://own-gateway.example/v1"))
    provider = _provider(
        provider="self_hosted", api_mode="chat_completions", base_url="https://llm.example/v1"
    )
    llm = resolver.get(_ctx(provider))
    assert isinstance(llm, ChatCompletionsLLM)
    assert str(llm._client.base_url).startswith("https://llm.example/v1")


def test_a_key_free_server_needs_no_stored_key(tmp_path: Path) -> None:
    resolver, _ = _resolver(_settings(""))
    provider = _provider(
        provider="self_hosted",
        api_mode="chat_completions",
        auth_type="none",
        base_url="https://llm.example/v1",
    )
    assert resolver.get(_ctx(provider))._client.api_key == NO_KEY_PLACEHOLDER  # type: ignore[attr-defined]


def test_a_missing_key_fails_the_call_instead_of_using_the_services_account(tmp_path: Path) -> None:
    resolver, _ = _resolver(_settings(_store(tmp_path, workspace=OTHER)))
    with pytest.raises(UpstreamError, match=KEY_MISSING):
        resolver.get(_ctx(_provider()))


def test_without_a_secret_store_here_every_workspace_keeps_the_services_own_llm() -> None:
    resolver, own = _resolver(_settings(""))
    assert resolver.get(_ctx(_provider())) is own


def test_an_unknown_api_mode_is_refused(tmp_path: Path) -> None:
    resolver, _ = _resolver(_settings(_store(tmp_path)))
    with pytest.raises(UpstreamError):
        resolver.get(_ctx(_provider(api_mode="something-else")))


def test_the_client_is_reused_until_the_settings_change_or_it_expires(tmp_path: Path) -> None:
    now = [0.0]
    resolver, _ = _resolver(_settings(_store(tmp_path), workspace_llm_ttl_s=60), lambda: now[0])
    first = resolver.get(_ctx(_provider()))
    assert resolver.get(_ctx(_provider())) is first
    assert resolver.get(_ctx(_provider(config_version=2))) is not first
    again = resolver.get(_ctx(_provider(config_version=2)))
    now[0] = 61.0
    assert resolver.get(_ctx(_provider(config_version=2))) is not again


def test_a_replaced_key_is_used_once_the_version_changes(tmp_path: Path) -> None:
    file = _store(tmp_path, key="sk-old")
    resolver, _ = _resolver(_settings(file))
    assert resolver.get(_ctx(_provider()))._client.api_key == "sk-old"  # type: ignore[attr-defined]
    _store(tmp_path, key="sk-new")
    new = resolver.get(_ctx(_provider(config_version=2)))
    assert new._client.api_key == "sk-new"  # type: ignore[attr-defined]


def test_one_workspace_never_gets_anothers_client(tmp_path: Path) -> None:
    file = _store(tmp_path, workspace=WORKSPACE, key="sk-a")
    # a second workspace in the same file
    data = json.loads(file.read_text(encoding="utf-8"))
    data[secret_path("", "test", OTHER, "openai")] = {
        "value": json.dumps({"schema_version": 1, "api_key": "sk-b"})
    }
    file.write_text(json.dumps(data), encoding="utf-8")
    resolver, _ = _resolver(_settings(file))
    a = resolver.get(_ctx(_provider(), WORKSPACE))
    b = resolver.get(_ctx(_provider(), OTHER))
    assert (a._client.api_key, b._client.api_key) == ("sk-a", "sk-b")  # type: ignore[attr-defined]


def _token(**claims: object) -> str:
    base: dict[str, object] = {
        "workspace_id": str(WORKSPACE),
        "actor_id": str(uuid.uuid4()),
        "aud": "revenue-engine-ai",
        "iss": "revenue-engine-api",
        "exp": int(time.time()) + 60,
    }
    base.update(claims)
    return jwt.encode(base, SECRET, algorithm="HS256")


def test_the_token_carries_the_workspaces_provider_settings() -> None:
    claim = {
        "provider": "openai",
        "model": "gpt-mine",
        "base_url": None,
        "api_mode": "responses",
        "auth_type": "api_key",
        "config_version": 3,
    }
    ctx = decode_service_token(_token(ai_provider=claim), _settings())
    assert ctx.ai_provider == AiProvider("openai", "gpt-mine", None, "responses", "api_key", 3)
    assert decode_service_token(_token(), _settings()).ai_provider is None


@pytest.mark.parametrize("claim", ["openai", {"provider": "openai"}, {"config_version": "x"}])
def test_a_malformed_provider_claim_rejects_the_token(claim: object) -> None:
    from revenue_engine_ai.core.errors import Unauthorized

    with pytest.raises(Unauthorized):
        decode_service_token(_token(ai_provider=claim), _settings())
