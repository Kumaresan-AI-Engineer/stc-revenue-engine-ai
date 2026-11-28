import time
import uuid

import jwt
import pytest

from revenue_engine_ai.agents.review import ReviewItem, review
from revenue_engine_ai.core.config import Settings
from revenue_engine_ai.core.errors import Unauthorized
from revenue_engine_ai.core.security import decode_service_token

from tests.fakes import FakeLLM

SECRET = "unit-test-secret-long-enough-for-hs256"


def _items(n: int) -> list[ReviewItem]:
    return [ReviewItem(id=str(i), statement="s", evidence=["e"], rule="r") for i in range(n)]


async def test_missing_verdict_fails_and_batches_are_split() -> None:
    llm = FakeLLM()
    llm.on(
        "review",
        lambda call: {
            "verdicts": [
                {"id": "0", "supported": True, "reason": "ok"},
                {"id": "1", "supported": False, "reason": "no"},
            ]
        },
    )
    result = await review(llm, _items(25), effort=None)
    assert llm.count("review") == 4  # 20 + 5, then each batch is re-asked for its missing ids
    assert result.failed["1"] == "no"
    assert result.failed["2"] == "The reviewer gave no verdict."
    assert "0" not in result.failed


async def test_a_missing_verdict_is_reasked_once_before_it_fails() -> None:
    llm = FakeLLM()
    calls = 0

    def answer(call: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"verdicts": [{"id": "0", "supported": True, "reason": "ok"}]}
        return {"verdicts": [{"id": str(i), "supported": True, "reason": "ok"} for i in range(3)]}

    llm.on("review", answer)
    result = await review(llm, _items(3), effort=None)
    assert llm.count("review") == 2
    assert result.failed == {}


async def test_nothing_to_review_makes_no_call() -> None:
    llm = FakeLLM()
    result = await review(llm, [], effort=None)
    assert result.failed == {} and llm.calls == []


def _settings(secret: str = SECRET) -> Settings:
    return Settings(service_token_secret=secret)  # type: ignore[arg-type]


def _token(**overrides: object) -> str:
    claims: dict[str, object] = {
        "workspace_id": str(uuid.uuid4()),
        "actor_id": str(uuid.uuid4()),
        "aud": "revenue-engine-ai",
        "iss": "revenue-engine-api",
        "exp": int(time.time()) + 60,
    }
    # An override of None leaves the claim out.
    payload = {k: v for k, v in (claims | overrides).items() if v is not None}
    return jwt.encode(payload, SECRET, algorithm="HS256")


def test_valid_token_gives_the_workspace_from_the_token() -> None:
    ws = uuid.uuid4()
    ctx = decode_service_token(_token(workspace_id=str(ws), can_see_restricted=True), _settings())
    assert ctx.workspace_id == ws and ctx.can_see_restricted


@pytest.mark.parametrize(
    "overrides",
    [
        {"exp": int(time.time()) - 10},
        {"aud": "someone-else"},
        {"iss": "someone-else"},
        {"iss": None},
        {"workspace_id": "not-a-uuid"},
    ],
)
def test_bad_tokens_are_refused(overrides: dict[str, object]) -> None:
    with pytest.raises(Unauthorized):
        decode_service_token(_token(**overrides), _settings())


def test_token_signed_with_another_secret_or_unconfigured_secret_is_refused() -> None:
    forged = jwt.encode(
        {"workspace_id": str(uuid.uuid4())}, "another-secret-long-enough-for-hs256", algorithm="HS256"
    )
    with pytest.raises(Unauthorized):
        decode_service_token(forged, _settings())
    with pytest.raises(Unauthorized):
        decode_service_token(_token(), _settings(secret=""))
