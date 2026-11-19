"""Asynchronous jobs through HTTP: submission, idempotency, the signed callbacks and their retries."""

import hashlib
import hmac
import uuid
from typing import Any

from conftest import Env
from fakes import approve_all
from test_api import CATALOGUE, RFP, _form

from core.config import get_settings

FILE_URL = "http://storage.test/ws/rfp.txt?X-Amz-Signature=abc"
CALLBACK = "http://backend.test/api/v1/proposal-drafts/ai-callback"


def _submission(key: str = "draft-1", **overrides: Any) -> dict[str, Any]:
    job: dict[str, Any] = {
        "type": "rfp_analysis",
        "input": {
            "file_url": FILE_URL,
            "filename": "rfp.txt",
            "mode": "upload_rfp",
            "client": "Acme",
            "industry": "Logistics",
            "engagement_model": "fixed",
            "catalogue": CATALOGUE,
        },
        "callback_url": CALLBACK,
        "idempotency_key": key,
        "priority": "interactive",
    }
    return job | overrides


def _script(env: Env) -> None:
    env.outside.files[FILE_URL] = RFP
    env.llm.on(
        "extraction",
        lambda call: _form(
            features=[{"key": "tracking", "quote": "real-time GPS tracking", "page": 1, "confidence": 0.9}]
        ),
    )
    env.llm.on("review", approve_all)


def _verified(raw: bytes, header: str) -> bool:
    """The backend's check (its core/signing.verify): HMAC-SHA256 of "<t>.<raw body>"."""
    fields = dict(part.split("=", 1) for part in header.split(","))
    secret = get_settings().callback_secret.get_secret_value().encode()
    expected = hmac.new(secret, f"{fields['t']}.".encode() + raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(fields["v1"], expected)


async def test_a_job_is_accepted_then_answered_with_signed_progress_and_result(env: Env) -> None:
    _script(env)
    r = await env.client.post("/v1/jobs", json=_submission(), headers=env.auth())
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    await env.jobs.drain()

    events = env.outside.events()
    assert [e["status"] for e in events] == ["progress", "progress", "progress", "succeeded"]
    assert all(_verified(c.content, c.headers["X-RE-Signature"]) for c in env.outside.callbacks)
    assert {str(c.url) for c in env.outside.callbacks} == {CALLBACK}
    assert len({e["event_id"] for e in events}) == 4
    done = events[-1]
    assert (done["job_id"], done["type"], done["idempotency_key"]) == (job_id, "rfp_analysis", "draft-1")
    assert done["workspace_id"] == str(env.workspace_id) and done["error"] is None
    assert done["result"]["core"][0]["key"] == "tracking" and done["result"]["pages"] == 1
    assert done["usage"]["calls"] == 2
    assert events[0]["result"] == {"message": "Reading the document", "done": 0, "total": 3}

    status = (await env.client.get(f"/v1/jobs/{job_id}", headers=env.auth())).json()
    assert status == {"job_id": job_id, "type": "rfp_analysis", "status": "succeeded", "error": None}


async def test_the_same_idempotency_key_never_starts_a_second_job(env: Env) -> None:
    _script(env)
    first = (await env.client.post("/v1/jobs", json=_submission(), headers=env.auth())).json()
    again = (await env.client.post("/v1/jobs", json=_submission(), headers=env.auth())).json()
    await env.jobs.drain()
    assert first == again and env.llm.count("extraction") == 1
    other = (await env.client.post("/v1/jobs", json=_submission("draft-2"), headers=env.auth())).json()
    assert other["job_id"] != first["job_id"]
    # Another workspace's key is its own: same key, separate job, and it cannot read the first one.
    elsewhere = uuid.uuid4()
    theirs = (await env.client.post("/v1/jobs", json=_submission(), headers=env.auth(elsewhere))).json()
    assert theirs["job_id"] != first["job_id"]
    r = await env.client.get(f"/v1/jobs/{first['job_id']}", headers=env.auth(elsewhere))
    assert r.status_code == 404


async def test_unknown_type_bad_input_and_foreign_links_are_refused(env: Env) -> None:
    r = await env.client.post("/v1/jobs", json=_submission(type="bulk_draft"), headers=env.auth())
    assert r.status_code == 422 and r.json()["rule"] == "job_type"
    bad = _submission()
    bad["input"] = {**bad["input"], "catalogue": []}
    r = await env.client.post("/v1/jobs", json=bad, headers=env.auth())
    assert r.status_code == 422 and r.json()["rule"] == "job_input"
    foreign = _submission()
    foreign["input"] = {**foreign["input"], "file_url": "http://169.254.169.254/latest/meta-data"}
    r = await env.client.post("/v1/jobs", json=foreign, headers=env.auth())
    assert r.status_code == 422 and r.json()["rule"] == "file_url"
    r = await env.client.post("/v1/jobs", json=_submission())
    assert r.status_code == 401
    assert env.outside.callbacks == []


async def test_a_job_that_cannot_finish_fails_with_plain_words(env: Env) -> None:
    # The link expired: storage answers 403.
    r = await env.client.post("/v1/jobs", json=_submission(), headers=env.auth())
    await env.jobs.drain()
    done = env.outside.events()[-1]
    assert done["status"] == "failed" and done["result"] is None
    assert "could not download the document" in done["error"] and len(done["error"]) <= 2000
    status = (await env.client.get(f"/v1/jobs/{r.json()['job_id']}", headers=env.auth())).json()
    assert status["status"] == "failed" and status["error"] == done["error"]


async def test_an_unreadable_document_fails_the_job_with_its_reason(env: Env) -> None:
    _script(env)
    env.outside.files[FILE_URL] = b"   "
    await env.client.post("/v1/jobs", json=_submission(), headers=env.auth())
    await env.jobs.drain()
    done = env.outside.events()[-1]
    assert done["status"] == "failed" and "has no readable text" in done["error"]


async def test_a_failed_delivery_is_retried_with_the_same_event_id(env: Env) -> None:
    _script(env)
    # Three progress events (one try each), then the result: 503 twice, then accepted.
    env.outside.callback_statuses = [200, 200, 200, 503, 503, 200]
    await env.client.post("/v1/jobs", json=_submission(), headers=env.auth())
    await env.jobs.drain()
    finals = [e for e in env.outside.events() if e["status"] == "succeeded"]
    assert len(finals) == 3 and len({e["event_id"] for e in finals}) == 1
