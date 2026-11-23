"""Asynchronous jobs (POST /v1/jobs): accepted at once, run in this process, answered on a signed callback.

submit: validate the type and its input (code) → one job per (workspace, idempotency key) → run in the
background, at most `max_parallel_jobs` at once → progress events while it runs → one `succeeded` or `failed`
event. Every event is signed (X-RE-Signature) and a failed delivery is retried with the same event_id.

A job type is a handler in HANDLERS: its input model, a check of that input, and what it runs.
"""

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from revenue_engine_ai.api.schemas import JobSubmission, RfpAnalysisInput, TemplateExtractInput
from revenue_engine_ai.core.config import Settings, get_settings
from revenue_engine_ai.core.errors import DomainError, RuleViolation
from revenue_engine_ai.core.security import ServiceContext
from revenue_engine_ai.core.signing import SIGNATURE_HEADER, sign
from revenue_engine_ai.documents import template_reader
from revenue_engine_ai.documents.template_reader import TemplateKind
from revenue_engine_ai.llm.ports import LLM
from revenue_engine_ai.llm.providers import get_llm
from revenue_engine_ai.services import analyse as analyse_service
from revenue_engine_ai.services import templates
from revenue_engine_ai.store.jobs import FAILED, SUCCEEDED, Job, JobStore, pg_job_store
from revenue_engine_ai.store.ports import ResultCache
from revenue_engine_ai.store.postgres import pg_standalone_result_cache

log = logging.getLogger(__name__)

ERROR_MAX = 2000
SOMETHING_WENT_WRONG = "Something went wrong while running this job on our side. Please try again."
NOT_WORD = "The file could not be read as a Word document. Upload it again from Word as .docx."
DOWNLOAD_FAILED = "We could not download the document (the link may have expired). Please try again."
# Pauses between delivery attempts of a final event: three tries in all.
RETRY_DELAYS = (1.0, 4.0)


@dataclass(frozen=True)
class JobDeps:
    settings: Settings
    http: httpx.AsyncClient  # downloads and callbacks
    llm: Callable[[], LLM]  # resolved when a job runs: a missing key fails that job, not the submission
    cache: ResultCache


class JobRun:
    """What a running job can do: download its file and report progress."""

    def __init__(self, runner: "JobRunner", ctx: ServiceContext, job: Job, callback_url: str) -> None:
        self.ctx = ctx
        self.deps = runner.deps
        self._runner = runner
        self._job = job
        self._callback_url = callback_url

    async def progress(self, message: str, done: int, total: int) -> None:
        # Best effort, one try: a lost progress event is replaced by the next one or by the result.
        result = {"message": message, "done": done, "total": total}
        await self._runner.deliver(self._job, self._callback_url, "progress", result=result, tries=1)

    async def download(self, url: str) -> bytes:
        limit = self.deps.settings.max_upload_mb * 1024 * 1024
        try:
            async with self.deps.http.stream("GET", url, follow_redirects=False) as response:
                if response.status_code != 200:
                    raise RuleViolation("file_url", DOWNLOAD_FAILED)
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > limit:
                        raise RuleViolation(
                            "file_size", f"Files can be at most {self.deps.settings.max_upload_mb} MB."
                        )
                return bytes(data)
        except httpx.HTTPError as exc:
            raise RuleViolation("file_url", DOWNLOAD_FAILED) from exc


@dataclass(frozen=True)
class JobHandler:
    input_model: type[BaseModel]
    check: Callable[[Any, Settings], None]
    run: Callable[[JobRun, Any], Awaitable[tuple[dict[str, Any], dict[str, Any]]]]  # (result, usage)


def _allowed_file_url(data: RfpAnalysisInput | TemplateExtractInput, settings: Settings) -> None:
    if not any(data.file_url.startswith(prefix) for prefix in settings.file_url_prefixes):
        raise RuleViolation("file_url", "The file link does not point at the backend's file storage.")


async def _rfp_analysis(run: JobRun, data: RfpAnalysisInput) -> tuple[dict[str, Any], dict[str, Any]]:
    content = await run.download(data.file_url)
    done = await analyse_service.analyse(
        run.ctx,
        data.filename,
        content,
        data.catalogue,
        run.deps.llm(),
        run.deps.cache,
        run.deps.settings,
        progress=run.progress,
    )
    return done.model_dump(), done.meta.usage.model_dump()


async def _template_extract(run: JobRun, data: TemplateExtractInput) -> tuple[dict[str, Any], dict[str, Any]]:
    content = await run.download(data.file_url)
    try:
        spec, usage = await templates.extract(
            content, TemplateKind(data.kind), run.deps.llm, run.deps.settings
        )
    except template_reader.UnreadableDocument as exc:
        raise RuleViolation("file", NOT_WORD) from exc
    return spec, asdict(usage)


HANDLERS: dict[str, JobHandler] = {
    "rfp_analysis": JobHandler(RfpAnalysisInput, _allowed_file_url, _rfp_analysis),
    "template_extract": JobHandler(TemplateExtractInput, _allowed_file_url, _template_extract),
}


def _plain(error: str) -> str:
    return " ".join(error.split())[:ERROR_MAX]


class JobRunner:
    def __init__(
        self, store: JobStore, deps: JobDeps, retry_delays: tuple[float, ...] = RETRY_DELAYS
    ) -> None:
        self.store = store
        self.deps = deps
        self._retry_delays = retry_delays
        self._slots = asyncio.Semaphore(deps.settings.max_parallel_jobs)
        self._tasks: set[asyncio.Task[None]] = set()

    async def submit(self, ctx: ServiceContext, submission: JobSubmission) -> str:
        handler = HANDLERS.get(submission.type)
        if handler is None:
            raise RuleViolation("job_type", f"Unknown job type '{submission.type}'.")
        try:
            data = handler.input_model.model_validate(submission.input)
        except ValidationError as exc:
            fields = ", ".join(".".join(str(p) for p in e["loc"]) for e in exc.errors())
            raise RuleViolation("job_input", f"The job input is not valid ({fields}).") from exc
        handler.check(data, self.deps.settings)
        job, created = await self.store.create_or_get(
            ctx.workspace_id, submission.type, submission.idempotency_key
        )
        if created:
            task = asyncio.create_task(self._run(ctx, job, handler, data, submission.callback_url))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        log.info(
            "job_submitted",
            extra={"fields": {"job_id": job.id, "type": job.type, "created": created, "status": job.status}},
        )
        return job.id

    async def drain(self) -> None:
        """Wait for every running job (tests, shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def _run(
        self, ctx: ServiceContext, job: Job, handler: JobHandler, data: Any, callback_url: str
    ) -> None:
        async with self._slots:
            await self.store.start(job.id)
            run = JobRun(self, ctx, job, callback_url)
            try:
                result, usage = await handler.run(run, data)
            except DomainError as exc:
                error = _plain(exc.detail)
            except Exception:
                log.exception("job_crashed", extra={"fields": {"job_id": job.id, "type": job.type}})
                error = SOMETHING_WENT_WRONG
            else:
                await self.store.finish(job.id, SUCCEEDED)
                await self.deliver(job, callback_url, SUCCEEDED, result=result, usage=usage)
                return
            await self.store.finish(job.id, FAILED, error)
            await self.deliver(job, callback_url, FAILED, error=error)

    async def deliver(
        self,
        job: Job,
        url: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        usage: dict[str, Any] | None = None,
        tries: int | None = None,
    ) -> None:
        """POST one signed event to the callback; a redelivery keeps its event_id (the backend dedupes)."""
        event = {
            "event_id": str(uuid.uuid4()),
            "job_id": job.id,
            "type": job.type,
            "workspace_id": str(job.workspace_id),
            "idempotency_key": job.idempotency_key,
            "status": status,
            "result": result,
            "error": error,
            "trace_id": job.id,
            "usage": usage,
        }
        body = json.dumps(event, ensure_ascii=False, default=str).encode()
        secret = self.deps.settings.callback_secret.get_secret_value()
        attempts = tries or len(self._retry_delays) + 1
        where = {"job_id": job.id, "event_id": event["event_id"], "status": status}
        for attempt in range(attempts):
            headers = {"Content-Type": "application/json", SIGNATURE_HEADER: sign(secret, body)}
            try:
                response = await self.deps.http.post(url, content=body, headers=headers)
                if response.status_code < 500:
                    if response.status_code >= 400:
                        log.warning(
                            "callback_refused", extra={"fields": {**where, "http": response.status_code}}
                        )
                    return
            except httpx.HTTPError as exc:
                log.info("callback_unreachable", extra={"fields": {**where, "error": type(exc).__name__}})
            if attempt + 1 < attempts:
                await asyncio.sleep(self._retry_delays[min(attempt, len(self._retry_delays) - 1)])
        log.warning("callback_not_delivered", extra={"fields": where})


@lru_cache
def get_job_runner() -> JobRunner:
    """The process's runner (a FastAPI dependency, so tests replace it)."""
    settings = get_settings()
    deps = JobDeps(settings, httpx.AsyncClient(timeout=30), get_llm, pg_standalone_result_cache())
    return JobRunner(pg_job_store(), deps)
