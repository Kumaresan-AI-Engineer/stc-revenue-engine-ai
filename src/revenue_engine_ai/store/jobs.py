"""Job records. A job runs after its request has answered, so every method here is its own short
transaction; nothing waits on a request's session."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from revenue_engine_ai.core.db import sessionmaker
from revenue_engine_ai.store.models import JobRow

QUEUED, RUNNING, SUCCEEDED, FAILED = "queued", "running", "succeeded", "failed"


@dataclass(frozen=True)
class Job:
    id: str
    workspace_id: uuid.UUID
    type: str
    idempotency_key: str
    status: str
    error: str | None


class JobStore(Protocol):
    async def create_or_get(
        self, workspace_id: uuid.UUID, job_type: str, idempotency_key: str
    ) -> tuple[Job, bool]:
        """The job for this key, and whether this call created it (only then does it run)."""
        ...

    async def get(self, workspace_id: uuid.UUID, job_id: str) -> Job | None: ...

    async def finish(self, job_id: str, status: str, error: str | None = None) -> None: ...

    async def start(self, job_id: str) -> None: ...


def _job(row: JobRow) -> Job:
    return Job(row.id, row.workspace_id, row.type, row.idempotency_key, row.status, row.error)


class PgJobStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create_or_get(
        self, workspace_id: uuid.UUID, job_type: str, idempotency_key: str
    ) -> tuple[Job, bool]:
        job_id = str(uuid.uuid4())
        async with self._sessions() as s, s.begin():
            stmt = (
                insert(JobRow)
                .values(
                    id=job_id,
                    workspace_id=workspace_id,
                    type=job_type,
                    idempotency_key=idempotency_key,
                    status=QUEUED,
                )
                .on_conflict_do_nothing(index_elements=["workspace_id", "idempotency_key"])
                .returning(JobRow.id)
            )
            created = (await s.execute(stmt)).scalar_one_or_none() is not None
            row = await s.scalar(
                select(JobRow).where(
                    JobRow.workspace_id == workspace_id, JobRow.idempotency_key == idempotency_key
                )
            )
            assert row is not None  # inserted now or by the earlier submission
            return _job(row), created

    async def get(self, workspace_id: uuid.UUID, job_id: str) -> Job | None:
        async with self._sessions() as s:
            row = await s.scalar(
                select(JobRow).where(JobRow.id == job_id, JobRow.workspace_id == workspace_id)
            )
            return _job(row) if row else None

    async def start(self, job_id: str) -> None:
        async with self._sessions() as s, s.begin():
            await s.execute(update(JobRow).where(JobRow.id == job_id).values(status=RUNNING))

    async def finish(self, job_id: str, status: str, error: str | None = None) -> None:
        async with self._sessions() as s, s.begin():
            await s.execute(
                update(JobRow)
                .where(JobRow.id == job_id)
                .values(status=status, error=error, finished_at=datetime.now(UTC))
            )


def pg_job_store() -> PgJobStore:
    return PgJobStore(sessionmaker())
