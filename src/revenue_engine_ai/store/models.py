"""Tables of the AI service's own database (Alembic migrations). Passage vectors are in Qdrant
(store/vector_db.py)."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import REAL, DateTime, Integer, String, Text, UniqueConstraint, Uuid, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from revenue_engine_ai.core.db import Base


class LibraryEntryIndex(Base):
    """One indexed past proposal: what was indexed and from which content."""

    __tablename__ = "library_entry_index"

    workspace_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    entry_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    client: Mapped[str] = mapped_column(String(200))
    title: Mapped[str] = mapped_column(String(300))
    industry: Mapped[str] = mapped_column(String(60))
    outcome: Mapped[str] = mapped_column(String(10))
    year: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64))
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EmbeddingCacheRow(Base):
    """Vectors by hash of (model, text): the same text always gets the very same vector."""

    __tablename__ = "embedding_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(ARRAY(REAL))


class ResultCacheRow(Base):
    """Validated agent results by hash of everything that shaped them (input, model, prompt version)."""

    __tablename__ = "result_cache"

    workspace_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class JobRow(Base):
    """An asynchronous job. One per (workspace, idempotency key): a repeated submission returns this one.
    Its input is not stored (it holds a signed file link); a job lost by a restart is submitted again by
    the backend with a new key."""

    __tablename__ = "job"
    __table_args__ = (UniqueConstraint("workspace_id", "idempotency_key"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[uuid.UUID] = mapped_column(Uuid, index=True)
    type: Mapped[str] = mapped_column(String(40))
    idempotency_key: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(16))
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
