"""Tables of the AI service's own database (created by Alembic migration 0001). Passage vectors are in
Qdrant (store/vector_db.py)."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import REAL, DateTime, Integer, String, Uuid, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from core.db import Base


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
