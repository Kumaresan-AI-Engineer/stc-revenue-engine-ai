"""Storage interfaces. Every read and write names the workspace; nothing crosses workspaces."""

import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from revenue_engine_ai.documents.chunking import Passage


@dataclass(frozen=True)
class EntryMeta:
    entry_id: str
    client: str
    title: str
    industry: str
    outcome: str  # won | lost | pending
    year: int
    kind: str = "proposal"  # proposal | casestudy | other


@dataclass(frozen=True)
class EntryFilter:
    """Which indexed entries a search looks at. An empty field does not narrow."""

    kinds: tuple[str, ...] = ()
    outcomes: tuple[str, ...] = ()
    industries: tuple[str, ...] = ()  # compared without case
    year_from: int | None = None
    year_to: int | None = None


@dataclass(frozen=True)
class StoredPassage:
    id: str
    entry_id: str
    section: str
    page: int
    text: str


@dataclass(frozen=True)
class Hit:
    passage: StoredPassage
    similarity: float


class LibraryIndex(Protocol):
    async def replace_entry(
        self,
        workspace_id: uuid.UUID,
        meta: EntryMeta,
        content_hash: str,
        passages: list[Passage],
        vectors: list[list[float]],
    ) -> None: ...

    async def delete_entry(self, workspace_id: uuid.UUID, entry_id: str) -> bool: ...

    async def entries(self, workspace_id: uuid.UUID, entry_ids: list[str] | None) -> dict[str, EntryMeta]: ...

    async def find_entries(self, workspace_id: uuid.UUID, scope: EntryFilter) -> dict[str, EntryMeta]: ...

    async def content_hashes(self, workspace_id: uuid.UUID, entry_ids: list[str]) -> dict[str, str]: ...

    async def search(
        self,
        workspace_id: uuid.UUID,
        vector: list[float],
        *,
        entry_ids: list[str] | None,
        sections: list[str] | None,
        limit: int,
    ) -> list[Hit]:
        """Most similar first; equal similarity ordered by passage id so results never reshuffle."""
        ...

    async def passage(self, workspace_id: uuid.UUID, passage_id: str) -> StoredPassage | None: ...


class EmbeddingCache(Protocol):
    async def get_many(self, keys: list[str]) -> dict[str, list[float]]: ...

    async def put_many(self, items: dict[str, list[float]]) -> None: ...


class ResultCache(Protocol):
    async def get(self, workspace_id: uuid.UUID, key: str) -> dict[str, Any] | None: ...

    async def put(self, workspace_id: uuid.UUID, key: str, kind: str, payload: dict[str, Any]) -> None: ...
