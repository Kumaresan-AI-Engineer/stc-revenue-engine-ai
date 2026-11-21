"""Test doubles for the provider and storage interfaces. No network, no database."""

import hashlib
import json
import math
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from pydantic import BaseModel

from revenue_engine_ai.core.config import EMBEDDING_DIMENSIONS
from revenue_engine_ai.documents.chunking import Passage
from revenue_engine_ai.llm.ports import AgentCall, Completion, Tool, Usage
from revenue_engine_ai.store.jobs import QUEUED, RUNNING, Job
from revenue_engine_ai.store.ports import EntryMeta, Hit, StoredPassage

Handler = Callable[[AgentCall[Any]], BaseModel | dict[str, Any]]
ToolScript = Callable[[AgentCall[Any], dict[str, Tool]], Awaitable[dict[str, Any]]]


class FakeLLM:
    """Answers each agent with a scripted handler and records every call."""

    model = "fake-model"

    def __init__(self) -> None:
        self.handlers: dict[str, Handler] = {}
        self.tool_scripts: dict[str, ToolScript] = {}
        self.calls: list[AgentCall[Any]] = []

    def on(self, agent: str, handler: Handler) -> None:
        self.handlers[agent] = handler

    def on_tools(self, agent: str, script: ToolScript) -> None:
        self.tool_scripts[agent] = script

    def count(self, agent: str) -> int:
        return sum(1 for c in self.calls if c.agent == agent)

    async def structured[T: BaseModel](self, call: AgentCall[T]) -> Completion[T]:
        self.calls.append(call)
        value = self.handlers[call.agent](call)
        data = value.model_dump() if isinstance(value, BaseModel) else value
        return Completion(call.output.model_validate(data), Usage(100, 20, 1, 0))

    async def with_tools[T: BaseModel](
        self, call: AgentCall[T], tools: list[Tool], max_tool_calls: int
    ) -> Completion[T]:
        self.calls.append(call)
        data = await self.tool_scripts[call.agent](call, {t.name: t for t in tools})
        return Completion(call.output.model_validate(data), Usage(200, 40, 2, 1))


def approve_all(call: AgentCall[Any]) -> dict[str, Any]:
    """Review handler that supports every item it is given."""
    items = json.loads(call.input)
    return {"verdicts": [{"id": i["id"], "supported": True, "reason": "ok"} for i in items]}


class FakeEmbedder:
    """Bag-of-words hashed into a unit vector: texts sharing words are similar, and it is deterministic."""

    model = "BAAI/bge-small-en-v1.5"

    def __init__(self) -> None:
        self.embedded: list[str] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [_vector(t) for t in texts]


def _vector(text: str) -> list[float]:
    v = [0.0] * EMBEDDING_DIMENSIONS
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBEDDING_DIMENSIONS] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class InMemoryIndex:
    def __init__(self) -> None:
        self.entries_: dict[tuple[uuid.UUID, str], tuple[EntryMeta, str]] = {}
        self.passages: dict[tuple[uuid.UUID, str], tuple[StoredPassage, list[float]]] = {}

    async def replace_entry(
        self,
        workspace_id: uuid.UUID,
        meta: EntryMeta,
        content_hash: str,
        passages: list[Passage],
        vectors: list[list[float]],
    ) -> None:
        await self.delete_entry(workspace_id, meta.entry_id)
        self.entries_[(workspace_id, meta.entry_id)] = (meta, content_hash)
        for p, v in zip(passages, vectors, strict=True):
            self.passages[(workspace_id, p.id)] = (
                StoredPassage(p.id, meta.entry_id, p.section, p.page, p.text),
                v,
            )

    async def delete_entry(self, workspace_id: uuid.UUID, entry_id: str) -> bool:
        existed = self.entries_.pop((workspace_id, entry_id), None) is not None
        for key in [
            k for k, (p, _) in self.passages.items() if k[0] == workspace_id and p.entry_id == entry_id
        ]:
            del self.passages[key]
        return existed

    async def entries(self, workspace_id: uuid.UUID, entry_ids: list[str] | None) -> dict[str, EntryMeta]:
        return {
            e: meta
            for (ws, e), (meta, _) in self.entries_.items()
            if ws == workspace_id and (entry_ids is None or e in entry_ids)
        }

    async def content_hashes(self, workspace_id: uuid.UUID, entry_ids: list[str]) -> dict[str, str]:
        return {e: h for (ws, e), (_, h) in self.entries_.items() if ws == workspace_id and e in entry_ids}

    async def search(
        self,
        workspace_id: uuid.UUID,
        vector: list[float],
        *,
        entry_ids: list[str] | None,
        sections: list[str] | None,
        limit: int,
    ) -> list[Hit]:
        hits = [
            Hit(p, round(_cosine(vector, v), 4))
            for (ws, _), (p, v) in self.passages.items()
            if ws == workspace_id
            and (entry_ids is None or p.entry_id in entry_ids)
            and (sections is None or p.section in sections)
        ]
        return sorted(hits, key=lambda h: (-h.similarity, h.passage.id))[:limit]

    async def passage(self, workspace_id: uuid.UUID, passage_id: str) -> StoredPassage | None:
        found = self.passages.get((workspace_id, passage_id))
        return found[0] if found else None


class InMemoryEmbeddingCache:
    def __init__(self) -> None:
        self.rows: dict[str, list[float]] = {}

    async def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        return {k: self.rows[k] for k in keys if k in self.rows}

    async def put_many(self, items: dict[str, list[float]]) -> None:
        self.rows.update(items)


class InMemoryResultCache:
    def __init__(self) -> None:
        self.rows: dict[tuple[uuid.UUID, str], dict[str, Any]] = {}

    async def get(self, workspace_id: uuid.UUID, key: str) -> dict[str, Any] | None:
        return self.rows.get((workspace_id, key))

    async def put(self, workspace_id: uuid.UUID, key: str, kind: str, payload: dict[str, Any]) -> None:
        self.rows.setdefault((workspace_id, key), payload)


class InMemoryJobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}

    async def create_or_get(
        self, workspace_id: uuid.UUID, job_type: str, idempotency_key: str
    ) -> tuple[Job, bool]:
        for job in self.jobs.values():
            if job.workspace_id == workspace_id and job.idempotency_key == idempotency_key:
                return job, False
        job = Job(str(uuid.uuid4()), workspace_id, job_type, idempotency_key, QUEUED, None)
        self.jobs[job.id] = job
        return job, True

    async def get(self, workspace_id: uuid.UUID, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        return job if job and job.workspace_id == workspace_id else None

    async def start(self, job_id: str) -> None:
        self.jobs[job_id] = replace(self.jobs[job_id], status=RUNNING)

    async def finish(self, job_id: str, status: str, error: str | None = None) -> None:
        self.jobs[job_id] = replace(self.jobs[job_id], status=status, error=error)
