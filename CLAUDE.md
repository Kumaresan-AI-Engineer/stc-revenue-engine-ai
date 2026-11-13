# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Revenue Engine — AI service

AI side of the proposal builder; the backend is `stc-revenue-engine-api` (sibling repo) and its `CLAUDE.md`
engineering standards apply here too. `README.md` describes the agents, endpoints and what is not done yet.

## Commands

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
docker compose up -d                                  # Postgres 16 + pgvector on 5433
.venv/Scripts/python -m alembic upgrade head
.venv/Scripts/python -m ruff check . && .venv/Scripts/python -m ruff format --check .
.venv/Scripts/python -m mypy src
.venv/Scripts/python -m pytest                        # single: pytest tests/api/test_api.py::test_name
cd src && ../.venv/Scripts/uvicorn main:app --reload --port 8001
```

`tests/integration` needs the migrated database and is **skipped silently** without it. Everything else runs on
fakes (`tests/fakes.py`: scripted `FakeLLM`, word-hash `FakeEmbedder`, in-memory stores) wired through
`dependency_overrides` in `tests/conftest.py`; no test calls OpenAI.

## How it fits together

- **Layers:** `api/router.py` (HTTP only) → `services/` (one use-case per step: analyse, library, drafting,
  clarification) → `agents/` (prompt + output schema + pure `check`) and `store/` (pgvector). `rules/` and
  `documents/` are pure. Agents depend on `llm/ports.py` (`LLM`, `Embedder`), never on OpenAI directly;
  `llm/openai_provider.py` is the only file that imports `openai`.
- **Every agent result goes:** agent → code `check` → `agents/review.py` → one retry with the problems as notes →
  drop / `needs_person`. Keep that shape when adding an agent.
- **Determinism is a feature:** changing a prompt means bumping its `PROMPT_VERSION` (part of every cache key);
  changing the embedding model needs a new migration (vector size) and a re-index; search order must stay
  stable (ties by id). Results are cached in `result_cache` keyed by everything that shaped them.
- **Tenancy:** the workspace comes only from the service JWT (`core/security.py`); every store method takes it.
  The drafting tools can only reach the selected entries.
- **Hard rules:** no prices (amounts are redacted at indexing, pricing questions are skipped, amounts in output
  fail the check); estimates stay in the backend.
- **OpenAI SDK is 3.x:** `responses.parse(text_format=...)`, `store=False` with encrypted reasoning carried
  between tool turns. Check the installed SDK before changing call shapes.
