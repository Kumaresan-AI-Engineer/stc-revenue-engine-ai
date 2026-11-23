---
name: new-agent
description: Use when adding or changing an AI agent in stc-revenue-engine-ai - a new agent, a new job type, or a change to a prompt or output schema. Walks the agent shape (prompt, output schema, pure check, review, one retry), the service and route, the cache key and PROMPT_VERSION, the backend contract and the tests on fakes. Trigger words - new agent, add an agent, change the prompt, new job type, agent output.
---

# Add or change an agent (Revenue Engine AI service)

Paths are under `src/revenue_engine_ai/`.

## 1. The agent (`agents/<name>.py`)

- `PROMPT_VERSION = "<name>-v1"`; bump it on any change to the prompt or the output schema (it is in every cache
  key, so a stale cached answer is never served).
- A Pydantic output model with limits (lengths, list sizes), and a pure `check` that returns the problems: no I/O,
  no LLM. Hard rules live here: no prices, quotes must be in the source, only known keys.
- Call the model through `llm/ports.py` (`LLM`), never through `openai` directly.

## 2. The service (`services/<name>.py`)

agent → `check` → `agents/review.py` → one retry with the problems as notes → drop or `needs_person`. Cache the
result in `result_cache`, keyed by everything that shaped it. A new asynchronous job is an entry in
`services/jobs.py` `HANDLERS`.

## 3. The route (`api/router.py`, `api/schemas.py`)

HTTP only, behind the service-JWT dependency, workspace from the token. Then update the backend's
`integrations/ai_client/contract.py` (sibling repo) to the same shape, and its fake.

## 4. Tests

- `tests/unit`: the `check` rules, with no LLM.
- `tests/api`: the route through scripted `FakeLLM` answers (`tests/fakes.py`), including a failed check that is
  retried and one that ends as `needs_person`.

## 5. Gates

`ruff check .`, `ruff format --check .`, `mypy src`, `pytest`. Update `README.md` (agents and endpoints) and
`CLAUDE.md` if the shape or a rule changed.
