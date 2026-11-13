# Revenue Engine — AI service

The AI side of the proposal builder. The backend (`stc-revenue-engine-api`) stays the system of record and
the policy engine: **this service never changes business state, never sees another workspace's data and never
writes a price.** It answers the backend with checked, structured results.

OpenAI for the agents (Responses API, structured outputs) · free local embeddings (fastembed,
`BAAI/bge-small-en-v1.5`, no key, documents never leave the service) · Python 3.12 · FastAPI · PostgreSQL 16 +
pgvector.

## The agents

Four agents with one job each, run in a fixed order by code (no AI decides what runs next).

| Step | Agent | What it does | Checked by |
|---|---|---|---|
| 1 → 2 Analyse | **Requirement Extraction** | Reads the RFP: catalogue features (an enum of the catalogue sent), custom requirements, the RFP's questions, required documents, facts, unclear points. Every item has a quote + page. | code: quote is on the page · Review Agent: features and custom requirements |
| 3 Matches | *Semantic search* (no LLM) | Past proposals covering each requirement (incl. custom ones), and similar projects | fixed threshold and order |
| 4 Responses | **Response Drafting** (tools) | Searches only the selected past proposals (`search_library`, `get_passage`, ≤ 5 calls) and drafts an answer whose every claim quotes a passage it read | code: passage was read, quote is in it, no amounts · Review Agent |
| 5 Queries | **Clarification** | Questions for the client: why we ask, a sample answer, the assumption we price on | code: topic id, no amounts, no repeats · Review Agent |
| all | **Review** | Independent second look: is the statement fully supported by its evidence and the rule? Missing verdict = fail | — |
| 6 Estimate | — | Never AI. Stays in the backend's `estimate.py`. | — |

Background: the **Library indexer** splits each past proposal into passages, removes amounts, embeds them and
stores them per workspace.

Every agent call follows the same loop: **agent → code check → review → one retry with the problems as notes →
anything still failing is dropped or marked `needs_person`**. Nothing is guessed.

## Predictable output

- **Fixed forms:** structured outputs (strict JSON schema); feature keys are an enum of the catalogue.
- **Grounding:** every quote must appear in its source (normalised case, spacing, quotes, dashes).
- **Pinned versions:** chat model snapshot (`gpt-5.5-2026-04-23`), embedding model, `PROMPT_VERSION` per agent;
  all returned in `meta` for the backend's trace.
- **Stored results:** results are cached by a hash of everything that shaped them (input, catalogue, model,
  prompt versions, library content), and query vectors are cached by text, so the same input returns the same
  answer. Search is an exact cosine scan with ties broken by id.
- **Free embeddings:** fastembed runs `BAAI/bge-small-en-v1.5` on CPU (384 dimensions); the model is downloaded once
  (baked into the Docker image). The match threshold (0.75) is a starting point to tune on the evaluation set.
- **No prices:** amounts are removed from indexed passages; pricing questions are never drafted; any amount in an
  answer or client question fails the check.

## Endpoints (`/v1`, service JWT required)

| Endpoint | Purpose |
|---|---|
| `POST /rfp/analyse` | multipart: `file` (PDF, DOCX, XLSX, TXT) + `request` = `{"catalogue": [...]}` |
| `PUT /library/entries/{entry_id}` | multipart: `meta` JSON + optional `rfp`, `proposal` files; unchanged content is not re-embedded |
| `DELETE /library/entries/{entry_id}` | remove from the index |
| `POST /library/similar` | requirements (+ RFP summary) → matching past proposals with the passage that proves it |
| `POST /proposals/draft-answers` | questions + selected entry ids → `drafted` / `needs_person` / `skipped_pricing` / `skipped` |
| `POST /proposals/client-queries` | requirements (covered or not) + unclear points → client questions |
| `GET /healthz` | liveness |

Errors are RFC 9457 problem+json, like the backend (`rule` on 422, 401 without a valid token, 502 when OpenAI
fails). Interactive docs at `/docs`.

**Auth:** the backend signs a short-lived HS256 JWT per call with `RE_AI_SERVICE_TOKEN_SECRET`, audience
`revenue-engine-ai`, claims `workspace_id`, `actor_id`, `exp` (+ `can_see_restricted`, `request_id`). The
workspace comes only from the token.

## Run

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
docker compose up -d                                   # Postgres 16 + pgvector on port 5433
cp .env.example .env                                   # set RE_AI_OPENAI_API_KEY and RE_AI_SERVICE_TOKEN_SECRET
.venv/Scripts/python -m alembic upgrade head
cd src && ../.venv/Scripts/uvicorn main:app --reload --port 8001
```

Checks (the CI gates): `ruff check .`, `ruff format --check .`, `mypy src`, `alembic check`, `pytest`.

## Backend integration

The backend (`stc-revenue-engine-api`, `integrations/ai_client`) calls every endpoint above from the proposal
builder: Analyse → `/rfp/analyse`; Find comparables → indexes un-indexed library entries, then
`/library/similar` for custom requirements; Responses → `/proposals/draft-answers`; Client queries →
`/proposals/client-queries`. Set the same `RE_AI_SERVICE_TOKEN_SECRET` on both, `RE_AI_SERVICE_URL` on the
backend (default `http://localhost:8001`) and `RE_MOCK_AI=false` there to use this service locally.

## Not done yet

- **Evaluation set:** 10–20 past RFPs with known answers to tune the match threshold and gate prompt changes.
- Asynchronous jobs with signed callbacks (ARCHITECTURE §10) for very long documents; calls are synchronous now.
- OCR for scanned PDFs; RS256 service tokens against the backend's JWKS; Batch API for bulk library imports.
