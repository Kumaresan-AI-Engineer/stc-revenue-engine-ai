# Revenue Engine — AI service

The AI side of the proposal builder. The backend (`stc-revenue-engine-api`) stays the system of record and
the policy engine: **this service never changes business state, never sees another workspace's data and never
writes a price.** It answers the backend with checked, structured results.

OpenAI for the agents (Responses API, structured outputs; or an OpenAI-compatible gateway such as Yantra through
Chat Completions: `RE_AI_OPENAI_BASE_URL`, `RE_AI_LLM_API=chat_completions`, answers validated and corrected once) · free local embeddings (fastembed,
`BAAI/bge-small-en-v1.5`, no key, documents never leave the service) · Python 3.12 · FastAPI · Qdrant (vector
database, embedded by default: a local folder, no server) · PostgreSQL 16 (entry index and caches, no extensions).

## The agents

Four agents with one job each, run in a fixed order by code (no AI decides what runs next).

| Step | Agent | What it does | Checked by |
|---|---|---|---|
| 1 → 2 Analyse | **Requirement Extraction** (job `rfp_analysis`) | Reads the RFP (scanned PDF pages by OCR): catalogue features with a confidence (`core`; `standard` by the catalogue's `commodity`), requirements no catalogue entry covers (`unmatched`), the RFP's questions, required documents, due/start date, budget hint, unclear points. Every item has its quote (`evidence`) + page. | code: quote is on the page (a misquote is asked again once) · Review Agent: features and custom requirements (rejected ones are dropped, not re-extracted) |
| 3 Matches | *Semantic search* (no LLM) | Past proposals covering each requirement (incl. custom ones), and similar projects | fixed threshold and order |
| 4 Responses | **Response Drafting** (tools) | Searches only the selected past proposals, never lost ones, won first (`search_library`, `get_passage`, ≤ 5 calls) and drafts an answer whose every claim quotes a passage it read; returns `citations`, `confidence` (0.9 first try, 0.7 after a correction) and `no_confident_source` | code: passage was read, quote is in it, no amounts · Review Agent |
| 5 Queries | **Clarification** | Questions for the client: why we ask, a sample answer, the assumption we price on; at most `limit` (Suggest with AI: 3) | code: topic id, no amounts, no (near-)repeat of an existing question or assumption · Review Agent |
| 8 Submit | **Final Proposal Check** | Reads the user's own final proposal (1–2 files): price (only with `can_see_restricted`), weeks, core features delivered, answers matching the approved responses, attachments referenced | code: every fact's quote is in the file, a figure's number is in its quote; the five findings are written by code |
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
| `POST /proposals/client-queries` | requirements (covered or not) + unclear points + existing questions and assumptions + `limit` → client questions |
| `POST /proposals/final-check` | multipart: `files` (1–2) + `request` = `{"draft": {price, weeks, core_features, responses, attachments, ...}}` → `findings` for Scope, Cost, Timeline, Responses, Attachments (in that order) |
| `POST /jobs` | `{type, input, callback_url, idempotency_key, priority}` → `202 {job_id}`; type `rfp_analysis` |
| `GET /jobs/{job_id}` | the job's status (`queued`, `running`, `succeeded`, `failed`) and error, in the token's workspace |
| `GET /healthz` | liveness |

Errors are RFC 9457 problem+json, like the backend (`rule` on 422, 401 without a valid token, 502 when OpenAI
fails). Interactive docs at `/docs`.

**Auth:** the backend signs a short-lived HS256 JWT per call with `RE_AI_SERVICE_TOKEN_SECRET`, audience
`revenue-engine-ai`, issuer `revenue-engine-api` (both checked), claims `workspace_id`, `actor_id`, `exp`
(+ `can_see_restricted`, `request_id`, `role`, `purpose`). The workspace comes only from the token.

## Jobs (`POST /v1/jobs`)

A job is accepted at once (`202 {job_id}`) and runs in this process (at most `RE_AI_MAX_PARALLEL_JOBS` at a
time). The same `idempotency_key` in a workspace returns the first job and never starts a second one (table
`job`, migration `0002`). An unknown type is `422` with rule `job_type`, invalid input `job_input`, and a
`file_url` outside `RE_AI_FILE_URL_PREFIXES` (the backend's object storage) `file_url`.

While it runs, the job POSTs events to `callback_url`: `progress` (`result = {message, done, total}`, one try),
then one `succeeded` (`result` = the job's result, with `usage`) or `failed` (`error` in plain words, at most 2000
characters). Each event is `{event_id, job_id, type, workspace_id, idempotency_key, status, result, error,
trace_id, usage}`, signed with `RE_AI_CALLBACK_SECRET` in `X-RE-Signature: t=<unix>,v1=<hex HMAC-SHA256 of
"<t>.<raw body>">`. A final event whose delivery fails (network error or 5xx) is retried, three tries in all,
with the same `event_id`. A job lost by a restart is not resumed: the backend submits it again with a new key.

**`rfp_analysis`** input: `{file_url, filename, mode, client, industry, engagement_model, catalogue: [{key,
label, commodity, kw}]}`. Result (also what `POST /rfp/analyse` returns, there with `meta`): `summary`, `core:
[{key, confidence, evidence, page}]`, `standard: [key]`, `standard_evidence`, `unmatched: [{label, evidence,
page}]`, `due_date`, `start_date`, `budget_hint`, `client`, `questions: [{text, page, evidence}]`,
`required_documents: [name]`, `documents: [{name, evidence, page}]`, `unclear_points`, `rejected`, `ocr`, `pages`,
`meta`.

**OCR:** PDF pages without a text layer are rendered with pdfium and read with RapidOCR (ONNX on CPU, local), at
most `RE_AI_OCR_MAX_PAGES` (100) per document; the result says `ocr: true`. A file with no readable text at all
fails with "has no readable text".

## Run

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
docker compose up -d                                   # Postgres 16 on port 5433 (Qdrant is embedded)
cp .env.example .env                                   # set RE_AI_OPENAI_API_KEY and RE_AI_SERVICE_TOKEN_SECRET
.venv/Scripts/python -m alembic upgrade head
cd src && ../.venv/Scripts/uvicorn main:app --reload --port 8001
```

Checks (the CI gates): `ruff check .`, `ruff format --check .`, `mypy src`, `alembic check`, `pytest`.

## Backend integration

The backend (`stc-revenue-engine-api`, `integrations/ai_client`) calls every endpoint above from the proposal
builder: Analyse → `/rfp/analyse`; Find comparables → indexes un-indexed library entries, then
`/library/similar` for custom requirements; Responses → `/proposals/draft-answers`; Suggest with AI →
`/proposals/client-queries` (`limit` 3); the uploaded final proposal → `/proposals/final-check`; the RFP is read
by an `rfp_analysis` job. Set the same `RE_AI_SERVICE_TOKEN_SECRET` and `RE_AI_CALLBACK_SECRET` on both,
`RE_AI_SERVICE_URL` on the backend and `RE_MOCK_AI=false` there to use this service locally.

## Not done yet

- **Evaluation set:** 10–20 past RFPs with known answers to tune the match threshold and gate prompt changes.
- Other job types of ARCHITECTURE §10 (`template_extract`, `bulk_draft`, `ingest_file`, ...): only `rfp_analysis`
  runs as a job; `priority` is accepted but not used yet; a job lost by a restart is not resumed.
- RS256 service tokens against the backend's JWKS; Batch API for bulk library imports.
