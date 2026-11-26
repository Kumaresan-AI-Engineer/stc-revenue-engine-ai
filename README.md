# Revenue Engine — AI service

The AI side of the proposal builder. The backend (`stc-revenue-engine-api`) stays the system of record and
the policy engine: **this service never changes business state, never sees another workspace's data and never
writes a price.** It answers the backend with checked, structured results.

OpenAI for the agents (Responses API, structured outputs; or an OpenAI-compatible gateway such as Yantra through
Chat Completions: `RE_AI_OPENAI_BASE_URL`, `RE_AI_LLM_API=chat_completions`, answers validated and corrected once) · free local embeddings (fastembed,
`BAAI/bge-small-en-v1.5`, no key, documents never leave the service) · Python 3.12 · FastAPI · Qdrant (vector
database, embedded by default: a local folder, no server) · PostgreSQL 16 (entry index and caches, no extensions).

## The agents

Agents with one job each, run in a fixed order by code (no AI decides what runs next).

| Step | Agent | What it does | Checked by |
|---|---|---|---|
| 1 → 2 Analyse | **Requirement Extraction** (job `rfp_analysis`) | Reads the RFP (scanned PDF pages by OCR): catalogue features with a confidence (`core`; `standard` by the catalogue's `commodity`), requirements no catalogue entry covers (`unmatched`), the RFP's questions, required documents, due/start date, budget hint, unclear points. Every item has its quote (`evidence`) + page. | code: quote is on the page (a misquote is asked again once) · Review Agent: features and custom requirements (rejected ones are dropped, not re-extracted) |
| 3 Matches | *Semantic search* (no LLM) | Past proposals covering each requirement (incl. custom ones), and similar projects | fixed threshold and order |
| 4 Responses | **Response Drafting** (tools) | Searches only the selected past proposals, never lost ones, won first (`search_library`, `get_passage`, ≤ 5 calls) and drafts an answer whose every claim quotes a passage it read; returns `citations`, `confidence` (0.9 first try, 0.7 after a correction) and `no_confident_source` | code: passage was read, quote is in it, no amounts · Review Agent |
| 5 Queries | **Clarification** | Questions for the client: why we ask, a sample answer, the assumption we price on; at most `limit` (Suggest with AI: 3) | code: topic id, no amounts, no (near-)repeat of an existing question or assumption · Review Agent |
| 5 Queries | **Assumption** | One conservative assumption per open question (`assumption` ≤ 400 chars, stated as a fact) with its `risk` (≤ 300) | code: exactly one per question id, unknown ids dropped, no amounts; one retry for missing ids, then a generic conservative assumption |
| 5 Queries | **Answer Check** | Does a typed answer actually answer its question? `acceptable`, a one-sentence `reason`, a `suggestion` | code pre-check without the model: no words, digits only (unless the question asks for a number), a single word |
| 6 Estimate | **Sizing** | A size class per requirement (XS–XL) by industry-standard delivery benchmarks, with a rationale | code: person-days per class (XS 5, S 10, M 20, L 40, XL 80); a requirement not sized gets M |
| 6 Ask AI | **Estimate Assistant** | Turns a request about the estimate ("extend the timeline to 30 weeks", "reduce the cost by 20%", "drop the driver app", "why 26 weeks?") into actions the backend applies: `leave_out`, `add_back`, `delivery_approach`, `set_timeline`, `set_team`, `reset_timeline`, `set_price` (an `[AMOUNT_n]` reference or a percent), `reset_price`; a question gets an answer and no action. It never sees a price | code: only actions the estimate allows (never a Must-have left out, team within its range, a price change only with `can_change_price`), one per group and item, no money amount in the message; one retry with the notes when an action was dropped |
| 7 Proposal | **Section Writer** | Writes each section from the facts only, following its template guidance; a missing fact reads `[to be completed]` | code: plain text (markup removed), 1–5 paragraphs ≤ 1200 chars, any paragraph with money dropped (`dropped`); one retry for empty sections |
| Setup | **Template Analysis** (job `template_extract`) | Reviews the sections the reader found in a DOCX template: section key, generated or boilerplate, and writing guidance in the template's voice. A template with < 2 headings but pictures: the headings are read from up to 4 pictures (vision) | code: merged onto the reader's result (keys valid and unique, titles, order and fixed text stay the document's, guidance without money); the model failing leaves the reader's result with a warning |
| 8 Submit | **Final Proposal Check** | Reads the user's own final proposal (1–2 files): price (only with `can_see_restricted`), weeks, core features delivered, answers matching the approved responses, attachments referenced | code: every fact's quote is in the file, a figure's number is in its quote; the five findings are written by code |
| Case studies | **Case Study Writer** | A case study from a won (customer story) or lost (lessons learned) proposal, written in the case study template's sections like the proposal's Section Writer (`sections`: each one's paragraphs, following its guidance), plus title, summary, metrics, the common layout (about the client, challenges, solution and pillars, features, why us) and the passages it used. Reads the proposal's indexed passages (three searches within the entry) when it is indexed, else the facts only | code: plain text within the limits, anything with money dropped (`dropped`; the engagement model's name is not money), a metric only when its number is in the facts or the passages; one retry when the title or summary is empty |
| all | **Review** | Independent second look: is the statement fully supported by its evidence and the rule? Missing verdict = fail | — |
| 6 Estimate | — | Prices never AI. The estimate stays in the backend's `estimate.py`; Sizing only suggests person-days. | — |

Background: the **Library indexer** splits each past proposal into passages, removes amounts, embeds them and
stores them per workspace. A page gives packed passages (whole sentences up to 1200 characters, the context a
writer quotes) and each list item or short line on its own: a requirement of a few words matches the bullet that
states it (0.85+) where a page of bullets scores like unrelated text (~0.6). `CHUNKING_VERSION` is part of the
content hash, so an entry is indexed again when the passages change. Besides past bids, the indexer takes
**standalone documents** (a case study or any other file; `library_entry_index.kind` `proposal` \| `casestudy` \|
`other`, migration `0003`): the same passages and redaction, no outcome. The report builder searches both through
`POST /library/search`; Step 3 (`/library/similar`) never sees a standalone document.

The **template reader** takes a template's sections from its Word heading styles or, in a document laid out by
hand without any, from its formatting: the bold lines at the size most of them share (a larger one is the
title, a smaller one a label), even one run on at the end of a paragraph. The Case Study Writer writes a
section whose template shows a list as `bullets`.

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
  answer or client question fails the check; written sections, assumptions, answer checks, size rationales and
  template guidance drop any text with a currency amount or a price/cost/fee word next to a number.

## Endpoints (`/v1`, service JWT required)

| Endpoint | Purpose |
|---|---|
| `POST /provider/probe` | Try a model before it is saved (admin > AI provider). `{api_mode: responses\|chat_completions, base_url, model, api_key}` (the key is used for this call only, never stored or logged) → `{verdict: ready\|limited\|unsuitable, checks: [{key: answer\|speed\|tools, status: pass\|warn\|fail, message}], seconds}`. Three tiny requests through the same client the agents use: a structured answer (at the extraction effort, which also proves the server is reached, the model exists and the output limit is accepted), its speed (a warning above 20 s) and a tool call. A failing model is a `200` answer with `unsuitable`; the whole test gives up after `RE_AI_PROVIDER_PROBE_TIMEOUT_S` (default 180). The backend calls it when the admin tests or saves a model |
| `POST /rfp/analyse` | multipart: `file` (PDF, DOCX, XLSX, TXT) + `request` = `{"catalogue": [...]}` |
| `PUT /library/entries/{entry_id}` | multipart: `meta` JSON + optional `rfp`, `proposal` files; unchanged content is not re-embedded |
| `PUT /library/documents/{entry_id}` | multipart: `meta` JSON (`kind` `casestudy` \| `other`, `client`, `title`, `industry`, `year`) + `file` (PDF, DOCX, XLSX or text): one standalone document, read on its own (a section `document` plus a summary line); unchanged content is not re-embedded |
| `DELETE /library/entries/{entry_id}` | remove from the index (a past proposal or a document) |
| `POST /library/similar` | requirements (+ RFP summary) → matching past proposals with the passage that proves it. Only past bids (`kind` `proposal`) take part: a standalone document is never a comparable |
| `POST /library/search` | The report builder's search. `{query (≤ 500), kinds, outcomes, industries, year_from, year_to, threshold (default 0.55), limit (≤ 100, default 25), passages_per_document (≤ 3)}` → `{results: [{entry_id, kind, client, title, industry, outcome, year, similarity, passages: [{passage_id, section, page, similarity, text}]}], candidates, threshold, embedding_model}`. The filters narrow the entries first (Postgres, `library_entry_index`), then passages are ranked by exact cosine; an entry's score is its best passage's, order is stable (ties by id). No model call, so no cost. A hit on an entry's one-line summary (title, client, year) has to clear the threshold by 0.10, since that short line scores 0.5–0.56 against any query. The 0.55 threshold is a starting point: on a handful of real queries the passage that answers a plain-words description scored 0.56–0.65 and unrelated text 0.35–0.52 (bge-small); tune it on the evaluation set |
| `POST /proposals/draft-answers` | questions + selected entry ids → `drafted` / `needs_person` / `skipped_pricing` / `skipped` |
| `POST /proposals/client-queries` | requirements (covered or not) + unclear points + existing questions and assumptions + `limit` → client questions |
| `POST /proposals/final-check` | multipart: `files` (1–2) + `request` = `{"draft": {price, weeks, core_features, responses, attachments, ...}}` → `findings` for Scope, Cost, Timeline, Responses, Attachments (in that order) |
| `POST /proposals/write-sections` | JWT `purpose` `write_sections`. `{sections: [{key, title, guidance}] (1–30, unique keys), facts: {client, title, summary, industry, engagement_model, delivery_approach, requirements, standard_features, modules, timeline_weeks, team_size, assumptions, clarifications: [{question, answer}], experience: [{client, title, year, outcome}]}}` → `{sections: [{key, paragraphs}], dropped, meta}` |
| `POST /proposals/clarifications/assume` | JWT `purpose` `assume_answers`. `{rfp_summary, engagement_model, requirements, questions: [{id, question, category, why}] (1–60, unique ids)}` → `{assumptions: [{id, assumption, risk}], meta}`, one per question in its order |
| `POST /proposals/clarifications/check-answer` | JWT `purpose` `check_answer`. `{question, why, category, answer}` → `{acceptable, reason, suggestion, meta}` |
| `POST /proposals/size-requirements` | JWT `purpose` `size_requirements`. `{industry, requirements: [{key, label, detail}] (1–80, unique keys)}` → `{sizes: [{key, size, days, rationale}], meta}`, one per requirement in its order |
| `POST /proposals/estimate-request` | JWT `purpose` `estimate_request`. `{ask (1–500, money replaced by [AMOUNT_n]), amounts: ["AMOUNT_1"] (≤ 5), state: {engagement_model, delivery_approach, delivery_approaches: [{key, label, note}], weeks, team, person_days, team_min, team_max, weeks_min, weeks_max, in_scope / out_of_scope: [{key, label, kind: core|custom|standard, days, must}], price_set, team_set, can_change_price}}` → `{understood, message, actions: [{kind, key, weeks, team, amount_ref, percent}], meta}`. No price is ever sent; the backend computes every figure |
| `POST /case-studies/write` | JWT `purpose` `write_case_study`. `{entry_id, facts: {client, title, industry, engagement_model, outcome: won|lost, reason, weeks, features}, template: [{key, title, guidance}] (≤ 30)}` → `{title, summary, metrics: [{v, l}], industry, scope, engagement, about_client, challenges: [{title, text}], solution_intro, solution_pillars: [{title, text, tag}], features, why_us, sections: [{key, paragraphs}], sources: [{passage_id, page}], dropped, meta}` |
| `POST /jobs` | `{type, input, callback_url, idempotency_key, priority}` → `202 {job_id}`; types `rfp_analysis`, `template_extract` |
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

**`template_extract`** input: `{file_url, filename, kind: "proposal" | "case_study"}`. Result: the backend's
`TemplateSpec` (`branding`, `page`, `header`, `footer`, `cover`, `sections: [{key, title, order, kind, fixed_text?,
guidance}]`, `confidence`, `warnings`). The DOCX is read by code first (headings, styles, page, cover); a template
with fewer than 2 headings but embedded pictures (`word/media`: PNG, JPEG, GIF, WebP) has its headings read from up
to 4 of them by a vision call (images sent as data URLs); then the Template Analysis Agent reviews every section and
code merges its advice. A model failure never fails the job: the code's reading is returned with a warning.
`usage` counts the model calls.

**OCR:** PDF pages without a text layer are rendered with pdfium and read with RapidOCR (ONNX on CPU, local), at
most `RE_AI_OCR_MAX_PAGES` (100) per document; the result says `ocr: true`. A file with no readable text at all
fails with "has no readable text".

## Run

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
docker compose up -d                                   # Postgres 16 on port 5433 (Qdrant is embedded)
cp .env.example .env                                   # set RE_AI_SERVICE_TOKEN_SECRET and RE_AI_SECRET_STORE_FILE (no AI key here)
.venv/Scripts/python -m alembic upgrade head
.venv/Scripts/python -m uvicorn revenue_engine_ai.main:app --app-dir src --reload --port 8001
```

Checks (the CI gates): `ruff check .`, `ruff format --check .`, `mypy src`, `alembic check`,
`pytest --cov --cov-fail-under=90`, and the Docker image build. `pre-commit install` runs ruff on every commit
(optional). Docker: `docker build -t revenue-engine-ai .` then run it with the `RE_AI_*` settings on port 8001.

## Try it with a local model (Ollama)

Free, and documents stay on your machine. It is slow on a CPU, and a small model follows the output format less
reliably than OpenAI, so use it to check the wiring and small documents, not for quality. The steps below are the
suggested starting point; they have not been run end to end on every machine.

1. **Install Ollama and pull a model that supports tools.** On a PC with about 16 GB of RAM and no GPU, start with
   `ollama pull qwen3:8b` (about 5 GB); `qwen3:4b` is faster and weaker, `llama3.1:8b` does not "think" first.
   Avoid models of 14B or more. Check what is installed with `ollama list`.
2. **Raise Ollama's context.** Its default (about 4k tokens on a CPU) silently cuts a long RFP short. Set
   `OLLAMA_CONTEXT_LENGTH=16384` (PowerShell: `setx OLLAMA_CONTEXT_LENGTH 16384`), quit Ollama from the tray and
   start it again. `ollama run qwen3:8b "hi"`, then `ollama ps`: the CONTEXT column should say 16384.
3. **Give this service more time.** In `.env` set `RE_AI_REQUEST_TIMEOUT_S=900` (the default 120 s fails on a CPU; a
   call is also retried twice) and `RE_AI_MAX_PARALLEL_LLM_CALLS=1`, then restart the service (`.env` is read at start).
4. **Set the provider in the app** (Admin › AI provider, as an admin):

   | Field | Value |
   |---|---|
   | Provider | Self-hosted (OpenAI-compatible) |
   | Base URL | `http://localhost:11434/v1` |
   | Model identifier | the name from `ollama list`, e.g. `qwen3:8b` |
   | API mode | Chat Completions |
   | Authentication | None (Ollama needs no key) |

   The form says the address must use `https`; with `RE_ENV=local` on the backend, `http://localhost` is allowed.
   Elsewhere only `https` and public addresses are, unless the host is on the backend's `RE_AI_ENDPOINT_ALLOWED_HOSTS`.
   The **Check key** button shows only with Authentication set to API key: to try the server first, choose that,
   type any value (Ollama ignores it) and it lists the models; then switch back to None to save.
5. **Try a small RFP** (a page or two) on the proposal builder's Scope step. Expect minutes, not seconds. While it
   runs `ollama ps` shows the model working.

Saving a provider replaces the workspace's current one, so test in a spare workspace, or keep the old key at hand.

When it fails, the log line says why: `llm_call_failed` (the server refused or could not be reached; the message
names an expired key, a wrong model name, an unreachable server or a rate limit), `llm_answer_truncated` (raise
`RE_AI_MAX_OUTPUT_TOKENS` or use a smaller input) or `llm_answer_invalid` (the model did not follow the JSON schema
twice: try another model). A timeout means `RE_AI_REQUEST_TIMEOUT_S` is still too low or the model is too big.

The form's **Test model** button, and saving, first run the provider probe (above) against the model: three tiny
requests. On a CPU a local model can take a minute or more for them; a model that does not finish within
`RE_AI_PROVIDER_PROBE_TIMEOUT_S` (default 180 s; the backend waits `RE_AI_SERVICE_TIMEOUT_S`, 300 s, so keep this below
it) is refused as too slow. A model that passes with warnings (slow, did not use the test tool) is saved.

## Layout

```text
src/revenue_engine_ai/   the service package (src layout)
  main.py                app factory (create_app)
  api/                   routes and request/response schemas (HTTP only)
  services/              one use-case per step; jobs.py runs asynchronous jobs
  agents/                prompt + output schema + pure check, one module per agent
  documents/  rules/     pure: parsing, chunking, OCR, template reader; text and matching rules
  llm/                   ports (LLM, Embedder) and the providers that implement them
  store/                 Postgres (index, caches, jobs) and Qdrant
  core/                  settings, database, errors, logging, service tokens, signing
migrations/              Alembic
tests/                   unit/ (pure), api/ (routes on fakes), integration/ (real Postgres); fakes.py
.claude/                 Claude Code settings and project skills
.github/                 CI, Dependabot, pull request template
```

## Backend integration

The backend (`stc-revenue-engine-api`, `integrations/ai_client`) calls every endpoint above from the proposal
builder: Analyse → `/rfp/analyse`; Find comparables → indexes un-indexed library entries, then
`/library/similar` for custom requirements; Responses → `/proposals/draft-answers`; Suggest with AI →
`/proposals/client-queries` (`limit` 3); the uploaded final proposal → `/proposals/final-check`; the RFP is read
by an `rfp_analysis` job. Set the same `RE_AI_SERVICE_TOKEN_SECRET` and `RE_AI_CALLBACK_SECRET` on both,
`RE_AI_SERVICE_URL` on the backend and `RE_MOCK_AI=false` there to use this service locally.

## Not done yet

- **Evaluation set:** 10–20 past RFPs with known answers to tune the match threshold and gate prompt changes.
- Other job types of ARCHITECTURE §10 (`bulk_draft`, `ingest_file`, ...): only `rfp_analysis` and
  `template_extract` run as jobs; `priority` is accepted but not used yet; a job lost by a restart is not resumed.
**A workspace's own AI provider is the only source of AI credentials.** An admin saves a provider (OpenAI, an
OpenAI-compatible gateway, or a local server such as Ollama or vLLM), model and key in the backend (Admin › AI
provider; the key is checked with the provider before it is stored). Each call for that workspace carries the
settings in the service token (`ai_provider`: model, endpoint, API type, config version), and this service reads
the key from the secret store the backend writes to, at a path built from the token's workspace id
(`llm/workspace_llm.py`), caches the client for `RE_AI_WORKSPACE_LLM_TTL_S` (300 s; a saved change is seen at
once because the config version changes) and runs the call on it. There is no service-level key and no fallback:
a workspace with no saved provider gets `ai-provider-not-set` (422), a key the provider rejects gets
`ai-credential-rejected` (422), and a key that cannot be read fails the call with a clear message. An unreachable
server, a wrong model name and a rate limit are told apart too (`llm/failures.py`). Only the local
`file` store is read so far (`RE_AI_SECRET_STORE_FILE`, the backend's `RE_SECRET_STORE_FILE`).

- RS256 service tokens against the backend's JWKS; Batch API for bulk library imports.
- Reading a workspace's key from AWS Parameter Store (only the local file store is read), and the reasoning
  effort settings still come from this service's `.env`, whatever model the workspace chose.
