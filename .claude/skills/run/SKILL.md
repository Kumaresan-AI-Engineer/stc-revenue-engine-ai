---
name: run
description: Use to start the Revenue Engine AI service locally and check a change against the real running service - Postgres up, migrations applied, uvicorn on port 8001, then call the changed endpoints and report honestly what was and was not verified. Trigger words - run the AI service, start the service, check it works, try the endpoint, verify the agent.
---

# Run (Revenue Engine AI service)

Tests run on fakes; this checks the real service, with the real model when `.env` has a provider key.

## 1. Services

`docker compose up -d` (Postgres 16 on 5433; Qdrant is embedded in `.qdrant/`). Copy `.env.example` to `.env`
once. First time: `uv venv --python 3.12 .venv && uv pip install --python .venv/Scripts/python.exe -e ".[dev]"`.

## 2. One server on port 8001

`netstat -ano | findstr :8001` first. If a server with `--reload` from this repo is already running, use it. Only
stop a process you started yourself; ask before stopping anyone else's.

## 3. Start (from the repository root)

```bash
.venv/Scripts/python -m alembic upgrade head
.venv/Scripts/python -m uvicorn revenue_engine_ai.main:app --app-dir src --reload --port 8001
```

Run it in the background and wait for `http://127.0.0.1:8001/healthz`. A failure to start is the finding.

## 4. Exercise the change

- `http://127.0.0.1:8001/openapi.json` lists the changed routes with the expected shapes.
- The usual way in is through the backend (port 8000), which calls this service: use its `/run` skill and drive
  the feature from there.
- To call this service directly, sign a service token with `RE_AI_SERVICE_TOKEN_SECRET` the way `token()` in
  `tests/conftest.py` does. Never print the secret or the token.
- Golden path, then edge cases: another workspace (no data), a missing file, a provider error.

## 5. Report

What was started, which calls were made and their status codes, what worked, what broke, and what could not be
verified (no provider key, no database). Passing tests are not a verified feature.
