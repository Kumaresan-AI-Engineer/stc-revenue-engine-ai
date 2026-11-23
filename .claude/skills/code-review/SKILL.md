---
name: code-review
description: Use when reviewing changes in stc-revenue-engine-ai for correctness bugs before merging - the current diff, a branch, a PR number or a path. Reports findings ranked most severe first, each with a concrete failure scenario; --fix applies them. Trigger words - review, code review, check this PR, look over this diff, second opinion, before merging.
---

# Code review (Revenue Engine AI service)

Replaces the built-in `/code-review` in this repo. Correctness first; style only when it hides a bug.

## 1. Scope

No argument: `git diff HEAD` plus untracked files. A branch: `git diff dev...<branch>`. A PR number:
`gh pr diff <n>`. A path: that file or folder. Read each changed function whole, and its callers.

## 2. What to check

- **Agent contract:** the output schema matches what the service and `api/schemas.py` return; the backend's
  `integrations/ai_client/contract.py` (sibling repo `stc-revenue-engine-api`) accepts the new shape.
- **Determinism and caching:** a changed prompt or output schema without a `PROMPT_VERSION` bump serves stale
  cached results; a cache key missing an input that shapes the result; search order without a stable tie-break.
- **Checks:** the code `check` lets through text the prompt forbids (prices, invented clients, quotes not in the
  source); a retry that loses the reviewer's notes; a dropped item that should be `needs_person`.
- **Tenancy:** every store call takes the workspace from the service JWT, never from the body; a job's file URL is
  checked against `RE_AI_FILE_URL_PREFIXES`; drafting tools reach only the selected entries.
- **No prices:** amounts redacted at indexing, pricing questions skipped, amounts in output fail the check.
- **Async and errors:** blocking work (OCR, embedding, docx parsing) on the event loop; provider errors that become
  500s instead of `UpstreamError`; a swallowed exception.
- **Migrations:** a model change without a migration (`alembic check`); a vector-size change without a re-index.
- **Tests:** a fix without a test that fails without it; a test that calls a real provider instead of
  `tests/fakes.py`.

## 3. Verify before reporting

For each candidate, trace the real inputs that reach it. Drop what you cannot turn into a concrete failure.

## 4. Report

Most severe first: file:line, one-sentence defect, the failure scenario (inputs → wrong output). With `--fix`,
apply the fixes, add the missing tests, and run ruff, mypy and pytest.
