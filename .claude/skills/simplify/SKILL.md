---
name: simplify
description: Use right after changing code in stc-revenue-engine-ai to tidy only what changed - reuse existing helpers instead of duplicates, remove needless abstraction, nesting, dead code and repeated work, match the surrounding module - and apply the edits directly, then rerun lint, format, types and tests. Quality only, no bug hunting. Trigger words - simplify, clean up, tidy, refactor what I wrote, before committing.
---

# Simplify (Revenue Engine AI service)

Replaces the built-in `/simplify` in this repo. It always edits the code. It does not hunt for bugs (`/code-review`)
or vulnerabilities (`/security-review`).

## 1. Scope: only what changed

`git diff HEAD` plus untracked files. Never touch code outside the changed lines' functions, and never reformat or
rename files you are not otherwise changing. With no changes, say so and stop.

## 2. Look for

- **Duplication:** logic an existing helper already does. Search first: `rules/text.py` and `rules/matching.py`,
  `documents/` (parsing, chunking), `services/common.py`, `agents/review.py`, `core/errors.py`.
- **Agent shape drift:** every agent is prompt + output schema + pure `check`, run through review and one retry.
  A second way of calling the LLM, or a check that does I/O, is a finding.
- **Premature abstraction:** parameters, settings or ports nothing uses yet.
- **Dead code:** unused imports, variables, branches, commented-out blocks, debug output.
- **Repeated work:** an embedding or LLM call made twice for the same input (use the caches), a Qdrant query in a
  loop.
- **Fit:** names in the business language, docstrings like the neighbouring code, comments only for the why.

## 3. Apply

Keep behaviour identical: same responses, same errors, same cache keys. Changing a prompt's text means bumping its
`PROMPT_VERSION`, so do not reword prompts here. A bug or vulnerability you spot goes under "Spotted, not changed".

## 4. Check

```bash
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m ruff format <files you touched>
.venv/Scripts/python -m mypy src
.venv/Scripts/python -m pytest -q
```

If a test fails because of a simplification, undo it.

## 5. Report

Each change (file, what and why), the check results, and "Spotted, not changed". Never commit.
