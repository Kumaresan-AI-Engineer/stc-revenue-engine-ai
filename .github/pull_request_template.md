## What and why

<!-- One or two sentences: what changes for the proposal builder, and why. -->

## How it was checked

- [ ] `ruff check .` and `ruff format --check .`
- [ ] `mypy src`
- [ ] `pytest` (and `tests/integration` against a migrated database, if the store changed)
- [ ] Tried against the running service (`/run`), if an endpoint or agent changed

## Agent and contract changes

- [ ] A changed prompt bumps its `PROMPT_VERSION`
- [ ] A changed request or response shape is matched in the backend's `integrations/ai_client/contract.py`
- [ ] No prices reach a prompt or an answer
- [ ] `README.md` / `CLAUDE.md` updated where the behaviour they describe changed
