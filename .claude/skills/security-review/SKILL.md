---
name: security-review
description: Use when checking the pending changes in stc-revenue-engine-ai for security vulnerabilities only - service JWT and workspace isolation, file fetching (SSRF), prompt injection from documents, data leaking between workspaces or into prompts, secrets, uploads, new dependencies. Report-only, most severe first. Trigger words - security review, is this secure, vulnerabilities, OWASP, SSRF, prompt injection, before merging.
---

# Security review (Revenue Engine AI service)

Report-only: no edits. Scope is `git diff HEAD` plus untracked files unless a branch or path is given.

## Look for

- **Authentication:** a route without the service-JWT dependency (`core/security.py`); `aud`, `iss` or expiry not
  checked; the workspace read from the body, path or a document instead of the token.
- **Workspace isolation:** a store, cache or Qdrant query without the workspace filter; a cache key that lets one
  workspace receive another's result.
- **File fetching (SSRF):** a job URL fetched without the `RE_AI_FILE_URL_PREFIXES` check, redirects followed to
  another host, no size or time limit.
- **Callbacks:** job events not signed with `core/signing.py`, or a signature compared without constant time.
- **Prompt injection:** document text placed where it can act as instructions (outside the quoted source block),
  tools that reach beyond the selected entries, model output trusted without the code `check`.
- **Data in prompts and logs:** prices or restricted values reaching a prompt or an answer; secrets, tokens or
  document text in logs.
- **Uploads and parsing:** unbounded file size, zip bombs in docx or xlsx, unsafe XML parsing.
- **Secrets and dependencies:** a key in code, config or tests; a new dependency (check it with `pip-audit`).

## Report

Most severe first: file:line, the vulnerability, how an attacker would exploit it, and the fix. Nothing found: say
so, and list what was checked.
