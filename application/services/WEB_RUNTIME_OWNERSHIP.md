# Web Runtime Ownership

- `web_runtime.py` — provider resolution and runtime contracts.
- `web_runtime_helpers.py` — small payload/status helpers.
- `web_fetch_helpers.py` — fetch draft construction and guard-safe fetch plumbing.
- `web_search_registry.py` / `web_fetch_registry.py` — provider registry and selection.

Rules of thumb:
- keep runtime adapters thin
- enforce guards before any LLM-backed fetch path
- keep prompt text in `agents/*.md`
