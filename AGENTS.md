# Code Review Rules — Multi-Agent Supervisor System

## Security

- `eval()` is only allowed in `agents.py:calculate()` within the math namespace (`safe_dict` with `"__builtins__": {}`). Any other use of `eval()` or `exec()` must be flagged as critical.
- The `input_guard` middleware (defined in `security.py`, wired as the graph entry point in `graph.py`) must always run before any LLM call. Changes that bypass or reorder this step must be rejected.
- No API keys, tokens, or secrets may be hardcoded. All must come from `os.getenv()` or `.env` via `dotenv`.
- URL and domain validation for web scraping tools must remain in place. Additions to `allowed_domains` require justification.
- Prompt injection patterns in `_BLOCKED_PATTERNS` and `_RISK_SIGNALS` must not be weakened or removed.

## LangGraph / Agent Patterns

- `AgentState.messages` reducer must remain append-only (`lambda x, y: x + y`). Replacing it with a non-append reducer breaks replay and tracing.
- `supervisor_node` must always route via `llm.with_structured_output(RoutingDecision)`. Inline string parsing of routing decisions is not acceptable.
- New agents must be added to `AgentName` (Literal) and registered in `route_agent()`. Skipping either step causes silent routing failures.
- `create_react_agent` from `langgraph.prebuilt` is the correct factory for all specialized agents. LangGraph emits a deprecation warning pointing to `langchain.agents.create_agent`, but that function has a different signature and does not return a `CompiledStateGraph`. Do not migrate until there is a drop-in replacement with equivalent behavior and passing end-to-end tests.
- Agent system prompts must be loaded via `load_agent_prompt()` from `agents/{name}.md`. Hardcoded prompts inside agent factory functions must be flagged.

## HITL (Human-in-the-Loop)

- `code_node` and `web_scraping_node` must always check `HITL_ENABLED` before execution. Removing or short-circuiting this check must be rejected.
- HITL confirmation must happen before the agent runs, not after.

## AgentDoG Guardrail

- Post-execution safety checks must run before updated state reaches the graph. Any change that moves the guardrail after state mutation must be flagged.
- `HIGH_RISK_NODES` (`code_node`, `web_scraping_node`) must always be evaluated. Removing a node from this set requires explicit justification.
- Audit log entries must include: node name, risk level, decision, and trajectory summary. Omitting any field silently degrades observability.
- The `AGENTDOG_POLICY` modes (`fail_open`, `fail_closed`, `fail_soft`) must remain respected. Adding a bypass path for any policy is not allowed.

## Web Scraping & Context Quarantine

- Raw HTML from web scraping must never be written directly to shared `AgentState`. Only the ≤200-word summary from the sub-agent may reach shared state.
- `scrape_website_dynamic` must use the sync Playwright API (`sync_playwright`). Async Playwright is reserved for `scrape_website_with_json_capture` only.
- JSON bundles saved to `data_trading/` must follow the `{slug}_{sha256_10}_{unix_ts}.json` naming convention.
- The in-memory scrape cache (`_SCRAPE_CACHE`) TTL is 60 seconds. Do not increase it without considering stale price data risk.

## Async Patterns

- Functions that call `scrape_website_with_json_capture` or `_scrape_dynamic_async` must be `async`. Blocking async calls with `asyncio.run()` inside a running event loop will deadlock.
- `asyncio.gather(*tasks, return_exceptions=True)` must be used when collecting JSON capture tasks to avoid silently dropped responses.

## Persistence & Sessions

- SQLite is the default backend (`USE_SQLITE=true`). Changes to session schema in `persistence.py` must include a migration path for the existing `sessions.db`.
- The JSONL legacy fallback (`USE_SQLITE=false`) must remain functional and not be silently broken by schema changes.
- Memory distillation (`memory.py:distill_memory()`) writes to `sessions/{id}/MEMORY.md`. This file is injected as `SystemMessage` on session start — its format must remain stable.

## LLM Provider / Config

- `config.py:get_llm()` must support all three providers: `openai`, `azure`, `ollama`. Removing a provider branch must be flagged.
- `OPENAI_API_KEY` absence must raise a clear `ValueError`, not silently default to an empty string.
- Temperature must come from `TEMPERATURE` env var, not hardcoded.

## General Python Quality

- Do not use mutable default arguments in function signatures.
- Exceptions in tool functions must be caught and returned as user-readable strings, not re-raised (agents cannot handle raw exceptions).
- `Optional[str]` return values from price APIs (`_price_coingecko`, `_price_binance`, `_price_coinbase`) must be checked before use — never assumed truthy.
- Imports inside tool functions (e.g., `import requests`, `from bs4 import BeautifulSoup`) are intentional to avoid slow startup. Do not move them to the module level.
