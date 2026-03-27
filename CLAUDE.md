# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
playwright install chromium   # required for dynamic web scraping
cp .env.example .env          # then add OPENAI_API_KEY
```

## Running

```bash
python main.py                          # interactive CLI (con historial persistente en sessions/)
docker compose up --build               # containerized (mounts data_trading/ as volume)
python supervisor.py                    # quick test run (test_graph en __main__)
pytest tests/test_routing.py -v        # tests de routing (no requieren API key)
python dashboard.py [audit.jsonl]      # mini-dashboard visual del audit log (--no-show para solo guardar PNG)
python analytics.py [audit.jsonl]      # strategy ranking + learning curve (--train para logistic regression)
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | yes* | — | OpenAI API key (*no requerida con `ollama`) |
| `LLM_PROVIDER` | no | `openai` | `openai` / `azure` / `ollama` |
| `OPENAI_MODEL` | no | `gpt-4o-mini` | Model name |
| `TEMPERATURE` | no | `0.7` | LLM temperature |
| `AZURE_OPENAI_ENDPOINT` | no | — | Solo si `LLM_PROVIDER=azure` |
| `AZURE_OPENAI_API_KEY` | no | — | Solo si `LLM_PROVIDER=azure` |
| `AZURE_OPENAI_DEPLOYMENT` | no | `gpt-4o-mini` | Solo si `LLM_PROVIDER=azure` |
| `OLLAMA_MODEL` | no | `llama3` | Solo si `LLM_PROVIDER=ollama` |
| `HITL_ENABLED` | no | `true` | Confirmación humana antes de `code_node`/`web_scraping_node` |
| `LANGCHAIN_TRACING_V2` | no | — | `true` para activar LangSmith |
| `LANGCHAIN_API_KEY` | no | — | API key de LangSmith |
| `LANGCHAIN_PROJECT` | no | `multi-agents` | Nombre del proyecto en LangSmith |
| `AGENTDOG_GUARD_URL` | no | — | AgentDoG guardrail endpoint (OpenAI-compatible) |
| `AGENTDOG_POLICY` | no | `fail_open` | `fail_open` / `fail_closed` / `fail_soft` |
| `AGENTDOG_EVAL_MODE` | no | `high_risk_only` | `all_nodes` / `high_risk_only` / `final_only` |
| `AGENTDOG_AUDIT_LOG` | no | stdout | Path for JSONL audit log |
| `AGENTDOG_API_KEY` | no | — | Bearer token for guard endpoint |
| `USE_SQLITE` | no | `true` | `true` = SQLite (sessions.db) / `false` = JSONL legacy |

## Architecture

The system follows the **Supervisor pattern** using LangGraph's `StateGraph`:

```
User → input_guard → supervisor_node → route_agent() → [math|analysis|code|web_scraping]_node → END
```

**Key files:**
- [config.py](config.py): `get_llm()` — selects provider via `LLM_PROVIDER` (openai/azure/ollama).
- [agents.py](agents.py): Four specialized `create_react_agent` agents; all tools use `Annotated`+`Field` for precise LLM schemas.
- [supervisor.py](supervisor.py): `StateGraph`, middleware, HITL, AgentDoG guardrail, and `create_supervisor_graph()`.
- [main.py](main.py): Async REPL with persistent session history in `sessions/`.
- [tests/test_routing.py](tests/test_routing.py): Routing tests (no API key needed, uses mocks).

**Shared state (`AgentState`):**
- `messages`: append-only list (uses `lambda x, y: x + y` as reducer)
- `next_agent`: set by `supervisor_node` via `RoutingDecision` Pydantic model, consumed by `route_agent()`

**Execution layers in order:**
1. **`input_guard`** — pre-execution middleware, blocks prompt injection patterns before any LLM call
2. **`supervisor_node`** — routes via `llm.with_structured_output(RoutingDecision)`; BTC price shortcut bypasses the LLM
3. **HITL** — `code_node` and `web_scraping_node` prompt the user for confirmation (`HITL_ENABLED=true`)
4. **Agent execution** — `create_react_agent` runs tools in a ReAct loop
5. **AgentDoG** — post-execution trajectory check; blocks unsafe results before they reach state
6. **Context quarantine** (web scraping only) — sub-agent absorbs raw HTML; only a ≤200-word summary reaches shared state

**Agent tools:**
- `math_agent` → `calculate` (safe `eval` with math namespace)
- `analysis_agent` → `analyze_data`
- `code_agent` → `write_code`
- `web_scraping_agent` → `scrape_website_simple` (requests+BS4), `scrape_website_dynamic` (Playwright sync, cached 60s), `scrape_website_with_json_capture` (Playwright async, saves JSON API responses to `data_trading/`), `extract_price_from_text`

**AgentDoG guardrail** (`supervisor.py`):
- Post-execution safety check on the agent trajectory (action/observation pairs from `AIMessage.tool_calls` + `ToolMessage`).
- `code_node` and `web_scraping_node` are `HIGH_RISK_NODES`.
- Audit events emitted as JSONL to `AGENTDOG_AUDIT_LOG` or stdout.

**Persistent sessions** (`sessions/`): SQLite backend (`sessions/sessions.db`) con migración one-shot desde JSONL legacy. Fallback JSONL disponible con `USE_SQLITE=false`. Módulo: [persistence.py](persistence.py).

**Análisis de sesiones con DuckDB** (requiere `pip install duckdb`): ver [analytics/queries.sql](analytics/queries.sql).
```bash
duckdb -c ".read analytics/queries.sql"   # ejecutar todas las queries
```
Queries incluidas: debugging (score < 0), ranking por `(category, strategy)`, vista completa por sesión, malas decisiones del sistema, comparativa API vs scraping, learning curve, counterfactual insight (bandit > ML).

**`gateway.py`**: `AgentGateway` + `LaneQueue`. Session Lane por `session_key` (concurrency=1, modo collect). Inspirado en OpenClaw `src/process/command-queue.ts`. Base para integraciones futuras (ej: Telegram): `response = await gateway.send(str(user.id), text)`.

**`memory.py`**: `distill_memory()` + `load_memory_context()`. Destilación de sesión al salir → `sessions/{id}/MEMORY.md`. Se inyecta como `SystemMessage` al inicio de sesiones subsiguientes.

**`agents/`**: System prompts en formato Markdown, uno por agente (`math_agent.md`, `analysis_agent.md`, `code_agent.md`, `web_scraping_agent.md`). Permite modificar el comportamiento de los agentes sin tocar código. Si `AGENT_HOT_RELOAD=true`, los cambios se reflejan en caliente sin necesidad de reiniciar la aplicación.

**`data_trading/`**: JSON files auto-saved by `scrape_website_with_json_capture`; filename format is `{url_slug}_{sha256_10chars}_{unix_ts}.json`.
