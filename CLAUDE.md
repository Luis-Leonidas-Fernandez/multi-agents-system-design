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
python main.py                          # interactive CLI (con historial persistente en data/sessions/)
python main.py --frontend-bridge        # WebSocket bridge para el frontend React
make dev                                # Vite + frontend bridge con restart automático del backend
docker compose up --build               # containerized (mounts data/web_scraping/data_trading/ as volume)
python application/composition/graph.py  # quick test run (test_graph en __main__)
pytest tests/test_routing.py -v        # tests de routing (no requieren API key)
python features/analytics/infrastructure/dashboard.py [audit.jsonl]      # mini-dashboard visual del audit log (--no-show para solo guardar PNG)
python features/analytics/infrastructure/country_news_analytics.py [audit.jsonl]
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
| `TAVILY_API_KEY` | yes* | — | Tavily Search API key (*requerida para `search_web`) |

## Architecture

The system follows the **Supervisor pattern** using LangGraph's `StateGraph`:

```
User → input_guard → supervisor_node → route_agent() → [math|analysis|code|web_scraping]_node → END
```

**Key files:**
- [core/helpers/config_flow_helpers.py](core/helpers/config_flow_helpers.py): `get_llm()` — selects provider via `LLM_PROVIDER` (openai/azure/ollama).
- [application/services/agents_factory.py](application/services/agents_factory.py): Four specialized `create_react_agent` agents; feature-owned tools live under `features/*/infrastructure/` and external integrations under `integrations/`.
- [application/composition/graph.py](application/composition/graph.py): `StateGraph`, middleware, HITL, AgentDoG guardrail, and `create_supervisor_graph()`.
- [main.py](main.py): Async REPL with persistent session history in `data/sessions/`.
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
- `web_scraping_agent` → `scrape_website_simple` (requests+BS4), `scrape_website_dynamic` (Playwright sync, cached 60s), `scrape_website_with_json_capture` (Playwright async, saves JSON API responses to `data/web_scraping/data_trading/`), `extract_price_from_text`

**AgentDoG guardrail** (`application/composition/graph.py` / `features/*/infrastructure/` / `core/helpers/audit_flow_helpers.py`):
- Post-execution safety check on the agent trajectory (action/observation pairs from `AIMessage.tool_calls` + `ToolMessage`).
- `code_node` and `web_scraping_node` are `HIGH_RISK_NODES`.
- Audit events emitted as JSONL to `AGENTDOG_AUDIT_LOG` or stdout.

**Persistent sessions** (`data/sessions/`): SQLite backend (`data/sessions/sessions.db`) con migración one-shot desde JSONL legacy. Fallback JSONL disponible con `USE_SQLITE=false`. Módulo: [features/sessions/infrastructure/persistence.py](features/sessions/infrastructure/persistence.py).

**Análisis de sesiones con DuckDB** (requiere `pip install duckdb`): ver [features/analytics/infrastructure/queries.sql](features/analytics/infrastructure/queries.sql).
```bash
duckdb -c ".read features/analytics/infrastructure/queries.sql"   # ejecutar todas las queries
```
Queries incluidas: debugging (score < 0), ranking por `(category, strategy)`, vista completa por sesión, malas decisiones del sistema, comparativa API vs scraping, learning curve, counterfactual insight (bandit > ML).

**`application/services/session_gateway.py`**: `AgentGateway` + `LaneQueue`. Session Lane por `session_key` (concurrency=1, modo collect). Inspirado en OpenClaw `src/process/command-queue.ts`. Base para integraciones futuras (ej: Telegram): `response = await gateway.send(str(user.id), text)`.

**`application/services/background_tasks.py`**: lifecycle de tareas largas con snapshots append-only, estado explícito, resumen por sesión y lookup por sesión/estado/request/trace.
**`application/services/background_tasks.py`**: soporta cancelación y reintento con tracking de `parent_task_id` y `attempt_number`.

**`application/services/session_replay.py`**: replay unificado de sesión (transcript, prompts, background tasks y audit trail) para inspección CLI.
**`main.py`**: `/replay` muestra la línea de tiempo unificada de la sesión actual.

**`application/services/memory_retrieval.py`**: búsqueda y ranking de `MEMORY.md` por sesión con comando `/memory`.
**`main.py`**: `/memory [buscar texto]` busca memorias destiladas entre sesiones.

**`application/services/tool_approval.py`**: vista previa de aprobación de tools con riesgo, permisos y prompt HITL.
**`main.py`**: `/tools` y `/tool <name> [json_args|key=value ...]` muestran catálogo y previsualización de aprobación.
**`application/services/tool_execution.py`**: el prompt HITL incluye descripción, args y motivo de la política antes de confirmar.

**`main.py`**: CLI interactiva con comandos de inspección (`/help`, `/inspect`, `/tasks`, `/task <id>`, `/artifact`) para ver estado delegado y artifacts de la sesión actual.
**`main.py`**: `/context [agente]`, `/bookmarks`, `/bookmark [nombre]` y `/checkpoint <id>` exponen presupuesto de contexto y checkpoints de sesión.
**`application/services/command_registry.py`**: registra slash commands, aliases y grupos para autogenerar ayuda y descubrir comandos.
**`main.py`**: `/commands` y `/command <nombre>` usan el registry para listar o describir comandos; `/status` y `/state` siguen siendo aliases útiles.
**`application/services/context_budget.py`**: reporta qué entra al contexto, qué viene resumido y qué queda afuera de la vista del turno.
**`application/services/session_bookmarks.py`**: persiste bookmarks/checkpoints por sesión con referencia a artifact, replay y budget.
**`application/services/tool_impact.py`**: estima archivos afectados, diff aproximado y side effects para tools de código/web.
**`main.py`**: `/impact <name> [json_args|key=value ...]` muestra el impacto estimado; `/tool` y el prompt HITL lo incluyen también.
**`application/services/tool_impact.py`**: cuando puede, cruza la tarea con archivos reales del repo y sube la confianza a `repo-aware`.
**`application/services/tool_impact.py`**: también cruza símbolos/imports reales (classes/defs/imports) para afinar la evidencia del preview.

**`application/services/prompt_versioning.py`**: snapshots versionados de prompts con hash estable, historial append-only y lookup por agente.

**`main.py`**: también expone `/prompts` y `/prompt <agente>` para inspeccionar snapshots de prompts persistidos.
**`main.py`**: además expone `/cancel <id>` y `/retryable` para operar sobre background tasks.

**`application/services/session_artifacts.py`**: el artifact ahora incluye prompt snapshots persistidos para trazabilidad completa.
**`application/services/session_artifacts.py`**: el artifact también agrega presupuesto de contexto y bookmarks persistidos para inspección.

**`features/sessions/infrastructure/memory.py`**: `distill_memory()` + `load_memory_context()`. Destilación de sesión al salir → `data/sessions/{id}/MEMORY.md`. Se inyecta como `SystemMessage` al inicio de sesiones subsiguientes.

**`agents/`**: System prompts en formato Markdown, uno por agente (`math_agent.md`, `analysis_agent.md`, `code_agent.md`, `web_scraping_agent.md`). Permite modificar el comportamiento de los agentes sin tocar código. Si `AGENT_HOT_RELOAD=true`, los cambios se reflejan en caliente sin necesidad de reiniciar la aplicación.

**`data/web_scraping/data_trading/`**: JSON files auto-saved by `scrape_website_with_json_capture`; filename format is `{url_slug}_{sha256_10chars}_{unix_ts}.json`.
