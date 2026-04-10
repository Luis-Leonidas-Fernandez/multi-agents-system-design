# Sistema Multi-Agentes con LangGraph

Sistema multi-agentes implementado con LangGraph siguiendo el patrón supervisor/coordinador con arquitectura hexagonal.

## Estructura

```
06_multi_agents/
├── main.py                # Punto de entrada (REPL interactivo con slash commands)
├── requirements.txt
├── application/           # Capa de aplicación
│   ├── use_cases/         # Flujos y casos de uso
│   ├── services/          # Registries, factories, gateway y servicios de sesión
│   ├── helpers/           # Helpers compartidos (config, audit, scraping, precio, etc.)
│   ├── policies/          # Guardrails, HITL y seguridad
│   └── composition/       # Composition root
│       └── graph.py       # Grafo supervisor/coordinador y wiring
├── domain/                # Modelos puros del dominio (AgentState, RoutingDecision, pricing)
├── infra/                 # Infraestructura (persistence.py, scraping_infra.py, memory.py)
├── ports/                 # Contratos/puertos (llm_port.py, confirmation_port.py)
├── nodes/                 # Nodos del grafo (adaptadores finos sobre use_cases)
├── tools/                 # Tools reutilizables para agentes (math, code, data, web, crypto)
├── agents/                # System prompts en Markdown por agente (hot-reload opcional)
├── prompts/               # Snapshots de prompts versionados por agente
├── ops/                   # Dashboards y scripts de observabilidad (dashboard.py, analytics.py)
├── analytics/             # Queries DuckDB sobre sesiones
├── ui/                    # Frontend alternativo (claude_app.py)
├── sessions/              # Historial persistido de sesiones (SQLite)
├── docs/                  # Documentación larga y material educativo
├── flujo/                 # Diagramas de flujo del sistema
├── tests/                 # Suite de tests (399 passing)
└── .env.example           # Ejemplo de variables de entorno
```

## Instalación

```bash
pip install -r requirements.txt
playwright install chromium   # requerido para web scraping dinámico
cp .env.example .env          # agregar OPENAI_API_KEY
```

## Ejecución

```bash
./run.sh                               # wrapper local: levanta la app y el auto-start de SearXNG
python main.py                           # REPL interactivo con historial en sessions/
docker compose up --build                # containerizado (monta data_trading/ como volumen)
python application/composition/graph.py  # test rápido del grafo (__main__)
pytest tests/ -v                         # suite completa (399 tests, no requieren API key)
python ops/dashboard.py [audit.jsonl]    # dashboard visual del audit log
python ops/analytics.py [audit.jsonl]    # strategy ranking + learning curve
```

`python main.py` intenta levantar `searxng` automáticamente vía `docker compose up -d searxng` cuando `SEARXNG_AUTO_START=true` y `SEARXNG_BASE_URL` apunta a un host local.

## Variables de entorno

| Variable | Requerida | Default | Descripción |
|---|---|---|---|
| `OPENAI_API_KEY` | sí* | — | OpenAI API key (*no requerida con `ollama`) |
| `LLM_PROVIDER` | no | `openai` | `openai` / `azure` / `ollama` |
| `OPENAI_MODEL` | no | `gpt-4o-mini` | Nombre del modelo |
| `TEMPERATURE` | no | `0.7` | Temperatura del LLM |
| `AZURE_OPENAI_ENDPOINT` | no | — | Solo si `LLM_PROVIDER=azure` |
| `AZURE_OPENAI_API_KEY` | no | — | Solo si `LLM_PROVIDER=azure` |
| `AZURE_OPENAI_DEPLOYMENT` | no | `gpt-4o-mini` | Solo si `LLM_PROVIDER=azure` |
| `OLLAMA_MODEL` | no | `llama3` | Solo si `LLM_PROVIDER=ollama` |
| `HITL_ENABLED` | no | `true` | Confirmación humana antes de `code_node`/`web_scraping_node` |
| `COORDINATOR_MODE` | no | `false` | `true` activa el modo coordinador con workers paralelos |
| `AGENT_HOT_RELOAD` | no | `false` | `true` recarga system prompts de `agents/` sin reiniciar |
| `USE_SQLITE` | no | `true` | `true` = SQLite / `false` = JSONL legacy |
| `TAVILY_API_KEY` | sí* | — | Tavily Search API key (*requerida si querés usar Tavily como provider) |
| `SEARXNG_BASE_URL` | no | — | URL base de tu instancia SearXNG (fallback sin API key) |
| `SEARXNG_AUTO_START` | no | `true` | `true` arranca `searxng` con `docker compose up -d searxng` cuando levantás `python main.py` |
| `SEARXNG_LIMITER` | no | `false` | `false` evita que el limiter/bot-detection bloquee las búsquedas locales |
| `SEARXNG_LANGUAGE` | no | — | Código de idioma para SearXNG, por ejemplo `es` |
| `LANGCHAIN_TRACING_V2` | no | — | `true` para activar LangSmith |
| `LANGCHAIN_API_KEY` | no | — | API key de LangSmith |
| `LANGCHAIN_PROJECT` | no | `multi-agents` | Nombre del proyecto en LangSmith |
| `AGENTDOG_GUARD_URL` | no | — | AgentDoG guardrail endpoint (OpenAI-compatible) |
| `AGENTDOG_POLICY` | no | `fail_open` | `fail_open` / `fail_closed` / `fail_soft` |
| `AGENTDOG_EVAL_MODE` | no | `high_risk_only` | `all_nodes` / `high_risk_only` / `final_only` |
| `AGENTDOG_AUDIT_LOG` | no | stdout | Path para el audit log JSONL |
| `AGENTDOG_API_KEY` | no | — | Bearer token para el guardrail endpoint |

## Arquitectura

### Flujo de ejecución

```
User → input_guard → supervisor / coordinator → route_agent() → [agente especializado] → END
```

**Modo supervisor** (default): el supervisor rutea directamente al agente especializado.

**Modo coordinador** (`COORDINATOR_MODE=true`): el coordinador spawnea workers dinámicos y puede ejecutar probes en paralelo antes de delegar. Para `web_scraping_agent` lanza un probe round y el flujo de noticias usa búsqueda web estilo OpenClaw para obtener múltiples fuentes.

### Capas de ejecución en orden

1. **`input_guard`** — bloquea patrones de prompt injection antes de cualquier llamada al LLM
2. **`supervisor_node`** — rutea via `llm.with_structured_output(RoutingDecision)`; el precio BTC tiene un shortcut que bypasea el LLM
3. **HITL** — `code_node` y `web_scraping_node` piden confirmación al usuario (`HITL_ENABLED=true`)
4. **Agent execution** — `create_react_agent` corre tools en un loop ReAct
5. **AgentDoG** — chequeo post-ejecución de la trayectoria; bloquea resultados inseguros antes de que lleguen al estado compartido
6. **Context quarantine** (solo web scraping) — el sub-agente absorbe el HTML crudo; solo un resumen ≤200 palabras llega al estado compartido

### Agentes y tools

| Agente | Tools |
|---|---|
| `math_agent` | `calculate` (safe `eval` con namespace matemático) |
| `analysis_agent` | `analyze_data` |
| `code_agent` | `write_code` |
| `web_scraping_agent` | `scrape_website_simple` (requests+BS4), `scrape_website_dynamic` (Playwright, cache 60s), `scrape_website_with_json_capture` (Playwright async, guarda JSON en `data_trading/`), `extract_price_from_text`, `search_web` (Tavily + SearXNG fallback) |

### Estado compartido (`AgentState`)

- `messages` — lista append-only (reducer `lambda x, y: x + y`)
- `next_agent` — seteado por `supervisor_node` via `RoutingDecision`, consumido por `route_agent()`

### Archivos clave

| Archivo | Responsabilidad |
|---|---|
| `application/composition/graph.py` | `StateGraph`, wiring, `create_supervisor_graph()` |
| `application/helpers/config_flow_helpers.py` | `get_llm()` — selecciona provider via `LLM_PROVIDER` |
| `application/services/agents_factory.py` | Construcción centralizada de agentes ReAct |
| `application/services/coordinator_mode.py` | Feature flag del modo coordinador |
| `application/services/coordinator_workers.py` | Spawn y ejecución de workers dinámicos |
| `application/services/session_gateway.py` | `AgentGateway` + `LaneQueue` (base para integración Telegram/etc.) |
| `infra/persistence.py` | SQLite backend para historial de sesiones |
| `infra/memory.py` | `distill_memory()` — destila sesión a `MEMORY.md` e inyecta como contexto |
| `domain/models.py` | `AgentState`, `RoutingDecision`, modelos Pydantic |
| `main.py` | REPL async con slash commands y sesiones persistidas |

## Slash commands del REPL

| Comando | Descripción |
|---|---|
| `/help` | Lista todos los comandos disponibles |
| `/commands` | Registry completo con aliases y grupos |
| `/command <nombre>` | Ayuda detallada de un comando |
| `/replay` | Línea de tiempo unificada de la sesión (transcript + prompts + tasks + audit) |
| `/memory [texto]` | Busca memorias destiladas entre sesiones |
| `/inspect` | Estado actual del grafo y agentes |
| `/tasks` | Background tasks activas |
| `/task <id>` | Detalle de una task específica |
| `/cancel <id>` | Cancela una background task |
| `/retryable` | Lista tasks reintentables |
| `/artifact` | Artifact completo de la sesión |
| `/tools` | Catálogo de tools con riesgo y modo de permiso |
| `/tool <name> [args]` | Preview HITL de una tool antes de ejecutarla |
| `/impact <name> [args]` | Impacto estimado (archivos afectados, diff, side effects) |
| `/context [agente]` | Presupuesto de contexto del turno actual |
| `/bookmarks` | Lista checkpoints de sesión |
| `/bookmark [nombre]` | Guarda un checkpoint de sesión |
| `/checkpoint <id>` | Consulta un checkpoint guardado |
| `/prompts` | Snapshots de prompts versionados |
| `/prompt <agente>` | Snapshot del prompt de un agente específico |
| `/status` / `/state` | Estado resumido de la sesión |

## Observabilidad

**AgentDoG guardrail**: chequeo post-ejecución sobre la trayectoria (pares acción/observación de `AIMessage.tool_calls` + `ToolMessage`). `code_node` y `web_scraping_node` son `HIGH_RISK_NODES`. Emite eventos JSONL a `AGENTDOG_AUDIT_LOG` o stdout.

**Dashboard visual**:
```bash
python ops/dashboard.py [audit.jsonl]   # genera PNG del audit log
python ops/analytics.py [audit.jsonl]   # strategy ranking + learning curve (--train para regresión logística)
```

**Análisis con DuckDB** (requiere `pip install duckdb`):
```bash
duckdb -c ".read analytics/queries.sql"
```
Queries incluidas: debugging (score < 0), ranking por `(category, strategy)`, malas decisiones del sistema, comparativa API vs scraping, learning curve, counterfactual insight.

**Memory distillation**: al salir del REPL, `infra/memory.py` destila la sesión en `sessions/{id}/MEMORY.md` y lo inyecta como `SystemMessage` al inicio de la siguiente sesión.

## Documentación

| Archivo | Contenido |
|---|---|
| `docs/ARCHITECTURE.md` | Mapa de capas y responsabilidades |
| `docs/RELEASE_NOTES.md` | Historial de releases del refactor arquitectónico |
| `docs/GUIA_EDUCATIVA.md` | Guía conceptual paso a paso |
| `docs/CODIGO_PASO_A_PASO.md` | Recorrido didáctico del código |
| `docs/DIAGRAMA_FLUJO.md` | Visualización del flujo entre agentes |
| `docs/DIAGRAMA_EJECUCION.md` | Flujo de ejecución del sistema |
| `docs/PLAN_CLASE.md` | Plan de clase de 90 minutos |

## Ejemplos de uso

```
"Calcula la raíz cuadrada de 144"
"Analiza un dataset de ventas del Q3"
"Escribe una función Python para calcular factoriales"
"Extrae el precio actual de BTC"
"Scrapea esta página y dame un resumen"
```

## Estado

- **399 tests passing**
- **Sin warnings conocidos**
