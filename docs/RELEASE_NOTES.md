# Release Notes

## 2026-04 — Refactor arquitectónico y cobertura de tests

### Cambios principales
- Centralización de agentes en `application/services/agent_registry.py`.
- Deduplicación de nodos especializados con `nodes/generic_node.py`.
- Construcción del chain del supervisor en `application/use_cases/supervisor_chain.py`.
- División fina del supervisor en `application/use_cases/supervisor_chain.py`, `supervisor_routing.py` y `supervisor_shortcuts.py`.
- Centralización de factories ReAct en `application/services/agents_factory.py` con `_build_specialized_agent`.
- Extracción de tools a `tools/` para código, datos y web scraping.
- Extracción de helpers de precio a `application/helpers/price_flow_helpers.py`.
- Extracción de helpers de seguridad a `application/policies/security_flow_helpers.py`.
- Extracción de helpers de audit a `application/helpers/audit_flow_helpers.py`.
- Extracción de helpers de persistencia a `application/helpers/persistence_flow_helpers.py`.
- Extracción de helpers de scraping a `application/helpers/scraping_flow_helpers.py`.
- Extracción de helpers de config a `application/helpers/config_flow_helpers.py`.
- Extracción del middleware de entrada a `application/policies/security_flow.py`.
- Extracción de HITL a `application/policies/hitl_flow.py`.
- Separación de `MODEL_PRICING` en `domain/model_pricing.py`.
- Temperatura configurable por agente desde el registry.
- Seguridad extensible en runtime para patrones de bloqueo y señales de riesgo.
- Truncado compartido en `application/helpers/text_truncation.py`.
- Abstracción de HITL con `ConfirmationHandler` en `application/policies/hitl_flow.py`.

### Cobertura agregada
- Integración del grafo supervisor.
- Validación crítica de HITL en `code_node`.
- Context quarantine y auto-retry en `web_scraping_node`.

### Estado actual
- Suite verificada: **66 tests passing**
- Warning conocido: deprecación de `create_react_agent` en LangGraph

### Nota
La migración de `create_react_agent` quedó pendiente por política del repositorio hasta tener un reemplazo drop-in equivalente.
