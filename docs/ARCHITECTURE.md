# Arquitectura actual

Este proyecto está evolucionando hacia una **arquitectura hexagonal pragmática**:

- **Dominio**: modelos y reglas puras.
- **Aplicación**: casos de uso que coordinan flujos.
- **Puertos**: contratos abstractos para dependencias externas.
- **Adaptadores**: implementación concreta de LLM, HITL, scraping y grafo.
- **Orquestación**: LangGraph como borde de composición.

## Capas actuales

### `core/domain/`
Fuente de verdad de los modelos compartidos.

- `core/domain/models.py`
  - `AgentName`
  - `RoutingDecision`
  - `AgentState`

### `core/ports/`
Contratos puros para desacoplar el dominio de la infraestructura.

- `core/ports/confirmation_port.py` — confirmación humana
- `core/ports/llm_port.py` — fábrica de LLM

### `features/*/application/`
Casos de uso concretos del sistema, agrupados por slice.

- `features/security/application/input_guard_flow.py` — genera `request_id` y ejecuta el guard
- `features/security/application/guard_decision.py` — decisión pura después del guard
- `features/supervisor/application/routing_decision.py` — decisión pura de routing
- `features/supervisor/application/supervisor_chain.py` — construye el chain estructurado
- `features/supervisor/application/supervisor_shortcuts.py` — atajos como el fast-path BTC
- `features/supervisor/application/supervisor_routing.py` — ejecuta el routing del supervisor
- `features/web_scraping/application/flow.py` — coordina scraping, retry, resumen, guardrails y HITL

### `application/services/`
- `agents_factory.py` — construye los agentes ReAct especializados
- `session_gateway.py` — maneja sesiones, persistence y LaneQueue
- `session_persistence.py` — frontera de aplicación para persistencia de sesiones
- `session_memory.py` — frontera de aplicación para carga/destilación de memoria
- `runtime.py` — orquestador de alto nivel por conversación (selección de sesión, snapshots, turnos y cierre con continuidad)
- `tool_registry.py` — catálogo unificado de tools y asignación por agente
- `tool_execution.py` — contrato de ejecución de tools registradas

### `application/policies/`
- `security_flow.py` — middleware de seguridad de entrada
- `hitl_flow.py` — confirmación humana
- `tool_permissions.py` — políticas de permiso para ejecución de tools

### `core/helpers/`
- `message_flow_helpers.py` — helpers compartidos para extraer texto de mensajes
- `trace_flow_helpers.py` — helpers compartidos de trazabilidad (`request_id`)
- `audit_flow_helpers.py` — métricas, follow-up y truncado de observabilidad
- `persistence_flow_helpers.py` — serialización y fallback JSONL de sesiones
- `scraping_flow_helpers.py` — validación, cache y parseo de scraping
- `config_flow_helpers.py` — validación de env y fábrica de LLM
- `security_flow_helpers.py` — patrones y parsing de seguridad
- `text_truncation.py` — truncado compartido
- `generic_node_factory.py` — factory compartida para nodos especializados

### `application/composition/graph.py`
Borde de orquestación y composition root.

Responsabilidades:
- construir el `StateGraph`
- registrar nodos
- definir `route_after_guard` y `route_agent`
- mantener el flujo `input_guard → supervisor → agente`
- traducir decisiones de aplicación a `END` y nodos concretos

## Flujo de ejecución

1. `input_guard_node` delega en `features/security/application/input_guard_flow.py`.
2. `route_after_guard` delega en `features/security/application/guard_decision.py`.
3. `supervisor_node` delega en `features/supervisor/application/supervisor_routing.py`, que usa `supervisor_chain.py` y `supervisor_shortcuts.py`.
4. `route_agent` delega en `features/supervisor/application/routing_decision.py` y traduce `__end__` a `END`.
5. El nodo especializado ejecuta su caso de uso.
6. `features/web_scraping/application/flow.py` y `core/helpers/generic_node_factory.py` aplican guardrails, HITL, retry y postcondiciones.

## Estado de migración

La migración no está completa todavía, pero ya existen fronteras claras:

- `core/domain/` para tipos puros.
- `core/ports/` para contratos.
- `features/*/application/` para coordinación del negocio y flujos por slice.
- `features/*/infrastructure/` para tools especializadas por feature (con `tools/` solo como compatibilidad temporal).
- `features/price/application/price_flow_helpers.py` para el fast path de precios cripto.
- `core/helpers/security_flow_helpers.py` para patrones y parsing de seguridad.
- `core/helpers/audit_flow_helpers.py` para métricas, follow-up y truncado de observabilidad.
- `core/helpers/persistence_flow_helpers.py` para serialización y fallback JSONL de sesiones.
- `core/helpers/scraping_flow_helpers.py` para validación, cache y parseo de scraping.
- `core/helpers/config_flow_helpers.py` para validación de env y fábrica de LLM.
- `application/policies/security_flow.py` para el middleware de seguridad de entrada.
- `application/policies/hitl_flow.py` para HITL y confirmación humana.
- `features/*/infrastructure/` y `application/composition/graph.py` como adaptadores/orquestación.
- `application/services/prompt_loader.py` para cargar/cached prompts por agente.
- `application/services/prompt_assembly.py` para componer prompt base + tools + permisos.
- `application/services/trace_context.py` para request_id/trace_id de sesión y turnos.
- `application/services/supervisor_prompt.py` para componer el prompt del supervisor.
- `application/services/tool_audit.py` para el audit trail de invocaciones de tools.
- `application/services/tool_audit_store.py` para persistir y consultar eventos de audit trail por sesión.
- `application/services/session_artifacts.py` para consolidar transcript, memoria y audit trail por sesión.

## Próximos pasos sugeridos

1. Extraer más casos de uso cuando aparezcan oportunidades claras.
2. Mover lógica repetida desde adaptadores hacia `application/` o `domain/`.
3. Mantener `application/composition/graph.py` como composition root, sin negocio adentro.
4. Usar `application/services/runtime.py` como frontera del CLI para mantener la orquestación fuera del entrypoint.
5. En la fase 1.1, mover el contexto explícito del turno a `RuntimeTurnContext` para que sesión, snapshot y request_id viajen juntos.
6. En la fase 4.1, `resolve_session()` unifica overview + turn context para que el CLI no arme contexto a mano.
7. En la fase 4.2, `prepare_turn()` ya no existe: todo el flujo de turnos pasa por `resolve_session(session_id, message)`.
8. En la fase 4.3, `build_session_view()` entrega el banner y prompt hint del CLI como un objeto único.
9. En la fase 4.4, `select_session_id()` encapsula la validación del ID elegido por el usuario.
10. En la fase 4.5, `SessionLifecycle` unifica selección, vista, resolución de turnos y cierre bajo una sola frontera.
11. En la fase 4.6, `features/sessions/application/session_persistence.py` encapsula el backend concreto para que runtime y gateway no dependan de la infraestructura directamente.
12. En la fase 4.7, `features/sessions/application/session_memory.py` encapsula carga y destilación de memoria para que gateway no dependa de la infraestructura directamente.
13. En la fase 4.8, `application/services/prompt_loader.py` y `application/services/prompt_assembly.py` separan carga del prompt base y composición del contexto extra.
14. En la fase 4.9, `application/services/trace_context.py` centraliza request_id/trace_id para turnos y cierres.
15. En la fase 4.10, `application/services/supervisor_prompt.py` centraliza el prompt del supervisor y saca la composición del use case.
16. En la fase 4.11, `application/services/tool_audit.py` registra eventos de invocación, decisión y cierre de tool calls.
17. En la fase 4.12, `application/services/tool_audit_store.py` persiste y consulta eventos de audit trail por sesión.
18. En la fase 4.13, `application/services/session_artifacts.py` consolida transcript, memoria y audit trail por sesión en un artefacto único.
19. En la fase 1.2, `request_id` se genera una vez en el runtime y se propaga al gateway, persistencia y guardrails como correlación de extremo a extremo.
20. En la fase 2, `application/services/tool_registry.py` centraliza tools por agente para evitar imports y wiring dispersos.
21. En la fase 3, `application/policies/tool_permissions.py` y `application/services/tool_execution.py` separan decisión de permiso y ejecución real.
22. En la fase 4, el runtime debe exponer continuidad de sesión explícita: snapshot antes/después del cierre, persistencia de mensajes y memoria destilada.
