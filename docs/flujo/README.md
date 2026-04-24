# Flujo visual: precio actual de Bitcoin

## Recorrido alto nivel

```mermaid
flowchart TD
  U[Usuario: "Dame el precio actual de Bitcoin"] --> M[main.py]
  M --> G[features.sessions.application.session_gateway.AgentGateway]
  G --> P1[features.sessions.*]
  G --> C[application.composition.graph.create_supervisor_graph]
  C --> IG[input_guard_node]
  IG --> S[application.policies.security_flow.input_guard]
  S --> D[features.security.application.guard_decision]
  D --> R[features.supervisor.application.supervisor_routing]
  R --> SC[features.supervisor.application.supervisor_shortcuts]
  SC -->|BTC detectado| WN[features.web_scraping.infrastructure.node]
  WN --> HITL[application.policies.hitl_flow]
  WN --> WF[features.web_scraping.application.flow]
  WF --> AG[application.policies.agentdog]
  WF --> ST[application.policies.scrape_tracker]
  WF --> H[core.helpers.*]
  WF --> T[features/web_scraping/infrastructure/*]
  WF --> A[Respuesta final]
  A --> G
  G --> P2[persistence + memory distillation]
```

## Secuencia detallada

```mermaid
sequenceDiagram
  autonumber
  participant U as Usuario
  participant M as main.py
  participant G as AgentGateway
  participant GR as graph.py
  participant IG as input_guard
  participant S as security_flow
  participant D as guard_decision
  participant SR as supervisor_routing
  participant SS as supervisor_shortcuts
  participant WN as web_scraping_node
  participant WF as web_scraping_flow
  participant AD as AgentDoG
  participant ST as scrape_tracker
  participant T as tools

  U->>M: Pide precio actual de Bitcoin
  M->>G: send(session_id, mensaje)
  G->>G: carga memoria + persistence
  G->>GR: create_supervisor_graph()
  GR->>IG: input_guard_node()
  IG->>S: ejecuta guard
  S-->>IG: permitido
  IG->>D: decide_after_guard()
  D-->>IG: supervisor
  IG->>SR: run_supervisor_routing()
  SR->>SS: detecta BTC
  SS-->>SR: web_scraping_agent
  SR->>WN: ejecuta nodo
  WN->>WF: run_web_scraping_flow()
  WF->>AD: evalúa seguridad/trajectoria
  WF->>ST: actualiza estrategia y tracking
  WF->>T: scrape / price helpers
  WF-->>WN: respuesta final
  WN-->>G: AIMessage final
  G-->>M: devuelve texto
  M-->>U: muestra precio
```

## Áreas que intervienen

- **Entrada/CLI**: `main.py`
- **Sesiones + persistencia**: `features.sessions.application.session_gateway`, `features.sessions.infrastructure.persistence`, `features.sessions.infrastructure.memory`
- **Composición**: `application.composition.graph`
- **Seguridad de entrada**: `application.policies.security_flow`
- **Decisión de flujo**: `features.security.application.guard_decision`, `features.supervisor.application.routing_decision`, `features.supervisor.application.supervisor_shortcuts`
- **Supervisor**: `features.supervisor.application.supervisor_chain`, `features.supervisor.application.supervisor_routing`
- **Scraping**: `features.web_scraping.infrastructure.node`, `features.web_scraping.application.flow`
- **Guardrails**: `application.policies.agentdog`, `application.policies.scrape_tracker`, `application.policies.hitl_flow`
- **Helpers compartidos**: `core.helpers.*`
- **Herramientas**: `features/*/infrastructure/*`

## Qué pasa en este caso

La clave es que la frase contiene **Bitcoin/BTC**, entonces el supervisor activa el **fast path** y deriva directamente a `web_scraping_agent`, evitando routing innecesario. Después, el flujo de scraping consulta precio público, aplica guardrails y devuelve una respuesta final corta y limpia.
