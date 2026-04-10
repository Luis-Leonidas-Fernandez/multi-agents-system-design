# Sistema Multi-Agentes con LangGraph

Sistema multi-agentes implementado con LangGraph siguiendo el patrón supervisor.

## 📋 Estructura

```
06_multi_agents/
├── README.md
├── requirements.txt
├── application/           # Casos de uso y helpers de aplicación
│   ├── use_cases/         # Flujos de aplicación
│   ├── services/          # Registries, factories y gateway
│   ├── helpers/           # Helpers compartidos
│   ├── policies/          # Guardrails, HITL y seguridad
│   └── composition/       # Composition root y wiring
│       └── graph.py       # Grafo supervisor y wiring
├── docs/                  # Documentación larga y material educativo
├── domain/                # Modelos puros del dominio
├── ports/                 # Contratos/puertos
├── tools/                 # Tools reutilizables para agentes
├── ops/                  # Dashboards y scripts de observabilidad
├── main.py                # Punto de entrada
├── tests/                 # Suite de tests y fixtures
└── .env.example           # Ejemplo de variables de entorno
```

## 🚀 Instalación

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Configurar variables de entorno:
```bash
cp .env.example .env
# Editar .env y agregar tu OPENAI_API_KEY
```

3. Ejecutar:
```bash
python main.py
```

## 🏗️ Arquitectura

El sistema utiliza un **patrón supervisor** donde:

- **Supervisor**: Coordina y delega tareas a agentes especializados
- **Agente de Matemáticas**: Resuelve problemas matemáticos
- **Agente de Análisis**: Analiza datos y genera reportes
- **Agente de Código**: Escribe y revisa código
- **Agente de Web Scraping**: Extrae información de páginas web

## 📚 Conceptos Clave

- **StateGraph**: Grafo de estado que define el flujo entre agentes
- **Nodes**: Nodos que representan cada agente
- **Edges**: Conexiones que definen cómo fluye la información
- **State**: Estado compartido entre agentes

## 📖 Guías Educativas

Este proyecto incluye guías completas para enseñanza:

### Para Estudiantes:
- **[GUIA_EDUCATIVA.md](./docs/GUIA_EDUCATIVA.md)**: Guía completa paso a paso con explicaciones teóricas y prácticas
- **[CODIGO_PASO_A_PASO.md](./docs/CODIGO_PASO_A_PASO.md)**: Código comentado línea por línea para seguir durante la implementación
- **[DIAGRAMA_FLUJO.md](./docs/DIAGRAMA_FLUJO.md)**: Visualizaciones y diagramas del flujo del sistema

### Para Instructores:
- **[PLAN_CLASE.md](./docs/PLAN_CLASE.md)**: Plan estructurado para una clase de 90 minutos con objetivos, timing y ejercicios

### Orden Recomendado de Estudio:

1. **Leer** `docs/GUIA_EDUCATIVA.md` para entender los conceptos
2. **Seguir** `docs/CODIGO_PASO_A_PASO.md` para implementar paso a paso
3. **Consultar** `docs/DIAGRAMA_FLUJO.md` para visualizar el flujo
4. **Usar** `docs/PLAN_CLASE.md` si estás enseñando

## 🗂️ Mapa de documentación

- `README.md` — visión general del proyecto y puntos de entrada.
- `docs/ARCHITECTURE.md` — mapa actual de capas y responsabilidades.
- `docs/RELEASE_NOTES.md` — nota de release del refactor arquitectónico.
- `docs/DIAGRAMA_EJECUCION.md` — flujo de ejecución del sistema.
- `docs/DIAGRAMA_FLUJO.md` — visualización del flujo entre agentes.
- `docs/GUIA_EDUCATIVA.md` — explicación paso a paso del sistema.
- `docs/CODIGO_PASO_A_PASO.md` — recorrido didáctico del código.
- `docs/PLAN_CLASE.md` — plan de clase para enseñar el proyecto.

## 🎯 Ejemplos de Uso

- "Calcula la raíz cuadrada de 144"
- "Analiza un dataset de ventas"
- "Escribe una función para calcular factoriales"
- "Extrae información de https://example.com"
- "Scrapea esta página web y dame un resumen"

## 📝 Changelog reciente

### Fase 4 — Refactor arquitectónico completado

- **Registry de agentes**: se centralizó la metadata y el wiring en `application/services/agent_registry.py`.
- **Nodos genéricos**: `math`, `analysis` y `code` comparten `nodes/generic_node.py`.
- **Pricing separado**: `MODEL_PRICING` vive en `domain/model_pricing.py` y lo usan los helpers de audit.
- **Temperatura por agente**: cada `AgentSpec` puede definir su `temperature`.
- **Seguridad configurable**: `_BLOCKED_PATTERNS` y `_RISK_SIGNALS` admiten overrides por entorno sin perder defaults.
- **Truncado compartido**: la lógica común se movió a `application/helpers/text_truncation.py`.
- **HITL abstraído**: `application/policies/hitl_flow.py` introdujo `ConfirmationHandler` para desacoplar la confirmación humana del flujo de los nodos.
- **Supervisor chain en aplicación**: `application/use_cases/supervisor_chain.py` construye el chain estructurado.
- **Supervisor más fino**: la lógica del supervisor quedó dividida en `application/use_cases/supervisor_chain.py`, `supervisor_routing.py` y `supervisor_shortcuts.py`.
- **Web scraping más delgado**: `nodes/web_scraping_node.py` quedó como adaptador fino sobre el caso de uso.
- **Factories de agentes menos repetidas**: `application/services/agents_factory.py` ahora centraliza la construcción ReAct en `_build_specialized_agent`.
- **Tools modularizadas**: `tools/` concentra las tools de código, datos y web.
- **Helpers de precio extraídos**: `application/helpers/price_flow_helpers.py` contiene el fast path cripto.
- **Helpers de seguridad extraídos**: `application/helpers/security_flow_helpers.py` concentra parsing y patrones.
- **Helpers de audit extraídos**: `application/helpers/audit_flow_helpers.py` concentra métricas y truncado.
- **Helpers de persistencia extraídos**: `application/helpers/persistence_flow_helpers.py` concentra serialización y JSONL.
- **Helpers de scraping extraídos**: `application/helpers/scraping_flow_helpers.py` concentra validación, cache y parseo HTML.
- **Helpers de config extraídos**: `application/helpers/config_flow_helpers.py` concentra validación y fábrica LLM.
- **Guard de entrada extraído**: `application/policies/security_flow.py` aloja el middleware de seguridad.
- **HITL extraído**: `application/policies/hitl_flow.py` centraliza confirmación y flag de alto riesgo.

### Fase 5 — Inspección y trazabilidad extendidas

- **Delegación inspectable**: tareas en background con lifecycle persistido y comandos CLI para ver estado y artefactos.
- **Prompt snapshots**: `prompts/{agent}/PROMPT_SNAPSHOT.json` + historial append-only por agente.
- **Artifacts enriquecidos**: `SESSION_ARTIFACT.json` ahora incluye prompt snapshots y sus rutas.

### Fase 6 — Replay unificado

- **Replay CLI**: `/replay` muestra una línea de tiempo unificada de la sesión.
- **Timeline**: combina snapshot, prompts, mensajes, background tasks y audit trail.
- **Objetivo**: facilitar debug y revisión sin saltar entre artifacts separados.

### Fase 7 — Memory retrieval

- **Memory CLI**: `/memory [buscar texto]` busca memorias destiladas entre sesiones.
- **Ranking**: score por coincidencia de términos + bonus de recencia.
- **Listado**: `/memory` sin query muestra sesiones con `MEMORY.md`.

### Fase 8 — Approval UX

- **Tool catalog**: `/tools` muestra tools, riesgo y modo de permiso.
- **Tool preview**: `/tool <name> [json_args|key=value ...]` muestra la previsualización HITL antes de ejecutar.
- **Prompt HITL**: la confirmación humana ahora incluye args y motivo de la política.
- **Objetivo**: reducir decisiones ciegas antes de ejecutar operaciones sensibles.

### Fase 9 — Contexto y checkpoints

- **Context budget**: `/context [agente]` muestra qué entra al contexto, qué viene resumido y qué queda afuera.
- **Bookmarks**: `/bookmarks`, `/bookmark [nombre]` y `/checkpoint <id>` guardan y consultan checkpoints de sesión.
- **Commands registry**: `/commands` lista los slash commands y `/command <nombre>` muestra ayuda detallada; aliases como `/status` y `/state` apuntan a la misma lógica.
- **Objetivo**: hacer visible el estado de la sesión y cortar puntos de reanudación útiles para debug.

### Fase 10 — Impact preview real

- **Impact preview**: `/impact <tool> [args]` estima archivos afectados, tamaño de diff y side effects antes de ejecutar tools de código/web.
- **Repo-aware**: usa archivos reales del repo como evidencia cuando la tarea coincide con módulos/tests existentes.
- **Symbols/imports**: también cruza símbolos reales del código (classes/defs/imports) para elevar la confianza.
- **Approval UX**: `/tool` y el prompt HITL ahora muestran también el impacto estimado.
- **Objetivo**: reemplazar “prompts bonitos” por señales concretas de cambio y riesgo.

### Estado de verificación

- Suite verificada: **66 tests passing**
- Warnings conocidos: deprecación de `create_react_agent` en LangGraph
- No se migró aún ese factory porque el repositorio indica esperar un reemplazo drop-in equivalente

### Release notes

- [RELEASE_NOTES.md](./docs/RELEASE_NOTES.md): nota de release del refactor arquitectónico y la cobertura de tests agregada.

### Cobertura de tests agregada

- **Integración del grafo**: `tests/test_graph_integration.py` valida el cableado `input_guard → supervisor → agente`.
- **HITL crítico**: `tests/test_code_node_hitl.py` asegura que `code_node` cancela antes de ejecutar el agente si el usuario rechaza.
- **Web scraping**: `tests/test_web_scraping_node.py` cubre `web_scraping_agent`, context quarantine y auto-retry cuando el contenido es insuficiente.
