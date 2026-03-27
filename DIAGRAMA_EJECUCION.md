# Diagrama de Ejecución del Sistema Multi-Agentes

```mermaid
flowchart TD
    USER(["👤 Usuario\nescribe pregunta"])
    REPL["main.py\ngraph.ainvoke(state)"]
    SUP["supervisor_node()\n• BTC shortcut check\n• LLM elige agente\n→ next_agent = X"]
    ROUTER{"route_agent()\nnext_agent?"}

    MATH_N["math_node()\nmatch_agent.ainvoke()"]
    ANA_N["analysis_node()\nanalysis_agent.ainvoke()"]
    CODE_N["code_node()\ncode_agent.ainvoke()"]
    WEB_N["web_scraping_node()\nweb_scraping_agent.ainvoke()"]

    GUARD_M{"AgentDoG\nguardrail\n(math_node)"}
    GUARD_A{"AgentDoG\nguardrail\n(analysis_node)"}
    GUARD_C{"AgentDoG\nguardrail\n⚠️ HIGH RISK"}
    GUARD_W{"AgentDoG\nguardrail\n⚠️ HIGH RISK"}

    BLOCKED["AIMessage:\n'Respuesta retenida\npor seguridad'"]
    END_NODE(["END\n→ state.messages[-1]"])

    %% Tools
    CALC["🔧 calculate()\neval() seguro"]
    ANALYZE["🔧 analyze_data()"]
    WRITECODE["🔧 write_code()"]
    SCRAPE_S["🔧 scrape_website_simple()\nrequests + BS4"]
    SCRAPE_D["🔧 scrape_website_dynamic()\nPlaywright sync\ncaché 60s"]
    SCRAPE_J["🔧 scrape_website_with_json_capture()\nPlaywright async\n→ data_trading/*.json"]
    EXTRACT_P["🔧 extract_price_from_text()"]

    USER --> REPL --> SUP --> ROUTER

    ROUTER -->|math_agent| MATH_N
    ROUTER -->|analysis_agent| ANA_N
    ROUTER -->|code_agent| CODE_N
    ROUTER -->|web_scraping_agent| WEB_N

    MATH_N --> CALC --> MATH_N
    MATH_N --> GUARD_M
    GUARD_M -->|safe| END_NODE
    GUARD_M -->|unsafe| BLOCKED --> END_NODE

    ANA_N --> ANALYZE --> ANA_N
    ANA_N --> GUARD_A
    GUARD_A -->|safe| END_NODE
    GUARD_A -->|unsafe| BLOCKED

    CODE_N --> WRITECODE --> CODE_N
    CODE_N --> GUARD_C
    GUARD_C -->|safe| END_NODE
    GUARD_C -->|unsafe| BLOCKED

    WEB_N --> SCRAPE_S & SCRAPE_D & SCRAPE_J & EXTRACT_P --> WEB_N
    WEB_N --> GUARD_W
    GUARD_W -->|safe| END_NODE
    GUARD_W -->|unsafe| BLOCKED

    classDef highRisk fill:#ff9999,stroke:#cc0000
    classDef tool fill:#d4edda,stroke:#28a745
    classDef guard fill:#fff3cd,stroke:#ffc107
    class CODE_N,WEB_N highRisk
    class CALC,ANALYZE,WRITECODE,SCRAPE_S,SCRAPE_D,SCRAPE_J,EXTRACT_P tool
    class GUARD_M,GUARD_A,GUARD_C,GUARD_W guard
```

## Notas

- **Rojo**: nodos `HIGH_RISK` — `code_node` y `web_scraping_node` siempre son evaluados por AgentDoG
- **Amarillo**: guardrail AgentDoG — evalúa la trayectoria completa (tool_calls + observaciones) post-ejecución
- **Verde**: herramientas disponibles para cada agente
- Los agentes van directo a `END`, no regresan al supervisor
- El supervisor tiene un shortcut para preguntas de precio BTC que bypasea el LLM
- `scrape_website_with_json_capture` guarda automáticamente en `data_trading/`
