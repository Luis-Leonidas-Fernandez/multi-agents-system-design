"""
Construcción del grafo supervisor multi-agentes.

Instancia los agentes, ensambla los nodos y retorna el grafo compilado.
Este módulo es el único punto donde se crean los agentes y se cablean
al grafo LangGraph — los módulos de nodos son puros (sin estado global).
"""
import uuid

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig

from config import get_llm
from agents import (
    create_math_agent,
    create_analysis_agent,
    create_code_agent,
    create_web_scraping_agent,
)
from state import AgentState, RoutingDecision
from security import input_guard
from nodes import (
    make_math_node,
    make_analysis_node,
    make_code_node,
    make_web_scraping_node,
)


# ==================== INSTANCIAS DE AGENTES (módulo scope) ====================
# Se crean una sola vez al importar graph.py. Los nodos reciben las instancias
# via closure — no hay globals en los módulos de nodes/.

math_agent         = create_math_agent()
analysis_agent     = create_analysis_agent()
code_agent         = create_code_agent()
web_scraping_agent = create_web_scraping_agent()


# ==================== NODOS CONCRETOS ====================

math_node         = make_math_node(math_agent)
analysis_node     = make_analysis_node(analysis_agent)
code_node         = make_code_node(code_agent)
web_scraping_node = make_web_scraping_node(web_scraping_agent, get_llm)


# ==================== SUPERVISOR NODE ====================

async def supervisor_node(state: AgentState) -> AgentState:
    """
    Nodo supervisor que decide qué agente usar (async).

    Usa structured output (Pydantic) para garantizar que el LLM
    devuelva siempre un agente válido sin necesidad de text-parsing.
    """
    messages     = state["messages"]
    last_message = messages[-1].content if messages else ""

    # Shortcut BTC: evita llamada al LLM para consultas comunes de precio.
    lm = last_message.lower()
    if ("bitcoin" in lm or "btc" in lm) and any(
        k in lm for k in ["precio", "price", "cotiza", "cotización", "cotizacion"]
    ):
        return {"next_agent": "web_scraping_agent"}

    llm            = get_llm()
    llm_structured = llm.with_structured_output(RoutingDecision)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Eres un supervisor que coordina un equipo de agentes especializados.\n\n"
            "Tienes acceso a cuatro agentes:\n"
            "- math_agent: problemas matemáticos, cálculos, álgebra, estadística numérica\n"
            "- analysis_agent: análisis de datos, insights, reportes, patrones en datasets\n"
            "- code_agent: escribir código, programación, desarrollo de software\n"
            "- web_scraping_agent: extraer información de URLs, scraping, obtener datos de páginas web\n\n"
            "Elige el agente más adecuado para la solicitud. "
            "Si no estás seguro, elige el que mejor se ajuste."
        )),
        ("user", "{input}"),
    ])

    chain = prompt | llm_structured
    try:
        decision: RoutingDecision = await chain.ainvoke(
            {"input": last_message},
            config=RunnableConfig(
                tags=["supervisor", "routing"],
                metadata={
                    "node":          "supervisor",
                    "input_chars":   len(last_message),
                    "history_turns": len(messages),
                    "risk_flag":     state.get("risk_flag", False),
                },
            ),
        )
        return {"next_agent": decision.agent}
    except Exception as e:
        print(f"[supervisor] routing falló ({type(e).__name__}: {e}), usando fallback math_agent")
        return {"next_agent": "math_agent"}


# ==================== ROUTER ====================

def route_agent(state: AgentState) -> str:
    """Enruta al nodo correcto según state['next_agent']."""
    if not state.get("messages"):
        return END

    next_agent = state.get("next_agent")
    if next_agent == "math_agent":
        return "math_agent"
    if next_agent == "analysis_agent":
        return "analysis_agent"
    if next_agent == "code_agent":
        return "code_agent"
    if next_agent == "web_scraping_agent":
        return "web_scraping_agent"
    return "supervisor"


# ==================== INPUT GUARD NODE ====================

async def input_guard_node(state: AgentState) -> AgentState:
    """Genera request_id del turno y aplica el middleware de seguridad."""
    rid     = str(uuid.uuid4())
    blocked = input_guard(state)
    if blocked:
        return {**blocked, "request_id": rid}
    return {**state, "request_id": rid}


def route_after_guard(state: AgentState) -> str:
    """Usa state['blocked'] en lugar de comparar el contenido del mensaje."""
    if state.get("blocked", False):
        return END
    return "supervisor"


# ==================== CONSTRUCCIÓN DEL GRAFO ====================

def create_supervisor_graph():
    """
    Crea y retorna el grafo supervisor compilado.

    Flujo:
      input_guard → supervisor → route_agent → [agente especializado] → END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("input_guard",       input_guard_node)
    workflow.add_node("supervisor",        supervisor_node)
    workflow.add_node("math_agent",        math_node)
    workflow.add_node("analysis_agent",    analysis_node)
    workflow.add_node("code_agent",        code_node)
    workflow.add_node("web_scraping_agent", web_scraping_node)

    workflow.set_entry_point("input_guard")
    workflow.add_conditional_edges(
        "input_guard",
        route_after_guard,
        {"supervisor": "supervisor", END: END},
    )
    workflow.add_conditional_edges(
        "supervisor",
        route_agent,
        {
            "math_agent":         "math_agent",
            "analysis_agent":     "analysis_agent",
            "code_agent":         "code_agent",
            "web_scraping_agent": "web_scraping_agent",
            END:                  END,
        },
    )

    workflow.add_edge("math_agent",         END)
    workflow.add_edge("analysis_agent",     END)
    workflow.add_edge("code_agent",         END)
    workflow.add_edge("web_scraping_agent", END)

    return workflow.compile()


__all__ = ["create_supervisor_graph"]
