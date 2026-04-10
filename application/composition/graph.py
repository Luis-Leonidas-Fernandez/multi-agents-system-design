"""
Construcción del grafo supervisor multi-agentes.

Instancia los agentes, ensambla los nodos y retorna el grafo compilado.
Este módulo es el único punto donde se crean los agentes y se cablean
al grafo LangGraph — los módulos de nodos son puros (sin estado global).
"""
import uuid
from typing import Any, cast

from langgraph.graph import StateGraph, END

from application.helpers.config_flow_helpers import get_llm
from application.services.agent_registry import AGENT_NAMES, get_registered_nodes
from application.services.coordinator_mode import is_coordinator_mode_enabled
from application.services.coordinator_workers import coordinator_runtime_service
from application.use_cases.supervisor_chain import build_supervisor_chain
from domain.models import AgentState
from application.policies.security_flow import input_guard



# ==================== SUPERVISOR — recursos cacheados ====================
# El chain se construye una sola vez en el primer turno (lazy init) y se
# reutiliza en turnos posteriores.
_supervisor_chain = None  # construido en el primer turno via _get_supervisor_chain()


def _get_supervisor_chain():
    global _supervisor_chain
    if _supervisor_chain is None:
        _supervisor_chain = build_supervisor_chain(get_llm)
    return _supervisor_chain


# ==================== SUPERVISOR NODE ====================

async def supervisor_node(state: AgentState) -> dict[str, Any]:
    """
    Nodo supervisor que decide qué agente usar (async).

    Usa structured output (Pydantic) para garantizar que el LLM
    devuelva siempre un agente válido sin necesidad de text-parsing.
    """
    from application.use_cases.supervisor_routing import run_supervisor_routing

    def chain_factory():
        chain = _get_supervisor_chain()
        return chain

    return await run_supervisor_routing(state, chain_factory)


async def coordinator_node(state: AgentState) -> dict[str, Any]:
    """Nodo coordinador que reutiliza la misma ruta base, pero queda como
    boundary explícita para evolucionar hacia spawn/messaging dinámico."""
    route = await supervisor_node(state)
    next_agent = route.get("next_agent", "")
    session_id = str(state.get("session_id") or "")
    last_message = str(state.get("messages", [])[-1].content if state.get("messages") else "")
    if session_id and next_agent in AGENT_NAMES:
        if next_agent == "web_scraping_agent":
            probe_result = await coordinator_runtime_service.execute_parallel_probe_round(
                session_id,
                "general",
                last_message,
                sender="coordinator",
            )
            route["coordinator_worker_id"] = ",".join(probe_result.get("worker_ids", [])) or ""
            route["coordinator_worker_agent"] = next_agent
            route["coordinator_probe_best_source"] = probe_result.get("best_source", "")
            route["coordinator_probe_sources"] = [result.get("source_name", "") for result in probe_result.get("probe_results", [])]
            route["messages"] = list(state.get("messages", []))
            if probe_result.get("response"):
                from langchain_core.messages import AIMessage

                route["messages"] = list(state.get("messages", [])) + [AIMessage(content=str(probe_result["response"]))]
            route["next_agent"] = "__end__"
            return route

        worker = await coordinator_runtime_service.spawn_worker(
            session_id,
            next_agent,
            worker_name=f"{next_agent}:{state.get('request_id', '')[:8]}" if state.get("request_id") else None,
            metadata={
                "request_id": state.get("request_id", ""),
                "reason": "coordinator_spawn",
                "target_agent": next_agent,
            },
        )
        worker_result = await coordinator_runtime_service.execute_worker_turn(
            session_id,
            worker.worker_id,
            last_message,
            sender="coordinator",
        )
        route["coordinator_worker_id"] = worker.worker_id
        route["coordinator_worker_agent"] = worker.agent_name
        route["messages"] = state.get("messages", []) + []
        if worker_result.get("response"):
            from langchain_core.messages import AIMessage

            route["messages"] = list(state.get("messages", [])) + [AIMessage(content=str(worker_result["response"]))]
        route["next_agent"] = "__end__"
    return route


# ==================== ROUTER ====================

def route_agent(state: dict[str, Any]) -> str:
    """Enruta al nodo correcto según state['next_agent']."""
    from application.use_cases.routing_decision import decide_agent_route

    decision = decide_agent_route(cast(AgentState, state), AGENT_NAMES)
    return END if decision == "__end__" else decision


def _entry_node_name() -> str:
    return "coordinator" if is_coordinator_mode_enabled() else "supervisor"


# ==================== INPUT GUARD NODE ====================

async def input_guard_node(state: AgentState) -> dict[str, Any]:
    """Genera request_id del turno y aplica el middleware de seguridad."""
    from application.use_cases.input_guard_flow import run_input_guard

    return await run_input_guard(state, input_guard, lambda: str(uuid.uuid4()))


def route_after_guard(state: dict[str, Any]) -> str:
    """Usa state['blocked'] en lugar de comparar el contenido del mensaje."""
    from application.use_cases.guard_decision import decide_after_guard

    decision = decide_after_guard(state)
    if decision == "__end__":
        return END
    return _entry_node_name()


# ==================== CONSTRUCCIÓN DEL GRAFO ====================

def create_supervisor_graph():
    """
    Crea y retorna el grafo supervisor compilado.

    Flujo:
      input_guard → coordinator/supervisor → route_agent → [agente especializado] → END
    """
    workflow = StateGraph(AgentState)
    registered_nodes = get_registered_nodes()

    workflow.add_node("input_guard",       cast(Any, input_guard_node))
    workflow.add_node("supervisor",        cast(Any, supervisor_node))
    workflow.add_node("coordinator",       cast(Any, coordinator_node))
    for agent_name in AGENT_NAMES:
        workflow.add_node(agent_name, cast(Any, registered_nodes[agent_name]))

    workflow.set_entry_point("input_guard")
    workflow.add_conditional_edges(
        "input_guard",
        route_after_guard,
        {"supervisor": "supervisor", "coordinator": "coordinator", END: END},
    )
    for entry_node in ("supervisor", "coordinator"):
        workflow.add_conditional_edges(
            entry_node,
            route_agent,
            {**{agent_name: agent_name for agent_name in AGENT_NAMES}, entry_node: entry_node, END: END},
        )

    for agent_name in AGENT_NAMES:
        workflow.add_edge(agent_name, END)

    return workflow.compile()


__all__ = [
    "create_supervisor_graph",
    "input_guard_node",
    "route_after_guard",
    "route_agent",
    "supervisor_node",
]
