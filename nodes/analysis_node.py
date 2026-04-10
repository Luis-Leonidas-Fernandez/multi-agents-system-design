"""
Nodo del agente de análisis.
Factory pattern: make_analysis_node(agent) → async analysis_node(state).
"""
from typing import Callable, Awaitable, Any

from nodes.generic_node import make_generic_agent_node
from domain.models import AgentState


def make_analysis_node(agent) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Retorna analysis_node con el agente inyectado como closure."""

    return make_generic_agent_node(
        agent,
        node_name="analysis_node",
        agent_name="analysis_agent",
        tags=("analysis", "agent"),
    )


__all__ = ["make_analysis_node"]
