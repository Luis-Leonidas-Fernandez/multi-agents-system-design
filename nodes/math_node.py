"""
Nodo del agente de matemáticas.
Factory pattern: make_math_node(agent) → async math_node(state).
"""
from typing import Callable, Awaitable, Any

from nodes.generic_node import make_generic_agent_node
from domain.models import AgentState


def make_math_node(agent) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Retorna math_node con el agente inyectado como closure."""

    return make_generic_agent_node(
        agent,
        node_name="math_node",
        agent_name="math_agent",
        tags=("math", "agent"),
    )


__all__ = ["make_math_node"]
