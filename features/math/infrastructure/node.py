"""Feature-level node adapter for the math agent."""

from typing import Any, Awaitable, Callable

from core.helpers.generic_node_factory import make_generic_agent_node
from core.domain.models import AgentState


def make_math_node(agent) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Retorna math_node con el agente inyectado como closure."""

    return make_generic_agent_node(
        agent,
        node_name="math_node",
        agent_name="math_agent",
        tags=("math", "agent"),
    )


__all__ = ["make_math_node"]
