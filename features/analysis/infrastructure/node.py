"""Feature-level node adapter for the analysis agent."""

from typing import Any, Awaitable, Callable

from core.helpers.generic_node_factory import make_generic_agent_node
from core.domain.models import AgentState


def make_analysis_node(agent) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Retorna analysis_node con el agente inyectado como closure."""

    return make_generic_agent_node(
        agent,
        node_name="analysis_node",
        agent_name="analysis_agent",
        tags=("analysis", "agent"),
    )


__all__ = ["make_analysis_node"]
