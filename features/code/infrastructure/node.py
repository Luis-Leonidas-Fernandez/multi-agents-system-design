"""Feature-level node adapter for the code agent."""

from typing import Any, Awaitable, Callable

from application.policies.agentdog import _should_evaluate_guard, evaluate_trajectory_safe
from core.domain.models import AgentState
from core.helpers.generic_node_factory import make_generic_agent_node


def make_code_node(agent) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Retorna code_node con el agente inyectado como closure."""

    return make_generic_agent_node(
        agent,
        node_name="code_node",
        agent_name="code_agent",
        tags=("code", "agent", "high_risk"),
        blocked_reason="agentdog",
        should_evaluate_guard_fn=_should_evaluate_guard,
        evaluate_trajectory_safe_fn=evaluate_trajectory_safe,
    )


__all__ = ["make_code_node"]
