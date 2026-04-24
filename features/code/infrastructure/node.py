"""Feature-level node adapter for the code agent."""

from typing import Any, Awaitable, Callable

import application.policies.hitl_flow as hitl_flow
from core.domain.models import AgentState
from core.helpers.generic_node_factory import make_generic_agent_node


def make_code_node(agent) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Retorna code_node con el agente inyectado como closure."""

    return make_generic_agent_node(
        agent,
        node_name="code_node",
        agent_name="code_agent",
        tags=("code", "agent", "high_risk"),
        hitl_prompt_label="code_agent",
        confirmation_handler=hitl_flow.get_confirmation_handler() if hitl_flow.HITL_ENABLED else None,
        blocked_reason="agentdog",
    )


__all__ = ["make_code_node"]
