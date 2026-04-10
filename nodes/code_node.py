"""
Nodo del agente de código con HITL pre-ejecución.
Factory pattern: make_code_node(agent) → async code_node(state).
"""
from typing import Callable, Awaitable, Any

from nodes.generic_node import make_generic_agent_node
import application.policies.hitl_flow as hitl_flow
from domain.models import AgentState


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
