"""Decisión pura de routing del supervisor."""

from typing import Iterable

from core.domain.models import AgentState


def decide_agent_route(state: AgentState, valid_agents: Iterable[str]) -> str:
    if not state.get("messages"):
        return "__end__"

    next_agent = state.get("next_agent")
    if next_agent in set(valid_agents):
        return next_agent
    if next_agent == "__end__":
        return "__end__"
    if next_agent == "__error__":
        return "__end__"
    return "supervisor"


__all__ = ["decide_agent_route"]
