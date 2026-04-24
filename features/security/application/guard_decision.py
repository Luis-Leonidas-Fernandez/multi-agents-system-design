"""Decisión pura después del guard de entrada."""
from typing import Any, Mapping


def decide_after_guard(state: Mapping[str, Any]) -> str:
    if state.get("blocked", False):
        return "__end__"
    return "supervisor"


__all__ = ["decide_after_guard"]
