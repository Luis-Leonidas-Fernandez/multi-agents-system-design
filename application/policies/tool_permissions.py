"""Políticas de permiso para la ejecución de tools."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal


ToolPermissionMode = Literal["allow_all", "confirm_high_risk", "deny_high_risk", "deny_all"]


@dataclass(frozen=True)
class ToolPermissionDecision:
    allowed: bool
    requires_confirmation: bool
    mode: ToolPermissionMode
    reason: str


def get_tool_permission_mode() -> ToolPermissionMode:
    mode = os.getenv("TOOL_PERMISSION_MODE", "confirm_high_risk").strip().lower()
    if mode not in {"allow_all", "confirm_high_risk", "deny_high_risk", "deny_all"}:
        return "confirm_high_risk"
    return mode  # type: ignore[return-value]


def decide_tool_permission(*, risk_level: str, permission_mode: str | None = None) -> ToolPermissionDecision:
    mode = permission_mode or get_tool_permission_mode()
    is_high_risk = risk_level == "high"

    if mode == "deny_all":
        return ToolPermissionDecision(False, False, mode, "deny_all")

    if mode == "deny_high_risk" and is_high_risk:
        return ToolPermissionDecision(False, False, mode, "deny_high_risk")

    if mode == "confirm_high_risk" and is_high_risk:
        return ToolPermissionDecision(True, True, mode, "confirm_high_risk")

    return ToolPermissionDecision(True, False, mode, "allow")


__all__ = ["ToolPermissionDecision", "ToolPermissionMode", "get_tool_permission_mode", "decide_tool_permission"]
