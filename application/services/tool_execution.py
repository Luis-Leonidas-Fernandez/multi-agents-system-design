"""Contrato de ejecución para tools registradas.

Este módulo separa la decisión de permiso de la invocación real. La idea es
tener una seam explícita para enforcement futuro sin depender de los agentes
o del runtime principal.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Awaitable, Callable, Mapping, Optional

from application.policies.tool_permissions import ToolPermissionDecision, decide_tool_permission
from application.services.tool_approval import tool_approval_service
from application.services.tool_audit import tool_audit_service
from application.services.tool_registry import get_tool_spec


@dataclass(frozen=True)
class ToolExecutionContext:
    request_id: str
    agent_name: str
    tool_name: str
    arguments: Mapping[str, Any]
    session_id: str | None = None
    trace_id: str | None = None


@dataclass(frozen=True)
class ToolExecutionResult:
    allowed: bool
    request_id: str
    agent_name: str
    tool_name: str
    output: Any
    decision: ToolPermissionDecision
    duration_ms: int


async def execute_registered_tool(
    context: ToolExecutionContext,
    confirm_fn: Optional[Callable[[str], Awaitable[bool]]] = None,
) -> ToolExecutionResult:
    started_at = time.monotonic()
    spec = get_tool_spec(context.tool_name)
    preview = tool_approval_service.build_preview(
        agent_name=context.agent_name,
        tool_name=context.tool_name,
        arguments=context.arguments,
        spec=spec,
    )
    tool_audit_service.requested(
        request_id=context.request_id,
        agent_name=context.agent_name,
        tool_name=context.tool_name,
        session_id=context.session_id,
        trace_id=context.trace_id,
        arguments=dict(context.arguments),
    )
    decision = decide_tool_permission(risk_level=spec.risk_level, permission_mode=spec.permission_mode)
    tool_audit_service.decided(
        request_id=context.request_id,
        agent_name=context.agent_name,
        tool_name=context.tool_name,
        outcome="blocked" if not decision.allowed else ("needs_confirmation" if decision.requires_confirmation else "requested"),
        risk_level=spec.risk_level,
        permission_mode=decision.mode,
        allowed=decision.allowed,
        requires_confirmation=decision.requires_confirmation,
        decision_reason=decision.reason,
        session_id=context.session_id,
        trace_id=context.trace_id,
    )

    if not decision.allowed:
        duration_ms = int((time.monotonic() - started_at) * 1000)
        tool_audit_service.completed(
            request_id=context.request_id,
            agent_name=context.agent_name,
            tool_name=context.tool_name,
            outcome="blocked",
            duration_ms=duration_ms,
            session_id=context.session_id,
            trace_id=context.trace_id,
            output_preview=f"Tool bloqueada por política: {decision.reason}",
        )
        return ToolExecutionResult(
            allowed=False,
            request_id=context.request_id,
            agent_name=context.agent_name,
            tool_name=context.tool_name,
            output=f"Tool bloqueada por política: {decision.reason}",
            decision=decision,
            duration_ms=duration_ms,
        )

    if decision.requires_confirmation:
        confirmed = False
        if confirm_fn is not None:
            confirmed = await confirm_fn(preview.confirmation_prompt or f"[HITL] {context.agent_name} quiere usar {context.tool_name}. ¿Confirmar? [s/n]: ")
        if not confirmed:
            duration_ms = int((time.monotonic() - started_at) * 1000)
            tool_audit_service.completed(
                request_id=context.request_id,
                agent_name=context.agent_name,
                tool_name=context.tool_name,
                outcome="cancelled",
                duration_ms=duration_ms,
                session_id=context.session_id,
                trace_id=context.trace_id,
                output_preview="Tool cancelada por el usuario.",
            )
            return ToolExecutionResult(
                allowed=False,
                request_id=context.request_id,
                agent_name=context.agent_name,
                tool_name=context.tool_name,
                output="Tool cancelada por el usuario.",
                decision=decision,
                duration_ms=duration_ms,
            )

    tool = spec.tool
    try:
        if hasattr(tool, "ainvoke"):
            output = await tool.ainvoke(dict(context.arguments))
        else:
            output = tool.invoke(dict(context.arguments))
        duration_ms = int((time.monotonic() - started_at) * 1000)
        tool_audit_service.completed(
            request_id=context.request_id,
            agent_name=context.agent_name,
            tool_name=context.tool_name,
            outcome="success",
            duration_ms=duration_ms,
            session_id=context.session_id,
            trace_id=context.trace_id,
            output_preview=str(output)[:250],
        )
        return ToolExecutionResult(
            allowed=True,
            request_id=context.request_id,
            agent_name=context.agent_name,
            tool_name=context.tool_name,
            output=output,
            decision=decision,
            duration_ms=duration_ms,
        )
    except Exception as exc:
        duration_ms = int((time.monotonic() - started_at) * 1000)
        tool_audit_service.completed(
            request_id=context.request_id,
            agent_name=context.agent_name,
            tool_name=context.tool_name,
            outcome="error",
            duration_ms=duration_ms,
            session_id=context.session_id,
            trace_id=context.trace_id,
            output_preview=f"Error al ejecutar tool: {exc}",
        )
        return ToolExecutionResult(
            allowed=False,
            request_id=context.request_id,
            agent_name=context.agent_name,
            tool_name=context.tool_name,
            output=f"Error al ejecutar tool: {exc}",
            decision=decision,
            duration_ms=duration_ms,
        )


__all__ = ["ToolExecutionContext", "ToolExecutionResult", "execute_registered_tool"]
