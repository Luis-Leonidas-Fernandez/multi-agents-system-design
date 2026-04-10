"""Audit trail para invocaciones de tools."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Literal
import logging
import time

from application.helpers.audit_flow_helpers import _emit_guard_audit
from application.services.tool_audit_store import tool_audit_store


_log = logging.getLogger(__name__)


ToolCallOutcome = Literal["requested", "needs_confirmation", "blocked", "cancelled", "success", "error"]


@dataclass(frozen=True)
class ToolCallAuditEvent:
    event_type: str
    request_id: str
    agent_name: str
    tool_name: str
    outcome: ToolCallOutcome
    ts_ms: int
    session_id: str | None = None
    trace_id: str | None = None
    risk_level: str | None = None
    permission_mode: str | None = None
    allowed: bool | None = None
    requires_confirmation: bool | None = None
    decision_reason: str | None = None
    duration_ms: int | None = None
    arguments: Mapping[str, Any] | None = None
    output_preview: str | None = None


@dataclass(frozen=True)
class ToolAuditService:
    """Servicio de auditoría para tool calls."""

    def record(self, event: ToolCallAuditEvent) -> None:
        payload = asdict(event)
        _emit_guard_audit(payload)
        try:
            tool_audit_store.append_event(event)
        except Exception:
            _log.warning("tool audit persistence failed", exc_info=True)

    def requested(
        self,
        *,
        request_id: str,
        agent_name: str,
        tool_name: str,
        session_id: str | None = None,
        trace_id: str | None = None,
        arguments: Mapping[str, Any] | None = None,
    ) -> None:
        self.record(
            ToolCallAuditEvent(
                event_type="tool_call_requested",
                request_id=request_id,
                agent_name=agent_name,
                tool_name=tool_name,
                outcome="requested",
                ts_ms=int(time.time() * 1000),
                session_id=session_id,
                trace_id=trace_id,
                arguments=arguments,
            )
        )

    def decided(
        self,
        *,
        request_id: str,
        agent_name: str,
        tool_name: str,
        outcome: ToolCallOutcome,
        risk_level: str,
        permission_mode: str,
        allowed: bool,
        requires_confirmation: bool,
        decision_reason: str,
        session_id: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        self.record(
            ToolCallAuditEvent(
                event_type="tool_call_decision",
                request_id=request_id,
                agent_name=agent_name,
                tool_name=tool_name,
                outcome=outcome,
                ts_ms=int(time.time() * 1000),
                session_id=session_id,
                trace_id=trace_id,
                risk_level=risk_level,
                permission_mode=permission_mode,
                allowed=allowed,
                requires_confirmation=requires_confirmation,
                decision_reason=decision_reason,
            )
        )

    def completed(
        self,
        *,
        request_id: str,
        agent_name: str,
        tool_name: str,
        outcome: ToolCallOutcome,
        duration_ms: int,
        session_id: str | None = None,
        trace_id: str | None = None,
        output_preview: str | None = None,
    ) -> None:
        self.record(
            ToolCallAuditEvent(
                event_type="tool_call_completed",
                request_id=request_id,
                agent_name=agent_name,
                tool_name=tool_name,
                outcome=outcome,
                ts_ms=int(time.time() * 1000),
                session_id=session_id,
                trace_id=trace_id,
                duration_ms=duration_ms,
                output_preview=output_preview,
            )
        )

    def load_session_events(self, session_id: str) -> list[dict[str, Any]]:
        return tool_audit_store.load_events(session_id)

    def find_session_events(
        self,
        session_id: str,
        *,
        request_id: str | None = None,
        trace_id: str | None = None,
        tool_name: str | None = None,
    ) -> list[dict[str, Any]]:
        return tool_audit_store.find_events(
            session_id,
            request_id=request_id,
            trace_id=trace_id,
            tool_name=tool_name,
        )

    def list_sessions(self) -> list[str]:
        return tool_audit_store.list_sessions()


tool_audit_service = ToolAuditService()


__all__ = ["ToolCallAuditEvent", "ToolCallOutcome", "ToolAuditService", "tool_audit_service"]
