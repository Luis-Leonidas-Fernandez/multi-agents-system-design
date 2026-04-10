"""Contexto de trazabilidad para sesiones y turnos.

Centraliza la generación de request_id/trace_id para que el runtime y el
gateway compartan una estructura de observabilidad explícita.
"""
from __future__ import annotations

from dataclasses import dataclass
import uuid


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    session_id: str
    request_id: str
    operation: str
    parent_trace_id: str | None = None


@dataclass(frozen=True)
class TraceContextService:
    """Generador de trace context para el runtime."""

    def create(self, session_id: str, operation: str, parent_trace_id: str | None = None) -> TraceContext:
        return TraceContext(
            trace_id=str(uuid.uuid4()),
            session_id=session_id,
            request_id=str(uuid.uuid4()),
            operation=operation,
            parent_trace_id=parent_trace_id,
        )


trace_context_service = TraceContextService()


__all__ = ["TraceContext", "TraceContextService", "trace_context_service"]
