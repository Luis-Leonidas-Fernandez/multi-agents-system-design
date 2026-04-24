"""Persistencia de sesiones con una frontera de aplicación.

Este módulo envuelve `infra.persistence` para que los servicios de aplicación
no dependan directamente del backend concreto.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import BaseMessage

import features.sessions.infrastructure.persistence as _infra_persistence


@dataclass
class SessionPersistence:
    """Adaptador de persistencia usado por runtime y gateway."""

    def list_sessions(self) -> list[str]:
        return _infra_persistence.list_sessions()

    def load_messages(self, session_id: str):
        return _infra_persistence.load_messages(session_id)

    def save_message(self, session_id: str, role: str, content: str, request_id: str | None = None) -> None:
        _infra_persistence.save_message(session_id, role, content, request_id=request_id)

    def save_session(self, session_id: str, messages: list[BaseMessage]) -> None:
        _infra_persistence.save_session(session_id, messages)


persistence = SessionPersistence()


__all__ = ["SessionPersistence", "persistence"]
