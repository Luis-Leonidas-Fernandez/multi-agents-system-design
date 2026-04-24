"""Memoria de sesión con una frontera de aplicación.

Este módulo envuelve `infra.memory` para que runtime y gateway no dependan
directamente del backend concreto de distillation/load.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import features.sessions.infrastructure.memory as _infra_memory
from core.helpers.config_flow_helpers import get_llm


_infra_memory.configure_llm_factory(get_llm)


@dataclass
class SessionMemory:
    """Adaptador de memoria usado por runtime y gateway."""

    def load_memory_context(self, session_id: str) -> str:
        return _infra_memory.load_memory_context(session_id)

    async def distill_memory(self, state: dict[str, Any], session_id: str) -> bool:
        return await _infra_memory.distill_memory(state, session_id)


memory = SessionMemory()


__all__ = ["SessionMemory", "memory"]
