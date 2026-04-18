"""Tipos de eventos del protocolo bridge Python ↔ Node UI.

Versión actual: 1 (campo 'type' obligatorio en todos los eventos).
Eventos futuros preparados pero no emitidos: TurnStartedEvent, TurnCompletedEvent.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class StateEvent:
    state: dict[str, Any]
    type: Literal["state"] = "state"


@dataclass(frozen=True)
class BusyEvent:
    status: str = "procesando…"
    type: Literal["busy"] = "busy"


@dataclass(frozen=True)
class ErrorEvent:
    message: str = ""
    type: Literal["error"] = "error"


@dataclass(frozen=True)
class ExitEvent:
    type: Literal["exit"] = "exit"


# Eventos futuros — definidos para reservar el contrato, no emitidos todavía.

@dataclass(frozen=True)
class TurnStartedEvent:
    request_id: str
    type: Literal["turn_started"] = "turn_started"


@dataclass(frozen=True)
class TurnCompletedEvent:
    request_id: str
    agent_name: str
    duration_ms: int
    type: Literal["turn_completed"] = "turn_completed"


BridgeEvent = StateEvent | BusyEvent | ErrorEvent | ExitEvent


def emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def event_to_dict(event: BridgeEvent) -> dict[str, Any]:
    from dataclasses import asdict
    return asdict(event)
