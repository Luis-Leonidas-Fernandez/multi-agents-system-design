"""Tipos de eventos y contrato del protocolo bridge Python ↔ Node UI.

Secuencia válida de eventos
---------------------------
1. ``hello``  — siempre primero; anuncia versión y capabilities.
2. ``state | busy | error | exit``  — cualquier orden posterior.

Reglas de invariante:
- Ningún evento puede preceder a ``hello``.
- Solo puede existir un ``hello`` por sesión.
- ``exit`` es terminal; no se emiten eventos posteriores.

Versioning (semver)
-------------------
``PROTOCOL_MAJOR.PROTOCOL_MINOR``  →  ``PROTOCOL_VERSION = "1.0"``

- Major bump → rompe compatibilidad. El Node UI debe rechazar la conexión.
- Minor bump → backward compatible. El Node UI puede aceptar features adicionales.

Schema versioning
-----------------
``StateEvent`` y ``ErrorEvent`` incluyen ``schema_version`` independiente
del protocolo, para que los dashboards/clientes detecten cambios de estructura
sin necesidad de bump de protocolo completo.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Versión de protocolo
# ---------------------------------------------------------------------------

PROTOCOL_MAJOR: int = 1
PROTOCOL_MINOR: int = 0
PROTOCOL_VERSION: str = f"{PROTOCOL_MAJOR}.{PROTOCOL_MINOR}"

# ---------------------------------------------------------------------------
# Schema versions (independientes del protocolo)
# ---------------------------------------------------------------------------

STATE_SCHEMA_VERSION: int = 1
ERROR_SCHEMA_VERSION: int = 1

# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BridgeCapabilities:
    streaming: bool = False
    tool_calls: bool = True
    audit_traces: bool = True
    hitl: bool = False
    cost_tracking: bool = False


# ---------------------------------------------------------------------------
# Eventos
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HelloEvent:
    protocol_version: str = PROTOCOL_VERSION
    capabilities: BridgeCapabilities = field(default_factory=BridgeCapabilities)
    type: Literal["hello"] = "hello"


@dataclass(frozen=True)
class StateEvent:
    state: dict[str, Any]
    schema_version: int = STATE_SCHEMA_VERSION
    type: Literal["state"] = "state"


@dataclass(frozen=True)
class BusyEvent:
    status: str = "procesando…"
    type: Literal["busy"] = "busy"


@dataclass(frozen=True)
class ErrorEvent:
    message: str = ""
    schema_version: int = ERROR_SCHEMA_VERSION
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


BridgeEvent = HelloEvent | StateEvent | BusyEvent | ErrorEvent | ExitEvent

# ---------------------------------------------------------------------------
# Emitter con fail-fast (Python side)
# ---------------------------------------------------------------------------


class BridgeEmitter:
    """Emitter tipado con guard: HelloEvent DEBE ser el primer evento.

    Usar una instancia por proceso/bridge — no compartir entre tests.
    """

    def __init__(self) -> None:
        self._hello_emitted: bool = False

    def emit(self, event: BridgeEvent) -> None:
        if not self._hello_emitted:
            if not isinstance(event, HelloEvent):
                raise RuntimeError(
                    f"First event must be HelloEvent, got {type(event).__name__}"
                )
            self._hello_emitted = True
        emit_json(asdict(event))


# ---------------------------------------------------------------------------
# Validador stateful (Node UI side)
# ---------------------------------------------------------------------------


class ProtocolMismatchError(Exception):
    pass


class BridgeProtocolValidator:
    """Valida el protocolo desde el lado del Node UI.

    Mantiene estado: el primer mensaje recibido DEBE ser ``hello`` con major
    version compatible. Mensajes posteriores no pueden ser ``hello``.
    """

    def __init__(self, *, supported_major: int = PROTOCOL_MAJOR) -> None:
        self._supported_major = supported_major
        self._initialized: bool = False

    def on_message(self, payload: dict[str, Any]) -> None:
        if not self._initialized:
            if payload.get("type") != "hello":
                raise ProtocolMismatchError(
                    f"Expected 'hello' as first event, got: {payload.get('type')!r}"
                )
            version_str = str(payload.get("protocol_version", ""))
            try:
                bridge_major = int(version_str.split(".")[0])
            except (ValueError, IndexError):
                raise ProtocolMismatchError(
                    f"Invalid protocol_version format: {version_str!r}"
                )
            if bridge_major != self._supported_major:
                raise ProtocolMismatchError(
                    f"Protocol major version mismatch: bridge={bridge_major}, ui={self._supported_major}"
                )
            self._initialized = True
            return

        if payload.get("type") == "hello":
            raise ProtocolMismatchError("Unexpected 'hello' after initialization")

    @property
    def initialized(self) -> bool:
        return self._initialized


# ---------------------------------------------------------------------------
# Helpers de bajo nivel
# ---------------------------------------------------------------------------


def emit_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def event_to_dict(event: BridgeEvent) -> dict[str, Any]:
    return asdict(event)
