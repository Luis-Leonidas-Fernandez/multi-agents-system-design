"""Tipos de eventos y contrato del protocolo bridge Python ↔ Node UI.

Secuencia válida de eventos
---------------------------
1. ``hello``     — siempre primero; anuncia versión y capabilities.
2. ``hello_ack`` — Node UI responde aceptando o rechazando (Node → Python).
3. ``state | busy | error | exit``  — cualquier orden posterior.

Reglas de invariante:
- Ningún evento puede preceder a ``hello``.
- Solo puede existir un ``hello`` por sesión.
- ``exit`` es terminal; no se emiten eventos posteriores.
- El bridge no procesa requests hasta recibir ``hello_ack`` con ``accepted=true``.

Versioning (semver)
-------------------
``PROTOCOL_MAJOR.PROTOCOL_MINOR``  →  ``PROTOCOL_VERSION = "1.0"``

- Major bump → rompe compatibilidad. El Node UI debe rechazar la conexión.
- Minor bump → backward compatible. El Node UI puede aceptar features adicionales.

Schema versioning
-----------------
``StatePayload`` incluye ``schema_version`` para que los dashboards/clientes
detecten cambios de estructura de estado sin necesidad de bump de protocolo.
``ErrorEvent`` incluye ``schema_version`` por el mismo motivo.

Correlation
-----------
``BridgeEmitter`` inyecta ``event_id`` (UUID), ``timestamp`` (Unix float) y
``session_id`` en cada evento emitido. Sin esto no se puede reconstruir sesiones
ni hacer tracing de auditoría.
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

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
class BridgeCapabilityFeatures:
    """Feature flags del bridge. Cada bool indica soporte real, no intención."""
    streaming: bool = False
    tool_calls: bool = True
    audit_traces: bool = True
    hitl: bool = False
    cost_tracking: bool = False


@dataclass(frozen=True)
class BridgeCapabilities:
    """Capabilities versionadas. ``version`` permite evolucionar la estructura
    sin romper clientes que no conocen campos nuevos."""
    version: int = 1
    features: BridgeCapabilityFeatures = field(default_factory=BridgeCapabilityFeatures)


# ---------------------------------------------------------------------------
# Payload tipado de estado (Python ↔ Node)
# ---------------------------------------------------------------------------


@dataclass
class StatePayload:
    """Estructura canónica del estado de sesión enviado al Node UI.

    ``phase`` mapea el estado interno a un enum auditable en lugar de strings
    libres, lo que facilita dashboards y trazas sin parsing.
    """
    session_id: str
    status: str
    phase: Literal["idle", "running", "error"]
    transcript: list[str]
    message_count: int
    has_memory: bool
    prompt: str = ""
    schema_version: int = STATE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Eventos salientes (Python → Node UI)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HelloEvent:
    protocol_version: str = PROTOCOL_VERSION
    capabilities: BridgeCapabilities = field(default_factory=BridgeCapabilities)
    type: Literal["hello"] = "hello"


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
    schema_version: int = ERROR_SCHEMA_VERSION
    type: Literal["error"] = "error"


@dataclass(frozen=True)
class ExitEvent:
    reason: Literal["completed", "error", "cancelled"] = "completed"
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
# Evento entrante (Node UI → Python)
# ---------------------------------------------------------------------------


@dataclass
class HelloAckPayload:
    """Confirmación de handshake enviada por el Node UI tras recibir ``hello``."""
    accepted: bool = True
    type: Literal["hello_ack"] = "hello_ack"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HelloAckPayload":
        return cls(accepted=bool(data.get("accepted", False)))


# ---------------------------------------------------------------------------
# Emitter tipado con fail-fast y correlación (Python side)
# ---------------------------------------------------------------------------


class BridgeEmitter:
    """Emitter tipado con tres garantías:

    1. Fail-fast: ``HelloEvent`` DEBE ser el primer evento emitido.
    2. Correlation: inyecta ``event_id``, ``timestamp`` y ``session_id`` en
       cada payload antes de serializarlo a JSON.
    3. Capabilities-aware: el método ``emit`` consulta las capabilities
       declaradas para gate comportamientos futuros (e.g. streaming).
    """

    def __init__(self, *, session_id: str = "") -> None:
        self._hello_emitted: bool = False
        self._session_id: str = session_id
        self._capabilities: BridgeCapabilities = BridgeCapabilities()

    @property
    def session_id(self) -> str:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        self._session_id = value

    @property
    def capabilities(self) -> BridgeCapabilities:
        return self._capabilities

    def emit(self, event: BridgeEvent) -> None:
        if not self._hello_emitted:
            if not isinstance(event, HelloEvent):
                raise RuntimeError(
                    f"First event must be HelloEvent, got {type(event).__name__}"
                )
            self._capabilities = event.capabilities
            self._hello_emitted = True

        payload = asdict(event)

        # Correlation envelope — clave para auditoría y tracing.
        payload["event_id"] = str(uuid.uuid4())
        payload["timestamp"] = time.time()
        if self._session_id:
            payload["session_id"] = self._session_id

        # Capabilities-gated behavior.
        if isinstance(event, StateEvent) and not self._capabilities.features.streaming:
            # Full state update path. When streaming=True: emit partial tokens here.
            pass

        emit_json(payload)

    def wait_for_hello_ack(
        self,
        *,
        timeout: float = 5.0,
        stdin: Any = None,
    ) -> bool:
        """Lee el primer mensaje de stdin esperando ``hello_ack``.

        Retorna ``True`` si el Node UI aceptó el handshake dentro del timeout.
        Retorna ``False`` en timeout, rechazo o mensaje inesperado.

        En producción usa ``select`` para respetar el timeout. Para objetos
        no seleccionables (e.g. ``io.StringIO`` en tests), hace ``readline``
        directo sin timeout.
        """
        import select as _select

        src = stdin if stdin is not None else sys.stdin
        try:
            readable, _, _ = _select.select([src], [], [], timeout)
            if not readable:
                return False
        except (OSError, ValueError):
            # Objeto no seleccionable (StringIO u otro sin fileno real).
            # Caemos a readline directo — válido en tests y entornos restringidos.
            pass

        raw = src.readline().strip()
        if not raw:
            return False
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return False
        if data.get("type") != "hello_ack":
            return False
        return bool(data.get("accepted", False))


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
