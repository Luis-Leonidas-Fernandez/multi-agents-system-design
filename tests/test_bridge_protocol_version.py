"""Tests de integración: versionado de protocolo bridge Python ↔ Node UI.

Escenarios cubiertos:
- Constantes y defaults del protocolo
- HelloEvent: semver, capabilities, schema versions
- BridgeEmitter: fail-fast — HelloEvent debe ser primero
- BridgeProtocolValidator: stateful — valida primer mensaje, major version, duplicados
- Compatibilidad: bridge v1.x + ui major=1 → ok, bridge v2.x + ui major=1 → error
- Forward compat: minor bump no rompe
"""
from __future__ import annotations

import pytest

from application.ui_bridge.protocol import (
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    PROTOCOL_VERSION,
    STATE_SCHEMA_VERSION,
    ERROR_SCHEMA_VERSION,
    BridgeCapabilities,
    BridgeEmitter,
    BridgeProtocolValidator,
    BusyEvent,
    ErrorEvent,
    ExitEvent,
    HelloEvent,
    ProtocolMismatchError,
    StateEvent,
    event_to_dict,
)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

def test_protocol_version_is_semver_string():
    assert PROTOCOL_VERSION == f"{PROTOCOL_MAJOR}.{PROTOCOL_MINOR}"
    assert PROTOCOL_MAJOR == 1
    assert PROTOCOL_MINOR == 0


def test_schema_versions_are_1():
    assert STATE_SCHEMA_VERSION == 1
    assert ERROR_SCHEMA_VERSION == 1


# ---------------------------------------------------------------------------
# HelloEvent
# ---------------------------------------------------------------------------

def test_hello_event_carries_semver_and_capabilities():
    payload = event_to_dict(HelloEvent())
    assert payload["type"] == "hello"
    assert payload["protocol_version"] == "1.0"
    assert isinstance(payload["capabilities"], dict)


def test_hello_capabilities_defaults():
    caps = event_to_dict(HelloEvent())["capabilities"]
    assert caps["tool_calls"] is True
    assert caps["audit_traces"] is True
    assert caps["streaming"] is False
    assert caps["hitl"] is False
    assert caps["cost_tracking"] is False


# ---------------------------------------------------------------------------
# Schema versions en eventos
# ---------------------------------------------------------------------------

def test_state_event_carries_schema_version():
    payload = event_to_dict(StateEvent(state={"session_id": "x"}))
    assert payload["schema_version"] == STATE_SCHEMA_VERSION
    assert payload["type"] == "state"


def test_error_event_carries_schema_version():
    payload = event_to_dict(ErrorEvent(message="boom"))
    assert payload["schema_version"] == ERROR_SCHEMA_VERSION
    assert payload["type"] == "error"


# ---------------------------------------------------------------------------
# BridgeEmitter — fail-fast Python side
# ---------------------------------------------------------------------------

def test_emitter_requires_hello_first(capsys):
    emitter = BridgeEmitter()
    with pytest.raises(RuntimeError, match="First event must be HelloEvent"):
        emitter.emit(BusyEvent())


def test_emitter_accepts_hello_then_other_events(capsys):
    emitter = BridgeEmitter()
    emitter.emit(HelloEvent())
    emitter.emit(StateEvent(state={}))
    emitter.emit(BusyEvent())
    emitter.emit(ErrorEvent(message="err"))
    emitter.emit(ExitEvent())
    out = capsys.readouterr().out.strip().split("\n")
    assert len(out) == 5
    import json
    types = [json.loads(line)["type"] for line in out]
    assert types == ["hello", "state", "busy", "error", "exit"]


def test_emitter_instances_are_independent(capsys):
    e1 = BridgeEmitter()
    e2 = BridgeEmitter()
    e1.emit(HelloEvent())
    # e2 todavía no emitió hello — debe fallar
    with pytest.raises(RuntimeError, match="First event must be HelloEvent"):
        e2.emit(StateEvent(state={}))


# ---------------------------------------------------------------------------
# BridgeProtocolValidator — stateful Node UI side
# ---------------------------------------------------------------------------

def test_validator_accepts_matching_major_version():
    validator = BridgeProtocolValidator(supported_major=1)
    validator.on_message({"type": "hello", "protocol_version": "1.0"})
    assert validator.initialized


def test_validator_accepts_minor_bump_forward_compat():
    """bridge 1.5 + ui major=1 → compatible (minor bump no rompe)."""
    validator = BridgeProtocolValidator(supported_major=1)
    validator.on_message({"type": "hello", "protocol_version": "1.5"})
    assert validator.initialized


def test_validator_rejects_major_version_mismatch():
    """bridge v2.x + ui major=1 → error claro."""
    validator = BridgeProtocolValidator(supported_major=1)
    with pytest.raises(ProtocolMismatchError, match="bridge=2, ui=1"):
        validator.on_message({"type": "hello", "protocol_version": "2.0"})


def test_validator_rejects_non_hello_as_first_event():
    validator = BridgeProtocolValidator(supported_major=1)
    with pytest.raises(ProtocolMismatchError, match="'state'"):
        validator.on_message({"type": "state", "state": {}})


def test_validator_rejects_duplicate_hello():
    validator = BridgeProtocolValidator(supported_major=1)
    validator.on_message({"type": "hello", "protocol_version": "1.0"})
    with pytest.raises(ProtocolMismatchError, match="Unexpected 'hello'"):
        validator.on_message({"type": "hello", "protocol_version": "1.0"})


def test_validator_rejects_invalid_version_format():
    validator = BridgeProtocolValidator(supported_major=1)
    with pytest.raises(ProtocolMismatchError, match="Invalid protocol_version"):
        validator.on_message({"type": "hello", "protocol_version": "not-a-version"})


def test_validator_accepts_normal_events_after_hello():
    validator = BridgeProtocolValidator(supported_major=1)
    validator.on_message({"type": "hello", "protocol_version": "1.0"})
    # No debe lanzar para eventos normales posteriores
    validator.on_message({"type": "state", "state": {}})
    validator.on_message({"type": "busy"})
    validator.on_message({"type": "error", "message": "x"})
