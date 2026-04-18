"""Tests de integración: versionado de protocolo bridge Python ↔ Node UI.

Cubre los 6 puntos de auditoría senior:
1. ACK del handshake (wait_for_hello_ack)
2. Capabilities wired en BridgeEmitter
3. Capabilities versionadas (version + features)
4. StatePayload tipado con phase
5. ExitEvent con reason semántico
6. Correlation IDs (event_id, session_id, timestamp) en cada evento
"""
from __future__ import annotations

import io
import json

import pytest

from application.ui_bridge.protocol import (
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    PROTOCOL_VERSION,
    STATE_SCHEMA_VERSION,
    ERROR_SCHEMA_VERSION,
    BridgeCapabilities,
    BridgeCapabilityFeatures,
    BridgeEmitter,
    BridgeProtocolValidator,
    BusyEvent,
    ErrorEvent,
    ExitEvent,
    HelloAckPayload,
    HelloEvent,
    ProtocolMismatchError,
    StateEvent,
    StatePayload,
    event_to_dict,
)


# ---------------------------------------------------------------------------
# Constantes y semver
# ---------------------------------------------------------------------------

def test_protocol_version_is_semver():
    assert PROTOCOL_VERSION == f"{PROTOCOL_MAJOR}.{PROTOCOL_MINOR}"
    assert PROTOCOL_MAJOR == 1
    assert PROTOCOL_MINOR == 0


def test_schema_versions_are_1():
    assert STATE_SCHEMA_VERSION == 1
    assert ERROR_SCHEMA_VERSION == 1


# ---------------------------------------------------------------------------
# 3. Capabilities versionadas
# ---------------------------------------------------------------------------

def test_hello_event_capabilities_have_version():
    payload = event_to_dict(HelloEvent())
    caps = payload["capabilities"]
    assert caps["version"] == 1
    assert isinstance(caps["features"], dict)


def test_hello_capabilities_feature_defaults():
    features = event_to_dict(HelloEvent())["capabilities"]["features"]
    assert features["tool_calls"] is True
    assert features["audit_traces"] is True
    assert features["streaming"] is False
    assert features["hitl"] is False
    assert features["cost_tracking"] is False


def test_bridge_capabilities_version_is_independent_of_protocol():
    caps = BridgeCapabilities(version=2, features=BridgeCapabilityFeatures(streaming=True))
    hello = HelloEvent(capabilities=caps)
    payload = event_to_dict(hello)
    assert payload["protocol_version"] == "1.0"
    assert payload["capabilities"]["version"] == 2
    assert payload["capabilities"]["features"]["streaming"] is True


# ---------------------------------------------------------------------------
# 4. StatePayload tipado con phase
# ---------------------------------------------------------------------------

def test_state_payload_phase_idle():
    p = StatePayload(session_id="s", status="listo", phase="idle",
                     transcript=[], message_count=0, has_memory=False)
    assert p.phase == "idle"
    assert p.schema_version == STATE_SCHEMA_VERSION


def test_state_payload_phase_running():
    p = StatePayload(session_id="s", status="procesando", phase="running",
                     transcript=[], message_count=0, has_memory=False)
    assert p.phase == "running"


def test_state_payload_to_dict_includes_schema_version():
    p = StatePayload(session_id="s", status="listo", phase="idle",
                     transcript=["you: hola"], message_count=1, has_memory=False)
    d = p.to_dict()
    assert d["schema_version"] == STATE_SCHEMA_VERSION
    assert d["phase"] == "idle"
    assert d["transcript"] == ["you: hola"]


def test_error_event_carries_schema_version():
    payload = event_to_dict(ErrorEvent(message="boom"))
    assert payload["schema_version"] == ERROR_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 5. ExitEvent con reason semántico
# ---------------------------------------------------------------------------

def test_exit_event_default_reason_is_completed():
    payload = event_to_dict(ExitEvent())
    assert payload["reason"] == "completed"


def test_exit_event_error_reason():
    payload = event_to_dict(ExitEvent(reason="error"))
    assert payload["reason"] == "error"


def test_exit_event_cancelled_reason():
    payload = event_to_dict(ExitEvent(reason="cancelled"))
    assert payload["reason"] == "cancelled"


# ---------------------------------------------------------------------------
# BridgeEmitter — fail-fast
# ---------------------------------------------------------------------------

def test_emitter_requires_hello_first():
    emitter = BridgeEmitter()
    with pytest.raises(RuntimeError, match="First event must be HelloEvent"):
        emitter.emit(BusyEvent())


def test_emitter_instances_are_independent(capsys):
    e1 = BridgeEmitter()
    e2 = BridgeEmitter()
    e1.emit(HelloEvent())
    with pytest.raises(RuntimeError, match="First event must be HelloEvent"):
        e2.emit(StateEvent(state={}))


# ---------------------------------------------------------------------------
# 6. Correlation IDs
# ---------------------------------------------------------------------------

def test_emitter_injects_event_id_and_timestamp(capsys):
    emitter = BridgeEmitter()
    emitter.emit(HelloEvent())
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert "event_id" in payload
    assert "timestamp" in payload
    assert isinstance(payload["event_id"], str) and len(payload["event_id"]) == 36  # UUID
    assert isinstance(payload["timestamp"], float)


def test_emitter_injects_session_id_when_set(capsys):
    emitter = BridgeEmitter(session_id="sess-abc")
    emitter.emit(HelloEvent())
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["session_id"] == "sess-abc"


def test_emitter_session_id_settable_after_construction(capsys):
    emitter = BridgeEmitter()
    emitter.emit(HelloEvent())
    emitter.session_id = "sess-xyz"
    emitter.emit(BusyEvent())
    lines = capsys.readouterr().out.strip().split("\n")
    hello_payload = json.loads(lines[0])
    busy_payload = json.loads(lines[1])
    assert "session_id" not in hello_payload  # no session_id at hello time
    assert busy_payload["session_id"] == "sess-xyz"


def test_emitter_all_events_have_correlation(capsys):
    emitter = BridgeEmitter(session_id="s1")
    emitter.emit(HelloEvent())
    emitter.emit(StateEvent(state={}))
    emitter.emit(BusyEvent())
    emitter.emit(ErrorEvent(message="x"))
    emitter.emit(ExitEvent())
    lines = capsys.readouterr().out.strip().split("\n")
    for line in lines:
        p = json.loads(line)
        assert "event_id" in p, f"Missing event_id in: {p}"
        assert "timestamp" in p, f"Missing timestamp in: {p}"
        assert p["session_id"] == "s1"


# ---------------------------------------------------------------------------
# 2. Capabilities wired en BridgeEmitter
# ---------------------------------------------------------------------------

def test_emitter_caches_capabilities_from_hello(capsys):
    custom_caps = BridgeCapabilities(
        version=1,
        features=BridgeCapabilityFeatures(streaming=True, tool_calls=False),
    )
    emitter = BridgeEmitter()
    emitter.emit(HelloEvent(capabilities=custom_caps))
    assert emitter.capabilities.features.streaming is True
    assert emitter.capabilities.features.tool_calls is False


# ---------------------------------------------------------------------------
# 1. ACK del handshake
# ---------------------------------------------------------------------------

def test_wait_for_hello_ack_returns_true_when_accepted(capsys):
    emitter = BridgeEmitter()
    emitter.emit(HelloEvent())
    stdin_mock = io.StringIO(json.dumps({"type": "hello_ack", "accepted": True}) + "\n")
    result = emitter.wait_for_hello_ack(stdin=stdin_mock, timeout=1.0)
    assert result is True


def test_wait_for_hello_ack_returns_false_when_rejected(capsys):
    emitter = BridgeEmitter()
    emitter.emit(HelloEvent())
    stdin_mock = io.StringIO(json.dumps({"type": "hello_ack", "accepted": False}) + "\n")
    result = emitter.wait_for_hello_ack(stdin=stdin_mock, timeout=1.0)
    assert result is False


def test_wait_for_hello_ack_returns_false_on_wrong_type(capsys):
    emitter = BridgeEmitter()
    emitter.emit(HelloEvent())
    stdin_mock = io.StringIO(json.dumps({"type": "state"}) + "\n")
    result = emitter.wait_for_hello_ack(stdin=stdin_mock, timeout=1.0)
    assert result is False


def test_wait_for_hello_ack_returns_false_on_invalid_json(capsys):
    emitter = BridgeEmitter()
    emitter.emit(HelloEvent())
    stdin_mock = io.StringIO("not-json\n")
    result = emitter.wait_for_hello_ack(stdin=stdin_mock, timeout=1.0)
    assert result is False


def test_hello_ack_payload_from_dict():
    ack = HelloAckPayload.from_dict({"type": "hello_ack", "accepted": True})
    assert ack.accepted is True
    ack2 = HelloAckPayload.from_dict({"type": "hello_ack", "accepted": False})
    assert ack2.accepted is False


# ---------------------------------------------------------------------------
# BridgeProtocolValidator — stateful Node UI side
# ---------------------------------------------------------------------------

def test_validator_accepts_matching_major():
    v = BridgeProtocolValidator(supported_major=1)
    v.on_message({"type": "hello", "protocol_version": "1.0"})
    assert v.initialized


def test_validator_accepts_minor_bump():
    v = BridgeProtocolValidator(supported_major=1)
    v.on_message({"type": "hello", "protocol_version": "1.9"})
    assert v.initialized


def test_validator_rejects_major_mismatch():
    v = BridgeProtocolValidator(supported_major=1)
    with pytest.raises(ProtocolMismatchError, match="bridge=2, ui=1"):
        v.on_message({"type": "hello", "protocol_version": "2.0"})


def test_validator_rejects_non_hello_first():
    v = BridgeProtocolValidator(supported_major=1)
    with pytest.raises(ProtocolMismatchError, match="'state'"):
        v.on_message({"type": "state", "state": {}})


def test_validator_rejects_duplicate_hello():
    v = BridgeProtocolValidator(supported_major=1)
    v.on_message({"type": "hello", "protocol_version": "1.0"})
    with pytest.raises(ProtocolMismatchError, match="Unexpected 'hello'"):
        v.on_message({"type": "hello", "protocol_version": "1.0"})


def test_validator_rejects_invalid_version_format():
    v = BridgeProtocolValidator(supported_major=1)
    with pytest.raises(ProtocolMismatchError, match="Invalid protocol_version"):
        v.on_message({"type": "hello", "protocol_version": "not-semver"})


def test_validator_accepts_normal_events_after_hello():
    v = BridgeProtocolValidator(supported_major=1)
    v.on_message({"type": "hello", "protocol_version": "1.0"})
    v.on_message({"type": "state", "state": {}})
    v.on_message({"type": "busy"})
    v.on_message({"type": "error", "message": "x"})
