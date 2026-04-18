"""Tests de integración: versionado de protocolo bridge Python ↔ Node UI.

Escenarios:
- bridge v1 + ui v1  → compatible, sin error
- bridge v2 + ui v1  → incompatible, error claro
"""
from __future__ import annotations

import pytest

from application.ui_bridge.protocol import (
    PROTOCOL_VERSION,
    HelloEvent,
    event_to_dict,
)


# ---------------------------------------------------------------------------
# Helpers — simulan la lógica de validación que el Node UI debe implementar
# ---------------------------------------------------------------------------

class ProtocolMismatchError(Exception):
    pass


def ui_validate_hello(hello_payload: dict, *, ui_version: int) -> None:
    """Valida el evento hello recibido desde el bridge.

    Lanza ProtocolMismatchError si la versión del bridge no coincide con la
    versión que la UI soporta.
    """
    if hello_payload.get("type") != "hello":
        raise ProtocolMismatchError(f"Se esperaba evento 'hello', llegó: {hello_payload.get('type')!r}")
    bridge_version = hello_payload.get("protocol_version")
    if bridge_version != ui_version:
        raise ProtocolMismatchError(
            f"Versión de protocolo incompatible: bridge={bridge_version}, ui={ui_version}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_protocol_version_constant_is_1():
    assert PROTOCOL_VERSION == 1


def test_hello_event_carries_protocol_version():
    payload = event_to_dict(HelloEvent())
    assert payload["type"] == "hello"
    assert payload["protocol_version"] == PROTOCOL_VERSION


def test_bridge_v1_ui_v1_compatible():
    """bridge v1 + ui v1 → sin error."""
    hello_payload = event_to_dict(HelloEvent())  # emite protocol_version=1
    ui_validate_hello(hello_payload, ui_version=1)  # no lanza


def test_bridge_v2_ui_v1_raises_clear_error():
    """bridge v2 + ui v1 → ProtocolMismatchError con mensaje claro."""
    hello_v2 = {"type": "hello", "protocol_version": 2}
    with pytest.raises(ProtocolMismatchError, match="bridge=2, ui=1"):
        ui_validate_hello(hello_v2, ui_version=1)


def test_wrong_first_event_type_raises():
    """Si el primer evento no es hello, la UI debe rechazarlo."""
    not_hello = {"type": "state", "state": {}}
    with pytest.raises(ProtocolMismatchError, match="'state'"):
        ui_validate_hello(not_hello, ui_version=1)
