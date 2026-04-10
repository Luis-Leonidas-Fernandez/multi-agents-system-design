"""Tests de helpers de trazabilidad de flujo."""


def test_get_or_create_request_id_uses_existing_value():
    from application.helpers.trace_flow_helpers import get_or_create_request_id

    state = {"request_id": "req-123"}
    assert get_or_create_request_id(state, lambda: "new") == "req-123"


def test_get_or_create_request_id_creates_when_missing():
    from application.helpers.trace_flow_helpers import get_or_create_request_id

    state = {}
    assert get_or_create_request_id(state, lambda: "new") == "new"
