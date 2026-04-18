from application.ui_bridge.state_mapper import (
    extract_latest_ai_text_from_live_state as _extract,
    merge_turn_response_into_ui_state as _merge,
)
from application.ui_bridge.protocol import StatePayload


def _make_payload(**kwargs) -> StatePayload:
    defaults = {
        "session_id": "sess-1",
        "status": "listo",
        "phase": "idle",
        "prompt": "Escribí",
        "transcript": [],
        "message_count": 0,
        "has_memory": False,
    }
    defaults.update(kwargs)
    return StatePayload(**defaults)


class _FakeMessage:
    def __init__(self, type_, content):
        self.type = type_
        self.content = content


def test_merge_appends_ai_when_transcript_is_stale():
    payload = _make_payload(transcript=["human: hola"], message_count=1)
    merged = _merge(payload, "respuesta final", message_count=2)
    assert merged.transcript == ["human: hola", "assistant: respuesta final"]
    assert merged.message_count == 2


def test_merge_does_not_duplicate_existing_ai():
    payload = _make_payload(
        transcript=["human: hola", "assistant: respuesta final"],
        message_count=2,
    )
    merged = _merge(payload, "respuesta final", message_count=2)
    assert merged.transcript == ["human: hola", "assistant: respuesta final"]
    assert merged.message_count == 2


def test_merge_ignores_empty_response():
    payload = _make_payload(transcript=["human: hola"], message_count=1)
    merged = _merge(payload, "   ", message_count=1)
    assert merged is payload


def test_extract_returns_latest_assistant_message():
    live_state = {
        "messages": [
            _FakeMessage("human", "hola"),
            _FakeMessage("ai", "respuesta 1"),
            _FakeMessage("assistant", "respuesta 2"),
        ]
    }
    assert _extract(live_state) == "respuesta 2"


def test_extract_returns_empty_when_missing():
    live_state = {"messages": [_FakeMessage("human", "hola")]}
    assert _extract(live_state) == ""
