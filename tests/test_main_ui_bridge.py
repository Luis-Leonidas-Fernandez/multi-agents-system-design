from main import (
    _extract_latest_ai_text_from_live_state,
    _merge_turn_response_into_ui_state,
)


class _FakeMessage:
    def __init__(self, type_, content):
        self.type = type_
        self.content = content


def test_merge_turn_response_into_ui_state_appends_ai_when_transcript_is_stale():
    state = {
        "session_id": "sess-1",
        "status": "listo",
        "prompt": "Escribí",
        "transcript": ["human: hola"],
        "message_count": 1,
        "has_memory": False,
    }

    merged = _merge_turn_response_into_ui_state(state, "respuesta final", message_count=2)

    assert merged["transcript"] == ["human: hola", "ai: respuesta final"]
    assert merged["message_count"] == 2


def test_merge_turn_response_into_ui_state_does_not_duplicate_existing_ai():
    state = {
        "session_id": "sess-1",
        "status": "listo",
        "prompt": "Escribí",
        "transcript": ["human: hola", "ai: respuesta final"],
        "message_count": 2,
        "has_memory": False,
    }

    merged = _merge_turn_response_into_ui_state(state, "respuesta final", message_count=2)

    assert merged["transcript"] == ["human: hola", "ai: respuesta final"]
    assert merged["message_count"] == 2


def test_merge_turn_response_into_ui_state_ignores_empty_response():
    state = {
        "session_id": "sess-1",
        "status": "listo",
        "prompt": "Escribí",
        "transcript": ["human: hola"],
        "message_count": 1,
        "has_memory": False,
    }

    merged = _merge_turn_response_into_ui_state(state, "   ", message_count=1)

    assert merged == state


def test_extract_latest_ai_text_from_live_state_returns_latest_assistant_message():
    live_state = {
        "messages": [
            _FakeMessage("human", "hola"),
            _FakeMessage("ai", "respuesta 1"),
            _FakeMessage("assistant", "respuesta 2"),
        ]
    }

    assert _extract_latest_ai_text_from_live_state(live_state) == "respuesta 2"


def test_extract_latest_ai_text_from_live_state_returns_empty_when_missing():
    live_state = {"messages": [_FakeMessage("human", "hola")]}

    assert _extract_latest_ai_text_from_live_state(live_state) == ""
