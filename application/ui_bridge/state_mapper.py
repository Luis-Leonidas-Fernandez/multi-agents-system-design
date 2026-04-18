"""Mapeo entre el estado interno del runtime y el payload JSON de la UI."""
from __future__ import annotations

from typing import Any, Optional, cast

from application.services.runtime import AgentRuntime

_UI_ROLE_MAP: dict[str, str] = {"ai": "assistant", "human": "you"}


def build_ui_state_payload(lifecycle: Any, runtime: AgentRuntime, status: str) -> dict[str, Any]:
    artifact = runtime.build_session_artifact(lifecycle.session_id)
    transcript: list[str] = []
    for item in artifact.transcript:
        role = _UI_ROLE_MAP.get(str(item.get("role", "message")).lower(), str(item.get("role", "message")).lower())
        content = str(item.get("content", ""))
        transcript.append(f"{role}: {content}".rstrip())
    view = lifecycle.view()
    return {
        "session_id": lifecycle.session_id,
        "status": status,
        "prompt": view.prompt_hint,
        "transcript": transcript,
        "message_count": view.snapshot.message_count,
        "has_memory": view.snapshot.has_memory,
    }


def merge_turn_response_into_ui_state(
    state: dict[str, Any],
    response: str,
    *,
    message_count: Optional[int] = None,
) -> dict[str, Any]:
    """Asegura que la UI vea la respuesta del turno aunque el artifact esté atrasado."""
    normalized_response = (response or "").strip()
    if not normalized_response:
        return state

    transcript = list(cast(list[str], state.get("transcript", [])))
    ai_line = f"assistant: {normalized_response}"
    if transcript and transcript[-1] == ai_line:
        merged = dict(state)
        if message_count is not None:
            merged["message_count"] = max(int(merged.get("message_count", 0) or 0), int(message_count))
        return merged

    merged = dict(state)
    merged["transcript"] = transcript + [ai_line]
    if message_count is not None:
        merged["message_count"] = max(int(merged.get("message_count", 0) or 0), int(message_count))
    elif "message_count" in merged:
        merged["message_count"] = int(merged.get("message_count", 0) or 0) + 1
    return merged


def extract_latest_ai_text_from_live_state(live_state: Optional[dict[str, Any]]) -> str:
    if not live_state:
        return ""
    messages = live_state.get("messages")
    if not isinstance(messages, list):
        return ""
    for item in reversed(messages):
        role = str(getattr(item, "type", getattr(item, "role", ""))).lower()
        if role in {"ai", "assistant"}:
            content = str(getattr(item, "content", "")).strip()
            if content:
                return content
    return ""
