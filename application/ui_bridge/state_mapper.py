"""Mapeo entre el estado interno del runtime y el payload tipado de la UI."""
from __future__ import annotations

from typing import Any, Literal, Optional

from application.services.runtime import AgentRuntime
from application.ui_bridge.protocol import StatePayload

_UI_ROLE_MAP: dict[str, str] = {"ai": "assistant", "human": "you"}


def _derive_phase(status: str) -> Literal["idle", "running", "error"]:
    s = status.lower()
    if "procesando" in s or "running" in s:
        return "running"
    if "error" in s:
        return "error"
    return "idle"


def build_ui_state_payload(lifecycle: Any, runtime: AgentRuntime, status: str) -> StatePayload:
    artifact = runtime.build_session_artifact(lifecycle.session_id)
    transcript: list[str] = []
    for item in artifact.transcript:
        role = _UI_ROLE_MAP.get(str(item.get("role", "message")).lower(), str(item.get("role", "message")).lower())
        content = str(item.get("content", ""))
        transcript.append(f"{role}: {content}".rstrip())
    view = lifecycle.view()
    return StatePayload(
        session_id=lifecycle.session_id,
        status=status,
        phase=_derive_phase(status),
        transcript=transcript,
        message_count=view.snapshot.message_count,
        has_memory=view.snapshot.has_memory,
        prompt=view.prompt_hint,
    )


def merge_turn_response_into_ui_state(
    payload: StatePayload,
    response: str,
    *,
    message_count: Optional[int] = None,
) -> StatePayload:
    """Asegura que la UI vea la respuesta del turno aunque el artifact esté atrasado."""
    normalized = (response or "").strip()
    if not normalized:
        return payload

    ai_line = f"assistant: {normalized}"
    transcript = list(payload.transcript)

    if transcript and transcript[-1] == ai_line:
        if message_count is not None:
            return StatePayload(
                session_id=payload.session_id,
                status=payload.status,
                phase=payload.phase,
                transcript=transcript,
                message_count=max(payload.message_count, message_count),
                has_memory=payload.has_memory,
                prompt=payload.prompt,
            )
        return payload

    return StatePayload(
        session_id=payload.session_id,
        status=payload.status,
        phase=payload.phase,
        transcript=transcript + [ai_line],
        message_count=max(payload.message_count, message_count) if message_count is not None else payload.message_count + 1,
        has_memory=payload.has_memory,
        prompt=payload.prompt,
    )


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
