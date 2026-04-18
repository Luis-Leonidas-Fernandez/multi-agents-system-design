"""Loop principal del bridge Python ↔ Node UI.

Lee mensajes JSON desde stdin, emite eventos JSON hacia stdout.
Protocolo compatible con ui/src/hooks/useBridgeSession.js.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import cast, Optional

from application.helpers.config_flow_helpers import validate_env
from application.services.runtime import AgentRuntime
from application.ui_bridge.protocol import emit_json
from application.ui_bridge.state_mapper import (
    build_ui_state_payload,
    extract_latest_ai_text_from_live_state,
    merge_turn_response_into_ui_state,
)

TURN_TIMEOUT_SECONDS = float(os.getenv("TURN_TIMEOUT_SECONDS", "180"))


def run_ui_bridge() -> None:
    # HITL usa input() interactivo — incompatible con bridge (stdin es JSON del Node UI).
    os.environ.setdefault("HITL_ENABLED", "false")
    validate_env()
    runtime = AgentRuntime()
    lifecycle = runtime.start_session_lifecycle(None)
    emit_json({"type": "state", "state": build_ui_state_payload(lifecycle, runtime, "listo")})
    try:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                emit_json({"type": "error", "message": "JSON inválido"})
                continue

            action = str(message.get("action", "")).lower()
            if action == "exit":
                break
            if action != "submit":
                emit_json({"type": "error", "message": f"Acción no soportada: {action}"})
                continue

            text = str(message.get("text", "")).strip()
            if not text:
                emit_json({"type": "state", "state": build_ui_state_payload(lifecycle, runtime, "listo")})
                continue

            if text.lower() in {"salir", "exit", "quit"}:
                asyncio.run(lifecycle.close())
                asyncio.run(runtime.shutdown())
                emit_json({"type": "exit"})
                break

            if text.startswith("/"):
                from application.services.cli_dispatch import dispatch_inspection_command
                result = dispatch_inspection_command(text, lifecycle, runtime)
                if result.handled:
                    base_state = build_ui_state_payload(lifecycle, runtime, "listo")
                    base_transcript = list(cast(list[str], base_state.get("transcript", [])))
                    base_transcript.append(f"command: {text}")
                    base_transcript.extend(result.lines)
                    emit_json({"type": "state", "state": {**base_state, "transcript": base_transcript}})
                    continue

            emit_json({"type": "busy", "status": "procesando…"})
            session = lifecycle.resolve(text)
            if session.turn_context is None:
                emit_json({"type": "state", "state": build_ui_state_payload(lifecycle, runtime, "listo")})
                continue
            try:
                turn = asyncio.run(asyncio.wait_for(runtime.execute_turn(session.turn_context), timeout=TURN_TIMEOUT_SECONDS))
                live_state = asyncio.run(runtime.get_live_state(lifecycle.session_id))
                effective_response = (turn.response or "").strip() or extract_latest_ai_text_from_live_state(
                    cast(Optional[dict], live_state)
                )
                state_payload = build_ui_state_payload(lifecycle, runtime, "listo")
                state_payload = merge_turn_response_into_ui_state(
                    state_payload,
                    effective_response,
                    message_count=turn.message_count,
                )
                emit_json({"type": "state", "state": state_payload})
            except asyncio.TimeoutError:
                emit_json({"type": "error", "message": f"El turno superó {int(TURN_TIMEOUT_SECONDS)}s. Revisá proveedor/red/modelo."})
                emit_json({"type": "state", "state": build_ui_state_payload(lifecycle, runtime, "listo")})
            except Exception as exc:
                emit_json({"type": "error", "message": str(exc)})
                emit_json({"type": "state", "state": build_ui_state_payload(lifecycle, runtime, "listo")})
    finally:
        try:
            asyncio.run(lifecycle.close())
        except Exception:
            pass
        try:
            asyncio.run(runtime.shutdown())
        except Exception:
            pass
