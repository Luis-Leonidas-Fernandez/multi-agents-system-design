"""WebSocket bridge for the frontend dashboard."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

from websockets.server import WebSocketServerProtocol, serve

from application.frontend_bridge.protocol import (
    DashboardAction,
    DashboardAgent,
    DashboardEvent,
    DashboardLog,
    DashboardSnapshot,
    DashboardStatus,
    DashboardTokens,
    parse_response_sections,
    to_jsonable,
)
from application.services.runtime import AgentRuntime, SessionLifecycle


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _initial_snapshot(session_id: str, runtime: AgentRuntime) -> DashboardSnapshot:
    artifact = runtime.build_session_artifact(session_id)
    reasoning, conclusion, final_response = "Waiting for agent action...", "No conclusion yet.", "No response yet."
    if artifact.transcript:
        last_ai = next((item for item in reversed(artifact.transcript) if str(item.get("role", "")).lower() in {"ai", "assistant"}), None)
        if last_ai:
            reasoning, conclusion, final_response = parse_response_sections(str(last_ai.get("content", "")))
            if not conclusion:
                conclusion = "Response received from backend."
    agent = DashboardAgent(id="analysis", name="Analysis", status="running")
    context = runtime.build_context_budget(session_id)
    report = context.to_dict() if hasattr(context, "to_dict") else context.__dict__
    tokens = DashboardTokens(
        prompt=int(report.get("estimated_context_chars", 0) // 4),
        completion=int(report.get("estimated_remaining_chars", 0) // 4),
        total=int(report.get("estimated_tokens", report.get("estimated_context_chars", 0) // 4)),
    )
    events = [
        DashboardEvent(id="boot", kind="info", title="Session ready", detail=session_id, at=_now(), agentId=agent.id),
    ]
    logs = [
        DashboardLog(id="boot-log", level="info", message=f"Connected to session {session_id}", at=_now()),
    ]
    return DashboardSnapshot(
        activeAgent=agent,
        reasoning=reasoning,
        conclusion=conclusion,
        finalResponse=final_response,
        events=events,
        logs=logs,
        tokens=tokens,
        sessionId=session_id,
    )


async def _send_message(ws: WebSocketServerProtocol, message_type: str, payload: object) -> None:
    await ws.send(json.dumps({"type": message_type, "payload": to_jsonable(payload)}, ensure_ascii=False))


async def _handle_connection(ws: WebSocketServerProtocol, runtime: AgentRuntime) -> None:
    query = parse_qs(urlparse(ws.path).query)
    session_id = (query.get("session_id", [""])[0] or "").strip()
    resolved_session_id = runtime.select_session_id(session_id or None)
    lifecycle = SessionLifecycle(runtime=runtime, session_id=resolved_session_id)
    session_id = lifecycle.session_id

    await _send_message(ws, "status", DashboardStatus(connected=True, mode="websocket"))
    await _send_message(ws, "snapshot", await _initial_snapshot(session_id, runtime))

    try:
        async for raw in ws:
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await _send_message(ws, "log", DashboardLog(id="bad-json", level="warn", message="Invalid JSON payload", at=_now()))
                continue

            if message.get("type") != "action":
                await _send_message(ws, "log", DashboardLog(id="bad-type", level="warn", message=f"Unsupported message type: {message.get('type')}", at=_now()))
                continue

            payload = message.get("payload") or {}
            action = DashboardAction(agentId=str(payload.get("agentId", "analysis")), message=str(payload.get("message", "")).strip())
            if not action.message:
                continue

            await _send_message(
                ws,
                "event",
                DashboardEvent(id=f"action-{session_id}", kind="action", title="Action sent", detail=action.message, at=_now(), agentId=action.agentId),
            )

            try:
                session = lifecycle.resolve(action.message)
                if session.turn_context is None:
                    continue
                turn = await runtime.execute_turn(session.turn_context)
                reasoning, conclusion, final_response = parse_response_sections(turn.response)
                if not final_response:
                    final_response = turn.response
                agent_name = str(action.agentId or "analysis")
                tokens = DashboardTokens(
                    prompt=max(0, len(action.message) // 4),
                    completion=max(0, len(final_response) // 4),
                    total=max(0, (len(action.message) + len(final_response)) // 4),
                )
                events = [
                    DashboardEvent(id=f"turn-{turn.request_id}", kind="success", title="Action processed", detail=agent_name, at=_now(), agentId=agent_name),
                ]
                logs = [
                    DashboardLog(id=f"log-{turn.request_id}", level="info", message=turn.response[:240], at=_now()),
                ]
                await _send_message(
                    ws,
                    "snapshot",
                    DashboardSnapshot(
                        activeAgent=DashboardAgent(id=agent_name, name=agent_name.title(), status="running"),
                        reasoning=reasoning or turn.response,
                        conclusion=conclusion or "Backend turn completed.",
                        finalResponse=final_response,
                        events=events,
                        logs=logs,
                        tokens=tokens,
                        sessionId=session_id,
                    ),
                )
            except Exception as exc:
                await _send_message(ws, "log", DashboardLog(id="turn-error", level="error", message=str(exc), at=_now()))
    finally:
        try:
            await lifecycle.close()
        finally:
            await runtime.shutdown()


async def serve_frontend_bridge(host: str = "127.0.0.1", port: int = 8787) -> None:
    async def handler(ws: WebSocketServerProtocol):
        runtime = AgentRuntime()
        await _handle_connection(ws, runtime)

    async with serve(handler, host, port):
        print(f"[frontend-bridge] websocket server en {host}:{port}")
        await asyncio.Future()
