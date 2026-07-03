"""WebSocket bridge for the frontend dashboard."""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from websockets.exceptions import ConnectionClosed
from websockets.server import WebSocketServerProtocol, serve

from application.frontend_bridge.protocol import (
    DashboardAction,
    DashboardAgent,
    DashboardArtifact,
    DashboardArtifactAction,
    DashboardEvent,
    DashboardLog,
    DashboardMoodleAuditTree,
    DashboardMoodleAuditTreeNode,
    DashboardMoodleAuditTreeStats,
    DashboardSnapshot,
    DashboardStatus,
    DashboardTokens,
    parse_response_sections,
    to_jsonable,
)
from application.services.request_runtime import use_request_runtime
from application.services.runtime import AgentRuntime, SessionLifecycle
from features.web_scraping.application.moodle_audit import load_moodle_audit_snapshot
from features.web_scraping.application.moodle_audit_tree import build_moodle_audit_tree
from features.web_scraping.application.moodle_artifacts import (
    approve_moodle_artifact,
    delete_moodle_artifact,
    list_session_moodle_artifacts,
)
from integrations.google_calendar_tools import create_calendar_events_from_validated_tasks_payload


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_artifacts(session_id: str) -> list[DashboardArtifact]:
    artifacts: list[DashboardArtifact] = []
    for item in list_session_moodle_artifacts(session_id):
        artifacts.append(DashboardArtifact(**item.to_dict()))
    print(
        f"[ARTIFACT_DEBUG][backend] session_id={session_id} artifacts_count={len(artifacts)} paths={[artifact.jsonPath for artifact in artifacts]}",
        flush=True,
    )
    return artifacts


_MOODLE_AUDIT_JSON_RE = re.compile(r"(?P<path>(?:/|data/)[^\s]+__moodle_audit_snapshot\.json)")
_MOODLE_AUDIT_SUMMARY_RE = re.compile(r"(?P<path>(?:/|data/)[^\s]+__moodle_audit_summary\.md)")


def _build_moodle_tree_node(payload: dict[str, object]) -> DashboardMoodleAuditTreeNode:
    return DashboardMoodleAuditTreeNode(
        id=str(payload.get("id") or ""),
        kind=str(payload.get("kind") or "unknown"),
        title=str(payload.get("title") or "Sin título"),
        url=str(payload.get("url") or ""),
        canonicalUrl=str(payload.get("canonicalUrl") or ""),
        previewUrl=str(payload.get("previewUrl") or ""),
        downloadUrl=str(payload.get("downloadUrl") or ""),
        redirectUrl=str(payload.get("redirectUrl") or ""),
        mimeType=str(payload.get("mimeType") or ""),
        subtitle=str(payload.get("subtitle") or ""),
        description=str(payload.get("description") or ""),
        badges=[str(item) for item in (payload.get("badges") or []) if str(item).strip()],
        metadata=dict(payload.get("metadata") or {}),
        children=[
            _build_moodle_tree_node(child)
            for child in (payload.get("children") or [])
            if isinstance(child, dict)
        ],
    )


def _extract_moodle_audit_tree(final_response: str) -> DashboardMoodleAuditTree | None:
    response = (final_response or "").strip()
    if not response:
        return None
    json_match = _MOODLE_AUDIT_JSON_RE.search(response)
    if not json_match:
        return None
    audit_path = Path(json_match.group("path"))
    if not audit_path.exists():
        return None
    summary_match = _MOODLE_AUDIT_SUMMARY_RE.search(response)
    summary_path = summary_match.group("path") if summary_match else ""
    try:
        snapshot = load_moodle_audit_snapshot(audit_path)
        payload = build_moodle_audit_tree(
            snapshot,
            audit_json_path=str(audit_path),
            summary_path=summary_path,
        )
    except Exception as exc:
        print(f"[MOODLE_AUDIT_TREE][backend] error={exc} path={audit_path}", flush=True)
        return None

    root_payload = payload.get("root")
    if not isinstance(root_payload, dict):
        return None
    stats_payload = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
    return DashboardMoodleAuditTree(
        jobUid=str(payload.get("jobUid") or ""),
        courseName=str(payload.get("courseName") or ""),
        auditPath=str(payload.get("auditPath") or audit_path),
        summaryPath=str(payload.get("summaryPath") or summary_path),
        stats=DashboardMoodleAuditTreeStats(
            pageCount=int(stats_payload.get("pageCount") or 0),
            retainedPageCount=int(stats_payload.get("retainedPageCount") or 0),
            externalRedirectCount=int(stats_payload.get("externalRedirectCount") or 0),
            downloadDocumentCount=int(stats_payload.get("downloadDocumentCount") or 0),
            assignmentLikeCount=int(stats_payload.get("assignmentLikeCount") or 0),
            resourceTypeCounts={
                str(key): int(value or 0)
                for key, value in dict(stats_payload.get("resourceTypeCounts") or {}).items()
            },
        ),
        root=_build_moodle_tree_node(root_payload),
    )


async def _build_snapshot(
    session_id: str,
    runtime: AgentRuntime,
    *,
    agent: DashboardAgent,
    reasoning: str = "",
    conclusion: str = "",
    final_response: str = "",
    turn_id: str = "",
    turn_latency_ms: int = 0,
    events: list[DashboardEvent] | None = None,
    logs: list[DashboardLog] | None = None,
    fallback_last_user_message: str = "",
) -> DashboardSnapshot:
    artifact = runtime.build_session_artifact(session_id)
    last_user_message = fallback_last_user_message
    if artifact.transcript:
        if not final_response:
            last_ai = next((item for item in reversed(artifact.transcript) if str(item.get("role", "")).lower() in {"ai", "assistant"}), None)
            if last_ai:
                reasoning, conclusion, final_response = parse_response_sections(str(last_ai.get("content", "")))
        last_user = next((item for item in reversed(artifact.transcript) if str(item.get("role", "")).lower() in {"human", "user", "you"}), None)
        if last_user:
            last_user_message = str(last_user.get("content", "")).strip()
    context = runtime.build_context_budget(session_id)
    report = context.to_dict() if hasattr(context, "to_dict") else context.__dict__
    tokens = DashboardTokens(
        prompt=int(report.get("estimated_context_chars", 0) // 4),
        completion=int(report.get("estimated_remaining_chars", 0) // 4),
        total=int(report.get("estimated_tokens", report.get("estimated_context_chars", 0) // 4)),
    )
    return DashboardSnapshot(
        activeAgent=agent,
        reasoning=reasoning,
        conclusion=conclusion,
        finalResponse=final_response,
        turnId=turn_id,
        turnLatencyMs=turn_latency_ms,
        messageCount=len(artifact.transcript),
        lastUserMessage=last_user_message,
        lastAssistantResponse=final_response,
        events=events or [],
        logs=logs or [],
        tokens=tokens,
        sessionId=session_id,
        artifacts=_build_artifacts(session_id),
        moodleAuditTree=_extract_moodle_audit_tree(final_response),
    )


async def _initial_snapshot(session_id: str, runtime: AgentRuntime) -> DashboardSnapshot:
    agent = DashboardAgent(id="analysis", name="Analysis", status="running")
    events = [
        DashboardEvent(id="boot", kind="info", title="Session ready", detail=session_id, at=_now(), agentId=agent.id),
    ]
    logs = [
        DashboardLog(id="boot-log", level="info", message=f"Connected to session {session_id}", at=_now()),
    ]
    return await _build_snapshot(session_id, runtime, agent=agent, events=events, logs=logs)


async def _send_message(ws: WebSocketServerProtocol, message_type: str, payload: object) -> None:
    try:
        await ws.send(json.dumps({"type": message_type, "payload": to_jsonable(payload)}, ensure_ascii=False))
    except ConnectionClosed:
        return


async def _handle_connection(ws: WebSocketServerProtocol, runtime: AgentRuntime) -> None:
    query = parse_qs(urlparse(ws.path).query)
    session_id = (query.get("session_id", [""])[0] or "").strip()
    resolved_session_id = runtime.select_session_id(session_id or None)
    lifecycle = SessionLifecycle(runtime=runtime, session_id=resolved_session_id)
    session_id = lifecycle.session_id
    current_turn_task: asyncio.Task | None = None

    async def send_refresh_snapshot(
        *,
        agent_name: str = "analysis",
        reasoning: str = "",
        conclusion: str = "",
        final_response: str = "",
        turn_id: str = "",
        turn_latency_ms: int = 0,
        events: list[DashboardEvent] | None = None,
        logs: list[DashboardLog] | None = None,
        fallback_last_user_message: str = "",
    ) -> None:
        snapshot = await _build_snapshot(
            session_id,
            runtime,
            agent=DashboardAgent(id=agent_name, name=agent_name.title(), status="running"),
            reasoning=reasoning,
            conclusion=conclusion,
            final_response=final_response,
            turn_id=turn_id,
            turn_latency_ms=turn_latency_ms,
            events=events,
            logs=logs,
            fallback_last_user_message=fallback_last_user_message,
        )
        print(
            f"[ARTIFACT_DEBUG][backend] send_refresh_snapshot session_id={session_id} artifacts_count={len(snapshot.artifacts)} final_response_chars={len(snapshot.finalResponse or '')}",
            flush=True,
        )
        await _send_message(ws, "snapshot", snapshot)

    async def run_turn(action: DashboardAction) -> None:
        nonlocal current_turn_task
        try:
            session = lifecycle.resolve(action.message, enabled_mcps=tuple(action.enabledMcps))
            if session.turn_context is None:
                return
            start_time = asyncio.get_running_loop().time()
            turn = await runtime.execute_turn(session.turn_context)
            latency_ms = max(0, int((asyncio.get_running_loop().time() - start_time) * 1000))
            reasoning, conclusion, final_response = parse_response_sections(turn.response)
            if not final_response:
                final_response = turn.response
            agent_name = str(action.agentId or "analysis")
            events = [
                DashboardEvent(id=f"turn-{turn.request_id}", kind="success", title="Action processed", detail=agent_name, at=_now(), agentId=agent_name),
            ]
            logs = [
                DashboardLog(id=f"log-{turn.request_id}", level="info", message=turn.response[:240], at=_now()),
            ]
            await send_refresh_snapshot(
                agent_name=agent_name,
                reasoning=reasoning or turn.response,
                conclusion=conclusion or "Backend turn completed.",
                final_response=final_response,
                turn_id=turn.request_id,
                turn_latency_ms=latency_ms,
                events=events,
                logs=logs,
                fallback_last_user_message=action.message,
            )
        except asyncio.CancelledError:
            try:
                await _send_message(ws, "log", DashboardLog(id="turn-aborted", level="info", message="Turn cancelled by user", at=_now()))
            except ConnectionClosed:
                return
            raise
        except Exception as exc:
            try:
                await _send_message(ws, "log", DashboardLog(id="turn-error", level="error", message=str(exc), at=_now()))
            except ConnectionClosed:
                return
        finally:
            current_turn_task = None

    async def run_artifact_action(action: DashboardArtifactAction) -> None:
        logs: list[DashboardLog] = []
        events: list[DashboardEvent] = []
        final_response = ""
        try:
            if action.kind == "approve":
                approve_moodle_artifact(action.artifactPath, approved=True)
                events.append(DashboardEvent(id=f"artifact-approve-{session_id}", kind="success", title="Artifact aprobado", detail=action.artifactPath, at=_now(), agentId="analysis"))
                logs.append(DashboardLog(id=f"artifact-approve-log-{session_id}", level="info", message=f"Artifact aprobado: {action.artifactPath}", at=_now()))
                final_response = "JSON aprobado. Quedó listo en el chat para crear eventos."
            elif action.kind == "delete":
                delete_moodle_artifact(action.artifactPath)
                events.append(DashboardEvent(id=f"artifact-delete-{session_id}", kind="warning", title="Artifact eliminado", detail=action.artifactPath, at=_now(), agentId="analysis"))
                logs.append(DashboardLog(id=f"artifact-delete-log-{session_id}", level="info", message=f"Artifact eliminado: {action.artifactPath}", at=_now()))
                final_response = "Artifact eliminado del chat."
            elif action.kind == "create_events":
                with use_request_runtime(session_id=session_id, request_id="artifact-action", enabled_mcps=tuple(action.enabledMcps)):
                    payload = create_calendar_events_from_validated_tasks_payload(action.artifactPath)
                if isinstance(payload, str):
                    raise RuntimeError(payload)
                created_count = int(payload.get("created_count") or 0)
                events.append(DashboardEvent(id=f"artifact-sync-{session_id}", kind="success", title="Eventos creados", detail=str(created_count), at=_now(), agentId="analysis"))
                logs.append(DashboardLog(id=f"artifact-sync-log-{session_id}", level="info", message=f"Se crearon {created_count} eventos desde {action.artifactPath}", at=_now()))
                final_response = json.dumps(payload, ensure_ascii=False, indent=2)
            else:
                raise RuntimeError(f"Acción de artifact no soportada: {action.kind}")

            await send_refresh_snapshot(
                agent_name="analysis",
                reasoning="Artifact workflow",
                conclusion="Acción de artifact ejecutada.",
                final_response=final_response,
                events=events,
                logs=logs,
            )
        except Exception as exc:
            await _send_message(ws, "log", DashboardLog(id="artifact-error", level="error", message=str(exc), at=_now()))

    try:
        await _send_message(ws, "status", DashboardStatus(connected=True, mode="websocket"))
        initial_snapshot = await _initial_snapshot(session_id, runtime)
        print(
            f"[ARTIFACT_DEBUG][backend] initial_snapshot session_id={session_id} artifacts_count={len(initial_snapshot.artifacts)}",
            flush=True,
        )
        await _send_message(ws, "snapshot", initial_snapshot)
    except ConnectionClosed:
        return

    try:
        async for raw in ws:
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await _send_message(ws, "log", DashboardLog(id="bad-json", level="warn", message="Invalid JSON payload", at=_now()))
                continue

            message_type = message.get("type")
            if message_type == "abort":
                if current_turn_task is None or current_turn_task.done():
                    await _send_message(ws, "log", DashboardLog(id="no-turn", level="warn", message="No active turn to abort", at=_now()))
                    continue
                current_turn_task.cancel()
                try:
                    await current_turn_task
                except asyncio.CancelledError:
                    pass
                continue

            if message_type == "artifact_action":
                payload = message.get("payload") or {}
                action = DashboardArtifactAction(
                    kind=str(payload.get("kind", "approve")),  # type: ignore[arg-type]
                    artifactPath=str(payload.get("artifactPath", "")).strip(),
                    enabledMcps=[str(item).strip() for item in (payload.get("enabledMcps") or []) if str(item).strip()],
                )
                if not action.artifactPath:
                    await _send_message(ws, "log", DashboardLog(id="artifact-missing-path", level="warn", message="Artifact path missing", at=_now()))
                    continue
                await run_artifact_action(action)
                continue

            if message_type != "action":
                await _send_message(ws, "log", DashboardLog(id="bad-type", level="warn", message=f"Unsupported message type: {message.get('type')}", at=_now()))
                continue

            payload = message.get("payload") or {}
            enabled_mcps = payload.get("enabledMcps") or []
            action = DashboardAction(
                agentId=str(payload.get("agentId", "analysis")),
                message=str(payload.get("message", "")).strip(),
                enabledMcps=[str(item).strip() for item in enabled_mcps if str(item).strip()],
            )
            if not action.message:
                continue

            await _send_message(
                ws,
                "event",
                DashboardEvent(id=f"action-{session_id}", kind="action", title="Action sent", detail=action.message, at=_now(), agentId=action.agentId),
            )

            if current_turn_task is not None and not current_turn_task.done():
                await _send_message(ws, "log", DashboardLog(id="busy-turn", level="warn", message="A turn is already running", at=_now()))
                continue

            current_turn_task = asyncio.create_task(run_turn(action))
    except ConnectionClosed:
        return
    finally:
        try:
            if current_turn_task is not None and not current_turn_task.done():
                current_turn_task.cancel()
                try:
                    await current_turn_task
                except asyncio.CancelledError:
                    pass
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
