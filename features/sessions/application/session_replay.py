"""Replay unificado de sesión para inspección en CLI.

Combina transcript, prompts, background tasks y audit trail en una vista
ordenada por secciones para facilitar debugging y revisión.
"""
from __future__ import annotations

from dataclasses import dataclass
from textwrap import shorten
from typing import Any

from features.sessions.application.session_artifacts import session_artifact_service


@dataclass(frozen=True)
class ReplayTimelineItem:
    section: str
    order: int
    title: str
    detail: str


@dataclass(frozen=True)
class SessionReplay:
    session_id: str
    generated_at_ms: int
    items: list[ReplayTimelineItem]


class SessionReplayService:
    def build_replay(self, session_id: str) -> SessionReplay:
        artifact = session_artifact_service.build_artifact(session_id)
        items: list[ReplayTimelineItem] = []

        items.append(
            ReplayTimelineItem(
                section="snapshot",
                order=0,
                title="session snapshot",
                detail=f"messages={artifact.message_count} memory={'yes' if artifact.has_memory else 'no'} existing={'yes' if artifact.is_existing_session else 'no'}",
            )
        )

        for index, snapshot in enumerate(artifact.prompt_snapshots, start=1):
            items.append(
                ReplayTimelineItem(
                    section="prompt",
                    order=10 + index,
                    title=str(snapshot.get("agent_name") or "prompt"),
                    detail=f"version={snapshot.get('prompt_version', '?')} hash={str(snapshot.get('prompt_hash', '?'))[:12]} path={snapshot.get('snapshot_path', '?')}",
                )
            )

        for index, message in enumerate(artifact.transcript, start=1):
            items.append(
                ReplayTimelineItem(
                    section="message",
                    order=100 + index,
                    title=str(message.get("role") or "message"),
                    detail=shorten(str(message.get("content") or ""), width=120, placeholder="…"),
                )
            )

        for index, task in enumerate(artifact.background_tasks, start=1):
            items.append(
                ReplayTimelineItem(
                    section="background-task",
                    order=200 + index,
                    title=str(task.get("task_id") or f"task-{index}"),
                    detail=f"status={task.get('status', '?')} attempt={task.get('attempt_number', 1)} parent={task.get('parent_task_id', '-') or '-'} title={task.get('title', '')}",
                )
            )

        for index, event in enumerate(sorted(artifact.audit_events, key=lambda event: int(event.get("ts_ms") or 0)), start=1):
            items.append(
                ReplayTimelineItem(
                    section="tool-audit",
                    order=300 + index,
                    title=str(event.get("event_type") or "tool_event"),
                    detail=f"tool={event.get('tool_name', '?')} outcome={event.get('outcome', '?')} request={event.get('request_id', '?')} trace={event.get('trace_id', '?')}",
                )
            )

        items.append(
            ReplayTimelineItem(
                section="summary",
                order=999,
                title="artifact summary",
                detail=f"audit={len(artifact.audit_events)} tasks={len(artifact.background_tasks)} prompts={len(artifact.prompt_snapshots)} traces={len(artifact.trace_ids)}",
            )
        )

        return SessionReplay(session_id=session_id, generated_at_ms=artifact.generated_at_ms, items=items)

    def list_sessions(self) -> list[str]:
        return session_artifact_service.list_sessions()


def format_session_replay(replay: SessionReplay) -> list[str]:
    lines = [f"[replay] sesión={replay.session_id} items={len(replay.items)} generated_at_ms={replay.generated_at_ms}"]
    for item in sorted(replay.items, key=lambda item: (item.order, item.section, item.title)):
        lines.append(f"  [{item.section}] {item.title} — {item.detail}")
    return lines


session_replay_service = SessionReplayService()


__all__ = ["ReplayTimelineItem", "SessionReplay", "SessionReplayService", "format_session_replay", "session_replay_service"]
