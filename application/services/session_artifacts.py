"""Artefactos de sesión consolidados.

Este módulo agrupa el transcript, la memoria, los eventos de audit, las
tareas delegadas en background, el presupuesto de contexto y los bookmarks
persistidos en un archivo JSON por sesión.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import time
from typing import Any

from application.services.session_memory import SessionMemory, memory as session_memory
from application.services.session_persistence import SessionPersistence, persistence as session_persistence
from application.services.background_tasks import BackgroundTaskService, background_task_service
from application.services.context_budget import SessionContextBudgetService, context_budget_service, context_budget_to_dict
from application.services.session_bookmarks import SessionBookmarkService, session_bookmark_service
from application.services.prompt_versioning import PromptVersionService, prompt_version_service
from application.services.tool_audit import ToolAuditService, tool_audit_service


@dataclass(frozen=True)
class SessionArtifact:
    session_id: str
    generated_at_ms: int
    message_count: int
    has_memory: bool
    is_existing_session: bool
    transcript: list[dict[str, Any]]
    memory_markdown: str
    audit_events: list[dict[str, Any]]
    background_tasks: list[dict[str, Any]]
    background_task_summary: dict[str, Any]
    prompt_snapshots: list[dict[str, Any]]
    context_budget: dict[str, Any]
    bookmarks: list[dict[str, Any]]
    trace_ids: list[str]


class SessionArtifactStore:
    """Store de artefactos por sesión en JSON."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or Path("sessions")

    def _session_dir(self, session_id: str) -> Path:
        return self._base_dir / session_id

    def _artifact_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "SESSION_ARTIFACT.json"

    def artifact_path(self, session_id: str) -> Path:
        return self._artifact_path(session_id)

    def save(self, artifact: SessionArtifact) -> None:
        path = self._artifact_path(artifact.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(artifact), ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, session_id: str) -> dict[str, Any] | None:
        path = self._artifact_path(session_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_sessions(self) -> list[str]:
        if not self._base_dir.exists():
            return []
        sessions: list[str] = []
        for session_dir in self._base_dir.iterdir():
            if session_dir.is_dir() and (session_dir / "SESSION_ARTIFACT.json").exists():
                sessions.append(session_dir.name)
        return sorted(sessions)


class SessionArtifactService:
    """Construye y persiste artefactos consolidados de sesión."""

    def __init__(
        self,
        persistence_backend: SessionPersistence | None = None,
        memory_backend: SessionMemory | None = None,
        audit_backend: ToolAuditService | None = None,
        background_task_backend: BackgroundTaskService | None = None,
        context_budget_backend: SessionContextBudgetService | None = None,
        bookmark_backend: SessionBookmarkService | None = None,
        prompt_version_backend: PromptVersionService | None = None,
        store: SessionArtifactStore | None = None,
    ) -> None:
        self._persistence = persistence_backend or session_persistence
        self._memory = memory_backend or session_memory
        self._audit = audit_backend or tool_audit_service
        self._background_tasks = background_task_backend or background_task_service
        self._context_budget = context_budget_backend or context_budget_service
        self._bookmarks = bookmark_backend or session_bookmark_service
        self._prompt_versions = prompt_version_backend or prompt_version_service
        self._store = store or SessionArtifactStore()

    def build_artifact(self, session_id: str) -> SessionArtifact:
        messages = self._persistence.load_messages(session_id)
        memory_markdown = self._memory.load_memory_context(session_id)
        transcript = [
            {
                "role": getattr(msg, "type", "ai"),
                "content": getattr(msg, "content", ""),
            }
            for msg in messages
        ]
        audit_events = self._audit.load_session_events(session_id)
        background_tasks = self._background_tasks.load_session_tasks(session_id)
        background_task_summary = asdict(self._background_tasks.describe_session(session_id))
        prompt_snapshots = [
            {
                **snapshot,
                "snapshot_path": str(self._prompt_versions.snapshot_path(agent)),
                "history_path": str(self._prompt_versions.history_path(agent)),
            }
            for agent in self._prompt_versions.list_agents()
            if (snapshot := self._prompt_versions.load_snapshot(agent)) is not None
        ]
        context_budget = context_budget_to_dict(self._context_budget.build_report(session_id))
        bookmarks = self._bookmarks.list(session_id)
        trace_ids = sorted({str(event.get("trace_id")) for event in audit_events if event.get("trace_id")})
        return SessionArtifact(
            session_id=session_id,
            generated_at_ms=int(time.time() * 1000),
            message_count=len(transcript),
            has_memory=bool(memory_markdown),
            is_existing_session=session_id in self._persistence.list_sessions(),
            transcript=transcript,
            memory_markdown=memory_markdown,
            audit_events=audit_events,
            background_tasks=background_tasks,
            background_task_summary=background_task_summary,
            prompt_snapshots=prompt_snapshots,
            context_budget=context_budget,
            bookmarks=bookmarks,
            trace_ids=trace_ids,
        )

    def export_artifact(self, session_id: str) -> SessionArtifact:
        artifact = self.build_artifact(session_id)
        self._store.save(artifact)
        return artifact

    def load_artifact(self, session_id: str) -> dict[str, Any] | None:
        return self._store.load(session_id)

    def list_sessions(self) -> list[str]:
        return self._store.list_sessions()

    def artifact_path(self, session_id: str) -> Path:
        return self._store.artifact_path(session_id)


session_artifact_store = SessionArtifactStore()
session_artifact_service = SessionArtifactService()


__all__ = ["SessionArtifact", "SessionArtifactStore", "SessionArtifactService", "session_artifact_store", "session_artifact_service"]
