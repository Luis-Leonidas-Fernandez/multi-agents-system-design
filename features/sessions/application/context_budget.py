"""Presupuesto y estado de contexto para sesiones.

Expone una vista legible de qué entra al contexto del turno, qué viene
resumido desde memoria persistida y qué queda afuera como observabilidad.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import os
import time
from typing import Any, Literal

from features.sessions.application.background_tasks import BackgroundTaskService, background_task_service
from features.sessions.application.memory_retrieval import MemoryRetrievalService, memory_retrieval_service
from features.sessions.application.prompt_versioning import PromptVersionService, prompt_version_service
from features.sessions.application.session_bookmarks import SessionBookmarkStore, session_bookmark_store
from features.sessions.application.session_memory import SessionMemory, memory as session_memory
from features.sessions.application.session_persistence import SessionPersistence, persistence as session_persistence
from application.services.tool_audit import ToolAuditService, tool_audit_service


ContextBudgetRole = Literal["included", "summarized", "excluded"]


@dataclass(frozen=True)
class ContextBudgetItem:
    section: str
    role: ContextBudgetRole
    chars: int
    detail: str


@dataclass(frozen=True)
class SessionContextBudget:
    session_id: str
    generated_at_ms: int
    budget_chars: int
    estimated_context_chars: int
    estimated_remaining_chars: int
    estimated_tokens: int
    status: str
    scope: str
    transcript_message_count: int
    memory_present: bool
    items: list[ContextBudgetItem]


class SessionContextBudgetService:
    """Calcula una vista de presupuesto de contexto para una sesión."""

    def __init__(
        self,
        persistence_backend: SessionPersistence | None = None,
        memory_backend: SessionMemory | None = None,
        prompt_version_backend: PromptVersionService | None = None,
        audit_backend: ToolAuditService | None = None,
        background_task_backend: BackgroundTaskService | None = None,
        bookmark_store: SessionBookmarkStore | None = None,
        budget_chars: int | None = None,
        memory_backend_retrieval: MemoryRetrievalService | None = None,
    ) -> None:
        self._persistence = persistence_backend or session_persistence
        self._memory = memory_backend or session_memory
        self._prompt_versions = prompt_version_backend or prompt_version_service
        self._audit = audit_backend or tool_audit_service
        self._background_tasks = background_task_backend or background_task_service
        self._bookmarks = bookmark_store or session_bookmark_store
        self._memory_retrieval = memory_backend_retrieval or memory_retrieval_service
        self._budget_chars = budget_chars or self._load_budget_chars()

    def _load_budget_chars(self) -> int:
        raw = os.getenv("CONTEXT_BUDGET_CHARS", "120000").strip()
        try:
            return max(1, int(raw))
        except ValueError:
            return 120000

    def _message_chars(self, messages: list[Any]) -> int:
        return sum(len(str(getattr(message, "content", ""))) for message in messages)

    def _snapshot_chars(self, snapshot: dict[str, Any] | None) -> int:
        if not snapshot:
            return 0
        return len(str(snapshot.get("system_prompt", ""))) + len(str(snapshot.get("extra_context", "")))

    def build_report(self, session_id: str, agent_name: str | None = None) -> SessionContextBudget:
        messages = self._persistence.load_messages(session_id)
        memory_markdown = self._memory.load_memory_context(session_id)
        prompt_snapshot = self._prompt_versions.load_snapshot(agent_name) if agent_name else None
        audit_events = self._audit.load_session_events(session_id)
        background_tasks = self._background_tasks.load_session_tasks(session_id)
        bookmarks = self._bookmarks.list(session_id)
        matched_sessions = self._memory_retrieval.list_sessions()

        transcript_chars = self._message_chars(messages)
        memory_chars = len(memory_markdown)
        prompt_chars = self._snapshot_chars(prompt_snapshot)
        estimated_context_chars = transcript_chars + memory_chars + prompt_chars
        estimated_remaining_chars = self._budget_chars - estimated_context_chars
        estimated_tokens = max(0, estimated_context_chars // 4)
        if estimated_context_chars > self._budget_chars:
            status = "exceeded"
        elif estimated_context_chars >= int(self._budget_chars * 0.85):
            status = "warn"
        else:
            status = "ok"

        items: list[ContextBudgetItem] = []
        if prompt_snapshot:
            items.append(
                ContextBudgetItem(
                    section="system_prompt",
                    role="included",
                    chars=len(str(prompt_snapshot.get("system_prompt", ""))),
                    detail=f"{agent_name} version={prompt_snapshot.get('prompt_version', '?')}",
                )
            )
            items.append(
                ContextBudgetItem(
                    section="extra_context",
                    role="included",
                    chars=len(str(prompt_snapshot.get("extra_context", ""))),
                    detail=f"{agent_name} extra context",
                )
            )

        items.append(
            ContextBudgetItem(
                section="transcript",
                role="included",
                chars=transcript_chars,
                detail=f"{len(messages)} mensajes persistidos",
            )
        )
        items.append(
            ContextBudgetItem(
                section="memory",
                role="summarized",
                chars=memory_chars,
                detail="MEMORY.md inyectado desde sesiones previas" if memory_markdown else "sin memoria destilada",
            )
        )
        items.append(
            ContextBudgetItem(
                section="observability",
                role="excluded",
                chars=0,
                detail=(
                    f"audit_events={len(audit_events)} background_tasks={len(background_tasks)} "
                    f"bookmarks={len(bookmarks)} memory_sessions={len(matched_sessions)}"
                ),
            )
        )

        return SessionContextBudget(
            session_id=session_id,
            generated_at_ms=int(time.time() * 1000),
            budget_chars=self._budget_chars,
            estimated_context_chars=estimated_context_chars,
            estimated_remaining_chars=estimated_remaining_chars,
            estimated_tokens=estimated_tokens,
            status=status,
            scope=agent_name or "session",
            transcript_message_count=len(messages),
            memory_present=bool(memory_markdown),
            items=items,
        )


def context_budget_to_dict(report: SessionContextBudget) -> dict[str, Any]:
    return asdict(report)


context_budget_service = SessionContextBudgetService()


__all__ = [
    "ContextBudgetItem",
    "ContextBudgetRole",
    "SessionContextBudget",
    "SessionContextBudgetService",
    "context_budget_service",
    "context_budget_to_dict",
]
