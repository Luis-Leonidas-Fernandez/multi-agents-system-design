"""Delegación de trabajo en background con persistencia por sesión.

Este servicio modela tareas largas como un lifecycle explícito:
queued -> running -> completed/failed/cancelled.
La persistencia es append-only por snapshot para que el estado se pueda
reconstruir tras reinicios sin perder historial.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import asyncio
import logging
from pathlib import Path
import time
import uuid
from typing import Any, Awaitable, Callable, Mapping, Literal, cast

from core.persistence.append_only_store import AppendOnlyStore


BackgroundTaskStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
BackgroundTaskStateKind = Literal["active", "terminal"]

BACKGROUND_TASK_ACTIVE_STATUSES: tuple[BackgroundTaskStatus, ...] = ("queued", "running")
BACKGROUND_TASK_TERMINAL_STATUSES: tuple[BackgroundTaskStatus, ...] = ("completed", "failed", "cancelled")


@dataclass(frozen=True)
class BackgroundTaskState:
    task_id: str
    session_id: str
    title: str
    status: BackgroundTaskStatus
    state_kind: BackgroundTaskStateKind
    attempt_number: int
    parent_task_id: str | None
    created_at_ms: int
    updated_at_ms: int
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    cancelled_at_ms: int | None = None
    cancel_reason: str | None = None
    request_id: str | None = None
    trace_id: str | None = None
    result: Any | None = None
    error: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class BackgroundTaskSummary:
    session_id: str
    total: int
    queued: int
    running: int
    completed: int
    failed: int
    cancelled: int
    active: int
    terminal: int
    retryable: int
    latest_updated_at_ms: int | None


@dataclass(frozen=True)
class BackgroundTaskRecord:
    task_id: str
    session_id: str
    title: str
    status: BackgroundTaskStatus
    attempt_number: int
    parent_task_id: str | None
    created_at_ms: int
    updated_at_ms: int
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    cancelled_at_ms: int | None = None
    cancel_reason: str | None = None
    request_id: str | None = None
    trace_id: str | None = None
    result: Any | None = None
    error: str | None = None
    metadata: Mapping[str, Any] | None = None


class BackgroundTaskStore:
    """Store append-only de snapshots de tareas por sesión."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or Path("infra") / "sessions"

    def _session_path(self, session_id: str) -> Path:
        return self._base_dir / session_id / "BACKGROUND_TASKS.jsonl"

    def append_record(self, record: BackgroundTaskRecord) -> None:
        AppendOnlyStore.append(self._session_path(record.session_id), record)

    def load_records(self, session_id: str) -> list[dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for record in AppendOnlyStore.load_lines(self._session_path(session_id)):
            task_id = str(record.get("task_id") or "")
            if task_id:
                latest[task_id] = record
        return sorted(latest.values(), key=lambda r: (r.get("created_at_ms") or 0, r.get("updated_at_ms") or 0, r.get("task_id") or ""))

    def list_sessions(self) -> list[str]:
        if not self._base_dir.exists():
            return []
        sessions: list[str] = []
        for session_dir in self._base_dir.iterdir():
            if session_dir.is_dir() and (session_dir / "BACKGROUND_TASKS.jsonl").exists():
                sessions.append(session_dir.name)
        return sorted(sessions)

    def get_record(self, session_id: str, task_id: str) -> dict[str, Any] | None:
        for record in self.load_records(session_id):
            if record.get("task_id") == task_id:
                return record
        return None

    def find_records(
        self,
        session_id: str,
        *,
        task_id: str | None = None,
        status: BackgroundTaskStatus | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> list[dict[str, Any]]:
        records = self.load_records(session_id)
        filtered: list[dict[str, Any]] = []
        for record in records:
            if task_id and record.get("task_id") != task_id:
                continue
            if status and record.get("status") != status:
                continue
            if request_id and record.get("request_id") != request_id:
                continue
            if trace_id and record.get("trace_id") != trace_id:
                continue
            filtered.append(record)
        return filtered


class BackgroundTaskService:
    """Crea y ejecuta tareas largas en background con estado persistido."""

    def __init__(
        self,
        store: BackgroundTaskStore | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._store = store or BackgroundTaskStore()
        self._clock = clock or time.time
        self._running: dict[str, asyncio.Task[Any]] = {}
        self._log = logging.getLogger(__name__)

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    def _persist(self, record: BackgroundTaskRecord) -> None:
        self._store.append_record(record)

    def _emit(self, record: BackgroundTaskRecord) -> None:
        self._log.info(
            "background_task session=%s task=%s status=%s title=%s",
            record.session_id,
            record.task_id,
            record.status,
            record.title,
        )

    def _save(self, record: BackgroundTaskRecord) -> BackgroundTaskRecord:
        self._persist(record)
        self._emit(record)
        return record

    def _update(
        self,
        record: BackgroundTaskRecord,
        *,
        status: BackgroundTaskStatus,
        result: Any | None = None,
        error: str | None = None,
        started_at_ms: int | None = None,
        finished_at_ms: int | None = None,
        cancelled_at_ms: int | None = None,
        cancel_reason: str | None = None,
    ) -> BackgroundTaskRecord:
        if status == "cancelled" and cancel_reason is None:
            cancel_reason = record.cancel_reason
        return replace(
            record,
            status=status,
            updated_at_ms=self._now_ms(),
            result=result if result is not None else record.result,
            error=error,
            started_at_ms=started_at_ms if started_at_ms is not None else record.started_at_ms,
            finished_at_ms=finished_at_ms if finished_at_ms is not None else record.finished_at_ms,
            cancelled_at_ms=cancelled_at_ms if cancelled_at_ms is not None else record.cancelled_at_ms,
            cancel_reason=cancel_reason if cancel_reason is not None else record.cancel_reason,
        )

    def _state_kind(self, status: BackgroundTaskStatus) -> BackgroundTaskStateKind:
        return "active" if status in BACKGROUND_TASK_ACTIVE_STATUSES else "terminal"

    def _to_state(self, record: dict[str, Any]) -> BackgroundTaskState:
        status = cast(BackgroundTaskStatus, record.get("status", "queued"))
        return BackgroundTaskState(
            task_id=str(record.get("task_id") or ""),
            session_id=str(record.get("session_id") or ""),
            title=str(record.get("title") or ""),
            status=status,
            state_kind=self._state_kind(status),
            attempt_number=int(record.get("attempt_number") or 1),
            parent_task_id=record.get("parent_task_id"),
            created_at_ms=int(record.get("created_at_ms") or 0),
            updated_at_ms=int(record.get("updated_at_ms") or 0),
            started_at_ms=record.get("started_at_ms"),
            finished_at_ms=record.get("finished_at_ms"),
            cancelled_at_ms=record.get("cancelled_at_ms"),
            cancel_reason=record.get("cancel_reason"),
            request_id=record.get("request_id"),
            trace_id=record.get("trace_id"),
            result=record.get("result"),
            error=record.get("error"),
            metadata=record.get("metadata"),
        )

    async def submit(
        self,
        session_id: str,
        title: str,
        runner: Callable[[], Awaitable[Any]],
        *,
        request_id: str | None = None,
        trace_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        parent_task_id: str | None = None,
        attempt_number: int = 1,
    ) -> BackgroundTaskRecord:
        task_id = str(uuid.uuid4())[:8]
        record = BackgroundTaskRecord(
            task_id=task_id,
            session_id=session_id,
            title=title,
            status="queued",
            attempt_number=attempt_number,
            parent_task_id=parent_task_id,
            created_at_ms=self._now_ms(),
            updated_at_ms=self._now_ms(),
            request_id=request_id,
            trace_id=trace_id,
            metadata=metadata,
        )
        self._save(record)

        async def _runner() -> None:
            running = self._save(self._update(record, status="running", started_at_ms=self._now_ms()))
            try:
                result = await runner()
            except asyncio.CancelledError:
                cancelled = self._save(
                    self._update(
                        running,
                        status="cancelled",
                        finished_at_ms=self._now_ms(),
                    )
                )
                self._running.pop(cancelled.task_id, None)
                raise
            except Exception as exc:
                failed = self._save(
                    self._update(
                        running,
                        status="failed",
                        finished_at_ms=self._now_ms(),
                        error=str(exc),
                    )
                )
                self._running.pop(failed.task_id, None)
            else:
                completed = self._save(
                    self._update(
                        running,
                        status="completed",
                        result=result,
                        finished_at_ms=self._now_ms(),
                    )
                )
                self._running.pop(completed.task_id, None)

        self._running[task_id] = asyncio.create_task(_runner())
        return record

    async def cancel_task(self, session_id: str, task_id: str, *, reason: str | None = None) -> dict[str, Any] | None:
        record = self.get_session_task(session_id, task_id)
        if record is None:
            return None
        if record.get("status") in BACKGROUND_TASK_TERMINAL_STATUSES:
            return record
        running = self._running.get(task_id)
        if running is None:
            cancelled = self._save(
                self._update(
                    self._to_record(record),
                    status="cancelled",
                    finished_at_ms=self._now_ms(),
                    cancelled_at_ms=self._now_ms(),
                    cancel_reason=reason,
                )
            )
            return asdict(cancelled)
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        cancelled = self.get_session_task(session_id, task_id)
        if cancelled is not None and reason is not None:
            cancelled = self._save(
                self._update(
                    self._to_record(cancelled),
                    status="cancelled",
                    finished_at_ms=self._now_ms(),
                    cancelled_at_ms=self._now_ms(),
                    cancel_reason=reason,
                )
            )
        if cancelled is None:
            return None
        return cancelled if isinstance(cancelled, dict) else asdict(cancelled)

    async def retry_task(
        self,
        session_id: str,
        task_id: str,
        runner: Callable[[], Awaitable[Any]],
        *,
        title: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> BackgroundTaskRecord | None:
        previous = self.get_session_task(session_id, task_id)
        if previous is None:
            return None
        previous_attempt = int(previous.get("attempt_number") or 1)
        merged_metadata: dict[str, Any] = dict(previous.get("metadata") or {})
        if metadata:
            merged_metadata.update(metadata)
        merged_metadata["retry_of"] = task_id
        return await self.submit(
            session_id,
            title or str(previous.get("title") or "retry task"),
            runner,
            request_id=request_id or previous.get("request_id"),
            trace_id=trace_id or previous.get("trace_id"),
            metadata=merged_metadata,
            parent_task_id=task_id,
            attempt_number=previous_attempt + 1,
        )

    def list_sessions(self) -> list[str]:
        return self._store.list_sessions()

    def load_session_tasks(self, session_id: str) -> list[dict[str, Any]]:
        return self._store.load_records(session_id)

    def list_active_session_tasks(self, session_id: str) -> list[dict[str, Any]]:
        return self.find_session_tasks(session_id, status="queued") + self.find_session_tasks(session_id, status="running")

    def describe_session(self, session_id: str) -> BackgroundTaskSummary:
        tasks = self.load_session_tasks(session_id)
        counts = {status: 0 for status in BACKGROUND_TASK_ACTIVE_STATUSES + BACKGROUND_TASK_TERMINAL_STATUSES}
        latest_updated_at_ms: int | None = None
        for task in tasks:
            status = task.get("status")
            if status in counts:
                counts[cast(BackgroundTaskStatus, status)] += 1
            updated_at_ms = task.get("updated_at_ms")
            if isinstance(updated_at_ms, int):
                latest_updated_at_ms = updated_at_ms if latest_updated_at_ms is None else max(latest_updated_at_ms, updated_at_ms)
        return BackgroundTaskSummary(
            session_id=session_id,
            total=len(tasks),
            queued=counts["queued"],
            running=counts["running"],
            completed=counts["completed"],
            failed=counts["failed"],
            cancelled=counts["cancelled"],
            active=counts["queued"] + counts["running"],
            terminal=counts["completed"] + counts["failed"] + counts["cancelled"],
            retryable=counts["failed"] + counts["cancelled"],
            latest_updated_at_ms=latest_updated_at_ms,
        )

    def describe_task(self, session_id: str, task_id: str) -> BackgroundTaskState | None:
        record = self.get_session_task(session_id, task_id)
        if record is None:
            return None
        return self._to_state(record)

    def find_session_tasks(
        self,
        session_id: str,
        *,
        task_id: str | None = None,
        status: BackgroundTaskStatus | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._store.find_records(
            session_id,
            task_id=task_id,
            status=status,
            request_id=request_id,
            trace_id=trace_id,
        )

    def get_session_task(self, session_id: str, task_id: str) -> dict[str, Any] | None:
        return self._store.get_record(session_id, task_id)

    def get_task(self, session_id: str, task_id: str) -> dict[str, Any] | None:
        return self.get_session_task(session_id, task_id)

    def _to_record(self, record: dict[str, Any]) -> BackgroundTaskRecord:
        return BackgroundTaskRecord(
            task_id=str(record.get("task_id") or ""),
            session_id=str(record.get("session_id") or ""),
            title=str(record.get("title") or ""),
            status=cast(BackgroundTaskStatus, record.get("status") or "queued"),
            attempt_number=int(record.get("attempt_number") or 1),
            parent_task_id=record.get("parent_task_id"),
            created_at_ms=int(record.get("created_at_ms") or self._now_ms()),
            updated_at_ms=int(record.get("updated_at_ms") or self._now_ms()),
            started_at_ms=record.get("started_at_ms"),
            finished_at_ms=record.get("finished_at_ms"),
            cancelled_at_ms=record.get("cancelled_at_ms"),
            cancel_reason=record.get("cancel_reason"),
            request_id=record.get("request_id"),
            trace_id=record.get("trace_id"),
            result=record.get("result"),
            error=record.get("error"),
            metadata=record.get("metadata"),
        )

    def running_task_count(self) -> int:
        return len(self._running)

    async def shutdown(self) -> None:
        if not self._running:
            return
        running = list(self._running.values())
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        self._running.clear()


background_task_store = BackgroundTaskStore()
background_task_service = BackgroundTaskService()


__all__ = [
    "BackgroundTaskRecord",
    "BackgroundTaskState",
    "BackgroundTaskSummary",
    "BackgroundTaskStatus",
    "BackgroundTaskStateKind",
    "BackgroundTaskStore",
    "BackgroundTaskService",
    "BACKGROUND_TASK_ACTIVE_STATUSES",
    "BACKGROUND_TASK_TERMINAL_STATUSES",
    "background_task_store",
    "background_task_service",
]
