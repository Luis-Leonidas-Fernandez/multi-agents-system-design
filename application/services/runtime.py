"""Runtime/orchestrator de alto nivel para el sistema multi-agentes.

Este módulo introduce una frontera explícita entre el CLI y la ejecución
del grafo. La idea viene de la capa de orquestación de Claude Code: un
runtime por conversación que conoce el estado de sesión, resuelve la
sesión activa y delega la ejecución de cada turno.
"""
# pyright: reportRedeclaration=false
from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Optional

from application.services.session_gateway import AgentGateway
from application.services.background_tasks import BackgroundTaskService, BackgroundTaskState, BackgroundTaskSummary, background_task_service
from application.services.coordinator_workers import coordinator_runtime_service
from application.services.context_budget import SessionContextBudget, SessionContextBudgetService, context_budget_service, context_budget_to_dict
from application.services.session_artifacts import session_artifact_service
from application.services.session_bookmarks import SessionBookmarkService, session_bookmark_service
from application.services.session_replay import session_replay_service, SessionReplay
from application.services.memory_retrieval import MemoryRetrievalService, memory_retrieval_service, MemorySearchHit
from application.services.tool_approval import ToolApprovalPreview, tool_approval_service
from application.services.trace_context import TraceContext, TraceContextService, trace_context_service
from application.services.session_persistence import SessionPersistence, persistence as session_persistence
from application.services.session_inspection import format_session_banner


@dataclass(frozen=True)
class RuntimeSessionSnapshot:
    """Resumen liviano de una sesión antes de procesarla."""

    session_id: str
    message_count: int
    has_memory: bool
    is_existing_session: bool


@dataclass(frozen=True)
class RuntimeTurnResult:
    """Resultado de un turno ejecutado por el runtime."""

    session_id: str
    request_id: str
    response: str
    message_count: int
    has_memory: bool
    trace: TraceContext


@dataclass(frozen=True)
class RuntimeSessionClosure:
    """Resultado de cerrar una sesión desde el runtime."""

    session_id: str
    before: RuntimeSessionSnapshot
    after: RuntimeSessionSnapshot
    memory_written: bool
    trace: TraceContext


@dataclass(frozen=True)
class RuntimeSessionResolution:
    """Contexto unificado de sesión para CLI y turnos."""

    session_id: str
    overview: RuntimeSessionSnapshot
    turn_context: RuntimeTurnContext | None = None


@dataclass(frozen=True)
class RuntimeSessionView:
    """Vista CLI de una sesión ya resuelta."""

    snapshot: RuntimeSessionSnapshot
    banner_lines: tuple[str, ...]
    prompt_hint: str


@dataclass(frozen=True)
class SessionLifecycle:
    """Lifecycle unitario de una sesión para el CLI."""

    runtime: "AgentRuntime"
    session_id: str

    def view(self) -> RuntimeSessionView:
        return self.runtime.build_session_view(self.session_id)

    def resolve(self, message: str | None = None) -> RuntimeSessionResolution:
        return self.runtime.resolve_session(self.session_id, message)

    async def close(self) -> RuntimeSessionClosure:
        return await self.runtime.close_session(self.session_id)

    def export_artifact(self):
        return self.runtime.export_session_artifact(self.session_id)

    def artifact_path(self):
        return self.runtime.session_artifact_path(self.session_id)

    def load_artifact(self):
        return self.runtime.load_session_artifact(self.session_id)

    def replay(self):
        return self.runtime.build_session_replay(self.session_id)

    def context_budget(self, agent_name: str | None = None):
        return self.runtime.build_context_budget(self.session_id, agent_name=agent_name)

    def list_bookmarks(self):
        return self.runtime.list_bookmarks(self.session_id)

    def describe_bookmark(self, checkpoint_id: str):
        return self.runtime.describe_bookmark(self.session_id, checkpoint_id)

    def create_bookmark(self, label: str | None = None, note: str = ""):
        return self.runtime.create_bookmark(self.session_id, label=label, note=note)

    def search_memory(self, query: str, limit: int = 5):
        return self.runtime.search_memory(query, limit=limit)

    def list_memory_sessions(self):
        return self.runtime.list_memory_sessions()

    def preview_tool(self, tool_name: str, arguments: dict | None = None):
        return self.runtime.preview_tool(tool_name, arguments=arguments)

    def tool_catalog(self):
        return self.runtime.tool_catalog()

    def list_background_tasks(self):
        return self.runtime.list_background_tasks(self.session_id)

    async def submit_background_task(self, title: str, runner):
        return await self.runtime.submit_background_task(self.session_id, title, runner)

    def background_task_summary(self):
        return self.runtime.background_task_summary(self.session_id)

    def describe_background_task(self, task_id: str):
        return self.runtime.describe_background_task(self.session_id, task_id)

    def list_retryable_background_tasks(self):
        return self.runtime.list_retryable_background_tasks(self.session_id)

    async def cancel_background_task(self, task_id: str, reason: str | None = None):
        return await self.runtime.cancel_background_task(self.session_id, task_id, reason=reason)

    async def retry_background_task(self, task_id: str, title: str, runner, *, metadata: dict | None = None):
        return await self.runtime.retry_background_task(self.session_id, task_id, title, runner, metadata=metadata)


@dataclass(frozen=True)
class RuntimeTurnContext:
    """Contexto explícito de un turno antes de ejecutarlo."""

    session_id: str
    request_id: str
    trace: TraceContext
    message: str
    snapshot: RuntimeSessionSnapshot


class AgentRuntime:
    """Orquestador de sesión encima de AgentGateway.

    Mantiene la lógica de selección de sesión y de snapshotting fuera del CLI.
    El gateway sigue siendo el motor de ejecución; este runtime es la capa
    que hace más explícito el ciclo: resolver sesión → describir estado →
    ejecutar turno → cerrar.
    """

    def __init__(self, gateway: Optional[AgentGateway] = None, persistence_backend: SessionPersistence | None = None, trace_backend: TraceContextService | None = None, background_task_backend: BackgroundTaskService | None = None, memory_backend: MemoryRetrievalService | None = None, context_budget_backend: SessionContextBudgetService | None = None, bookmark_backend: SessionBookmarkService | None = None) -> None:
        self._gateway = gateway or AgentGateway()
        self._persistence = persistence_backend or session_persistence
        self._trace = trace_backend or trace_context_service
        self._artifacts = session_artifact_service
        self._background_tasks = background_task_backend or background_task_service
        self._memory_retrieval = memory_backend or memory_retrieval_service
        self._context_budget = context_budget_backend or context_budget_service
        self._bookmarks = bookmark_backend or session_bookmark_service
        self._coordinator = coordinator_runtime_service

    def list_sessions(self) -> list[str]:
        return self._persistence.list_sessions()

    def select_session_id(self, session_input: Optional[str] = None) -> str:
        """Selecciona una sesión válida o crea una nueva si el input no coincide."""
        sessions = self.list_sessions()
        if session_input and session_input in sessions:
            return session_input
        return str(uuid.uuid4())[:8]

    def start_session_lifecycle(self, session_input: Optional[str] = None) -> SessionLifecycle:
        """Crea el lifecycle de una sesión ya seleccionada."""
        return SessionLifecycle(runtime=self, session_id=self.select_session_id(session_input))

    def snapshot_session(self, session_id: str) -> RuntimeSessionSnapshot:
        message_count, has_memory = self._gateway.load_session_info(session_id)
        return RuntimeSessionSnapshot(
            session_id=session_id,
            message_count=message_count,
            has_memory=has_memory,
            is_existing_session=session_id in self._persistence.list_sessions(),
        )

    def resolve_session(self, session_id: str, message: str | None = None) -> RuntimeSessionResolution:
        """Resuelve la sesión y, opcionalmente, arma el contexto del turno."""
        overview = self.snapshot_session(session_id)
        turn_context = None
        if message is not None:
            trace = self._trace.create(session_id, "turn")
            turn_context = RuntimeTurnContext(
                session_id=session_id,
                request_id=trace.request_id,
                trace=trace,
                message=message,
                snapshot=overview,
            )
        return RuntimeSessionResolution(
            session_id=session_id,
            overview=overview,
            turn_context=turn_context,
        )

    def build_session_view(self, session_id: str) -> RuntimeSessionView:
        """Construye la vista textual de sesión para el CLI."""
        snapshot = self.snapshot_session(session_id)
        context_budget = self._context_budget.build_report(session_id)
        return RuntimeSessionView(
            snapshot=snapshot,
            banner_lines=tuple(format_session_banner(snapshot, context_budget)),
            prompt_hint="Escribí un mensaje, /help o salir para terminar",
        )

    async def execute_turn(self, turn: RuntimeTurnContext) -> RuntimeTurnResult:
        response = await self._gateway.send(turn.session_id, turn.message, request_id=turn.request_id, trace_id=turn.trace.trace_id)
        message_count, has_memory = self._gateway.load_session_info(turn.session_id)
        return RuntimeTurnResult(
            session_id=turn.session_id,
            request_id=turn.request_id,
            response=response,
            message_count=message_count,
            has_memory=has_memory,
            trace=turn.trace,
        )

    async def send_turn(self, session_id: str, message: str) -> RuntimeTurnResult:
        resolution = self.resolve_session(session_id, message)
        if resolution.turn_context is None:
            raise RuntimeError("turn_context no fue resuelto")
        return await self.execute_turn(resolution.turn_context)

    async def close_session(self, session_id: str) -> RuntimeSessionClosure:
        trace = self._trace.create(session_id, "close")
        before = self.snapshot_session(session_id)
        memory_written = await self._gateway.close_session(session_id)
        after = self.snapshot_session(session_id)
        return RuntimeSessionClosure(
            session_id=session_id,
            before=before,
            after=after,
            memory_written=memory_written,
            trace=trace,
        )

    def export_session_artifact(self, session_id: str):
        return self._artifacts.export_artifact(session_id)

    def build_session_artifact(self, session_id: str):
        return self._artifacts.build_artifact(session_id)

    def load_session_artifact(self, session_id: str):
        return self._artifacts.load_artifact(session_id)

    async def get_live_state(self, session_id: str):
        return await self._gateway.get_state(session_id)

    def build_session_replay(self, session_id: str) -> SessionReplay:
        return session_replay_service.build_replay(session_id)

    def build_context_budget(self, session_id: str, agent_name: str | None = None) -> SessionContextBudget:
        return self._context_budget.build_report(session_id, agent_name=agent_name)

    def list_session_replays(self) -> list[str]:
        return session_replay_service.list_sessions()

    def list_memory_sessions(self) -> list[str]:
        return self._memory_retrieval.list_sessions()

    def search_memory(self, query: str, limit: int = 5) -> list[MemorySearchHit]:
        return self._memory_retrieval.search(query, limit=limit)

    def preview_tool(self, tool_name: str, arguments: dict | None = None) -> ToolApprovalPreview:
        return tool_approval_service.build_preview(agent_name="cli", tool_name=tool_name, arguments=arguments or {})

    async def spawn_worker(self, session_id: str, agent_name: str, *, worker_name: str | None = None, parent_worker_id: str | None = None, metadata: dict | None = None):
        return await self._coordinator.spawn_worker(
            session_id,
            agent_name,
            worker_name=worker_name,
            parent_worker_id=parent_worker_id,
            metadata=metadata,
        )

    async def send_worker_message(self, session_id: str, worker_id: str, content: str, *, sender: str = "coordinator"):
        return await self._coordinator.send_message(session_id, worker_id, content, sender=sender)

    def list_workers(self, session_id: str):
        return self._coordinator.list_workers(session_id)

    def list_worker_messages(self, session_id: str):
        return self._coordinator.list_messages(session_id)

    def tool_catalog(self) -> list[str]:
        return tool_approval_service.render_catalog_lines()

    def list_session_artifacts(self) -> list[str]:
        return self._artifacts.list_sessions()

    def session_artifact_path(self, session_id: str):
        return self._artifacts.artifact_path(session_id)

    def list_bookmarks(self, session_id: str) -> list[dict[str, object]]:
        return self._bookmarks.list(session_id)

    def describe_bookmark(self, session_id: str, checkpoint_id: str) -> dict[str, object] | None:
        return self._bookmarks.describe(session_id, checkpoint_id)

    def create_bookmark(self, session_id: str, label: str | None = None, note: str = ""):
        artifact = self._artifacts.export_artifact(session_id)
        replay = self.build_session_replay(session_id)
        budget = self.build_context_budget(session_id)
        prompt_agents = [str(snapshot.get("agent_name", "")) for snapshot in artifact.prompt_snapshots]
        return self._bookmarks.create(
            session_id=session_id,
            label=label,
            artifact_path=str(self.session_artifact_path(session_id)),
            message_count=artifact.message_count,
            has_memory=artifact.has_memory,
            replay_item_count=len(replay.items),
            context_budget={"report": context_budget_to_dict(budget)},
            prompt_agents=[agent for agent in prompt_agents if agent],
            note=note,
        )

    def list_background_tasks(self, session_id: str) -> list[dict[str, object]]:
        return self._background_tasks.load_session_tasks(session_id)

    def list_retryable_background_tasks(self, session_id: str) -> list[dict[str, object]]:
        return self._background_tasks.find_session_tasks(session_id, status="failed") + self._background_tasks.find_session_tasks(session_id, status="cancelled")

    def background_task_summary(self, session_id: str) -> BackgroundTaskSummary:
        return self._background_tasks.describe_session(session_id)

    def describe_background_task(self, session_id: str, task_id: str) -> BackgroundTaskState | None:
        return self._background_tasks.describe_task(session_id, task_id)

    async def cancel_background_task(self, session_id: str, task_id: str, reason: str | None = None):
        return await self._background_tasks.cancel_task(session_id, task_id, reason=reason)

    async def retry_background_task(self, session_id: str, task_id: str, title: str, runner, *, metadata: dict | None = None):
        trace = self._trace.create(session_id, "background-task-retry")
        return await self._background_tasks.retry_task(
            session_id,
            task_id,
            runner,
            title=title,
            metadata=metadata,
            request_id=trace.request_id,
            trace_id=trace.trace_id,
        )

    async def submit_background_task(
        self,
        session_id: str,
        title: str,
        runner,
    ):
        trace = self._trace.create(session_id, "background-task")
        return await self._background_tasks.submit(
            session_id,
            title,
            runner,
            request_id=trace.request_id,
            trace_id=trace.trace_id,
        )

    async def shutdown(self) -> None:
        await self._gateway.shutdown()
        await self._background_tasks.shutdown()


__all__ = [
    "AgentRuntime",
    "RuntimeSessionSnapshot",
    "RuntimeSessionClosure",
    "RuntimeSessionResolution",
    "RuntimeSessionView",
    "SessionLifecycle",
    "RuntimeTurnContext",
    "RuntimeTurnResult",
]
