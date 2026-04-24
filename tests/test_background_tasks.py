"""Tests para la delegación de tareas en background."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_background_task_service_persiste_lifecycle_completo(tmp_path):
    from features.sessions.application.background_tasks import BackgroundTaskService, BackgroundTaskStore

    store = BackgroundTaskStore(base_dir=tmp_path)
    service = BackgroundTaskService(store=store, clock=lambda: 1000.0)

    started = asyncio.Event()
    release = asyncio.Event()

    async def runner():
        started.set()
        await release.wait()
        return {"ok": True}

    record = await service.submit(
        "sess-1",
        "procesar reporte",
        runner,
        request_id="req-1",
        trace_id="trace-1",
        metadata={"priority": "high"},
    )

    assert record.status == "queued"

    await started.wait()
    running = service.get_session_task("sess-1", record.task_id)
    assert running is not None
    assert running["status"] == "running"

    release.set()
    while service.running_task_count() > 0:
        await asyncio.sleep(0)

    loaded = service.get_session_task("sess-1", record.task_id)
    assert loaded is not None
    assert loaded["status"] == "completed"
    assert loaded["result"] == {"ok": True}
    assert loaded["request_id"] == "req-1"
    assert loaded["trace_id"] == "trace-1"

    summary = service.describe_session("sess-1")
    assert summary.total == 1
    assert summary.completed == 1
    assert summary.active == 0
    assert summary.terminal == 1

    state = service.describe_task("sess-1", record.task_id)
    assert state is not None
    assert state.status == "completed"
    assert state.state_kind == "terminal"

    path = tmp_path / "sess-1" / "BACKGROUND_TASKS.jsonl"
    assert path.exists()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


@pytest.mark.asyncio
async def test_background_task_service_marca_error_y_lo_persiste(tmp_path):
    from features.sessions.application.background_tasks import BackgroundTaskService, BackgroundTaskStore

    service = BackgroundTaskService(store=BackgroundTaskStore(base_dir=tmp_path), clock=lambda: 2000.0)

    async def runner():
        raise RuntimeError("falló el background")

    record = await service.submit("sess-2", "sincronizar", runner)
    while service.running_task_count() > 0:
        await asyncio.sleep(0)

    loaded = service.get_session_task("sess-2", record.task_id)
    assert loaded is not None
    assert loaded["status"] == "failed"
    assert "falló el background" in loaded["error"]

    summary = service.describe_session("sess-2")
    assert summary.failed == 1
    assert summary.total == 1


@pytest.mark.asyncio
async def test_runtime_submit_background_task_delega_en_servicio():
    from application.services.runtime import AgentRuntime

    runtime = AgentRuntime(gateway=MagicMock())
    runtime._background_tasks = MagicMock()  # type: ignore[attr-defined]
    runtime._background_tasks.submit = AsyncMock(return_value={"task_id": "task-1"})  # type: ignore[attr-defined]

    async def runner():
        return "ok"

    result = await runtime.submit_background_task("sess-3", "delegar", runner)

    assert result == {"task_id": "task-1"}
    runtime._background_tasks.submit.assert_awaited_once()  # type: ignore[attr-defined]
    args, kwargs = runtime._background_tasks.submit.await_args  # type: ignore[attr-defined]
    assert args[0] == "sess-3"
    assert args[1] == "delegar"
    assert kwargs["request_id"]
    assert kwargs["trace_id"]


def test_session_lifecycle_list_background_tasks_delega_en_runtime():
    from application.services.runtime import AgentRuntime, SessionLifecycle
    from unittest.mock import MagicMock

    runtime = AgentRuntime(gateway=MagicMock())
    runtime._background_tasks = MagicMock()  # type: ignore[attr-defined]
    runtime._background_tasks.load_session_tasks.return_value = [{"task_id": "task-9"}]  # type: ignore[attr-defined]

    lifecycle = SessionLifecycle(runtime=runtime, session_id="sess-4")
    result = lifecycle.list_background_tasks()

    assert result == [{"task_id": "task-9"}]
    runtime._background_tasks.load_session_tasks.assert_called_once_with("sess-4")  # type: ignore[attr-defined]


def test_session_lifecycle_background_task_summary_delega_en_runtime():
    from application.services.runtime import AgentRuntime, SessionLifecycle

    runtime = AgentRuntime(gateway=MagicMock())
    runtime._background_tasks = MagicMock()  # type: ignore[attr-defined]
    runtime._background_tasks.describe_session.return_value = {"session_id": "sess-5", "running": 1}  # type: ignore[attr-defined]

    lifecycle = SessionLifecycle(runtime=runtime, session_id="sess-5")
    result = lifecycle.background_task_summary()

    assert result == {"session_id": "sess-5", "running": 1}
    runtime._background_tasks.describe_session.assert_called_once_with("sess-5")  # type: ignore[attr-defined]


def test_session_lifecycle_describe_background_task_delega_en_runtime():
    from application.services.runtime import AgentRuntime, SessionLifecycle

    runtime = AgentRuntime(gateway=MagicMock())
    runtime._background_tasks = MagicMock()  # type: ignore[attr-defined]
    runtime._background_tasks.describe_task.return_value = {"task_id": "task-12", "state_kind": "terminal"}  # type: ignore[attr-defined]

    lifecycle = SessionLifecycle(runtime=runtime, session_id="sess-6")
    result = lifecycle.describe_background_task("task-12")

    assert result == {"task_id": "task-12", "state_kind": "terminal"}
    runtime._background_tasks.describe_task.assert_called_once_with("sess-6", "task-12")  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_background_task_service_cancela_tarea_en_ejecucion(tmp_path):
    from features.sessions.application.background_tasks import BackgroundTaskService, BackgroundTaskStore

    service = BackgroundTaskService(store=BackgroundTaskStore(base_dir=tmp_path), clock=lambda: 4000.0)
    started = asyncio.Event()
    release = asyncio.Event()

    async def runner():
        started.set()
        await release.wait()
        return "ok"

    record = await service.submit("sess-9", "cancelable", runner)
    await started.wait()

    cancelled = await service.cancel_task("sess-9", record.task_id, reason="manual stop")
    assert cancelled is not None
    assert cancelled["status"] == "cancelled"
    assert cancelled["cancel_reason"] == "manual stop"
    stored = service.get_session_task("sess-9", record.task_id)
    assert stored is not None
    assert stored["status"] == "cancelled"


@pytest.mark.asyncio
async def test_background_task_service_retry_creates_new_attempt(tmp_path):
    from features.sessions.application.background_tasks import BackgroundTaskService, BackgroundTaskStore

    service = BackgroundTaskService(store=BackgroundTaskStore(base_dir=tmp_path), clock=lambda: 5000.0)

    async def failing_runner():
        raise RuntimeError("boom")

    record = await service.submit("sess-10", "reintentar", failing_runner)
    while service.running_task_count() > 0:
        await asyncio.sleep(0)

    async def success_runner():
        return {"ok": True}

    retried = await service.retry_task("sess-10", record.task_id, success_runner)
    assert retried is not None
    assert retried.parent_task_id == record.task_id
    assert retried.attempt_number == 2
    assert retried.metadata and retried.metadata["retry_of"] == record.task_id


def test_runtime_list_retryable_background_tasks_delega_en_servicio():
    from application.services.runtime import AgentRuntime

    runtime = AgentRuntime(gateway=MagicMock())
    runtime._background_tasks = MagicMock()  # type: ignore[attr-defined]
    runtime._background_tasks.find_session_tasks.side_effect = [[{"task_id": "f1"}], [{"task_id": "c1"}]]  # type: ignore[attr-defined]

    result = runtime.list_retryable_background_tasks("sess-11")

    assert result == [{"task_id": "f1"}, {"task_id": "c1"}]
    assert runtime._background_tasks.find_session_tasks.call_count == 2  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_session_lifecycle_cancel_background_task_delega_en_runtime():
    from application.services.runtime import AgentRuntime, SessionLifecycle

    runtime = AgentRuntime(gateway=MagicMock())
    runtime._background_tasks = MagicMock()  # type: ignore[attr-defined]
    runtime._background_tasks.cancel_task = AsyncMock(return_value={"task_id": "task-99", "status": "cancelled"})  # type: ignore[attr-defined]

    lifecycle = SessionLifecycle(runtime=runtime, session_id="sess-12")
    result = await lifecycle.cancel_background_task("task-99", reason="stop")

    assert result == {"task_id": "task-99", "status": "cancelled"}
    runtime._background_tasks.cancel_task.assert_awaited_once_with("sess-12", "task-99", reason="stop")  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_session_lifecycle_retry_background_task_delega_en_runtime():
    from application.services.runtime import AgentRuntime, SessionLifecycle

    runtime = AgentRuntime(gateway=MagicMock())
    runtime._background_tasks = MagicMock()  # type: ignore[attr-defined]
    runtime._background_tasks.retry_task = AsyncMock(return_value={"task_id": "task-100", "parent_task_id": "task-1"})  # type: ignore[attr-defined]

    lifecycle = SessionLifecycle(runtime=runtime, session_id="sess-13")

    async def runner():
        return "ok"

    result = await lifecycle.retry_background_task("task-1", "reintentar", runner)

    assert result == {"task_id": "task-100", "parent_task_id": "task-1"}
    runtime._background_tasks.retry_task.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_session_lifecycle_submit_background_task_delega_en_runtime():
    from application.services.runtime import AgentRuntime, SessionLifecycle

    runtime = AgentRuntime(gateway=MagicMock())
    runtime._background_tasks = MagicMock()  # type: ignore[attr-defined]
    runtime._background_tasks.submit = AsyncMock(return_value={"task_id": "task-10"})  # type: ignore[attr-defined]
    lifecycle = SessionLifecycle(runtime=runtime, session_id="sess-5")

    async def runner():
        return "ok"

    result = await lifecycle.submit_background_task("procesar", runner)

    assert result == {"task_id": "task-10"}
    runtime._background_tasks.submit.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_background_task_service_shutdown_cancela_tareas_activas(tmp_path):
    from features.sessions.application.background_tasks import BackgroundTaskService, BackgroundTaskStore

    service = BackgroundTaskService(store=BackgroundTaskStore(base_dir=tmp_path), clock=lambda: 3000.0)
    started = asyncio.Event()

    async def runner():
        started.set()
        await asyncio.Event().wait()

    await service.submit("sess-8", "bloqueada", runner)
    await started.wait()

    await service.shutdown()

    assert service.running_task_count() == 0
    loaded = service.get_session_task("sess-8", service.load_session_tasks("sess-8")[0]["task_id"])
    assert loaded is not None
    assert loaded["status"] == "cancelled"


def test_runtime_background_task_summary_delega_en_servicio():
    from application.services.runtime import AgentRuntime

    runtime = AgentRuntime(gateway=MagicMock())
    runtime._background_tasks = MagicMock()  # type: ignore[attr-defined]
    runtime._background_tasks.describe_session.return_value = {"session_id": "sess-6", "total": 2}  # type: ignore[attr-defined]

    result = runtime.background_task_summary("sess-6")

    assert result == {"session_id": "sess-6", "total": 2}
    runtime._background_tasks.describe_session.assert_called_once_with("sess-6")  # type: ignore[attr-defined]


def test_runtime_describe_background_task_delega_en_servicio():
    from application.services.runtime import AgentRuntime

    runtime = AgentRuntime(gateway=MagicMock())
    runtime._background_tasks = MagicMock()  # type: ignore[attr-defined]
    runtime._background_tasks.describe_task.return_value = {"task_id": "task-11", "status": "running"}  # type: ignore[attr-defined]

    result = runtime.describe_background_task("sess-7", "task-11")

    assert result == {"task_id": "task-11", "status": "running"}
    runtime._background_tasks.describe_task.assert_called_once_with("sess-7", "task-11")  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_runtime_shutdown_cierra_gateway_y_background_tasks():
    from application.services.runtime import AgentRuntime

    gateway = MagicMock()
    gateway.shutdown = AsyncMock(return_value=None)
    runtime = AgentRuntime(gateway=gateway)
    runtime._background_tasks = MagicMock()  # type: ignore[attr-defined]
    runtime._background_tasks.shutdown = AsyncMock(return_value=None)  # type: ignore[attr-defined]

    await runtime.shutdown()

    gateway.shutdown.assert_awaited_once()
    runtime._background_tasks.shutdown.assert_awaited_once()  # type: ignore[attr-defined]
