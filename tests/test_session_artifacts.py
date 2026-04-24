"""Tests para artefactos consolidados de sesión."""
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage


def test_session_artifact_service_exporta_bundle(tmp_path):
    from features.sessions.application.session_artifacts import SessionArtifactService, SessionArtifactStore

    persistence_backend = MagicMock()
    persistence_backend.load_messages.return_value = [
        HumanMessage(content="hola"),
        AIMessage(content="respuesta"),
    ]
    persistence_backend.list_sessions.return_value = ["sess-1"]

    memory_backend = MagicMock()
    memory_backend.load_memory_context.return_value = "memoria de prueba"

    audit_backend = MagicMock()
    audit_backend.load_session_events.return_value = [
        {"trace_id": "trace-1", "tool_name": "calculate"},
        {"trace_id": "trace-2", "tool_name": "write_code"},
    ]

    background_task_backend = MagicMock()
    background_task_backend.load_session_tasks.return_value = [
        {"task_id": "task-1", "status": "completed"},
    ]
    from features.sessions.application.background_tasks import BackgroundTaskSummary

    background_task_backend.describe_session.return_value = BackgroundTaskSummary(
        session_id="sess-1",
        total=1,
        queued=0,
        running=0,
        completed=1,
        failed=0,
        cancelled=0,
        active=0,
        terminal=1,
        retryable=0,
        latest_updated_at_ms=2,
    )

    prompt_version_backend = MagicMock()
    prompt_version_backend.list_agents.return_value = ["math_agent"]
    prompt_version_backend.load_snapshot.return_value = {
        "agent_name": "math_agent",
        "prompt_version": "v1",
        "prompt_hash": "abc",
    }
    prompt_version_backend.snapshot_path.return_value = tmp_path / "prompts" / "math_agent" / "PROMPT_SNAPSHOT.json"
    prompt_version_backend.history_path.return_value = tmp_path / "prompts" / "math_agent" / "PROMPT_HISTORY.jsonl"

    from features.sessions.application.context_budget import ContextBudgetItem, SessionContextBudget

    context_budget_backend = MagicMock()
    context_budget_backend.build_report.return_value = SessionContextBudget(
        session_id="sess-1",
        generated_at_ms=1,
        budget_chars=1000,
        estimated_context_chars=100,
        estimated_remaining_chars=900,
        estimated_tokens=25,
        status="ok",
        scope="session",
        transcript_message_count=2,
        memory_present=True,
        items=[ContextBudgetItem(section="transcript", role="included", chars=10, detail="2 mensajes")],
    )

    bookmark_backend = MagicMock()
    bookmark_backend.list.return_value = []

    store = SessionArtifactStore(base_dir=tmp_path)
    service = SessionArtifactService(
        persistence_backend=persistence_backend,
        memory_backend=memory_backend,
        audit_backend=audit_backend,
        background_task_backend=background_task_backend,
        context_budget_backend=context_budget_backend,
        bookmark_backend=bookmark_backend,
        prompt_version_backend=prompt_version_backend,
        store=store,
    )

    artifact = service.export_artifact("sess-1")
    loaded = service.load_artifact("sess-1")

    assert artifact.session_id == "sess-1"
    assert artifact.message_count == 2
    assert artifact.has_memory is True
    assert artifact.trace_ids == ["trace-1", "trace-2"]
    assert artifact.background_tasks == [{"task_id": "task-1", "status": "completed"}]
    assert artifact.background_task_summary["completed"] == 1
    assert artifact.prompt_snapshots == [{"agent_name": "math_agent", "prompt_version": "v1", "prompt_hash": "abc", "snapshot_path": str(tmp_path / "prompts" / "math_agent" / "PROMPT_SNAPSHOT.json"), "history_path": str(tmp_path / "prompts" / "math_agent" / "PROMPT_HISTORY.jsonl")}]
    assert artifact.context_budget["session_id"] == "sess-1"
    assert artifact.bookmarks == []
    assert loaded is not None
    assert loaded["session_id"] == "sess-1"
    assert loaded["memory_markdown"] == "memoria de prueba"
    assert loaded["background_tasks"] == [{"task_id": "task-1", "status": "completed"}]
    assert loaded["background_task_summary"]["completed"] == 1
    assert loaded["prompt_snapshots"] == [{"agent_name": "math_agent", "prompt_version": "v1", "prompt_hash": "abc", "snapshot_path": str(tmp_path / "prompts" / "math_agent" / "PROMPT_SNAPSHOT.json"), "history_path": str(tmp_path / "prompts" / "math_agent" / "PROMPT_HISTORY.jsonl")}]
    assert loaded["context_budget"]["session_id"] == "sess-1"
    assert loaded["bookmarks"] == []


def test_session_artifact_store_lista_y_carga(tmp_path):
    from features.sessions.application.session_artifacts import SessionArtifact, SessionArtifactStore

    store = SessionArtifactStore(base_dir=tmp_path)
    artifact = SessionArtifact(
        session_id="sess-2",
        generated_at_ms=1,
        message_count=0,
        has_memory=False,
        is_existing_session=False,
        transcript=[],
        memory_markdown="",
        audit_events=[],
        background_tasks=[],
        background_task_summary={},
        prompt_snapshots=[],
        context_budget={},
        bookmarks=[],
        trace_ids=[],
    )

    store.save(artifact)

    assert store.list_sessions() == ["sess-2"]
    loaded = store.load("sess-2")
    assert loaded is not None
    assert loaded["session_id"] == "sess-2"


def test_runtime_export_session_artifact_delega_en_service():
    from application.services.runtime import AgentRuntime

    runtime = AgentRuntime(gateway=MagicMock())
    runtime._artifacts = MagicMock()  # type: ignore[attr-defined]
    runtime._artifacts.export_artifact.return_value = {"session_id": "sess-3"}

    result = runtime.export_session_artifact("sess-3")

    assert result == {"session_id": "sess-3"}
    runtime._artifacts.export_artifact.assert_called_once_with("sess-3")


def test_session_lifecycle_export_artifact_delega_en_runtime():
    from application.services.runtime import AgentRuntime, SessionLifecycle

    runtime = AgentRuntime(gateway=MagicMock())
    runtime._artifacts = MagicMock()  # type: ignore[attr-defined]
    runtime._artifacts.export_artifact.return_value = {"session_id": "sess-4"}
    lifecycle = SessionLifecycle(runtime=runtime, session_id="sess-4")

    result = lifecycle.export_artifact()

    assert result == {"session_id": "sess-4"}
    runtime._artifacts.export_artifact.assert_called_once_with("sess-4")


def test_runtime_session_artifact_path_delega_en_servicio():
    from application.services.runtime import AgentRuntime

    runtime = AgentRuntime(gateway=MagicMock())
    runtime._artifacts = MagicMock()  # type: ignore[attr-defined]
    runtime._artifacts.artifact_path.return_value = "data/sessions/sess-5/SESSION_ARTIFACT.json"  # type: ignore[attr-defined]

    result = runtime.session_artifact_path("sess-5")

    assert result == "data/sessions/sess-5/SESSION_ARTIFACT.json"
    runtime._artifacts.artifact_path.assert_called_once_with("sess-5")  # type: ignore[attr-defined]


def test_session_lifecycle_artifact_path_delega_en_runtime():
    from application.services.runtime import AgentRuntime, SessionLifecycle

    runtime = AgentRuntime(gateway=MagicMock())
    runtime._artifacts = MagicMock()  # type: ignore[attr-defined]
    runtime._artifacts.artifact_path.return_value = "data/sessions/sess-6/SESSION_ARTIFACT.json"  # type: ignore[attr-defined]

    lifecycle = SessionLifecycle(runtime=runtime, session_id="sess-6")
    result = lifecycle.artifact_path()

    assert result == "data/sessions/sess-6/SESSION_ARTIFACT.json"
    runtime._artifacts.artifact_path.assert_called_once_with("sess-6")  # type: ignore[attr-defined]
