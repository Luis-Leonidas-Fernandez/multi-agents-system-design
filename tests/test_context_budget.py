"""Tests para el presupuesto de contexto de sesión."""
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage


def test_context_budget_service_reporta_incluido_resumido_y_afuera():
    from features.sessions.application.context_budget import SessionContextBudgetService

    persistence_backend = MagicMock()
    persistence_backend.load_messages.return_value = [
        HumanMessage(content="hola"),
        AIMessage(content="respuesta"),
    ]

    memory_backend = MagicMock()
    memory_backend.load_memory_context.return_value = "memoria resumida"

    prompt_version_backend = MagicMock()
    prompt_version_backend.load_snapshot.return_value = {
        "system_prompt": "system prompt",
        "extra_context": "extra context",
        "prompt_version": "v1",
    }

    audit_backend = MagicMock()
    audit_backend.load_session_events.return_value = [{"event_type": "tool_call_requested"}]

    background_task_backend = MagicMock()
    background_task_backend.load_session_tasks.return_value = [{"task_id": "task-1"}]

    bookmark_store = MagicMock()
    bookmark_store.list.return_value = [{"checkpoint_id": "chk-1"}]

    memory_retrieval_backend = MagicMock()
    memory_retrieval_backend.list_sessions.return_value = ["sess-1"]

    service = SessionContextBudgetService(
        persistence_backend=persistence_backend,
        memory_backend=memory_backend,
        prompt_version_backend=prompt_version_backend,
        audit_backend=audit_backend,
        background_task_backend=background_task_backend,
        bookmark_store=bookmark_store,
        budget_chars=1000,
        memory_backend_retrieval=memory_retrieval_backend,
    )

    report = service.build_report("sess-1", agent_name="math_agent")

    assert report.session_id == "sess-1"
    assert report.scope == "math_agent"
    assert report.estimated_context_chars > 0
    assert report.memory_present is True
    assert any(item.section == "transcript" and item.role == "included" for item in report.items)
    assert any(item.section == "memory" and item.role == "summarized" for item in report.items)
    assert any(item.section == "observability" and item.role == "excluded" for item in report.items)


def test_runtime_context_budget_delega_en_servicio():
    from application.services.runtime import AgentRuntime

    runtime = AgentRuntime(gateway=MagicMock())
    runtime._context_budget = MagicMock()  # type: ignore[attr-defined]
    runtime._context_budget.build_report.return_value = {"session_id": "sess-2"}  # type: ignore[attr-defined]

    result = runtime.build_context_budget("sess-2", agent_name="math_agent")

    assert result == {"session_id": "sess-2"}
    runtime._context_budget.build_report.assert_called_once_with("sess-2", agent_name="math_agent")  # type: ignore[attr-defined]
