"""Tests para los helpers de inspección de sesión y CLI."""
from types import SimpleNamespace
from unittest.mock import MagicMock


def _handle_inspection_command(user_input, lifecycle, runtime=None):
    from application.services.cli_dispatch import dispatch_inspection_command

    result = dispatch_inspection_command(user_input, lifecycle, runtime)
    for line in result.lines:
        print(line)
    return result.handled


def test_format_background_task_summary_muestra_resumen():
    from application.services.session_inspection import format_background_task_summary

    lines = format_background_task_summary(
        {
            "session_id": "sess-1",
            "total": 3,
            "queued": 1,
            "running": 1,
            "completed": 1,
            "failed": 0,
            "cancelled": 0,
            "active": 2,
            "terminal": 1,
            "retryable": 0,
            "latest_updated_at_ms": 123,
        }
    )

    assert any("sess-1" in line for line in lines)
    assert any("active=2" in line for line in lines)
    assert any("retryable=0" in line for line in lines)
    assert any("latest_update_ms=123" in line for line in lines)


def test_format_background_task_state_muestra_error_y_resultado():
    from application.services.session_inspection import format_background_task_state

    lines = format_background_task_state(
        {
            "task_id": "task-1",
            "session_id": "sess-1",
            "title": "hacer algo",
            "status": "failed",
            "state_kind": "terminal",
            "attempt_number": 2,
            "parent_task_id": "task-0",
            "request_id": "req-1",
            "trace_id": "trace-1",
            "error": "boom",
            "cancel_reason": "manual",
        }
    )

    assert any("task-1" in line for line in lines)
    assert any("failed" in line for line in lines)
    assert any("boom" in line for line in lines)
    assert any("attempt=2" in line for line in lines)
    assert any("parent=task-0" in line for line in lines)


def test_format_session_artifact_muestra_artefacto_consolidado():
    from application.services.session_inspection import format_session_artifact

    lines = format_session_artifact(
        {
            "session_id": "sess-1",
            "message_count": 4,
            "has_memory": True,
            "audit_events": [{}, {}],
            "background_tasks": [{}, {}],
            "prompt_snapshots": [{}, {}],
            "context_budget": {"report": {"scope": "session", "status": "ok", "estimated_context_chars": 42, "estimated_remaining_chars": 958}},
            "bookmarks": [{}],
            "trace_ids": ["a", "b"],
            "background_task_summary": {"total": 2, "active": 1, "completed": 1, "failed": 0, "cancelled": 0},
        }
    )

    assert any("artifact" in line for line in lines)
    assert any("messages=4" in line for line in lines)
    assert any("background_tasks=2" in line for line in lines)
    assert any("prompt_snapshots=2" in line for line in lines)
    assert any("context scope=session" in line for line in lines)
    assert any("bookmarks=1" in line for line in lines)


def test_format_context_budget_muestra_incluido_resumido_y_afuera():
    from application.services.session_inspection import format_context_budget

    lines = format_context_budget(
        {
            "session_id": "sess-ctx",
            "scope": "math_agent",
            "status": "warn",
            "budget_chars": 1000,
            "estimated_context_chars": 850,
            "estimated_remaining_chars": 150,
            "estimated_tokens": 212,
            "transcript_message_count": 4,
            "memory_present": True,
            "items": [
                {"section": "system_prompt", "role": "included", "chars": 120, "detail": "math_agent version=v1"},
                {"section": "transcript", "role": "included", "chars": 500, "detail": "4 mensajes persistidos"},
                {"section": "memory", "role": "summarized", "chars": 230, "detail": "MEMORY.md"},
                {"section": "observability", "role": "excluded", "chars": 0, "detail": "audit_events=3 background_tasks=1 bookmarks=2"},
            ],
        }
    )

    assert any("scope=math_agent" in line for line in lines)
    assert any("incluido" in line for line in lines)
    assert any("resumido" in line for line in lines)
    assert any("afuera" in line for line in lines)


def test_format_session_banner_muestra_contexto_y_atajos():
    from application.services.session_inspection import format_session_banner

    lines = format_session_banner(
        {"session_id": "sess-ui", "message_count": 9, "has_memory": True},
        {"scope": "session", "status": "ok", "estimated_context_chars": 600, "estimated_remaining_chars": 400, "estimated_tokens": 150},
    )

    assert any("sess-ui" in line for line in lines)
    assert any("[contexto]" in line for line in lines)
    assert any("[atajos]" in line for line in lines)


def test_format_cli_chrome_muestra_estado_y_agentes():
    from application.services.session_inspection import format_cli_chrome

    lines = format_cli_chrome(
        {"session_id": "sess-ui", "message_count": 3, "has_memory": False},
        {"scope": "session", "status": "ok", "estimated_context_chars": 120, "estimated_remaining_chars": 880, "estimated_tokens": 20},
        [type("A", (), {"name": "math_agent", "risk_level": "low"})(), type("B", (), {"name": "code_agent", "risk_level": "high"})()],
    )

    assert any("sess-ui" in line for line in lines)
    assert any("math_agent[low]" in line for line in lines)
    assert any("code_agent[high]" in line for line in lines)


def test_format_session_selector_y_chat_block():
    from application.services.session_inspection import format_chat_block, format_session_selector, format_session_transcript, format_shell_frame

    selector = format_session_selector(["sess-a", "sess-b"])
    chat = format_chat_block("user", "hola\nsegundo renglón")
    transcript = format_session_transcript([
        {"role": "human", "content": "hola"},
        {"role": "ai", "content": "qué tal"},
    ])

    assert any("1. sess-a" in line for line in selector)
    assert any("Enter = nueva sesión" in line for line in selector)
    assert any("HOLA" in line.upper() or "USER" in line.upper() for line in chat)
    assert any("segundo renglón" in line for line in chat)
    assert any("[chat]" in line for line in transcript)
    assert any("HUMAN" in line for line in transcript)

    shell = format_shell_frame(
        {"session_id": "sess-ui", "message_count": 2, "has_memory": True},
        {"scope": "session", "status": "ok", "estimated_context_chars": 100, "estimated_remaining_chars": 900, "estimated_tokens": 20},
        [type("A", (), {"name": "math_agent", "risk_level": "low"})()],
        [{"role": "human", "content": "hola"}],
        "Escribí un mensaje, /help o salir para terminar",
    )

    assert any("conversation" in line for line in shell)
    assert any("math_agent[low]" in line for line in shell)


def test_format_bookmark_list_y_detail():
    from application.services.session_inspection import format_bookmark_detail, format_bookmark_list

    list_lines = format_bookmark_list([
        {"checkpoint_id": "chk-1", "label": "base", "message_count": 2, "has_memory": True}
    ])
    detail_lines = format_bookmark_detail(
        {
            "checkpoint_id": "chk-1",
            "label": "base",
            "session_id": "sess-1",
            "created_at_ms": 1,
            "message_count": 2,
            "has_memory": True,
            "replay_item_count": 7,
            "artifact_path": "sessions/sess-1/SESSION_ARTIFACT.json",
            "context_budget": {"report": {"scope": "session", "status": "ok", "estimated_context_chars": 10, "estimated_remaining_chars": 90}},
            "prompt_agents": ["math_agent"],
        }
    )

    assert any("chk-1" in line for line in list_lines)
    assert any("checkpoint" in line for line in detail_lines)
    assert any("artifact_path" in line for line in detail_lines)


def test_format_command_registry_y_detail():
    from application.services.command_registry import SlashCommandSpec
    from application.services.session_inspection import format_command_detail, format_command_registry

    registry_lines = format_command_registry({"general": [SlashCommandSpec(name="help", summary="muestra ayuda", usage="/help", group="general", aliases=("?",))]})
    detail_lines = format_command_detail({"name": "help", "group": "general", "usage": "/help", "summary": "muestra ayuda", "aliases": ["?"]})

    assert any("[commands]" in line for line in registry_lines)
    assert any("/help" in line for line in registry_lines)
    assert any("[command]" in line for line in detail_lines)
    assert any("aliases" in line for line in detail_lines)


def test_handle_inspection_command_reconoce_inspect(capsys):
    from application.services.runtime import AgentRuntime

    runtime = AgentRuntime(gateway=MagicMock())
    lifecycle = SimpleNamespace(
        context_budget=lambda agent_name=None: {
            "session_id": "sess-1",
            "scope": agent_name or "session",
            "status": "ok",
            "budget_chars": 1000,
            "estimated_context_chars": 100,
            "estimated_remaining_chars": 900,
            "estimated_tokens": 25,
            "transcript_message_count": 1,
            "memory_present": False,
            "items": [],
        },
        background_task_summary=lambda: {
            "session_id": "sess-1",
            "total": 1,
            "queued": 0,
            "running": 1,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "active": 1,
            "terminal": 0,
            "latest_updated_at_ms": None,
        },
        export_artifact=lambda: {
            "session_id": "sess-1",
            "message_count": 1,
            "has_memory": False,
            "audit_events": [],
            "background_tasks": [],
            "prompt_snapshots": [],
            "trace_ids": [],
            "background_task_summary": {"total": 1, "active": 1, "completed": 0, "failed": 0, "cancelled": 0},
        },
        list_background_tasks=lambda: [{"task_id": "task-1", "status": "running", "title": "hacer algo"}],
        describe_background_task=lambda task_id: {"task_id": task_id},
        list_bookmarks=lambda: [],
        describe_bookmark=lambda checkpoint_id: None,
        create_bookmark=lambda label=None, note="": {"checkpoint_id": "chk-1", "label": label or "checkpoint-1", "session_id": "sess-1", "created_at_ms": 1, "message_count": 1, "has_memory": False, "replay_item_count": 1, "artifact_path": "sessions/sess-1/SESSION_ARTIFACT.json", "context_budget": {"report": {"scope": "session", "status": "ok", "estimated_context_chars": 10, "estimated_remaining_chars": 90}}, "prompt_agents": []},
    )

    handled = _handle_inspection_command("/inspect", lifecycle, runtime=runtime)
    output = capsys.readouterr().out

    assert handled is True
    assert "[context]" in output
    assert "[tasks]" in output
    assert "[artifact]" in output


def test_handle_inspection_command_reconoce_task_individual(capsys):
    from application.services.runtime import AgentRuntime

    runtime = AgentRuntime(gateway=MagicMock())
    lifecycle = SimpleNamespace(
        background_task_summary=lambda: {"session_id": "sess-1", "total": 0, "queued": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0, "active": 0, "terminal": 0, "latest_updated_at_ms": None},
        export_artifact=lambda: {},
        list_background_tasks=lambda: [],
        describe_background_task=lambda task_id: {"task_id": task_id, "status": "running", "state_kind": "active", "title": "algo"},
    )

    handled = _handle_inspection_command("/task task-1", lifecycle, runtime=runtime)
    output = capsys.readouterr().out

    assert handled is True
    assert "task-1" in output


def test_handle_inspection_command_lista_prompt_snapshots(monkeypatch, capsys):
    from application.services.runtime import AgentRuntime

    monkeypatch.setattr("application.services.cli_dispatch.prompt_version_service.list_agents", lambda: ["math_agent", "code_agent"])
    runtime = AgentRuntime(gateway=MagicMock())
    lifecycle = SimpleNamespace(
        context_budget=lambda agent_name=None: {"session_id": "sess-1", "scope": agent_name or "session", "status": "ok", "budget_chars": 1000, "estimated_context_chars": 100, "estimated_remaining_chars": 900, "estimated_tokens": 25, "transcript_message_count": 1, "memory_present": False, "items": []},
        background_task_summary=lambda: {"session_id": "sess-1", "total": 0, "queued": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0, "active": 0, "terminal": 0, "latest_updated_at_ms": None},
        export_artifact=lambda: {"prompt_snapshots": [], "background_tasks": [], "audit_events": [], "trace_ids": []},
        list_background_tasks=lambda: [],
        describe_background_task=lambda task_id: None,
        list_bookmarks=lambda: [],
        describe_bookmark=lambda checkpoint_id: None,
    )

    handled = _handle_inspection_command("/prompts", lifecycle, runtime=runtime)
    output = capsys.readouterr().out

    assert handled is True
    assert "math_agent" in output
    assert "code_agent" in output


def test_handle_inspection_command_muestra_prompt_individual(monkeypatch, capsys):
    from application.services.runtime import AgentRuntime

    monkeypatch.setattr("application.services.cli_dispatch.prompt_version_service.load_snapshot", lambda agent: {"agent_name": agent, "prompt_version": "v1", "prompt_hash": "hash", "created_at_ms": 1, "extra_context": "ctx", "system_prompt": "prompt"})
    monkeypatch.setattr("application.services.cli_dispatch.prompt_version_service.load_history", lambda agent: [{"prompt_version": "v1"}])
    runtime = AgentRuntime(gateway=MagicMock())
    lifecycle = SimpleNamespace(
        context_budget=lambda agent_name=None: {"session_id": "sess-1", "scope": agent_name or "session", "status": "ok", "budget_chars": 1000, "estimated_context_chars": 100, "estimated_remaining_chars": 900, "estimated_tokens": 25, "transcript_message_count": 1, "memory_present": False, "items": []},
        background_task_summary=lambda: {"session_id": "sess-1", "total": 0, "queued": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0, "active": 0, "terminal": 0, "latest_updated_at_ms": None},
        export_artifact=lambda: {"prompt_snapshots": [], "background_tasks": [], "audit_events": [], "trace_ids": []},
        list_background_tasks=lambda: [],
        describe_background_task=lambda task_id: None,
        list_bookmarks=lambda: [],
        describe_bookmark=lambda checkpoint_id: None,
    )

    handled = _handle_inspection_command("/prompt math_agent", lifecycle, runtime=runtime)
    output = capsys.readouterr().out

    assert handled is True
    assert "math_agent" in output
    assert "historial=1" in output
    assert "snapshot_path=" in output
    assert "history_path=" in output


def test_handle_inspection_command_muestra_ruta_de_artifact(capsys):
    from application.services.runtime import AgentRuntime

    runtime = AgentRuntime(gateway=MagicMock())
    runtime._artifacts = MagicMock()  # type: ignore[attr-defined]
    runtime._artifacts.artifact_path.return_value = "sessions/sess-9/SESSION_ARTIFACT.json"  # type: ignore[attr-defined]
    lifecycle = SimpleNamespace(
        session_id="sess-9",
        context_budget=lambda agent_name=None: {"session_id": "sess-9", "scope": agent_name or "session", "status": "ok", "budget_chars": 1000, "estimated_context_chars": 100, "estimated_remaining_chars": 900, "estimated_tokens": 25, "transcript_message_count": 1, "memory_present": False, "items": []},
        background_task_summary=lambda: {"session_id": "sess-9", "total": 0, "queued": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0, "active": 0, "terminal": 0, "latest_updated_at_ms": None},
        export_artifact=lambda: {"session_id": "sess-9", "message_count": 0, "has_memory": False, "audit_events": [], "background_tasks": [], "prompt_snapshots": [], "trace_ids": [], "background_task_summary": {}},
        list_background_tasks=lambda: [],
        describe_background_task=lambda task_id: None,
        artifact_path=lambda: "sessions/sess-9/SESSION_ARTIFACT.json",
        list_bookmarks=lambda: [],
        describe_bookmark=lambda checkpoint_id: None,
    )

    handled = _handle_inspection_command("/artifact", lifecycle, runtime=runtime)
    output = capsys.readouterr().out

    assert handled is True
    assert "SESSION_ARTIFACT.json" in output


def test_handle_inspection_command_replay(capsys):
    from application.services.runtime import AgentRuntime

    runtime = AgentRuntime(gateway=MagicMock())
    runtime.build_session_replay = MagicMock(return_value=SimpleNamespace(  # type: ignore[attr-defined]
        session_id="sess-20",
        generated_at_ms=1,
        items=[SimpleNamespace(section="snapshot", title="session snapshot", detail="messages=0")],
    ))
    lifecycle = SimpleNamespace(session_id="sess-20")

    handled = _handle_inspection_command("/replay", lifecycle, runtime=runtime)
    output = capsys.readouterr().out

    assert handled is True
    assert "[replay]" in output


def test_handle_inspection_command_cancela_task(monkeypatch, capsys):
    from application.services.runtime import AgentRuntime

    scheduled = []
    monkeypatch.setattr("application.services.cli_dispatch.asyncio.create_task", lambda coro: (scheduled.append(coro), coro.close())[0])
    runtime = AgentRuntime(gateway=MagicMock())

    async def cancel_background_task(task_id, reason=None):
        return {"task_id": task_id, "status": "cancelled"}

    lifecycle = SimpleNamespace(
        session_id="sess-10",
        context_budget=lambda agent_name=None: {"session_id": "sess-10", "scope": agent_name or "session", "status": "ok", "budget_chars": 1000, "estimated_context_chars": 100, "estimated_remaining_chars": 900, "estimated_tokens": 25, "transcript_message_count": 1, "memory_present": False, "items": []},
        background_task_summary=lambda: {"session_id": "sess-10", "total": 0, "queued": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0, "retryable": 0, "active": 0, "terminal": 0, "latest_updated_at_ms": None},
        export_artifact=lambda: {"prompt_snapshots": [], "background_tasks": [], "audit_events": [], "trace_ids": []},
        list_background_tasks=lambda: [],
        list_retryable_background_tasks=lambda: [],
        describe_background_task=lambda task_id: None,
        artifact_path=lambda: "sessions/sess-10/SESSION_ARTIFACT.json",
        list_bookmarks=lambda: [],
        describe_bookmark=lambda checkpoint_id: None,
        cancel_background_task=cancel_background_task,
    )

    handled = _handle_inspection_command("/cancel task-1", lifecycle, runtime=runtime)
    output = capsys.readouterr().out

    assert handled is True
    assert "cancel requested" in output
    assert scheduled


def test_handle_inspection_command_lista_retryables(capsys):
    from application.services.runtime import AgentRuntime

    runtime = AgentRuntime(gateway=MagicMock())
    lifecycle = SimpleNamespace(
        session_id="sess-11",
        context_budget=lambda agent_name=None: {"session_id": "sess-11", "scope": agent_name or "session", "status": "ok", "budget_chars": 1000, "estimated_context_chars": 100, "estimated_remaining_chars": 900, "estimated_tokens": 25, "transcript_message_count": 1, "memory_present": False, "items": []},
        background_task_summary=lambda: {"session_id": "sess-11", "total": 0, "queued": 0, "running": 0, "completed": 0, "failed": 1, "cancelled": 1, "retryable": 2, "active": 0, "terminal": 2, "latest_updated_at_ms": None},
        export_artifact=lambda: {"prompt_snapshots": [], "background_tasks": [], "audit_events": [], "trace_ids": []},
        list_background_tasks=lambda: [],
        list_retryable_background_tasks=lambda: [{"task_id": "task-1", "status": "failed", "attempt_number": 1, "title": "algo"}],
        describe_background_task=lambda task_id: None,
        artifact_path=lambda: "sessions/sess-11/SESSION_ARTIFACT.json",
        cancel_background_task=lambda *args, **kwargs: None,
        list_bookmarks=lambda: [],
        describe_bookmark=lambda checkpoint_id: None,
    )

    handled = _handle_inspection_command("/retryable", lifecycle, runtime=runtime)
    output = capsys.readouterr().out

    assert handled is True
    assert "task-1" in output


def test_handle_inspection_command_context_y_bookmark(capsys):
    from application.services.runtime import AgentRuntime

    runtime = AgentRuntime(gateway=MagicMock())
    runtime.build_session_replay = MagicMock(return_value=SimpleNamespace(session_id="sess-ctx", generated_at_ms=1, items=[]))  # type: ignore[attr-defined]
    lifecycle = SimpleNamespace(
        session_id="sess-ctx",
        context_budget=lambda agent_name=None: {"session_id": "sess-ctx", "scope": agent_name or "session", "status": "ok", "budget_chars": 1000, "estimated_context_chars": 100, "estimated_remaining_chars": 900, "estimated_tokens": 25, "transcript_message_count": 2, "memory_present": True, "items": [{"section": "transcript", "role": "included", "chars": 100, "detail": "2 mensajes"}]},
        background_task_summary=lambda: {"session_id": "sess-ctx", "total": 0, "queued": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0, "active": 0, "terminal": 0, "latest_updated_at_ms": None},
        export_artifact=lambda: {"session_id": "sess-ctx", "message_count": 2, "has_memory": True, "audit_events": [], "background_tasks": [], "prompt_snapshots": [], "bookmarks": [], "context_budget": {"report": {"scope": "session", "status": "ok", "estimated_context_chars": 100, "estimated_remaining_chars": 900}}, "trace_ids": [], "background_task_summary": {}},
        list_background_tasks=lambda: [],
        list_retryable_background_tasks=lambda: [],
        describe_background_task=lambda task_id: None,
        artifact_path=lambda: "sessions/sess-ctx/SESSION_ARTIFACT.json",
        list_bookmarks=lambda: [{"checkpoint_id": "chk-1", "label": "base", "message_count": 2, "has_memory": True}],
        describe_bookmark=lambda checkpoint_id: {"checkpoint_id": checkpoint_id, "label": "base", "session_id": "sess-ctx", "created_at_ms": 1, "message_count": 2, "has_memory": True, "replay_item_count": 1, "artifact_path": "sessions/sess-ctx/SESSION_ARTIFACT.json", "context_budget": {"report": {"scope": "session", "status": "ok", "estimated_context_chars": 100, "estimated_remaining_chars": 900}}, "prompt_agents": ["math_agent"]},
        create_bookmark=lambda label=None, note="": {"checkpoint_id": "chk-2", "label": label or "checkpoint-2", "session_id": "sess-ctx", "created_at_ms": 2, "message_count": 2, "has_memory": True, "replay_item_count": 1, "artifact_path": "sessions/sess-ctx/SESSION_ARTIFACT.json", "context_budget": {"report": {"scope": "session", "status": "ok", "estimated_context_chars": 100, "estimated_remaining_chars": 900}}, "prompt_agents": ["math_agent"]},
    )

    handled_context = _handle_inspection_command("/context math_agent", lifecycle, runtime=runtime)
    handled_bookmark = _handle_inspection_command("/bookmark base first checkpoint", lifecycle, runtime=runtime)
    handled_bookmarks = _handle_inspection_command("/bookmarks", lifecycle, runtime=runtime)
    handled_checkpoint = _handle_inspection_command("/checkpoint chk-1", lifecycle, runtime=runtime)
    output = capsys.readouterr().out

    assert handled_context is True
    assert handled_bookmark is True
    assert handled_bookmarks is True
    assert handled_checkpoint is True
    assert "[context]" in output
    assert "[bookmark] checkpoint guardado" in output
    assert "[bookmarks]" in output
    assert "[checkpoint]" in output


def test_handle_inspection_command_commands_y_command(capsys):
    from application.services.runtime import AgentRuntime

    runtime = AgentRuntime(gateway=MagicMock())
    lifecycle = SimpleNamespace(
        session_id="sess-commands",
        context_budget=lambda agent_name=None: {"session_id": "sess-commands", "scope": agent_name or "session", "status": "ok", "budget_chars": 1000, "estimated_context_chars": 100, "estimated_remaining_chars": 900, "estimated_tokens": 25, "transcript_message_count": 1, "memory_present": False, "items": []},
        background_task_summary=lambda: {"session_id": "sess-commands", "total": 0, "queued": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0, "active": 0, "terminal": 0, "latest_updated_at_ms": None},
        export_artifact=lambda: {"session_id": "sess-commands", "message_count": 0, "has_memory": False, "audit_events": [], "background_tasks": [], "prompt_snapshots": [], "bookmarks": [], "context_budget": {"report": {"scope": "session", "status": "ok", "estimated_context_chars": 100, "estimated_remaining_chars": 900}}, "trace_ids": [], "background_task_summary": {}},
        list_background_tasks=lambda: [],
        list_retryable_background_tasks=lambda: [],
        describe_background_task=lambda task_id: None,
        artifact_path=lambda: "sessions/sess-commands/SESSION_ARTIFACT.json",
        list_bookmarks=lambda: [],
        describe_bookmark=lambda checkpoint_id: None,
        create_bookmark=lambda label=None, note="": {},
    )

    handled_commands = _handle_inspection_command("/commands", lifecycle, runtime=runtime)
    handled_command = _handle_inspection_command("/command help", lifecycle, runtime=runtime)
    handled_alias = _handle_inspection_command("/status", lifecycle, runtime=runtime)
    output = capsys.readouterr().out

    assert handled_commands is True
    assert handled_command is True
    assert handled_alias is True
    assert "[commands]" in output
    assert "[command]" in output
