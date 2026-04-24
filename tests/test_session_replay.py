"""Tests para el replay unificado de sesión."""
from types import SimpleNamespace
from unittest.mock import MagicMock


def _handle_inspection_command(user_input, lifecycle, runtime=None):
    from application.services.cli_dispatch import dispatch_inspection_command

    result = dispatch_inspection_command(user_input, lifecycle, runtime)
    for line in result.lines:
        print(line)
    return result.handled


def test_session_replay_service_combina_eventos_en_secciones(monkeypatch):
    from features.sessions.application.session_replay import SessionReplayService

    artifact = SimpleNamespace(
        session_id="sess-1",
        generated_at_ms=10,
        message_count=2,
        has_memory=True,
        is_existing_session=True,
        transcript=[
            {"role": "human", "content": "hola"},
            {"role": "ai", "content": "respuesta"},
        ],
        memory_markdown="memoria",
        audit_events=[
            {"event_type": "tool_call_requested", "tool_name": "calculate", "outcome": "requested", "ts_ms": 2, "request_id": "req-1", "trace_id": "trace-1"},
            {"event_type": "tool_call_completed", "tool_name": "calculate", "outcome": "success", "ts_ms": 3, "request_id": "req-1", "trace_id": "trace-1"},
        ],
        background_tasks=[{"task_id": "task-1", "status": "completed", "attempt_number": 1, "title": "hacer algo"}],
        background_task_summary={"total": 1},
        prompt_snapshots=[{"agent_name": "math_agent", "prompt_version": "v1", "prompt_hash": "abc", "snapshot_path": "agents/snapshots/math_agent/PROMPT_SNAPSHOT.json"}],
        trace_ids=["trace-1"],
    )

    from application.services import session_replay as replay_module

    monkeypatch.setattr(replay_module.session_artifact_service, "build_artifact", MagicMock(return_value=artifact))

    replay = SessionReplayService().build_replay("sess-1")

    assert replay.session_id == "sess-1"
    assert replay.items[0].section == "snapshot"
    assert any(item.section == "prompt" for item in replay.items)
    assert any(item.section == "message" for item in replay.items)
    assert any(item.section == "background-task" for item in replay.items)
    assert any(item.section == "tool-audit" for item in replay.items)
    assert replay.items[-1].section == "summary"


def test_format_session_replay_muestra_timeline():
    from features.sessions.application.session_replay import ReplayTimelineItem, SessionReplay
    from features.sessions.application.session_inspection import format_replay_timeline

    replay = SessionReplay(
        session_id="sess-2",
        generated_at_ms=1,
        items=[
            ReplayTimelineItem(section="snapshot", order=0, title="session snapshot", detail="messages=1"),
            ReplayTimelineItem(section="message", order=1, title="human", detail="hola"),
        ],
    )

    lines = format_replay_timeline(replay)

    assert any("sess-2" in line for line in lines)
    assert any("[message]" in line for line in lines)


def test_handle_inspection_command_replay(monkeypatch, capsys):
    from application.services.runtime import AgentRuntime

    runtime = AgentRuntime(gateway=MagicMock())
    runtime.build_session_replay = MagicMock(return_value=SimpleNamespace(  # type: ignore[attr-defined]
        session_id="sess-3",
        generated_at_ms=1,
        items=[SimpleNamespace(section="snapshot", title="session snapshot", detail="messages=0")],
    ))
    lifecycle = SimpleNamespace(session_id="sess-3")

    handled = _handle_inspection_command("/replay", lifecycle, runtime=runtime)
    output = capsys.readouterr().out

    assert handled is True
    assert "[replay]" in output
    runtime.build_session_replay.assert_called_once_with("sess-3")  # type: ignore[attr-defined]
