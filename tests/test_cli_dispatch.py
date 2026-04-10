"""Tests para el dispatcher compartido de comandos CLI/TUI."""
from unittest.mock import MagicMock


def test_dispatch_inspection_command_help_returns_help_lines():
    from application.services.cli_dispatch import dispatch_inspection_command

    result = dispatch_inspection_command("/help", MagicMock(), MagicMock())

    assert result.handled is True
    assert any("/commands" in line for line in result.lines)


def test_dispatch_inspection_command_tool_parses_arguments():
    from application.services.cli_dispatch import dispatch_inspection_command

    runtime = MagicMock()
    runtime.preview_tool.return_value = {"tool_name": "echo", "allowed": True, "requires_confirmation": False, "permission_mode": "auto", "risk_level": "low"}

    result = dispatch_inspection_command("/tool echo enabled=true count=2", MagicMock(), runtime)

    assert result.handled is True
    runtime.preview_tool.assert_called_once()
    assert any("echo" in line for line in result.lines)


def test_dispatch_inspection_command_workers_and_mailbox():
    from application.services.cli_dispatch import dispatch_inspection_command

    runtime = MagicMock()
    runtime.list_workers.return_value = [
        type("W", (), {"worker_id": "w-1", "worker_name": "math", "agent_name": "math_agent", "status": "idle", "session_id": "sess-1", "created_at_ms": 1, "updated_at_ms": 2})()
    ]
    runtime.list_worker_messages.return_value = [{"created_at_ms": 1, "sender": "coord", "recipient": "w-1", "kind": "direct", "content": "hola"}]

    workers = dispatch_inspection_command("/workers", MagicMock(session_id="sess-1"), runtime)
    mailbox = dispatch_inspection_command("/mailbox", MagicMock(session_id="sess-1"), runtime)
    detail = dispatch_inspection_command("/worker w-1", MagicMock(session_id="sess-1"), runtime)

    assert workers.handled is True
    assert any("[workers]" in line for line in workers.lines)
    assert mailbox.handled is True
    assert any("[mailbox]" in line for line in mailbox.lines)
    assert detail.handled is True
    assert any("[worker]" in line for line in detail.lines)
