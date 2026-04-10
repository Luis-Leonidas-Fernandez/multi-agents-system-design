"""Tests para el audit trail de tool calls."""
from unittest.mock import patch


def test_tool_audit_service_emite_evento_serializado():
    from application.services.tool_audit import ToolAuditService, ToolCallAuditEvent

    service = ToolAuditService()
    event = ToolCallAuditEvent(
        event_type="tool_call_requested",
        request_id="req-1",
        agent_name="math_agent",
        tool_name="calculate",
        outcome="requested",
        ts_ms=123,
        arguments={"expression": "2 + 2"},
    )

    with patch("application.services.tool_audit._emit_guard_audit") as mock_emit, \
         patch("application.services.tool_audit.tool_audit_store.append_event") as mock_append:
        service.record(event)

    mock_emit.assert_called_once()
    mock_append.assert_called_once_with(event)
    payload = mock_emit.call_args.args[0]
    assert payload["tool_name"] == "calculate"
    assert payload["outcome"] == "requested"


def test_tool_audit_service_helpers_llenan_contexto():
    from application.services.tool_audit import ToolAuditService

    service = ToolAuditService()

    with patch("application.services.tool_audit._emit_guard_audit") as mock_emit, \
         patch("application.services.tool_audit.tool_audit_store.append_event"):
        service.requested(
            request_id="req-2",
            agent_name="code_agent",
            tool_name="write_code",
            session_id="sess-1",
            trace_id="trace-1",
            arguments={"task": "hola"},
        )

    payload = mock_emit.call_args.args[0]
    assert payload["session_id"] == "sess-1"
    assert payload["trace_id"] == "trace-1"
    assert payload["event_type"] == "tool_call_requested"


def test_tool_audit_service_no_rompe_si_persistencia_falla():
    from application.services.tool_audit import ToolAuditService, ToolCallAuditEvent

    service = ToolAuditService()
    event = ToolCallAuditEvent(
        event_type="tool_call_requested",
        request_id="req-3",
        agent_name="math_agent",
        tool_name="calculate",
        outcome="requested",
        ts_ms=123,
    )

    with patch("application.services.tool_audit._emit_guard_audit"), \
         patch("application.services.tool_audit.tool_audit_store.append_event", side_effect=RuntimeError("boom")):
        service.record(event)


def test_tool_audit_service_query_helpers_delegan_en_store():
    from application.services.tool_audit import ToolAuditService

    service = ToolAuditService()

    with patch("application.services.tool_audit.tool_audit_store.load_events", return_value=[{"request_id": "req-1"}]) as mock_load, \
         patch("application.services.tool_audit.tool_audit_store.find_events", return_value=[{"request_id": "req-1"}]) as mock_find, \
         patch("application.services.tool_audit.tool_audit_store.list_sessions", return_value=["sess-1"]) as mock_list:
        assert service.load_session_events("sess-1") == [{"request_id": "req-1"}]
        assert service.find_session_events("sess-1", request_id="req-1") == [{"request_id": "req-1"}]
        assert service.list_sessions() == ["sess-1"]

    mock_load.assert_called_once_with("sess-1")
    mock_find.assert_called_once_with("sess-1", request_id="req-1", trace_id=None, tool_name=None)
    mock_list.assert_called_once()
