"""Tests para la persistencia del audit trail de tools."""


def test_tool_audit_store_persiste_y_lee_eventos(tmp_path):
    from application.services.tool_audit import ToolCallAuditEvent
    from application.services.tool_audit_store import ToolAuditStore

    store = ToolAuditStore(audit_dir=tmp_path)
    event = ToolCallAuditEvent(
        event_type="tool_call_completed",
        request_id="req-1",
        agent_name="math_agent",
        tool_name="calculate",
        outcome="success",
        ts_ms=1,
        session_id="sess-1",
        duration_ms=12,
        output_preview="ok",
    )

    store.append_event(event)
    loaded = store.load_events("sess-1")

    assert len(loaded) == 1
    assert loaded[0]["tool_name"] == "calculate"
    assert loaded[0]["duration_ms"] == 12


def test_tool_audit_store_filtra_por_session_y_request(tmp_path):
    from application.services.tool_audit import ToolCallAuditEvent
    from application.services.tool_audit_store import ToolAuditStore

    store = ToolAuditStore(audit_dir=tmp_path)
    store.append_event(
        ToolCallAuditEvent(
            event_type="tool_call_requested",
            request_id="req-2",
            agent_name="code_agent",
            tool_name="write_code",
            outcome="requested",
            ts_ms=1,
            session_id="sess-2",
        )
    )
    store.append_event(
        ToolCallAuditEvent(
            event_type="tool_call_completed",
            request_id="req-3",
            agent_name="code_agent",
            tool_name="write_code",
            outcome="success",
            ts_ms=2,
            session_id="sess-2",
        )
    )

    events = store.find_events("sess-2", request_id="req-3")
    assert len(events) == 1
    assert events[0]["request_id"] == "req-3"


def test_tool_audit_store_lista_sesiones(tmp_path):
    from application.services.tool_audit import ToolCallAuditEvent
    from application.services.tool_audit_store import ToolAuditStore

    store = ToolAuditStore(audit_dir=tmp_path)
    store.append_event(
        ToolCallAuditEvent(
            event_type="tool_call_requested",
            request_id="req-4",
            agent_name="math_agent",
            tool_name="calculate",
            outcome="requested",
            ts_ms=1,
            session_id="sess-a",
        )
    )
    store.append_event(
        ToolCallAuditEvent(
            event_type="tool_call_requested",
            request_id="req-5",
            agent_name="math_agent",
            tool_name="calculate",
            outcome="requested",
            ts_ms=1,
            session_id="sess-b",
        )
    )

    assert store.list_sessions() == ["sess-a", "sess-b"]
