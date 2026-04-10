"""Tests para el contexto de trazabilidad."""


def test_trace_context_service_crea_ids_unicos():
    from application.services.trace_context import TraceContextService

    service = TraceContextService()
    trace = service.create("sess-1", "turn")

    assert trace.session_id == "sess-1"
    assert trace.operation == "turn"
    assert trace.request_id
    assert trace.trace_id


def test_trace_context_service_acepta_parent_trace_id():
    from application.services.trace_context import TraceContextService

    service = TraceContextService()
    trace = service.create("sess-1", "close", parent_trace_id="parent-1")

    assert trace.parent_trace_id == "parent-1"
