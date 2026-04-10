"""Tests unitarios para el runtime/orchestrator de alto nivel."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def test_snapshot_session_retorna_resumen_y_detecta_sesion_existente():
    from application.services.runtime import AgentRuntime

    mock_gateway = MagicMock()
    mock_gateway.load_session_info.return_value = (3, True)

    with patch("application.services.runtime.AgentGateway", return_value=mock_gateway), \
         patch("application.services.runtime.session_persistence.list_sessions", return_value=["sess-1"]):
        runtime = AgentRuntime()
        snapshot = runtime.snapshot_session("sess-1")

    assert snapshot.session_id == "sess-1"
    assert snapshot.message_count == 3
    assert snapshot.has_memory is True
    assert snapshot.is_existing_session is True


def test_resolve_session_construye_contexto_con_snapshot_y_request_id():
    from application.services.runtime import AgentRuntime

    mock_gateway = MagicMock()
    mock_gateway.load_session_info.return_value = (2, False)

    with patch("application.services.runtime.AgentGateway", return_value=mock_gateway), \
         patch("application.services.runtime.session_persistence.list_sessions", return_value=["sess-3"]):
        runtime = AgentRuntime()
        resolution = runtime.resolve_session("sess-3", "hola")
        turn = resolution.turn_context

    assert turn is not None
    assert turn.session_id == "sess-3"
    assert turn.message == "hola"
    assert turn.request_id
    assert turn.trace.session_id == "sess-3"
    assert turn.trace.request_id == turn.request_id
    assert turn.trace.operation == "turn"
    assert turn.snapshot.session_id == "sess-3"
    assert turn.snapshot.message_count == 2
    assert turn.snapshot.has_memory is False


def test_resolve_session_construye_overview_sin_turno():
    from application.services.runtime import AgentRuntime

    mock_gateway = MagicMock()
    mock_gateway.load_session_info.return_value = (7, True)

    with patch("application.services.runtime.AgentGateway", return_value=mock_gateway), \
         patch("application.services.runtime.session_persistence.list_sessions", return_value=["sess-6"]):
        runtime = AgentRuntime()
        resolution = runtime.resolve_session("sess-6")

    assert resolution.session_id == "sess-6"
    assert resolution.overview.session_id == "sess-6"
    assert resolution.overview.message_count == 7
    assert resolution.overview.has_memory is True
    assert resolution.turn_context is None


def test_resolve_session_construye_overview_y_turn_context():
    from application.services.runtime import AgentRuntime

    mock_gateway = MagicMock()
    mock_gateway.load_session_info.return_value = (2, False)

    with patch("application.services.runtime.AgentGateway", return_value=mock_gateway), \
         patch("application.services.runtime.session_persistence.list_sessions", return_value=["sess-7"]):
        runtime = AgentRuntime()
        resolution = runtime.resolve_session("sess-7", "hola")

    assert resolution.session_id == "sess-7"
    assert resolution.turn_context is not None
    assert resolution.turn_context.message == "hola"
    assert resolution.turn_context.snapshot.message_count == 2
    assert resolution.turn_context.trace.operation == "turn"


def test_build_session_view_retorna_banner_y_prompt():
    from application.services.runtime import AgentRuntime

    mock_gateway = MagicMock()
    mock_gateway.load_session_info.return_value = (6, True)

    with patch("application.services.runtime.AgentGateway", return_value=mock_gateway), \
         patch("application.services.runtime.session_persistence.list_sessions", return_value=["sess-8"]):
        runtime = AgentRuntime()
        view = runtime.build_session_view("sess-8")

    assert view.snapshot.session_id == "sess-8"
    assert view.snapshot.has_memory is True
    assert len(view.banner_lines) == 3
    assert "memoria cargada" in view.banner_lines[0]
    assert "[contexto]" in view.banner_lines[1]
    assert "[atajos]" in view.banner_lines[2]
    assert view.prompt_hint == "Escribí un mensaje, /help o salir para terminar"


def test_start_session_lifecycle_encadena_seleccion_y_vistas():
    from application.services.runtime import AgentRuntime

    mock_gateway = MagicMock()
    mock_gateway.load_session_info.return_value = (1, False)

    with patch("application.services.runtime.AgentGateway", return_value=mock_gateway), \
         patch("application.services.runtime.session_persistence.list_sessions", return_value=["sess-9"]):
        runtime = AgentRuntime()
        lifecycle = runtime.start_session_lifecycle("sess-9")
        view = lifecycle.view()
        resolution = lifecycle.resolve("hola")

    assert lifecycle.session_id == "sess-9"
    assert view.snapshot.session_id == "sess-9"
    assert resolution.session_id == "sess-9"
    assert resolution.turn_context is not None
    assert resolution.turn_context.trace.operation == "turn"


@pytest.mark.asyncio
async def test_send_turn_delega_en_gateway_y_devuelve_metadata():
    from application.services.runtime import AgentRuntime

    mock_gateway = MagicMock()
    mock_gateway.load_session_info.return_value = (4, False)
    mock_gateway.send = AsyncMock(return_value="respuesta final")

    with patch("application.services.runtime.AgentGateway", return_value=mock_gateway):
        runtime = AgentRuntime()
        result = await runtime.send_turn("sess-2", "hola")

    mock_gateway.send.assert_awaited_once_with("sess-2", "hola", request_id=result.request_id, trace_id=result.trace.trace_id)
    assert result.response == "respuesta final"
    assert result.request_id
    assert result.message_count == 4
    assert result.has_memory is False
    assert result.trace.operation == "turn"


@pytest.mark.asyncio
async def test_execute_turn_usa_contexto_explicito():
    from application.services.runtime import AgentRuntime

    mock_gateway = MagicMock()
    mock_gateway.load_session_info.return_value = (1, True)
    mock_gateway.send = AsyncMock(return_value="ok")

    with patch("application.services.runtime.AgentGateway", return_value=mock_gateway), \
         patch("application.services.runtime.session_persistence.list_sessions", return_value=["sess-4"]):
        runtime = AgentRuntime()
        resolution = runtime.resolve_session("sess-4", "mensaje")
        context = resolution.turn_context
        assert context is not None
        result = await runtime.execute_turn(context)

    mock_gateway.send.assert_awaited_once_with("sess-4", "mensaje", request_id=context.request_id, trace_id=context.trace.trace_id)
    assert result.session_id == "sess-4"
    assert result.request_id == context.request_id
    assert result.response == "ok"
    assert result.trace == context.trace


@pytest.mark.asyncio
async def test_close_session_retorna_resumen_before_after_y_memoria():
    from application.services.runtime import AgentRuntime

    mock_gateway = MagicMock()
    mock_gateway.load_session_info.return_value = (5, False)
    mock_gateway.close_session = AsyncMock(return_value=True)

    with patch("application.services.runtime.AgentGateway", return_value=mock_gateway), \
         patch("application.services.runtime.session_persistence.list_sessions", return_value=["sess-5"]):
        runtime = AgentRuntime()
        closure = await runtime.close_session("sess-5")

    assert closure.session_id == "sess-5"
    assert closure.before.session_id == "sess-5"
    assert closure.after.session_id == "sess-5"
    assert closure.before.message_count == 5
    assert closure.after.message_count == 5
    assert closure.memory_written is True
    assert closure.trace.operation == "close"
    mock_gateway.close_session.assert_awaited_once_with("sess-5")


def test_select_session_id_preserva_id_existente():
    from application.services.runtime import AgentRuntime

    runtime = AgentRuntime(gateway=MagicMock())
    with patch("application.services.runtime.session_persistence.list_sessions", return_value=["manual-id"]):
        assert runtime.select_session_id("manual-id") == "manual-id"


def test_select_session_id_retorna_sesion_existente_o_nueva():
    from application.services.runtime import AgentRuntime

    runtime = AgentRuntime(gateway=MagicMock())

    with patch("application.services.runtime.session_persistence.list_sessions", return_value=["sess-x"]):
        assert runtime.select_session_id("sess-x") == "sess-x"

    with patch("application.services.runtime.session_persistence.list_sessions", return_value=["sess-x"]):
        selected = runtime.select_session_id("no-existe")
        assert selected
        assert selected != "no-existe"


@pytest.mark.asyncio
async def test_runtime_worker_methods_delegan_en_coordinator():
    from application.services.runtime import AgentRuntime

    mock_gateway = MagicMock()
    mock_coordinator = MagicMock()
    mock_coordinator.spawn_worker = AsyncMock(return_value={"worker_id": "w1"})
    mock_coordinator.send_message = AsyncMock(return_value={"task_id": "t1"})
    mock_coordinator.execute_worker_turn = AsyncMock(return_value={"response": "ok"})
    mock_coordinator.list_workers.return_value = ["worker-1"]
    mock_coordinator.list_messages.return_value = [{"content": "ok"}]

    with patch("application.services.runtime.AgentGateway", return_value=mock_gateway), \
         patch("application.services.runtime.coordinator_runtime_service", mock_coordinator):
        runtime = AgentRuntime()
        runtime._coordinator = mock_coordinator

        worker = await runtime.spawn_worker("sess-coord", "math_agent", worker_name="math-worker")
        message = await runtime.send_worker_message("sess-coord", "w1", "hola")
        execution = await runtime._coordinator.execute_worker_turn("sess-coord", "w1", "hola")
        workers = runtime.list_workers("sess-coord")
        messages = runtime.list_worker_messages("sess-coord")

    assert worker == {"worker_id": "w1"}
    assert message == {"task_id": "t1"}
    assert execution == {"response": "ok"}
    assert workers == ["worker-1"]
    assert messages == [{"content": "ok"}]
    mock_coordinator.spawn_worker.assert_awaited_once_with(
        "sess-coord",
        "math_agent",
        worker_name="math-worker",
        parent_worker_id=None,
        metadata=None,
    )
    mock_coordinator.send_message.assert_awaited_once_with("sess-coord", "w1", "hola", sender="coordinator")
    mock_coordinator.execute_worker_turn.assert_awaited_once_with("sess-coord", "w1", "hola")
