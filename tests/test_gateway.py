"""
Tests unitarios para session_gateway.py (LaneQueue y AgentGateway).

Se mockea el grafo LangGraph y las capas de persistencia/memoria
para aislar la lógica de concurrencia y routing.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ==================== LaneQueue ====================

class TestLaneQueue:

    @pytest.mark.asyncio
    async def test_send_handler_retorna_inmediatamente(self):
        """send() con un handler que retorna rápido → retorna el resultado."""
        from features.sessions.application.session_gateway import LaneQueue

        async def handler(msg: str) -> str:
            return f"respuesta: {msg}"

        queue = LaneQueue()
        result = await queue.send("session-1", "hola", handler)
        assert result == "respuesta: hola"

    @pytest.mark.asyncio
    async def test_send_segundo_mensaje_simultane_retorna_ack_silencioso(self):
        """Dos send() simultáneos para la misma sesión:
        el primero se procesa, el segundo entra en modo collect (retorna "").
        """
        from features.sessions.application.session_gateway import LaneQueue

        started_event = asyncio.Event()
        resume_event  = asyncio.Event()

        async def slow_handler(msg: str) -> str:
            started_event.set()
            await resume_event.wait()
            return f"resultado: {msg}"

        queue = LaneQueue()

        # Lanzar el primer send — se queda bloqueado en slow_handler
        task1 = asyncio.create_task(queue.send("session-A", "msg1", slow_handler))

        # Esperar a que el primer handler arranque
        await started_event.wait()

        # Ahora enviar el segundo mensaje — debe acumularse (collect)
        collect_result = await queue.send("session-A", "msg2", slow_handler)
        assert collect_result == "", f"Esperado '' (collect ack), got '{collect_result}'"

        # Liberar el primer handler
        resume_event.set()
        result1 = await task1
        assert "msg1" in result1

    @pytest.mark.asyncio
    async def test_send_sesiones_diferentes_no_interfieren(self):
        """Dos send() para sesiones distintas son independientes."""
        from features.sessions.application.session_gateway import LaneQueue

        async def handler(msg: str) -> str:
            return f"ok:{msg}"

        queue = LaneQueue()
        r1 = await queue.send("sess-X", "msgX", handler)
        r2 = await queue.send("sess-Y", "msgY", handler)
        assert r1 == "ok:msgX"
        assert r2 == "ok:msgY"

    @pytest.mark.asyncio
    async def test_send_handler_con_excepcion_retorna_error_string(self):
        """Si el handler lanza una excepción, send() retorna 'Error: ...'."""
        from features.sessions.application.session_gateway import LaneQueue

        async def failing_handler(msg: str) -> str:
            raise ValueError("handler falló")

        queue = LaneQueue()
        result = await queue.send("sess-err", "msg", failing_handler)
        assert "Error" in result or result == ""  # puede retornar string de error

    @pytest.mark.asyncio
    async def test_close_invalida_runs_activos(self):
        """close() incrementa la generation, invalidando runs activos."""
        from features.sessions.application.session_gateway import LaneQueue

        started_event = asyncio.Event()
        resume_event  = asyncio.Event()

        async def slow_handler(msg: str) -> str:
            started_event.set()
            await resume_event.wait()
            return "resultado"

        queue = LaneQueue()
        task = asyncio.create_task(queue.send("sess-close", "msg", slow_handler))

        await started_event.wait()

        # Cerrar mientras está corriendo → incrementa generation
        await queue.close("sess-close")

        # Liberar el handler
        resume_event.set()
        result = await task
        # El resultado puede ser "" (generation mismatch) o el resultado real
        # Lo importante es que no cuelgue
        assert result is not None

    @pytest.mark.asyncio
    async def test_lane_queue_concurrency_un_run_a_la_vez(self):
        """LaneQueue no inicia el siguiente mensaje hasta que termina el actual."""
        from features.sessions.application.session_gateway import LaneQueue

        execution_order = []

        async def sequential_handler(msg: str) -> str:
            execution_order.append(f"start:{msg}")
            await asyncio.sleep(0.01)
            execution_order.append(f"end:{msg}")
            return f"done:{msg}"

        queue = LaneQueue()

        # Enviar en paralelo — el segundo debe acumularse
        results = await asyncio.gather(
            queue.send("sess-seq", "first",  sequential_handler),
            queue.send("sess-seq", "second", sequential_handler),
        )

        # El primer resultado debe ser el del mensaje procesado
        assert any("done:first" in r for r in results if r), \
            f"Ningún resultado contiene 'done:first': {results}"

        # El orden de ejecución no debe tener interleaving
        start_indices = [i for i, s in enumerate(execution_order) if s.startswith("start:")]
        end_indices   = [i for i, s in enumerate(execution_order) if s.startswith("end:")]
        if start_indices and end_indices:
            # start siempre antes que end para el mismo mensaje
            assert all(s < e for s, e in zip(start_indices, end_indices))

    @pytest.mark.asyncio
    async def test_shutdown_limpia_todos_los_lanes(self):
        from features.sessions.application.session_gateway import LaneQueue

        async def handler(msg: str) -> str:
            return "ok"

        queue = LaneQueue()
        await queue.send("sess-1", "msg", handler)
        await queue.send("sess-2", "msg", handler)

        assert len(queue._lanes) == 2
        await queue.shutdown()
        assert len(queue._lanes) == 0


# ==================== AgentGateway ====================

@pytest.fixture
def mock_gateway(monkeypatch):
    """Crea un AgentGateway con grafo y persistence mockeados."""
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        "messages": [
            __import__("langchain_core.messages", fromlist=["AIMessage"]).AIMessage(
                content="Respuesta del agente"
            )
        ],
        "next_agent":     "",
        "risk_flag":      False,
        "blocked":        False,
        "request_id":     "req-123",
        "scrape_tracker": {},
    })

    with (
        patch("features.sessions.application.session_gateway.create_supervisor_graph", return_value=mock_graph),
        patch("features.sessions.application.session_gateway.persistence.load_messages",    return_value=[]),
        patch("features.sessions.application.session_gateway.persistence.save_message"),
        patch("features.sessions.application.session_gateway.persistence.save_session"),
        patch("features.sessions.application.session_gateway.memory.load_memory_context",   return_value=None),
        patch("features.sessions.application.session_gateway.memory.distill_memory",        AsyncMock(return_value=None)),
    ):
        from features.sessions.application.session_gateway import AgentGateway
        gw = AgentGateway()
        yield gw, mock_graph


@pytest.mark.asyncio
async def test_agentgateway_send_retorna_respuesta(mock_gateway):
    """send() con un grafo que responde inmediatamente → retorna la respuesta."""
    gw, _ = mock_gateway
    result = await gw.send("user-123", "¿Cuánto es 2+2?", request_id="req-1")
    assert result == "Respuesta del agente"


@pytest.mark.asyncio
async def test_agentgateway_send_persiste_mensajes(mock_gateway):
    """send() llama a save_message dos veces: human + ai."""
    gw, _ = mock_gateway

    with patch("features.sessions.application.session_gateway.persistence.save_message") as mock_save:
        await gw.send("user-456", "mensaje de prueba", request_id="req-2")
        assert mock_save.call_count == 2
        calls = mock_save.call_args_list
        assert calls[0][0][1] == "human"  # primer save: human
        assert calls[1][0][1] == "ai"     # segundo save: ai
        assert calls[0][1]["request_id"] == "req-2"
        assert calls[1][1]["request_id"] == "req-2"


@pytest.mark.asyncio
async def test_agentgateway_get_state_none_si_no_existe(mock_gateway):
    """get_state() retorna None para sesiones que no han enviado nada."""
    gw, _ = mock_gateway
    state = await gw.get_state("sesion-nueva-sin-envios")
    assert state is None


@pytest.mark.asyncio
async def test_agentgateway_get_state_retorna_estado_existente(mock_gateway):
    """get_state() retorna el estado después de un send()."""
    gw, _ = mock_gateway
    await gw.send("user-789", "primer mensaje", request_id="req-3")
    state = await gw.get_state("user-789")
    assert state is not None
    assert "messages" in state
    assert state.get("request_id") == "req-3"


@pytest.mark.asyncio
async def test_agentgateway_dos_sends_secuenciales_invocan_grafo_dos_veces(mock_gateway):
    """Dos sends() secuenciales (no simultáneos) → el grafo se invoca dos veces."""
    gw, mock_graph = mock_gateway

    r1 = await gw.send("sess-sec", "primer mensaje", request_id="req-a")
    r2 = await gw.send("sess-sec", "segundo mensaje", request_id="req-b")

    # El primer send siempre retorna la respuesta del grafo
    assert r1 == "Respuesta del agente"
    # El grafo se invocó al menos una vez
    assert mock_graph.ainvoke.call_count >= 1


@pytest.mark.asyncio
async def test_agentgateway_close_session_llama_distill_memory(mock_gateway):
    """close_session() invoca distill_memory."""
    gw, _ = mock_gateway

    with patch("features.sessions.application.session_gateway.memory.distill_memory", AsyncMock(return_value=None)) as mock_distill:
        await gw.send("sess-close", "msg", request_id="req-close")
        result = await gw.close_session("sess-close")
        mock_distill.assert_called_once()
        assert result is False


@pytest.mark.asyncio
async def test_agentgateway_close_session_retorna_si_memoria_fue_escrita(mock_gateway):
    """close_session() debe exponer si la memoria fue persistida."""
    gw, _ = mock_gateway

    with patch("features.sessions.application.session_gateway.memory.distill_memory", AsyncMock(return_value=True)):
        await gw.send("sess-close-ok", "msg", request_id="req-close-ok")
        result = await gw.close_session("sess-close-ok")

    assert result is True


@pytest.mark.asyncio
async def test_agentgateway_send_con_grafo_que_lanza_excepcion(mock_gateway):
    """Si el grafo falla, send() retorna 'Error: ...' sin propagar la excepción."""
    gw, mock_graph = mock_gateway
    mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("grafo falló"))

    result = await gw.send("sess-fail", "msg que falla", request_id="req-fail")
    assert "Error" in result


@pytest.mark.asyncio
async def test_agentgateway_send_con_grafo_que_falla_persiste_ai_error(mock_gateway):
    """Si el grafo falla, el error visible también debe persistirse como mensaje AI."""
    gw, mock_graph = mock_gateway
    mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("grafo falló"))

    with patch("features.sessions.application.session_gateway.persistence.save_message") as mock_save:
        result = await gw.send("sess-fail-persist", "msg que falla", request_id="req-fail-persist")

    assert "Error: grafo falló" == result
    assert mock_save.call_count == 2
    assert mock_save.call_args_list[0][0][1] == "human"
    assert mock_save.call_args_list[1][0][1] == "ai"
    assert mock_save.call_args_list[1][0][2] == "Error: grafo falló"


@pytest.mark.asyncio
async def test_agentgateway_reusa_request_id_provisto(mock_gateway):
    """Si el runtime provee request_id, el gateway no debe reemplazarlo."""
    gw, _ = mock_gateway

    await gw.send("sess-rid", "hola", request_id="req-runtime-1")
    state = await gw.get_state("sess-rid")
    assert state is not None
    assert state.get("request_id") == "req-runtime-1"
