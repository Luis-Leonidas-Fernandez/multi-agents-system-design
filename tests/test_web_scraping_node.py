"""
Tests unitarios para nodes/web_scraping_node.py.

Usa mocks para aislar: agente, LLM, HITL, AgentDoG, y las funciones
de scrape_tracker. Sin Playwright real ni API calls.
"""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage

# Desactivar HITL y guard por defecto en todos los tests de este módulo
os.environ.setdefault("HITL_ENABLED", "false")
os.environ.setdefault("AGENTDOG_GUARD_URL", "")
os.environ.setdefault("AGENTDOG_POLICY",    "fail_open")
os.environ.setdefault("AGENTDOG_EVAL_MODE", "high_risk_only")


# ==================== HELPERS ====================

def _make_agent_result(response_text: str) -> dict:
    """Crea un resultado de agente con un AIMessage final."""
    return {
        "messages": [
            HumanMessage(content="query"),
            AIMessage(content=response_text),
        ]
    }


def _make_state(message: str = "Scrapea https://example.com") -> dict:
    return {
        "messages":      [HumanMessage(content=message)],
        "next_agent":    "web_scraping_agent",
        "risk_flag":     False,
        "blocked":       False,
        "request_id":    "test-rid-123",
        "scrape_tracker": {},
    }


# ==================== HITL disabled → agente se invoca ====================

@pytest.mark.asyncio
async def test_hitl_disabled_agente_se_invoca():
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(return_value=_make_agent_result("Información de la página " * 15))

    mock_llm_fn = MagicMock(return_value=MagicMock())

    with (
        patch("nodes.web_scraping_node._HITL_ENABLED", False),
        patch("nodes.web_scraping_node.evaluate_trajectory_safe",
              AsyncMock(return_value=(True, {"label": "safe"}))),
        patch("nodes.web_scraping_node.get_runtime_policy", return_value={}),
    ):
        from nodes.web_scraping_node import make_web_scraping_node
        node = make_web_scraping_node(mock_agent, mock_llm_fn)
        result = await node(_make_state())

    mock_agent.ainvoke.assert_called_once()
    assert "messages" in result
    assert len(result["messages"]) == 1


# ==================== HITL enabled + confirmado → agente se invoca ====================

@pytest.mark.asyncio
async def test_hitl_enabled_usuario_confirma_agente_se_invoca():
    """Con HITL habilitado y confirmación, el agente debe ser invocado al menos una vez."""
    # Usar una respuesta con suficientes palabras para evitar el auto-retry (≥50 palabras)
    long_response = "Contenido extraído correctamente de la página web solicitada " * 5
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(return_value=_make_agent_result(long_response))

    mock_llm_fn = MagicMock(return_value=MagicMock())

    with (
        patch("nodes.web_scraping_node._HITL_ENABLED", True),
        patch("nodes.web_scraping_node._ask_confirmation", AsyncMock(return_value=True)),
        patch("nodes.web_scraping_node.evaluate_trajectory_safe",
              AsyncMock(return_value=(True, {"label": "safe"}))),
        patch("nodes.web_scraping_node.get_runtime_policy", return_value={}),
    ):
        from nodes.web_scraping_node import make_web_scraping_node
        node = make_web_scraping_node(mock_agent, mock_llm_fn)
        result = await node(_make_state())

    assert mock_agent.ainvoke.call_count >= 1, "El agente debe invocarse al menos una vez"
    assert "messages" in result


# ==================== HITL enabled + rechazado → retorna cancelación ====================

@pytest.mark.asyncio
async def test_hitl_enabled_usuario_rechaza_no_invoca_agente():
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(return_value=_make_agent_result("nunca debería llegar"))

    mock_llm_fn = MagicMock()

    with (
        patch("nodes.web_scraping_node._HITL_ENABLED", True),
        patch("nodes.web_scraping_node._ask_confirmation", AsyncMock(return_value=False)),
        patch("nodes.web_scraping_node.get_runtime_policy", return_value={}),
    ):
        from nodes.web_scraping_node import make_web_scraping_node
        node = make_web_scraping_node(mock_agent, mock_llm_fn)
        result = await node(_make_state())

    # El agente NO debe invocarse
    mock_agent.ainvoke.assert_not_called()

    # Debe retornar mensaje de cancelación
    assert "messages" in result
    msg_content = result["messages"][0].content
    assert "cancelada" in msg_content.lower() or "cancelled" in msg_content.lower() or "usuario" in msg_content.lower()


# ==================== AgentDoG bloquea → mensaje de bloqueo ====================

@pytest.mark.asyncio
async def test_agentdog_bloquea_retorna_mensaje_de_bloqueo():
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(return_value=_make_agent_result("contenido potencialmente peligroso " * 10))

    mock_llm_fn = MagicMock()

    with (
        patch("nodes.web_scraping_node._HITL_ENABLED", False),
        patch("nodes.web_scraping_node.evaluate_trajectory_safe",
              AsyncMock(return_value=(False, {"label": "unsafe", "reason": "policy_block"}))),
        patch("nodes.web_scraping_node._should_evaluate_guard", return_value=True),
        patch("nodes.web_scraping_node.get_runtime_policy", return_value={}),
    ):
        from nodes.web_scraping_node import make_web_scraping_node
        node = make_web_scraping_node(mock_agent, mock_llm_fn)
        result = await node(_make_state("fetch malicious content"))

    assert "messages" in result
    content = result["messages"][0].content
    assert "seguridad" in content.lower() or "retenida" in content.lower() or "política" in content.lower()


# ==================== AgentDoG aprueba → retorna resultado ====================

@pytest.mark.asyncio
async def test_agentdog_aprueba_retorna_resultado_del_agente():
    expected_text = "Datos relevantes extraídos de la página " * 10

    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(return_value=_make_agent_result(expected_text))

    mock_llm_fn = MagicMock()

    with (
        patch("nodes.web_scraping_node._HITL_ENABLED", False),
        patch("nodes.web_scraping_node.evaluate_trajectory_safe",
              AsyncMock(return_value=(True, {"label": "safe"}))),
        patch("nodes.web_scraping_node._should_evaluate_guard", return_value=True),
        patch("nodes.web_scraping_node.get_runtime_policy", return_value={}),
    ):
        from nodes.web_scraping_node import make_web_scraping_node
        node = make_web_scraping_node(mock_agent, mock_llm_fn)
        result = await node(_make_state())

    assert "messages" in result
    # El resultado debe contener texto del agente (posiblemente resumido)
    assert isinstance(result["messages"][0], AIMessage)
    assert len(result["messages"][0].content) > 0


# ==================== Context quarantine: raw_messages no en state ====================

@pytest.mark.asyncio
async def test_context_quarantine_raw_messages_no_en_state_retornado():
    """El nodo NO debe incluir las raw_messages del sub-agente en el estado retornado.

    Solo debe retornar el resumen final en messages (1 AIMessage).
    """
    # Simular resultado del agente con múltiples mensajes (historial interno)
    agent_result = {
        "messages": [
            HumanMessage(content="query del agente"),
            AIMessage(content="", tool_calls=[{"id": "tc1", "name": "scrape", "args": {}}]),
            # ToolMessage con HTML crudo masivo — NO debe llegar al estado
            MagicMock(
                content="<html>" + "x" * 5000 + "</html>",
                tool_call_id="tc1",
                __class__=__import__("langchain_core.messages", fromlist=["ToolMessage"]).ToolMessage,
            ),
            AIMessage(content="Resumen final limpio de 50 palabras " * 3),
        ]
    }

    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(return_value=agent_result)
    mock_llm_fn = MagicMock()

    with (
        patch("nodes.web_scraping_node._HITL_ENABLED", False),
        patch("nodes.web_scraping_node.evaluate_trajectory_safe",
              AsyncMock(return_value=(True, {"label": "safe"}))),
        patch("nodes.web_scraping_node._should_evaluate_guard", return_value=True),
        patch("nodes.web_scraping_node.get_runtime_policy", return_value={}),
    ):
        from nodes.web_scraping_node import make_web_scraping_node
        node = make_web_scraping_node(mock_agent, mock_llm_fn)
        result = await node(_make_state())

    # El estado retornado debe tener EXACTAMENTE 1 mensaje (el resumen)
    # No el historial interno del sub-agente
    assert len(result["messages"]) == 1
    returned_msg = result["messages"][0]
    assert isinstance(returned_msg, AIMessage)

    # El HTML crudo NO debe estar en el mensaje retornado
    assert "<html>" not in returned_msg.content


# ==================== scrape_tracker actualizado en resultado exitoso ====================

@pytest.mark.asyncio
async def test_resultado_exitoso_actualiza_scrape_tracker():
    """El estado retornado debe incluir scrape_tracker actualizado."""
    response_text = "Información extraída correctamente de la web " * 10

    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(return_value=_make_agent_result(response_text))
    mock_llm_fn = MagicMock()

    with (
        patch("nodes.web_scraping_node._HITL_ENABLED", False),
        patch("nodes.web_scraping_node.evaluate_trajectory_safe",
              AsyncMock(return_value=(True, {"label": "safe"}))),
        patch("nodes.web_scraping_node._should_evaluate_guard", return_value=True),
        patch("nodes.web_scraping_node.get_runtime_policy", return_value={}),
    ):
        from nodes.web_scraping_node import make_web_scraping_node
        node = make_web_scraping_node(mock_agent, mock_llm_fn)
        # Usar "noticias" para evitar el fast-path de crypto/api_price
        result = await node(_make_state("últimas noticias de tecnología"))

    assert "scrape_tracker" in result
    assert isinstance(result["scrape_tracker"], dict)
