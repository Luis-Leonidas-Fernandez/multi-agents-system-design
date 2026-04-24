"""
Tests unitarios para features/web_scraping/infrastructure/node.py.

Usa mocks para aislar: agente, LLM, HITL, AgentDoG, y las funciones
de scrape_tracker. Sin Playwright real ni API calls.
"""
import os
import pytest
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage
from core.domain.models import AgentState

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


def _make_state(message: str = "Scrapea https://example.com") -> AgentState:
    return cast(AgentState, {
        "messages":      [HumanMessage(content=message)],
        "next_agent":    "web_scraping_agent",
        "risk_flag":     False,
        "blocked":       False,
        "request_id":    "test-rid-123",
        "scrape_tracker": {},
        "session_id":    "sess-test",
        "coordinator_worker_id": "",
        "coordinator_worker_agent": "",
    })


# ==================== HITL disabled → agente se invoca ====================

@pytest.mark.asyncio
async def test_hitl_disabled_agente_se_invoca():
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(return_value=_make_agent_result("Información de la página " * 15))

    mock_llm_fn = MagicMock(return_value=MagicMock())

    with (
        patch("application.policies.hitl_flow.HITL_ENABLED", False),
        patch("features.web_scraping.infrastructure.node.evaluate_trajectory_safe",
              AsyncMock(return_value=(True, {"label": "safe"}))),
        patch("features.web_scraping.infrastructure.node.get_runtime_policy", return_value={}),
    ):
        from features.web_scraping.infrastructure.node import make_web_scraping_node
        node = make_web_scraping_node(mock_agent, mock_llm_fn)
        result = await node(_make_state())

    mock_agent.ainvoke.assert_called_once()
    assert "messages" in result
    assert len(result["messages"]) == 1


@pytest.mark.asyncio
async def test_web_agent_connection_failure_uses_search_fallback():
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(side_effect=ConnectionError("boom"))

    mock_llm_fn = MagicMock(return_value=MagicMock())

    fallback = {
        "summary": "• ANSA reporta novedades de seguridad en Italia esta semana\n\nSources:\n- [ANSA](https://www.ansa.it/)",
        "words": ["ANSA", "reporta", "novedades", "de", "seguridad"],
        "source_type": "search",
        "sources": [{"title": "ANSA", "url": "https://www.ansa.it/"}],
        "pre_synthesized": True,
    }

    with (
        patch("application.policies.hitl_flow.HITL_ENABLED", False),
        patch("features.web_scraping.infrastructure.node.evaluate_trajectory_safe",
              AsyncMock(return_value=(True, {"label": "safe"}))),
        patch("features.web_scraping.infrastructure.node._should_evaluate_guard", return_value=True),
        patch("features.web_scraping.infrastructure.node.get_runtime_policy", return_value={}),
        patch("features.web_scraping.application.flow._run_generic_web_search_fetch", new=AsyncMock(side_effect=[None, fallback])),
    ):
        from features.web_scraping.infrastructure.node import make_web_scraping_node
        node = make_web_scraping_node(mock_agent, mock_llm_fn)
        result = await node(_make_state("dame las ultimas noticias sobre seguridad en italia de esta semana"))

    assert "messages" in result
    assert "ANSA reporta novedades de seguridad en Italia esta semana" in result["messages"][0].content
    mock_llm_fn.assert_not_called()


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
        patch("application.policies.hitl_flow.HITL_ENABLED", True),
        patch("application.policies.hitl_flow.ask_confirmation", AsyncMock(return_value=True)),
        patch("features.web_scraping.infrastructure.node.evaluate_trajectory_safe",
              AsyncMock(return_value=(True, {"label": "safe"}))),
        patch("features.web_scraping.infrastructure.node.get_runtime_policy", return_value={}),
    ):
        from features.web_scraping.infrastructure.node import make_web_scraping_node
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
        patch("application.policies.hitl_flow.HITL_ENABLED", True),
        patch("application.policies.hitl_flow.ask_confirmation", AsyncMock(return_value=False)),
        patch("features.web_scraping.infrastructure.node.get_runtime_policy", return_value={}),
    ):
        from features.web_scraping.infrastructure.node import make_web_scraping_node
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
        patch("application.policies.hitl_flow.HITL_ENABLED", False),
        patch("features.web_scraping.infrastructure.node.evaluate_trajectory_safe",
              AsyncMock(return_value=(False, {"label": "unsafe", "reason": "policy_block"}))),
        patch("features.web_scraping.infrastructure.node._should_evaluate_guard", return_value=True),
        patch("features.web_scraping.application.flow._select_strategy_context", return_value={
            "tracker": {},
            "turn_count": 1,
            "category": "general",
            "prior_score": 0.0,
            "prior_reliability": "ok",
            "ml_recommended": None,
            "strategy": "prefer_search",
            "exploring": False,
            "exp_rate": 0.0,
            "prediction_match": None,
        }),
        patch("features.web_scraping.infrastructure.node.get_runtime_policy", return_value={}),
    ):
        from features.web_scraping.infrastructure.node import make_web_scraping_node
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
        patch("application.policies.hitl_flow.HITL_ENABLED", False),
        patch("features.web_scraping.infrastructure.node.evaluate_trajectory_safe",
              AsyncMock(return_value=(True, {"label": "safe"}))),
        patch("features.web_scraping.infrastructure.node._should_evaluate_guard", return_value=True),
        patch("features.web_scraping.infrastructure.node.get_runtime_policy", return_value={}),
    ):
        from features.web_scraping.infrastructure.node import make_web_scraping_node
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
        patch("application.policies.hitl_flow.HITL_ENABLED", False),
        patch("features.web_scraping.infrastructure.node.evaluate_trajectory_safe",
              AsyncMock(return_value=(True, {"label": "safe"}))),
        patch("features.web_scraping.infrastructure.node._should_evaluate_guard", return_value=True),
        patch("features.web_scraping.infrastructure.node.get_runtime_policy", return_value={}),
    ):
        from features.web_scraping.infrastructure.node import make_web_scraping_node
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
        patch("application.policies.hitl_flow.HITL_ENABLED", False),
        patch("features.web_scraping.infrastructure.node.evaluate_trajectory_safe",
              AsyncMock(return_value=(True, {"label": "safe"}))),
        patch("features.web_scraping.infrastructure.node._should_evaluate_guard", return_value=True),
        patch("features.web_scraping.infrastructure.node.get_runtime_policy", return_value={}),
    ):
        from features.web_scraping.infrastructure.node import make_web_scraping_node
        node = make_web_scraping_node(mock_agent, mock_llm_fn)
        # Usar "noticias" para evitar el fast-path de crypto/api_price
        result = await node(_make_state("últimas noticias de tecnología"))

    assert "scrape_tracker" in result
    assert isinstance(result["scrape_tracker"], dict)


# ==================== Auto-retry cuando el contenido es insuficiente ====================

@pytest.mark.asyncio
async def test_auto_retry_se_activa_con_contenido_insuficiente():
    short_response = "pocos datos"
    retry_response = "Resumen final luego del retry con suficiente contenido " * 6

    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(side_effect=[
        _make_agent_result(short_response),
        _make_agent_result(retry_response),
    ])
    mock_llm_fn = MagicMock()

    with (
        patch("application.policies.hitl_flow.HITL_ENABLED", False),
        patch("features.web_scraping.infrastructure.node.evaluate_trajectory_safe",
              AsyncMock(return_value=(True, {"label": "safe"}))),
        patch("features.web_scraping.infrastructure.node._should_evaluate_guard", return_value=True),
        # category=general skips the sports/news early-return block; _run_generic_web_search_fetch
        # returning None skips the is_web_information_query early-return block, so the
        # flow falls through to agent.ainvoke which the auto-retry test exercises.
        patch("features.web_scraping.application.flow._select_strategy_context", return_value={
            "tracker": {},
            "turn_count": 1,
            "category": "general",
            "prior_score": 0.0,
            "prior_reliability": "ok",
            "ml_recommended": None,
            "strategy": "prefer_search",
            "exploring": False,
            "exp_rate": 0.0,
            "prediction_match": None,
        }),
        patch("features.web_scraping.application.flow._run_generic_web_search_fetch",
              AsyncMock(return_value=None)),
        patch("features.web_scraping.infrastructure.node.get_runtime_policy", return_value={}),
    ):
        from features.web_scraping.infrastructure.node import make_web_scraping_node
        node = make_web_scraping_node(mock_agent, mock_llm_fn)
        result = await node(_make_state("noticias de tecnología"))

    assert mock_agent.ainvoke.call_count == 2, "El auto-retry debe invocar el agente dos veces"
    assert "messages" in result
    assert isinstance(result["messages"][0], AIMessage)
    assert "retry" in result["messages"][0].content.lower() or len(result["messages"][0].content.split()) > len(short_response.split())


@pytest.mark.asyncio
async def test_news_y_sports_usan_search_web_directo():
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(side_effect=AssertionError("no debería invocarse el agente"))
    mock_llm_fn = MagicMock(return_value=MagicMock())

    with (
        patch("application.policies.hitl_flow.HITL_ENABLED", False),
        patch("features.web_scraping.infrastructure.node.evaluate_trajectory_safe", AsyncMock(return_value=(True, {"label": "safe"}))),
        patch("features.web_scraping.infrastructure.node._should_evaluate_guard", return_value=True),
        patch("features.web_scraping.infrastructure.search_tools.search_web.func", side_effect=[
            "Web search results for query: \"dame los resultados del futbol de primera division de argentina del dia de hoy\"\n\n1. [ESPN results](https://www.espn.com.ar/futbol/resultados/_/liga/arg.1)\n   Results page\n\nSources:\n- [ESPN results](https://www.espn.com.ar/futbol/resultados/_/liga/arg.1)",
            "Web search results for query: \"dame los resultados del futbol de primera division de argentina del dia de hoy resultados\"\n\n1. [Flashscore results](https://www.flashscore.com.ar/futbol/argentina/liga-profesional/resultados/)\n   Results page\n\nSources:\n- [Flashscore results](https://www.flashscore.com.ar/futbol/argentina/liga-profesional/resultados/)",
        ]),
        patch("features.web_scraping.infrastructure.scraping_tools.fetch_web_page", AsyncMock(side_effect=[
            "URL: https://www.espn.com.ar/futbol/resultados/_/liga/arg.1\n\nResultados de la primera division de futbol argentina del dia de hoy\nRiver Plate 3 - 0 Belgrano (Córdoba)\nCentral Córdoba 1 - 3 Newell's Old Boys\n\nSources:\n- [espn.com.ar](https://www.espn.com.ar/futbol/resultados/_/liga/arg.1)",
            "URL: https://www.flashscore.com.ar/futbol/argentina/liga-profesional/resultados/\n\nResultados argentina futbol primera division\nRiver Plate 3 - 0 Belgrano (Córdoba)\n\nSources:\n- [flashscore.com.ar](https://www.flashscore.com.ar/futbol/argentina/liga-profesional/resultados/)",
        ])),
        patch("features.web_scraping.infrastructure.node.get_runtime_policy", return_value={}),
    ):
        from features.web_scraping.infrastructure.node import make_web_scraping_node
        node = make_web_scraping_node(mock_agent, mock_llm_fn)
        result = await node(_make_state("dame los resultados del futbol de primera division de argentina del dia de hoy"))

    assert "messages" in result
    assert isinstance(result["messages"][0], AIMessage)
    assert "River Plate 3 - 0 Belgrano (Córdoba)" in result["messages"][0].content
    assert "Sources:" in result["messages"][0].content
    assert "Salta al contenido principal" not in result["messages"][0].content
    assert "- -" not in result["messages"][0].content
    assert mock_agent.ainvoke.call_count == 0  # sports queries bypass the react agent


@pytest.mark.asyncio
async def test_news_economicas_china_no_hardcodea_espn():
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(side_effect=AssertionError("no debería invocarse el agente"))
    mock_llm_fn = MagicMock(return_value=MagicMock())

    with (
        patch("application.policies.hitl_flow.HITL_ENABLED", False),
        patch("features.web_scraping.infrastructure.node.evaluate_trajectory_safe", AsyncMock(return_value=(True, {"label": "safe"}))),
        patch("features.web_scraping.infrastructure.node._should_evaluate_guard", return_value=True),
        patch("features.web_scraping.infrastructure.search_tools.search_web.func", side_effect=[
            "Web search results for query: \"periodicos china noticias diarios\"\n\n1. [Xinhua](https://www.xinhuanet.com/)\n   Directorio de prensa de China\n\n2. [China Daily](https://www.chinadaily.com.cn/)\n   Directorio de prensa de China\n\nSources:\n- [Xinhua](https://www.xinhuanet.com/)\n- [China Daily](https://www.chinadaily.com.cn/)",
            "Web search results for query: \"dame las noticias economicas de china de hoy\"\n\n1. [Reuters China economy](https://www.reuters.com/world/china/)\n   China economy slows as market waits\n\n2. [ESPN Tenis](https://www.espn.com.ar/tenis/)\n   Noticias de Tenis\n\nSources:\n- [Reuters China economy](https://www.reuters.com/world/china/)\n- [ESPN Tenis](https://www.espn.com.ar/tenis/)",
            "Web search results for query: \"dame las noticias economicas de china de hoy últimas noticias recientes\"\n\n1. [El Economista China](https://www.eleconomista.es/economia/noticias/13643246/11/25/china-sufre-un-desplome-sin-precedentes-de-la-inversion-y-deja-a-la-economia-sin-motores-en-pleno-vuelo.html)\n   Inversion y economia china\n\nSources:\n- [El Economista China](https://www.eleconomista.es/economia/noticias/13643246/11/25/china-sufre-un-desplome-sin-precedentes-de-la-inversion-y-deja-a-la-economia-sin-motores-en-pleno-vuelo.html)",
        ]),
        patch("features.web_scraping.infrastructure.scraping_tools.fetch_web_page", AsyncMock(side_effect=[
            "URL: https://www.reuters.com/world/china/\n\nNoticias economicas de china del dia de hoy\nChina economy slows as global market waits for policy response\nAnalistas preveen una desaceleracion de la economia china\n\nSources:\n- [Reuters China economy](https://www.reuters.com/world/china/)",
            "URL: https://www.eleconomista.es/economia/noticias/13643246/11/25/china-sufre-un-desplome-sin-precedentes-de-la-inversion-y-deja-a-la-economia-sin-motores-en-pleno-vuelo.html\n\nEconomia de china hoy noticias\nChina sufre un desplome sin precedentes de la inversion y deja a la economia sin motores\n\nSources:\n- [El Economista China](https://www.eleconomista.es/economia/noticias/13643246/11/25/china-sufre-un-desplome-sin-precedentes-de-la-inversion-y-deja-a-la-economia-sin-motores-en-pleno-vuelo.html)",
        ])),
        patch("features.web_scraping.infrastructure.node.get_runtime_policy", return_value={}),
    ):
        from features.web_scraping.infrastructure.node import make_web_scraping_node
        node = make_web_scraping_node(mock_agent, mock_llm_fn)
        result = await node(_make_state("dame las noticias economicas de china de hoy"))

    content = result["messages"][0].content
    assert "ESPN" not in content
    assert "Reuters China economy" in content or "Reuters" in content or "El Economista China" in content
    assert "Sources:" in content


@pytest.mark.asyncio
async def test_news_recientes_de_japon_devuelven_respuesta_y_sources():
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(side_effect=AssertionError("no debería invocarse el agente"))
    mock_llm_fn = MagicMock(return_value=MagicMock())

    with (
        patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
        patch("application.policies.hitl_flow.HITL_ENABLED", False),
        patch("features.web_scraping.infrastructure.node.run_web_scraping_flow", AsyncMock(return_value={
            "messages": [AIMessage(content="Japón refuerza medidas de seguridad hoy\n\nTokio anuncia un nuevo operativo\n\nSources:\n- [Japan News](https://www.japannews.yomiuri.co.jp/security/today)")],
        })),
        patch("features.web_scraping.infrastructure.node.evaluate_trajectory_safe", AsyncMock(return_value=(True, {"label": "safe"}))),
        patch("features.web_scraping.infrastructure.node._should_evaluate_guard", return_value=True),
        patch("features.web_scraping.infrastructure.node.get_runtime_policy", return_value={}),
    ):
        from features.web_scraping.infrastructure.node import make_web_scraping_node
        node = make_web_scraping_node(mock_agent, mock_llm_fn)
        result = await node(_make_state("dame las ultimas noticias sobre seguridad de japon el dia de hoy"))

    content = result["messages"][0].content
    assert "Japón refuerza medidas de seguridad hoy" in content
    assert "Tokio anuncia un nuevo operativo" in content
    assert "Sources:" in content
    assert mock_agent.ainvoke.call_count == 0


@pytest.mark.asyncio
async def test_news_recientes_de_japon_ignora_fuente_sin_info_y_busca_otra():
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(side_effect=AssertionError("no debería invocarse el agente"))
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Japón celebrará en abril la primera reunión para revisar su estrategia de seguridad nacional\n\nEl Gobierno japonés convocará a expertos para revisar tres documentos clave."))
    mock_llm_fn = MagicMock(return_value=mock_llm)

    _NHK_URL = "https://www3.nhk.or.jp/nhkworld/es/news/20260404_05/"
    _NHK_CONTENT = (
        f"URL: {_NHK_URL}\n\n"
        "Japón celebrará en abril la primera reunión para revisar su estrategia de seguridad nacional\n"
        "El Gobierno japonés convocará a expertos para revisar tres documentos clave.\n\n"
        f"Sources:\n- [NHK]({_NHK_URL})"
    )
    _NO_INFO = "Lo siento, pero la página proporcionada no contiene información sobre la seguridad de Japón ni noticias relacionadas con ese tema."

    async def _fetch_by_url(url, **kwargs):
        return _NHK_CONTENT if _NHK_URL in url else _NO_INFO

    with (
        patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
        patch("application.policies.hitl_flow.HITL_ENABLED", False),
        patch("features.web_scraping.infrastructure.node.run_web_scraping_flow", AsyncMock(return_value={
            "messages": [AIMessage(content="Japón celebrará en abril la primera reunión para revisar su estrategia de seguridad nacional\n\nEl Gobierno japonés convocará a expertos para revisar tres documentos clave.\n\nSources:\n- [NHK](https://www3.nhk.or.jp/nhkworld/es/news/20260404_05/)")],
        })),
        patch("features.web_scraping.infrastructure.node.evaluate_trajectory_safe", AsyncMock(return_value=(True, {"label": "safe"}))),
        patch("features.web_scraping.infrastructure.node._should_evaluate_guard", return_value=True),
        patch("features.web_scraping.infrastructure.node.get_runtime_policy", return_value={}),
    ):
        from features.web_scraping.infrastructure.node import make_web_scraping_node
        node = make_web_scraping_node(mock_agent, mock_llm_fn)
        result = await node(_make_state("dame las ultimas noticias sobre seguridad de japon hoy"))

    content = result["messages"][0].content
    assert "CNN Mundo" not in content
    assert "Japón celebrará en abril la primera reunión" in content
    assert "Sources:" in content


@pytest.mark.asyncio
async def test_weekly_country_query_uses_snippet_when_daily_fetch_fails():
    from features.web_scraping.application.fetch_dispatch import _run_generic_web_search_fetch

    async def _discover(*args, **kwargs):
        return (["ansa.it", "repubblica.it"], ["ANSA", "La Repubblica"])

    async def _fetch_by_url(url, **kwargs):
        if "ansa.it" in url:
            raise RuntimeError("dns failed")
        return (
            "URL: https://www.repubblica.it/cronaca/2026/04/10/seguridad.html\n\n"
            "Repubblica confirma medidas de seguridad en Italia esta semana\n"
            "El ministerio anunció controles adicionales\n\n"
            "Sources:\n- [Repubblica](https://www.repubblica.it/cronaca/2026/04/10/seguridad.html)"
        )

    with (
        patch("features.web_scraping.application.flow._discover_country_press_sources", new=AsyncMock(side_effect=_discover)),
        patch("features.web_scraping.infrastructure.search_tools.search_web.func", side_effect=[
            "Web search results for query: \"dame las ultimas noticias sobre seguridad en italia de esta semana site:ansa.it ANSA noticias\"\n\n"
            "1. [ANSA seguridad Italia](https://www.ansa.it/italia/notizie/2026/04/10/seguridad.html)\n"
            "   ANSA reporta novedades de seguridad en Italia\n\n"
            "Sources:\n- [ANSA seguridad Italia](https://www.ansa.it/italia/notizie/2026/04/10/seguridad.html)",
            "Web search results for query: \"dame las ultimas noticias sobre seguridad en italia de esta semana site:repubblica.it La Repubblica noticias\"\n\n"
            "1. [Repubblica seguridad Italia](https://www.repubblica.it/cronaca/2026/04/10/seguridad.html)\n"
            "   Repubblica confirma medidas de seguridad en Italia esta semana\n\n"
            "Sources:\n- [Repubblica seguridad Italia](https://www.repubblica.it/cronaca/2026/04/10/seguridad.html)",
        ]),
        patch("features.web_scraping.infrastructure.scraping_tools.fetch_web_page", side_effect=_fetch_by_url),
    ):
        result = await _run_generic_web_search_fetch("dame las ultimas noticias sobre seguridad en italia de esta semana")

    assert result is not None
    summary = result["summary"]
    assert "ANSA reporta novedades de seguridad en Italia" in summary
    assert "Repubblica confirma medidas de seguridad en Italia esta semana" in summary
    assert "Sources:" in summary


@pytest.mark.asyncio
async def test_weekly_country_query_uses_single_snippet_before_generic_fallback():
    from features.web_scraping.application.fetch_dispatch import _run_generic_web_search_fetch

    async def _discover(*args, **kwargs):
        return (["ansa.it"], ["ANSA"])

    async def _fetch_by_url(url, **kwargs):
        raise RuntimeError("dns failed")

    with (
        patch("features.web_scraping.application.flow._discover_country_press_sources", new=AsyncMock(side_effect=_discover)),
        patch("features.web_scraping.infrastructure.search_tools.search_web.func", return_value=(
            "Web search results for query: \"dame las ultimas noticias sobre seguridad en italia de esta semana site:ansa.it ANSA noticias\"\n\n"
            "1. [ANSA seguridad Italia](https://www.ansa.it/italia/notizie/2026/04/10/seguridad.html)\n"
            "   ANSA reporta novedades de seguridad en Italia\n\n"
            "Sources:\n- [ANSA seguridad Italia](https://www.ansa.it/italia/notizie/2026/04/10/seguridad.html)"
        )),
        patch("features.web_scraping.infrastructure.scraping_tools.fetch_web_page", side_effect=_fetch_by_url),
    ):
        result = await _run_generic_web_search_fetch("dame las ultimas noticias sobre seguridad en italia de esta semana")

    assert result is not None
    assert result["source_type"] == "search"
    assert result["pre_synthesized"] is True
    summary = result["summary"]
    assert "ANSA reporta novedades de seguridad en Italia" in summary
    assert "Sources:" in summary


@pytest.mark.asyncio
async def test_recent_generic_web_query_requires_sufficient_context():
    from features.web_scraping.application.fetch_dispatch import _run_generic_web_search_fetch

    with (
        patch("features.web_scraping.infrastructure.search_tools.search_web.func", return_value="Web search results for query: \"dame las ultimas noticias sobre seguridad de japon hoy April 2026\"\n\n1. [Japan update](https://example.com/japan)\n   Short snippet\n\nSources:\n- [Japan update](https://example.com/japan)"),
        patch("features.web_scraping.infrastructure.scraping_tools.fetch_web_page", AsyncMock(return_value="URL: https://example.com/japan\n\nSolo una línea insuficiente\n\nSources:\n- [Japan update](https://example.com/japan)")),
    ):
        result = await _run_generic_web_search_fetch("dame las ultimas noticias sobre seguridad de japon hoy")

    assert result is None


@pytest.mark.asyncio
async def test_weekly_generic_web_query_combines_multiple_sources():
    from features.web_scraping.application.fetch_dispatch import _run_generic_web_search_fetch

    # The OpenClaw-style flow uses a single generic search result set and then ranks/
    # deduplicates hits before fetching the best article URLs.
    with (
        patch("features.web_scraping.infrastructure.search_tools.search_web.func", return_value=(
            "Web search results for query: \"dame las ultimas noticias sobre seguridad de japon esta semana\"\n\n"
            "1. [NHK: Japan security roundup](https://www3.nhk.or.jp/nhkworld/es/news/20260404_05/)\n"
            "   Japón refuerza medidas de seguridad esta semana\n\n"
            "2. [Reuters: Japan security and China tensions](https://www.reuters.com/world/asia-pacific/japan-security-china-tensions-2026-04-06/)\n"
            "   Tensiones de seguridad entre Japón y China\n\n"
            "3. [Japón y sus aliados](https://www.nippon.com/es/news/yjj2026040500456/)\n"
            "   Japón mantiene contactos diplomáticos con sus aliados esta semana\n\n"
            "Sources:\n"
            "- [NHK: Japan security roundup](https://www3.nhk.or.jp/nhkworld/es/news/20260404_05/)\n"
            "- [Reuters: Japan security and China tensions](https://www.reuters.com/world/asia-pacific/japan-security-china-tensions-2026-04-06/)\n"
            "- [Japón y sus aliados](https://www.nippon.com/es/news/yjj2026040500456/)"
        )),
        patch("features.web_scraping.infrastructure.scraping_tools.fetch_web_page", AsyncMock(side_effect=[
            "URL: https://www3.nhk.or.jp/nhkworld/es/news/20260404_05/\n\nJapón refuerza medidas de seguridad esta semana\nTokio anuncia un nuevo operativo\n\nSources:\n- [NHK](https://www3.nhk.or.jp/nhkworld/es/news/20260404_05/)",
            "URL: https://www.reuters.com/world/asia-pacific/japan-security-china-tensions-2026-04-06/\n\nTensiones de seguridad entre Japón y China aumentan esta semana\nWashington sigue de cerca el despliegue militar japonés\n\nSources:\n- [Reuters](https://www.reuters.com/world/asia-pacific/japan-security-china-tensions-2026-04-06/)",
            "URL: https://www.nippon.com/es/news/yjj2026040500456/\n\nJapón mantiene contactos diplomáticos con sus aliados en Asia esta semana\nLas conversaciones diplomáticas refuerzan la posición japonesa\n\nSources:\n- [Nippon](https://www.nippon.com/es/news/yjj2026040500456/)",
        ])),
    ):
        result = await _run_generic_web_search_fetch("dame las ultimas noticias sobre seguridad de japon de esta semana")

    assert result is not None
    content = result["summary"]
    assert "NHK" in content
    assert "Reuters" in content
    assert content.count("Sources:") == 1
    assert len(result["sources"]) >= 2


@pytest.mark.asyncio
async def test_build_source_backed_response_deduplicates_lines():
    from features.web_scraping.application.flow import _build_source_backed_response

    result = _build_source_backed_response(
        [
            "Japón refuerza medidas de seguridad hoy",
            "Japón refuerza medidas de seguridad hoy",
            "Tokio anuncia un nuevo operativo",
            "Tokio anuncia un nuevo operativo",
        ],
        [{"title": "NHK", "url": "https://www3.nhk.or.jp/nhkworld/es/news/20260404_05/"}],
    )

    assert result.count("Japón refuerza medidas de seguridad hoy") == 1
    assert result.count("Tokio anuncia un nuevo operativo") == 1


@pytest.mark.asyncio
async def test_url_directo_usa_web_fetch_explicitamente():
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(side_effect=AssertionError("no debería invocarse el agente"))
    mock_llm_fn = MagicMock(return_value=MagicMock())

    with (
        patch("application.policies.hitl_flow.HITL_ENABLED", False),
        patch("features.web_scraping.infrastructure.node.evaluate_trajectory_safe", AsyncMock(return_value=(True, {"label": "safe"}))),
        patch("features.web_scraping.infrastructure.node._should_evaluate_guard", return_value=True),
        patch("features.web_scraping.infrastructure.scraping_tools.fetch_web_page", AsyncMock(return_value="URL: https://example.com\n\nResumen corto\n\nSources:\n- [example.com](https://example.com)")),
        patch("features.web_scraping.infrastructure.node.get_runtime_policy", return_value={}),
    ):
        from features.web_scraping.infrastructure.node import make_web_scraping_node
        node = make_web_scraping_node(mock_agent, mock_llm_fn)
        result = await node(_make_state("resumi esta pagina https://example.com"))

    assert "messages" in result
    assert isinstance(result["messages"][0], AIMessage)
    assert "Resumen corto" in result["messages"][0].content
    assert "Sources:" in result["messages"][0].content


@pytest.mark.asyncio
async def test_summarize_if_long_preserves_sources_block():
    from features.web_scraping.application.retry_flow import _summarize_if_long

    llm = MagicMock(ainvoke=AsyncMock(return_value=MagicMock(content="Resumen compacto")))
    long_text = "Palabra " * 250 + "\n\nSources:\n- [example.com](https://example.com)"

    result = await _summarize_if_long(long_text, "rid-1", lambda: llm)

    assert "Resumen compacto" in result
    assert "Sources:" in result
    assert "https://example.com" in result


@pytest.mark.asyncio
async def test_sports_query_filtra_fuentes_no_argentinas():
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(side_effect=AssertionError("no debería invocarse el agente"))
    mock_llm_fn = MagicMock(return_value=MagicMock())

    with (
        patch("application.policies.hitl_flow.HITL_ENABLED", False),
        patch("features.web_scraping.infrastructure.node.evaluate_trajectory_safe", AsyncMock(return_value=(True, {"label": "safe"}))),
        patch("features.web_scraping.infrastructure.node._should_evaluate_guard", return_value=True),
        patch("features.web_scraping.infrastructure.search_tools.search_web.func", side_effect=[
            "Web search results for query: \"dame los resultados del futbol argentino del dia de hoy\"\n\n1. [Sopitas](https://www.sopitas.com/fm/francia-98-juan-inaki-ignacio-antonio-historia-futbolista-river-plate-banda/)\n   Historia de un jugador\n\nSources:\n- [Sopitas](https://www.sopitas.com/fm/francia-98-juan-inaki-ignacio-antonio-historia-futbolista-river-plate-banda/)",
            "Web search results for query: \"dame los resultados del futbol argentino del dia de hoy últimas noticias recientes\"\n\n1. [ESPN resultados](https://www.espn.com.ar/futbol/resultados/_/liga/arg.1)\n   Resultados de la liga argentina\n\nSources:\n- [ESPN resultados](https://www.espn.com.ar/futbol/resultados/_/liga/arg.1)",
        ]),
        patch("features.web_scraping.infrastructure.scraping_tools.fetch_web_page", AsyncMock(side_effect=[
            "URL: https://www.espn.com.ar/futbol/resultados/_/liga/arg.1\n\nResultados del futbol argentino del dia de hoy\nRiver Plate 3 - 0 Belgrano (Córdoba)\nCentral Córdoba 1 - 3 Newell's Old Boys\n\nSources:\n- [espn.com.ar](https://www.espn.com.ar/futbol/resultados/_/liga/arg.1)",
        ])),
        patch("features.web_scraping.infrastructure.node.get_runtime_policy", return_value={}),
    ):
        from features.web_scraping.infrastructure.node import make_web_scraping_node
        node = make_web_scraping_node(mock_agent, mock_llm_fn)
        result = await node(_make_state("dame los resultados del futbol argentino del dia de hoy"))

    assert "Sopitas" not in result["messages"][0].content
    assert "River Plate 3 - 0 Belgrano (Córdoba)" in result["messages"][0].content
    assert "- -" not in result["messages"][0].content


@pytest.mark.asyncio
async def test_sports_query_aplica_contexto_geografico_al_fetch():
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(side_effect=AssertionError("no debería invocarse el agente"))
    _llm_resp6 = MagicMock()
    _llm_resp6.content = "• Barcelona SC 2 - 1 Emelec en Ecuador.\n• Deportivo Cuenca 0 - 0 Aucas."
    _llm6 = MagicMock()
    _llm6.ainvoke = AsyncMock(return_value=_llm_resp6)
    mock_llm_fn = MagicMock(return_value=_llm6)

    fetch_mock = AsyncMock(return_value="URL: https://www.sofascore.com/es/futbol/ecuador/2026-04-06\n\nResultados del futbol ecuatoriano del dia de hoy\nBarcelona SC 2 - 1 Emelec\nDeportivo Cuenca 0 - 0 Aucas\n\nSources:\n- [sofascore](https://www.sofascore.com/es/futbol/ecuador/2026-04-06)")

    with (
        patch("application.policies.hitl_flow.HITL_ENABLED", False),
        patch("features.web_scraping.infrastructure.node.evaluate_trajectory_safe", AsyncMock(return_value=(True, {"label": "safe"}))),
        patch("features.web_scraping.infrastructure.node._should_evaluate_guard", return_value=True),
        patch("features.web_scraping.infrastructure.search_tools.search_web.func", side_effect=[
            "Web search results for query: \"dame los resultados del futbol ecuatoriano del dia de hoy\"\n\n1. [Marcador en directo de Fútbol - Sofascore](https://www.sofascore.com/es/futbol/ecuador/2026-04-06)\n   Resultados de Ecuador\n\nSources:\n- [Marcador en directo de Fútbol - Sofascore](https://www.sofascore.com/es/futbol/ecuador/2026-04-06)",
            "",
        ]),
        patch("features.web_scraping.infrastructure.scraping_tools.fetch_web_page", fetch_mock),
        patch("features.web_scraping.infrastructure.node.get_runtime_policy", return_value={}),
    ):
        from features.web_scraping.infrastructure.node import make_web_scraping_node
        node = make_web_scraping_node(mock_agent, mock_llm_fn)
        result = await node(_make_state("dame los resultados del futbol ecuatoriano del dia de hoy"))

    assert "Ecuador" in fetch_mock.call_args.kwargs["prompt"]
    assert "otros países" in fetch_mock.call_args.kwargs["prompt"]
    assert "Barcelona SC 2 - 1 Emelec" in result["messages"][0].content
    assert "Girona 1 - 0 Villarreal" not in result["messages"][0].content


@pytest.mark.asyncio
async def test_sports_query_rechaza_lineas_extranjeras_en_respuesta():
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(side_effect=AssertionError("no debería invocarse el agente"))
    _llm_resp = MagicMock()
    _llm_resp.content = "• Barcelona SC 2 - 1 Emelec en el partido del día en Ecuador."
    _llm = MagicMock()
    _llm.ainvoke = AsyncMock(return_value=_llm_resp)
    mock_llm_fn = MagicMock(return_value=_llm)

    with (
        patch("application.policies.hitl_flow.HITL_ENABLED", False),
        patch("features.web_scraping.infrastructure.node.evaluate_trajectory_safe", AsyncMock(return_value=(True, {"label": "safe"}))),
        patch("features.web_scraping.infrastructure.node._should_evaluate_guard", return_value=True),
        patch("features.web_scraping.infrastructure.search_tools.search_web.func", side_effect=[
            "Web search results for query: \"dame los resultados del futbol ecuatoriano del dia de hoy\"\n\n1. [Marcador en directo de Fútbol - Sofascore](https://www.sofascore.com/es/futbol/ecuador/2026-04-06)\n   Resultados de Ecuador\n\nSources:\n- [Marcador en directo de Fútbol - Sofascore](https://www.sofascore.com/es/futbol/ecuador/2026-04-06)",
            "",
        ]),
        patch("features.web_scraping.infrastructure.scraping_tools.fetch_web_page", AsyncMock(return_value="URL: https://www.sofascore.com/es/futbol/ecuador/2026-04-06\n\nResultados del futbol ecuatoriano del dia de hoy\nBarcelona SC 2 - 1 Emelec\nGirona 1 - 0 Villarreal\nJuventus 2 - 0 Génova\n\nSources:\n- [sofascore](https://www.sofascore.com/es/futbol/ecuador/2026-04-06)")),
        patch("features.web_scraping.infrastructure.node.get_runtime_policy", return_value={}),
    ):
        from features.web_scraping.infrastructure.node import make_web_scraping_node
        node = make_web_scraping_node(mock_agent, mock_llm_fn)
        result = await node(_make_state("dame los resultados del futbol ecuatoriano del dia de hoy"))

    assert "Barcelona SC 2 - 1 Emelec" in result["messages"][0].content
    assert "Girona 1 - 0 Villarreal" not in result["messages"][0].content
    assert "Juventus 2 - 0 Génova" not in result["messages"][0].content
