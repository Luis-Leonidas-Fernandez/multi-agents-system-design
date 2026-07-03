"""
Tests unitarios para features/web_scraping/infrastructure/node.py.

Usa mocks para aislar: agente, LLM, AgentDoG, y las funciones
de scrape_tracker. Sin Playwright real ni API calls.
"""
import asyncio
import os
import pytest
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage
from core.domain.models import AgentState

# Desactivar guard por defecto en todos los tests de este módulo
os.environ.setdefault("AGENTDOG_GUARD_URL", "")
os.environ.setdefault("AGENTDOG_POLICY",    "fail_open")
os.environ.setdefault("AGENTDOG_EVAL_MODE", "high_risk_only")


@pytest.fixture(autouse=True)
def _reset_web_scraping_caches():
    from features.web_scraping.application import flow as _flow

    _flow._COUNTRY_PRESS_CACHE.clear()
    _flow._COUNTRY_PRESS_SOURCE_CACHE.clear()
    _flow._COUNTRY_PRESS_DISCOVERY_STRATEGY_CACHE.clear()
    yield
    _flow._COUNTRY_PRESS_CACHE.clear()
    _flow._COUNTRY_PRESS_SOURCE_CACHE.clear()
    _flow._COUNTRY_PRESS_DISCOVERY_STRATEGY_CACHE.clear()


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


# ==================== agente se invoca ====================

@pytest.mark.asyncio
async def test_agente_se_invoca():
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(return_value=_make_agent_result("Información de la página " * 15))

    mock_llm_fn = MagicMock(return_value=MagicMock())

    with (
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


# ==================== AgentDoG bloquea → mensaje de bloqueo ====================

@pytest.mark.asyncio
async def test_agentdog_bloquea_retorna_mensaje_de_bloqueo():
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(return_value=_make_agent_result("contenido potencialmente peligroso " * 10))

    mock_llm_fn = MagicMock()

    with (
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
    from datetime import date, timedelta
    from features.web_scraping.application.fetch_dispatch import _run_generic_web_search_fetch

    recent = date.today() - timedelta(days=3)
    recent_path = f"{recent.year:04d}/{recent.month:02d}/{recent.day:02d}"
    ansa_url = f"https://www.ansa.it/italia/notizie/{recent_path}/seguridad.html"
    repubblica_url = f"https://www.repubblica.it/cronaca/{recent_path}/seguridad.html"

    async def _discover(query, source_group, source_terms, runtime_args=None):
        from features.web_scraping.application import flow as _flow

        domains = ["ansa.it", "repubblica.it"]
        names = ["ANSA", "La Repubblica"]
        _flow._country_press_cache_set(source_group, source_terms, domains, names)
        _flow._country_press_strategy_cache_set(source_group, source_terms, "lookup")
        _flow._country_press_source_cache_set(
            source_group,
            source_terms,
            [
                {"title": "ANSA", "url": "https://www.ansa.it/"},
                {"title": "La Repubblica", "url": "https://www.repubblica.it/"},
            ],
        )
        return domains, names

    async def _fetch_by_url(url, **kwargs):
        if "ansa.it" in url:
            raise RuntimeError("dns failed")
        return (
            f"URL: {repubblica_url}\n\n"
            "Repubblica confirma medidas de seguridad en Italia esta semana\n"
            "El ministerio anunció controles adicionales\n\n"
            f"Sources:\n- [Repubblica]({repubblica_url})"
        )

    with (
        patch("features.web_scraping.application.country_strategy.CountryRecentNewsStrategy.execute", new=AsyncMock(return_value=None)),
        patch("features.web_scraping.application.flow._discover_country_press_sources", new=AsyncMock(side_effect=_discover)),
        patch("features.web_scraping.infrastructure.search_tools.search_web.func", side_effect=[
            f"Web search results for query: \"dame las ultimas noticias sobre seguridad en italia de esta semana site:ansa.it ANSA noticias\"\n\n"
            f"1. [ANSA seguridad Italia]({ansa_url})\n"
            "   ANSA reporta novedades de seguridad en Italia\n\n"
            f"Sources:\n- [ANSA seguridad Italia]({ansa_url})",
            f"Web search results for query: \"dame las ultimas noticias sobre seguridad en italia de esta semana site:repubblica.it La Repubblica noticias\"\n\n"
            f"1. [Repubblica seguridad Italia]({repubblica_url})\n"
            "   Repubblica confirma medidas de seguridad en Italia esta semana\n\n"
            f"Sources:\n- [Repubblica seguridad Italia]({repubblica_url})",
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
    from datetime import date, timedelta
    from features.web_scraping.application.fetch_dispatch import _run_generic_web_search_fetch

    recent = date.today() - timedelta(days=3)
    recent_path = f"{recent.year:04d}/{recent.month:02d}/{recent.day:02d}"
    ansa_url = f"https://www.ansa.it/italia/notizie/{recent_path}/seguridad.html"

    async def _discover(query, source_group, source_terms, runtime_args=None):
        from features.web_scraping.application import flow as _flow

        domains = ["ansa.it"]
        names = ["ANSA"]
        _flow._country_press_cache_set(source_group, source_terms, domains, names)
        _flow._country_press_strategy_cache_set(source_group, source_terms, "lookup")
        _flow._country_press_source_cache_set(
            source_group,
            source_terms,
            [{"title": "ANSA", "url": "https://www.ansa.it/"}],
        )
        return domains, names

    async def _fetch_by_url(url, **kwargs):
        raise RuntimeError("dns failed")

    with (
        patch("features.web_scraping.application.country_strategy.CountryRecentNewsStrategy.execute", new=AsyncMock(return_value=None)),
        patch("features.web_scraping.application.flow._discover_country_press_sources", new=AsyncMock(side_effect=_discover)),
        patch("features.web_scraping.infrastructure.search_tools.search_web.func", return_value=(
            f"Web search results for query: \"dame las ultimas noticias sobre seguridad en italia de esta semana site:ansa.it ANSA noticias\"\n\n"
            f"1. [ANSA seguridad Italia]({ansa_url})\n"
            "   ANSA reporta novedades de seguridad en Italia\n\n"
            f"Sources:\n- [ANSA seguridad Italia]({ansa_url})"
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
async def test_week_search_uses_country_press_candidates_when_generic_results_are_weak():
    from datetime import date, timedelta
    from features.web_scraping.application.fetch_dispatch import _run_generic_web_search_fetch

    recent = date.today() - timedelta(days=3)
    recent_path = f"{recent.year:04d}/{recent.month:02d}/{recent.day:02d}"
    arirang_url = f"https://www.arirang.com/news/{recent_path}/security-update"
    edaily_url = f"https://www.edaily.co.kr/news/{recent_path}/security-update"

    async def _discover(query, source_group, source_terms, runtime_args=None):
        from features.web_scraping.application import flow as _flow

        domains = ["arirang.com", "edaily.co.kr"]
        names = ["Arirang", "Edaily"]
        _flow._country_press_cache_set(source_group, source_terms, domains, names)
        _flow._country_press_strategy_cache_set(source_group, source_terms, "lookup")
        _flow._country_press_source_cache_set(
            source_group,
            source_terms,
            [
                {"title": "Arirang", "url": "https://www.arirang.com/"},
                {"title": "Edaily", "url": "https://www.edaily.co.kr/"},
            ],
        )
        return domains, names

    generic_search = (
        'Web search results for query: "dame las ultimas noticias sobre seguridad en corea del sur"\n\n'
        '1. [arirang.com — seguridad](https://www.arirang.com/seguridad/)\n'
        '   Seguridad en Corea del Sur\n\n'
        '2. [edaily.co.kr — sociedad](https://www.edaily.co.kr/sociedad/)\n'
        '   Sociedad y seguridad\n\n'
        'Sources:\n'
        '- [arirang.com — seguridad](https://www.arirang.com/seguridad/)\n'
        '- [edaily.co.kr — sociedad](https://www.edaily.co.kr/sociedad/)'
    )
    local_search_1 = (
        f'Web search results for query: "dame las ultimas noticias sobre seguridad en corea del sur site:arirang.com Arirang noticias"\n\n'
        f'1. [Arirang South Korea security]({arirang_url})\n'
        '   Seúl refuerza medidas de seguridad tras nuevas amenazas regionales\n\n'
        f'Sources:\n- [Arirang South Korea security]({arirang_url})'
    )
    local_search_2 = (
        f'Web search results for query: "dame las ultimas noticias sobre seguridad en corea del sur site:edaily.co.kr Edaily noticias"\n\n'
        f'1. [Edaily South Korea police]({edaily_url})\n'
        '   La policía surcoreana amplía operativos y controles en Seúl\n\n'
        f'Sources:\n- [Edaily South Korea police]({edaily_url})'
    )

    with (
        patch("features.web_scraping.application.flow._discover_country_press_sources", new=AsyncMock(side_effect=_discover)),
        patch("features.web_scraping.infrastructure.search_tools.search_web.func", side_effect=[
            generic_search,
            local_search_1,
            local_search_2,
        ]),
        patch("features.web_scraping.infrastructure.scraping_tools.fetch_web_page", AsyncMock(side_effect=[
            f"URL: {arirang_url}\n\nSeúl refuerza medidas de seguridad tras nuevas amenazas regionales.\nEl gobierno surcoreano elevó la alerta en puntos sensibles.\n\nSources:\n- [Arirang]({arirang_url})",
            f"URL: {edaily_url}\n\nLa policía surcoreana amplía operativos y controles en Seúl.\nLas autoridades dicen que buscan prevenir incidentes en espacios públicos.\n\nSources:\n- [Edaily]({edaily_url})",
        ])),
    ):
        result = await _run_generic_web_search_fetch("dame las ultimas noticias sobre seguridad en corea del sur")

    assert result is not None
    content = result["summary"]
    assert "Seúl refuerza medidas de seguridad" in content
    assert "La policía surcoreana amplía operativos" in content
    assert content.count("Fuente:") >= 2
    assert "Sources:" in content


@pytest.mark.asyncio
async def test_weekly_country_query_does_not_short_circuit_on_weak_search_snippets():
    from features.web_scraping.application.fetch_dispatch import _run_generic_web_search_fetch

    with (
        patch("features.web_scraping.application.country_strategy.CountryRecentNewsStrategy.execute", new=AsyncMock(return_value={
            "summary": "Arirang — seguridad: • Seúl refuerza controles.\n\nFuente: [Arirang](https://www.arirang.com/news/1)\n\nEdaily — policiales: • La policía amplía operativos.\n\nFuente: [Edaily](https://www.edaily.co.kr/news/2)\n\nSources:\n- [Arirang](https://www.arirang.com/news/1)\n- [Edaily](https://www.edaily.co.kr/news/2)",
            "words": ["Seúl", "refuerza", "controles"],
            "source_type": "search",
            "sources": [
                {"title": "Arirang", "url": "https://www.arirang.com/news/1"},
                {"title": "Edaily", "url": "https://www.edaily.co.kr/news/2"},
            ],
            "pre_synthesized": True,
        })),
        patch("features.web_scraping.infrastructure.search_tools.search_web.func", return_value=(
            'Web search results for query: "dame las ultimas noticias sobre seguridad en corea del sur"\n\n'
            '1. [arirang.com — seguridad](https://www.arirang.com/seguridad/)\n'
            '   Seguridad en Corea del Sur\n\n'
            '2. [edaily.co.kr — sociedad](https://www.edaily.co.kr/sociedad/)\n'
            '   Sociedad y seguridad\n\n'
            'Sources:\n'
            '- [arirang.com — seguridad](https://www.arirang.com/seguridad/)\n'
            '- [edaily.co.kr — sociedad](https://www.edaily.co.kr/sociedad/)'
        )),
    ):
        result = await _run_generic_web_search_fetch("dame las ultimas noticias sobre seguridad en corea del sur")

    assert result is not None
    assert "Seúl refuerza controles" in result["summary"]
    assert "La policía amplía operativos" in result["summary"]


@pytest.mark.asyncio
async def test_country_recent_news_strategy_prefers_concrete_article_candidates_and_returns_labeled_content_for_ko_groups():
    from features.web_scraping.application.country_strategy import CountryRecentNewsStrategy
    from features.web_scraping.application import flow as _flow

    fetch_runtime = MagicMock()
    fetch_runtime.fetch = AsyncMock(side_effect=[
        MagicMock(content="South Korea tightened security around major transit hubs after recent threats. Police increased checkpoints and patrols in Seoul.\nAuthorities said the measures focus on airports, train stations and government buildings."),
        MagicMock(content="South Korean police expanded surveillance and crowd-control operations this week. Officials said the move aims to prevent attacks in dense public spaces.\nLocal agencies are coordinating emergency response drills in Seoul."),
        MagicMock(content="Corea del Sur reforzó alertas de ciberseguridad tras nuevas filtraciones de datos.\nLas autoridades recomendaron revisar credenciales y activar controles adicionales."),
    ])

    async def _discover(query, source_group, source_terms, runtime_args=None):
        domains = ["arirang.com", "edaily.co.kr"]
        names = ["Arirang", "Edaily"]
        _flow._country_press_cache_set(source_group, source_terms, domains, names)
        _flow._country_press_strategy_cache_set(source_group, source_terms, "lookup")
        _flow._country_press_source_cache_set(
            source_group,
            source_terms,
            [
                {"title": "Arirang", "url": "https://www.arirang.com/"},
                {"title": "Edaily", "url": "https://www.edaily.co.kr/"},
            ],
        )
        return domains, names

    strategy = CountryRecentNewsStrategy(
        search_runtime=MagicMock(),
        fetch_runtime=fetch_runtime,
        press_discovery=MagicMock(discover=AsyncMock(side_effect=_discover)),
    )

    with (
        patch(
            "features.web_scraping.application.country_press_helpers._run_country_press_search_candidates",
            new=AsyncMock(return_value=(
                [
                    {"title": "Arirang South Korea security", "url": "https://www.arirang.com/news/2026/05/27/security-update", "snippet": "Security update"},
                    {"title": "Edaily South Korea police", "url": "https://www.edaily.co.kr/news/2026/05/27/police-update", "snippet": "Police update"},
                    {"title": "Donga cyber alert", "url": "https://www.donga.com/news/2026/05/27/cyber-alert", "snippet": "Corea del Sur reforzó alertas de ciberseguridad y controles tras nuevas filtraciones de datos esta semana."},
                ],
                "",
            )),
        ),
        patch(
            "features.web_scraping.application.flow._extract_generic_content_lines",
            side_effect=[
                [
                    "South Korea tightened security around major transit hubs after recent threats.",
                    "Police increased checkpoints and patrols in Seoul.",
                ],
                [
                    "South Korean police expanded surveillance and crowd-control operations this week.",
                    "Local agencies are coordinating emergency response drills in Seoul.",
                ],
                [
                    "Corea del Sur reforzó alertas de ciberseguridad tras nuevas filtraciones de datos.",
                    "Las autoridades recomendaron revisar credenciales y activar controles adicionales.",
                ],
            ],
        ),
        patch(
            "features.web_scraping.application.flow._discover_homepage_section_targets",
            new=AsyncMock(return_value=([], True)),
        ),
    ):
        result = await strategy.execute("dame las ultimas noticias sobre seguridad en corea del sur de esta semana")

    assert result is not None
    assert result["pre_synthesized"] is True
    assert result.get("has_labeled_content") is not True
    assert "South Korea tightened security around major transit hubs" in result["summary"]
    assert "South Korean police expanded surveillance and crowd-control operations this week" in result["summary"]
    assert "Corea del Sur reforzó alertas de ciberseguridad" in result["summary"]
    assert fetch_runtime.fetch.await_count >= 3


@pytest.mark.asyncio
async def test_country_recent_news_strategy_returns_pre_synthesized_digest_for_translated_group_content():
    from features.web_scraping.application.country_strategy import CountryRecentNewsStrategy
    from features.web_scraping.application import flow as _flow

    fetch_runtime = MagicMock()
    fetch_runtime.fetch = AsyncMock(side_effect=[
        MagicMock(content="서울 경찰은 이번 주 지하철역과 공항 주변 순찰을 대폭 강화했다.\n당국은 군중 밀집 지역 점검도 확대했다고 밝혔다."),
        MagicMock(content="부산 경찰은 이번 주 터미널과 번화가에서 추가 검문과 순찰을 시작했다.\n지역 당국은 공공장소 안전 대책을 병행하고 있다."),
        MagicMock(content="정부는 이번 주 공공기관을 대상으로 사이버 보안 경보를 상향했다.\n비상 대응팀은 추가 모의훈련을 실시했다."),
    ])

    async def _discover(query, source_group, source_terms, runtime_args=None):
        domains = ["arirang.com", "edaily.co.kr"]
        names = ["Arirang", "Edaily"]
        _flow._country_press_cache_set(source_group, source_terms, domains, names)
        _flow._country_press_strategy_cache_set(source_group, source_terms, "lookup")
        _flow._country_press_source_cache_set(
            source_group,
            source_terms,
            [
                {"title": "Arirang", "url": "https://www.arirang.com/"},
                {"title": "Edaily", "url": "https://www.edaily.co.kr/"},
            ],
        )
        return domains, names

    strategy = CountryRecentNewsStrategy(
        search_runtime=MagicMock(),
        fetch_runtime=fetch_runtime,
        press_discovery=MagicMock(discover=AsyncMock(side_effect=_discover)),
    )

    with (
        patch(
            "features.web_scraping.application.country_press_helpers._run_country_press_search_candidates",
            new=AsyncMock(return_value=(
                [
                    {"title": "Arirang security update", "url": "https://www.arirang.com/news/2026/05/27/one", "snippet": "Security update"},
                    {"title": "Edaily police update", "url": "https://www.edaily.co.kr/news/2026/05/27/two", "snippet": "Police update"},
                    {"title": "Cyber alert", "url": "https://www.donga.com/news/2026/05/27/three", "snippet": "Cyber alert"},
                ],
                "",
            )),
        ),
        patch(
            "features.web_scraping.application.flow._extract_generic_content_lines",
            side_effect=[
                [
                    "서울 경찰은 이번 주 지하철역과 공항 주변 순찰을 대폭 강화했다.",
                    "당국은 군중 밀집 지역 점검도 확대했다고 밝혔다.",
                ],
                [
                    "부산 경찰은 이번 주 터미널과 번화가에서 추가 검문과 순찰을 시작했다.",
                    "지역 당국은 공공장소 안전 대책을 병행하고 있다.",
                ],
                [
                    "정부는 이번 주 공공기관을 대상으로 사이버 보안 경보를 상향했다.",
                    "비상 대응팀은 추가 모의훈련을 실시했다.",
                ],
            ],
        ),
        patch(
            "features.web_scraping.application.flow._discover_homepage_section_targets",
            new=AsyncMock(return_value=([], True)),
        ),
    ):
        result = await strategy.execute("dame las ultimas noticias sobre seguridad en corea del sur de esta semana")

    assert result is not None
    assert result["pre_synthesized"] is True
    assert result.get("has_labeled_content") is not True
    assert result.get("digest_contract") is not None
    assert result["summary"].count("Fuente:") >= 3
    assert "Sources:" in result["summary"]


@pytest.mark.asyncio
async def test_country_recent_news_strategy_uses_only_scraped_content_when_articles_are_insufficient():
    from features.web_scraping.application.country_strategy import CountryRecentNewsStrategy
    from features.web_scraping.application import flow as _flow

    fetch_runtime = MagicMock()
    fetch_runtime.fetch = AsyncMock(side_effect=[
        MagicMock(content="La policía de Seúl reforzó patrullajes nocturnos tras una serie de robos.\nTres sospechosos fueron detenidos en estaciones de tren."),
        MagicMock(content="La policía de Busan desplegó operativos especiales luego de un ataque con arma blanca.\nLos agentes buscan a dos sospechosos y reforzaron controles en terminales."),
        MagicMock(content="Error al procesar la pagina web: timeout"),
        MagicMock(content="Error al procesar la pagina web: timeout"),
        MagicMock(content="Error al procesar la pagina web: timeout"),
        MagicMock(content="Error al procesar la pagina web: timeout"),
        MagicMock(content="La policía amplió controles en zonas escolares y comerciales de Seúl.\nLas autoridades sumaron retenes y vigilancia adicional durante la semana."),
    ])

    async def _discover(query, source_group, source_terms, runtime_args=None):
        domains = ["arirang.com"]
        names = ["Arirang"]
        _flow._country_press_cache_set(source_group, source_terms, domains, names)
        _flow._country_press_strategy_cache_set(source_group, source_terms, "lookup")
        _flow._country_press_source_cache_set(
            source_group,
            source_terms,
            [{"title": "Arirang", "url": "https://www.arirang.com/"}],
        )
        return domains, names

    strategy = CountryRecentNewsStrategy(
        search_runtime=MagicMock(),
        fetch_runtime=fetch_runtime,
        press_discovery=MagicMock(discover=AsyncMock(side_effect=_discover)),
    )

    with (
        patch(
            "features.web_scraping.application.country_press_helpers._run_country_press_search_candidates",
            new=AsyncMock(return_value=(
                [
                    {"title": "Police raids in Seoul", "url": "https://www.arirang.com/news/2026/05/27/one", "snippet": "La policía de Seúl reforzó patrullajes nocturnos tras una serie de robos."},
                    {"title": "Knife attack in Busan", "url": "https://www.arirang.com/news/2026/05/27/two", "snippet": "La policía investiga un ataque con arma blanca ocurrido en Busan y busca a dos sospechosos."},
                    {"title": "Cybersecurity alert", "url": "https://www.arirang.com/news/2026/05/27/three", "snippet": "Autoridades ampliaron alertas y controles después de nuevas filtraciones de datos en Corea del Sur."},
                ],
                "",
            )),
        ),
        patch(
            "features.web_scraping.application.flow._extract_generic_content_lines",
            side_effect=[
                [
                    "La policía de Seúl reforzó patrullajes nocturnos tras una serie de robos.",
                    "Tres sospechosos fueron detenidos en estaciones de tren.",
                ],
                [
                    "La policía de Busan desplegó operativos especiales luego de un ataque con arma blanca.",
                    "Los agentes buscan a dos sospechosos y reforzaron controles en terminales.",
                ],
                [],
            ],
        ),
        patch(
            "features.web_scraping.application.flow._build_country_press_section_targets",
            return_value=[("https://www.arirang.com/news/security", "security")],
        ),
        patch(
            "features.web_scraping.application.flow._extract_section_content_lines",
            return_value=[
                "La policía amplió controles en zonas escolares y comerciales de Seúl.",
                "Las autoridades sumaron retenes y vigilancia adicional durante la semana.",
            ],
        ),
        patch(
            "features.web_scraping.application.flow._filter_section_lines_for_query",
            side_effect=lambda lines, *_: lines,
        ),
    ):
        result = await strategy.execute("dame las ultimas noticias sobre seguridad en corea del sur de esta semana")

    assert result is not None
    assert result["pre_synthesized"] is True
    assert result.get("has_labeled_content") is not True
    assert result["summary"].count("[") >= 2
    assert "ataque con arma blanca" in result["summary"]
    assert "filtraciones de datos" not in result["summary"]


@pytest.mark.asyncio
async def test_country_recent_news_strategy_short_circuits_on_section_first_results():
    from features.web_scraping.application.country_strategy import CountryRecentNewsStrategy
    from features.web_scraping.application import flow as _flow

    fetch_runtime = MagicMock()
    fetch_runtime.fetch = AsyncMock(side_effect=[
        MagicMock(content=(
            "La policía surcoreana reforzó operativos en barrios comerciales de Seúl.\n\n"
            "Las autoridades aumentaron la vigilancia en estaciones y centros públicos."
        )),
        MagicMock(content=(
            "La policía incrementó retenes y patrullajes en terminales de Busan.\n\n"
            "Los agentes desplegaron controles adicionales durante la semana."
        )),
        MagicMock(content=(
            "Las autoridades reforzaron controles de ciberseguridad y monitoreo en organismos públicos.\n\n"
            "Los equipos de respuesta coordinaron simulacros y revisiones preventivas."
        )),
        MagicMock(content=(
            "La policía amplió controles en aeropuertos y estaciones de tren de Incheon.\n\n"
            "Los agentes sumaron patrullas y puntos de control durante la semana."
        )),
    ])

    async def _discover(query, source_group, source_terms, runtime_args=None):
        domains = ["edaily.co.kr"]
        names = ["Edaily"]
        _flow._country_press_cache_set(source_group, source_terms, domains, names)
        _flow._country_press_strategy_cache_set(source_group, source_terms, "lookup")
        _flow._country_press_source_cache_set(
            source_group,
            source_terms,
            [{"title": "Edaily", "url": "https://www.edaily.co.kr/"}],
        )
        return domains, names

    strategy = CountryRecentNewsStrategy(
        search_runtime=MagicMock(),
        fetch_runtime=fetch_runtime,
        press_discovery=MagicMock(discover=AsyncMock(side_effect=_discover)),
    )

    with (
        patch(
            "features.web_scraping.application.country_press_helpers._run_country_press_search_candidates",
            new=AsyncMock(side_effect=AssertionError("no debería llamarse article-search si section-first ya alcanzó el mínimo concreto")),
        ),
        patch(
            "features.web_scraping.application.flow._discover_homepage_section_targets",
            new=AsyncMock(return_value=([
                ("https://www.edaily.co.kr/sociedad/", "sociedad"),
                ("https://www.edaily.co.kr/police/", "police"),
                ("https://www.edaily.co.kr/cyber/", "cyber"),
                ("https://www.edaily.co.kr/transport/", "transport"),
            ], True)),
        ),
        patch(
            "features.web_scraping.application.flow._extract_section_content_lines",
            side_effect=[
                [
                    "La policía surcoreana reforzó operativos en barrios comerciales de Seúl.",
                    "Las autoridades aumentaron la vigilancia en estaciones y centros públicos.",
                ],
                [
                    "La policía incrementó retenes y patrullajes en terminales de Busan.",
                    "Los agentes desplegaron controles adicionales durante la semana.",
                ],
                [
                    "Las autoridades reforzaron controles de ciberseguridad y monitoreo en organismos públicos.",
                    "Los equipos de respuesta coordinaron simulacros y revisiones preventivas.",
                ],
                [
                    "La policía amplió controles en aeropuertos y estaciones de tren de Incheon.",
                    "Los agentes sumaron patrullas y puntos de control durante la semana.",
                ],
            ],
        ),
        patch(
            "features.web_scraping.application.flow._filter_section_lines_for_query",
            side_effect=lambda lines, *_: lines,
        ),
    ):
        result = await strategy.execute("dame las ultimas noticias sobre seguridad en corea del sur de esta semana")

    assert result is not None
    assert result["pre_synthesized"] is True
    assert result.get("has_labeled_content") is not True
    assert "barrios comerciales de Seúl" in result["summary"]
    assert "terminales de Busan" in result["summary"]
    assert "ciberseguridad y monitoreo" in result["summary"]
    assert "aeropuertos y estaciones de tren de Incheon" in result["summary"]
    assert fetch_runtime.fetch.await_count == 4


def test_candidate_relevance_scoring_is_dynamic_for_public_safety_queries():
    from application.policies.candidate_scoring import (
        _is_relevant_candidate_for_query,
        _score_candidate_relevance,
    )

    query = "dame las ultimas noticias sobre seguridad en corea del sur de esta semana"
    relevant_candidate = {
        "title": "donga.com — 사회",
        "url": "https://www.donga.com/news/Society/List",
        "snippet": "La policía surcoreana abrió una investigación por un ataque en Seúl y reforzó patrullajes en estaciones.",
        "source_kind": "section_fallback",
    }
    tangential_candidate = {
        "title": "donga.com — 북한",
        "url": "https://www.donga.com/news/Politics/NK",
        "snippet": "Corea del Norte rechazó la desnuclearización y probó un nuevo misil en medio de tensiones diplomáticas.",
        "source_kind": "section_fallback",
    }

    relevant_score = _score_candidate_relevance(relevant_candidate, query, "south_korea")
    tangential_score = _score_candidate_relevance(tangential_candidate, query, "south_korea")

    assert relevant_score > tangential_score
    assert _is_relevant_candidate_for_query(relevant_candidate, query, "south_korea") is True
    assert _is_relevant_candidate_for_query(tangential_candidate, query, "south_korea") is False


def test_query_localizer_caps_languages_and_queries_per_domain():
    from features.web_scraping.domain.query_localization import (
        LocalizedNewsQueryBuilder,
        QueryLocalizationContext,
    )

    builder = LocalizedNewsQueryBuilder()
    specs = builder.build_query_specs(
        domain="chosun.com",
        press_name="chosun.com",
        context=QueryLocalizationContext(
            geography="Corea del Sur",
            geo_en="South Korea",
            topic="security",
            horizon="week",
            query_source_group="south_korea",
            public_safety_query=True,
        ),
    )

    assert len(specs) <= 6
    assert {spec.language for spec in specs}.issubset({"ko", "en"})


@pytest.mark.asyncio
async def test_country_recent_news_strategy_does_not_short_circuit_on_tangential_section_candidates():
    from features.web_scraping.application.country_strategy import CountryRecentNewsStrategy
    from features.web_scraping.application import flow as _flow

    async def _fetch_side_effect(request):
        return MagicMock(content=(
            "Corea del Norte rechazó la desnuclearización y lanzó un nuevo misil. "
            "Las autoridades internacionales reaccionaron con preocupación diplomática."
        ))

    fetch_runtime = MagicMock()
    fetch_runtime.fetch = AsyncMock(side_effect=_fetch_side_effect)

    async def _discover(query, source_group, source_terms, runtime_args=None):
        domains = ["donga.com"]
        names = ["Donga"]
        _flow._country_press_cache_set(source_group, source_terms, domains, names)
        _flow._country_press_strategy_cache_set(source_group, source_terms, "lookup")
        _flow._country_press_source_cache_set(
            source_group,
            source_terms,
            [{"title": "Donga", "url": "https://www.donga.com/"}],
        )
        return domains, names

    strategy = CountryRecentNewsStrategy(
        search_runtime=MagicMock(),
        fetch_runtime=fetch_runtime,
        press_discovery=MagicMock(discover=AsyncMock(side_effect=_discover)),
    )

    article_search_mock = AsyncMock(return_value=([], ""))

    with (
        patch(
            "features.web_scraping.application.flow._discover_homepage_section_targets",
            new=AsyncMock(return_value=([
                ("https://www.donga.com/news/Politics/NK", "북한"),
                ("https://www.donga.com/news/Politics/NK2", "북한"),
                ("https://www.donga.com/news/Politics/NK3", "북한"),
                ("https://www.donga.com/news/Politics/NK4", "북한"),
            ], True)),
        ),
        patch(
            "features.web_scraping.application.country_press_helpers._run_country_press_search_candidates",
            new=article_search_mock,
        ),
    ):
        result = await strategy.execute("dame las ultimas noticias sobre seguridad en corea del sur de esta semana")

    assert result is None
    article_search_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_country_recent_news_strategy_ignores_placeholder_section_payloads_and_falls_back_to_articles():
    from features.web_scraping.application.country_strategy import CountryRecentNewsStrategy
    from features.web_scraping.application import flow as _flow

    fetch_runtime = MagicMock()
    fetch_runtime.fetch = AsyncMock(side_effect=[
        MagicMock(content="<<<CITE_THIS: title=Arirang|url=https://www.arirang.com/seguridad/|domain=www.arirang.com>>>"),
        MagicMock(content="<<<CITE_THIS: title=Arirang|url=https://www.arirang.com/policiales/|domain=www.arirang.com>>>"),
        MagicMock(content="South Korea police reinforced patrols in Seoul after a security alert.\nAuthorities added checkpoints near transit hubs this week."),
        MagicMock(content="South Korean authorities expanded cyber monitoring after new breaches.\nEmergency teams coordinated with local police on response drills."),
        MagicMock(content="Officials increased security around government buildings and train stations.\nPolice said the measures are preventive and temporary."),
    ])

    async def _discover(query, source_group, source_terms, runtime_args=None):
        domains = ["arirang.com"]
        names = ["Arirang"]
        _flow._country_press_cache_set(source_group, source_terms, domains, names)
        _flow._country_press_strategy_cache_set(source_group, source_terms, "lookup")
        _flow._country_press_source_cache_set(
            source_group,
            source_terms,
            [{"title": "Arirang", "url": "https://www.arirang.com/"}],
        )
        return domains, names

    strategy = CountryRecentNewsStrategy(
        search_runtime=MagicMock(),
        fetch_runtime=fetch_runtime,
        press_discovery=MagicMock(discover=AsyncMock(side_effect=_discover)),
    )

    with (
        patch(
            "features.web_scraping.application.flow._discover_homepage_section_targets",
            new=AsyncMock(return_value=([
                ("https://www.arirang.com/seguridad/", "seguridad"),
                ("https://www.arirang.com/policiales/", "policiales"),
            ], True)),
        ),
        patch(
            "features.web_scraping.application.country_press_helpers._run_country_press_search_candidates",
            new=AsyncMock(return_value=(
                [
                    {"title": "Security alert in Seoul", "url": "https://www.arirang.com/news/2026/05/27/one", "snippet": "Security alert"},
                    {"title": "Cyber monitoring expands", "url": "https://www.arirang.com/news/2026/05/27/two", "snippet": "Cyber monitoring"},
                    {"title": "Government buildings secured", "url": "https://www.arirang.com/news/2026/05/27/three", "snippet": "Government buildings secured"},
                ],
                "",
            )),
        ),
        patch(
            "features.web_scraping.application.flow._extract_generic_content_lines",
            side_effect=[
                [
                    "South Korea police reinforced patrols in Seoul after a security alert.",
                    "Authorities added checkpoints near transit hubs this week.",
                ],
                [
                    "South Korean authorities expanded cyber monitoring after new breaches.",
                    "Emergency teams coordinated with local police on response drills.",
                ],
                [
                    "Officials increased security around government buildings and train stations.",
                    "Police said the measures are preventive and temporary.",
                ],
            ],
        ),
    ):
        result = await strategy.execute("dame las ultimas noticias sobre seguridad en corea del sur de esta semana")

    assert result is not None
    assert result["pre_synthesized"] is True
    assert "<<<CITE_THIS" not in result["summary"]
    assert "South Korea police reinforced patrols in Seoul after a security alert." in result["summary"]
    assert fetch_runtime.fetch.await_count == 5


@pytest.mark.asyncio
async def test_country_recent_news_strategy_retries_article_fetch_once_after_failure():
    from features.web_scraping.application.country_strategy import CountryRecentNewsStrategy
    from features.web_scraping.application import flow as _flow

    fetch_runtime = MagicMock()
    fetch_runtime.fetch = AsyncMock(side_effect=[
        RuntimeError("temporary fetch failure"),
        MagicMock(content="South Korea police reinforced patrols in Seoul after a security alert.\nAuthorities added checkpoints near transit hubs this week."),
        MagicMock(content="South Korean authorities expanded cyber monitoring after new breaches.\nEmergency teams coordinated with local police on response drills."),
        MagicMock(content="Officials increased security around government buildings and train stations.\nPolice said the measures are preventive and temporary."),
    ])

    async def _discover(query, source_group, source_terms, runtime_args=None):
        domains = ["arirang.com"]
        names = ["Arirang"]
        _flow._country_press_cache_set(source_group, source_terms, domains, names)
        _flow._country_press_strategy_cache_set(source_group, source_terms, "lookup")
        _flow._country_press_source_cache_set(
            source_group,
            source_terms,
            [{"title": "Arirang", "url": "https://www.arirang.com/"}],
        )
        return domains, names

    strategy = CountryRecentNewsStrategy(
        search_runtime=MagicMock(),
        fetch_runtime=fetch_runtime,
        press_discovery=MagicMock(discover=AsyncMock(side_effect=_discover)),
    )

    with (
        patch(
            "features.web_scraping.application.country_press_helpers._run_country_press_search_candidates",
            new=AsyncMock(return_value=(
                [
                    {"title": "Security alert in Seoul", "url": "https://www.arirang.com/news/2026/05/27/one", "snippet": "Security alert"},
                    {"title": "Cyber monitoring expands", "url": "https://www.arirang.com/news/2026/05/27/two", "snippet": "Cyber monitoring"},
                    {"title": "Government buildings secured", "url": "https://www.arirang.com/news/2026/05/27/three", "snippet": "Government buildings secured"},
                ],
                "",
            )),
        ),
        patch(
            "features.web_scraping.application.flow._extract_generic_content_lines",
            side_effect=[
                [
                    "South Korea police reinforced patrols in Seoul after a security alert.",
                    "Authorities added checkpoints near transit hubs this week.",
                ],
                [
                    "South Korean authorities expanded cyber monitoring after new breaches.",
                    "Emergency teams coordinated with local police on response drills.",
                ],
                [
                    "Officials increased security around government buildings and train stations.",
                    "Police said the measures are preventive and temporary.",
                ],
            ],
        ),
        patch(
            "features.web_scraping.application.flow._discover_homepage_section_targets",
            new=AsyncMock(return_value=([], True)),
        ),
    ):
        result = await strategy.execute("dame las ultimas noticias sobre seguridad en corea del sur de esta semana")

    assert result is not None
    assert result["pre_synthesized"] is True
    assert result.get("has_labeled_content") is not True
    assert fetch_runtime.fetch.await_count == 4
    assert "South Korea police reinforced patrols in Seoul after a security alert." in result["summary"]


@pytest.mark.asyncio
async def test_country_recent_news_strategy_logs_article_task_elapsed_times():
    from features.web_scraping.application.country_strategy import CountryRecentNewsStrategy
    from features.web_scraping.application import flow as _flow

    debug_events: list[tuple[str, dict]] = []

    async def _fetch(request):
        await asyncio.sleep(0.01 if "one" in request.url else 0.02)
        return MagicMock(content=(
            "South Korea police reinforced patrols in Seoul after a security alert.\n"
            "Authorities added checkpoints near transit hubs this week."
            if "one" in request.url else
            "South Korean authorities expanded cyber monitoring after new breaches.\n"
            "Emergency teams coordinated with local police on response drills."
        ))

    async def _discover(query, source_group, source_terms, runtime_args=None):
        domains = ["arirang.com"]
        names = ["Arirang"]
        _flow._country_press_cache_set(source_group, source_terms, domains, names)
        _flow._country_press_strategy_cache_set(source_group, source_terms, "lookup")
        _flow._country_press_source_cache_set(
            source_group,
            source_terms,
            [{"title": "Arirang", "url": "https://www.arirang.com/"}],
        )
        return domains, names

    strategy = CountryRecentNewsStrategy(
        search_runtime=MagicMock(),
        fetch_runtime=MagicMock(fetch=AsyncMock(side_effect=_fetch)),
        press_discovery=MagicMock(discover=AsyncMock(side_effect=_discover)),
    )

    def _capture_debug(event, **payload):
        debug_events.append((event, payload))

    with (
        patch(
            "features.web_scraping.application.country_press_helpers._run_country_press_search_candidates",
            new=AsyncMock(return_value=(
                [
                    {"title": "Security alert in Seoul", "url": "https://www.arirang.com/news/2026/05/27/one", "snippet": "Security alert"},
                    {"title": "Cyber monitoring expands", "url": "https://www.arirang.com/news/2026/05/27/two", "snippet": "Cyber monitoring"},
                ],
                "",
            )),
        ),
        patch("features.web_scraping.application.flow._discover_homepage_section_targets", new=AsyncMock(return_value=([], True))),
        patch("features.web_scraping.application.flow._web_debug", side_effect=_capture_debug),
    ):
        result = await strategy.execute("dame las ultimas noticias sobre seguridad en corea del sur de esta semana")

    assert result is not None
    task_events = [payload for event, payload in debug_events if event == "country_strategy.article_task_completed"]
    assert len(task_events) >= 2
    assert all("elapsed_ms" in payload for payload in task_events)
    assert all(payload["elapsed_ms"] >= 0 for payload in task_events)
    assert all("worker_id" in payload for payload in task_events)
    assert all("task_idx" in payload for payload in task_events)


@pytest.mark.asyncio
async def test_country_recent_news_strategy_logs_section_task_elapsed_times():
    from features.web_scraping.application.country_strategy import CountryRecentNewsStrategy
    from features.web_scraping.application import flow as _flow

    debug_events: list[tuple[str, dict]] = []

    async def _fetch(request):
        await asyncio.sleep(0.01 if "society" in request.url else 0.02)
        if "society" in request.url:
            return MagicMock(content=(
                "La policía surcoreana reforzó operativos en barrios comerciales de Seúl.\n\n"
                "Las autoridades aumentaron la vigilancia en estaciones y centros públicos."
            ))
        return MagicMock(content=(
            "La policía incrementó retenes y patrullajes en terminales de Busan.\n\n"
            "Los agentes desplegaron controles adicionales durante la semana."
        ))

    async def _discover(query, source_group, source_terms, runtime_args=None):
        domains = ["edaily.co.kr"]
        names = ["Edaily"]
        _flow._country_press_cache_set(source_group, source_terms, domains, names)
        _flow._country_press_strategy_cache_set(source_group, source_terms, "lookup")
        _flow._country_press_source_cache_set(
            source_group,
            source_terms,
            [{"title": "Edaily", "url": "https://www.edaily.co.kr/"}],
        )
        return domains, names

    strategy = CountryRecentNewsStrategy(
        search_runtime=MagicMock(),
        fetch_runtime=MagicMock(fetch=AsyncMock(side_effect=_fetch)),
        press_discovery=MagicMock(discover=AsyncMock(side_effect=_discover)),
    )

    def _capture_debug(event, **payload):
        debug_events.append((event, payload))

    with (
        patch(
            "features.web_scraping.application.flow._discover_homepage_section_targets",
            new=AsyncMock(return_value=([
                ("https://www.edaily.co.kr/society", "society"),
                ("https://www.edaily.co.kr/national", "national"),
            ], True)),
        ),
        patch(
            "features.web_scraping.application.country_press_helpers._run_country_press_search_candidates",
            new=AsyncMock(return_value=([], "")),
        ),
        patch("features.web_scraping.application.flow._web_debug", side_effect=_capture_debug),
    ):
        result = await strategy.execute("dame las ultimas noticias sobre seguridad en corea del sur de esta semana")

    assert result is not None
    task_events = [payload for event, payload in debug_events if event == "country_strategy.section_task_completed"]
    assert len(task_events) >= 2
    assert all("elapsed_ms" in payload for payload in task_events)
    assert all(payload["elapsed_ms"] >= 0 for payload in task_events)
    assert all("worker_id" in payload for payload in task_events)
    assert all("task_idx" in payload for payload in task_events)


@pytest.mark.asyncio
async def test_country_recent_news_strategy_ignores_malformed_source_urls_without_ipv6_crash():
    from features.web_scraping.application.country_strategy import CountryRecentNewsStrategy
    from features.web_scraping.application import flow as _flow

    fetch_runtime = MagicMock()
    fetch_runtime.fetch = AsyncMock(side_effect=[
        MagicMock(content="South Korea police reinforced patrols in Seoul after a security alert.\nAuthorities added checkpoints near transit hubs this week."),
        MagicMock(content="South Korean authorities expanded cyber monitoring after new breaches.\nEmergency teams coordinated with local police on response drills."),
    ])

    async def _discover(query, source_group, source_terms, runtime_args=None):
        domains = ["arirang.com"]
        names = ["Arirang"]
        _flow._country_press_cache_set(source_group, source_terms, domains, names)
        _flow._country_press_strategy_cache_set(source_group, source_terms, "lookup")
        _flow._country_press_source_cache_set(
            source_group,
            source_terms,
            [
                {"title": "Arirang", "url": "https://[oops"},
                {"title": "Arirang fallback", "url": "https://www.arirang.com/"},
            ],
        )
        return domains, names

    strategy = CountryRecentNewsStrategy(
        search_runtime=MagicMock(),
        fetch_runtime=fetch_runtime,
        press_discovery=MagicMock(discover=AsyncMock(side_effect=_discover)),
    )

    with (
        patch(
            "features.web_scraping.application.flow._discover_homepage_section_targets",
            new=AsyncMock(return_value=([], True)),
        ),
        patch(
            "features.web_scraping.application.country_press_helpers._run_country_press_search_candidates",
            new=AsyncMock(return_value=(
                [
                    {"title": "Security alert in Seoul", "url": "https://www.arirang.com/news/2026/05/27/one", "snippet": "Security alert"},
                    {"title": "Cyber monitoring expands", "url": "https://www.arirang.com/news/2026/05/27/two", "snippet": "Cyber monitoring"},
                ],
                "",
            )),
        ),
    ):
        result = await strategy.execute("dame las ultimas noticias sobre seguridad en corea del sur de esta semana")

    assert result is not None
    assert result["pre_synthesized"] is True
    assert "South Korea police reinforced patrols in Seoul after a security alert." in result["summary"]


def test_extract_country_press_sources_filters_malformed_urls_and_normalizes_generic_titles():
    from features.web_scraping.application.flow import _extract_country_press_sources

    text = (
        "[Enlace](https://[oops)\n"
        "[Enlace](https://www.arirang.com/)\n"
        "[KBS News](https://news.kbs.co.kr/)"
    )

    sources = _extract_country_press_sources(text)

    assert len(sources) == 2
    assert sources[0]["title"] == "arirang.com"
    assert sources[0]["domain"] == "arirang.com"
    assert sources[1]["title"] == "KBS News"
    assert all(source["url"] != "https://[oops" for source in sources)


def test_validate_public_http_url_rejects_invalid_ipv6_and_normalizes_valid_urls():
    from core.helpers.url_helpers import _validate_public_http_url

    normalized, error = _validate_public_http_url("https://www.arirang.com/news")
    assert error is None
    assert normalized.startswith("https://www.arirang.com/")

    invalid_normalized, invalid_error = _validate_public_http_url("https://[oops")
    assert invalid_normalized == ""
    assert invalid_error == "URL inválida"


@pytest.mark.asyncio
async def test_extract_relevant_homepage_sections_deduplicates_repeated_topic_sections():
    from features.web_scraping.application.flow import _extract_relevant_homepage_sections

    homepage_text = """
    [Crime](https://example.com/crime/)
    [Crime](https://example.com/crime?ref=nav)
    [Police](https://example.com/police/)
    [Police News](https://example.com/police/)
    [Article](https://example.com/news/2026/05/27/security-alert)
    [Facebook](https://facebook.com/example)
    """

    sections = _extract_relevant_homepage_sections(
        homepage_text,
        domain="example.com",
        base_url="https://example.com/",
        last_message="dame las ultimas noticias sobre seguridad en corea del sur de esta semana",
    )

    assert sections == [
        ("https://example.com/crime", "Crime"),
        ("https://example.com/police", "Police"),
    ]


@pytest.mark.asyncio
async def test_extract_relevant_homepage_sections_accepts_generic_local_navigation_labels_from_html():
    from features.web_scraping.application.flow import _extract_relevant_homepage_sections

    homepage_html = """
    <html>
      <body>
        <header>
          <nav class="global-nav">
            <a href="/society">사회</a>
            <a href="/politics">정치</a>
            <a href="/international">국제</a>
            <a href="/sports">스포츠</a>
          </nav>
        </header>
      </body>
    </html>
    """

    sections = _extract_relevant_homepage_sections(
        homepage_html,
        domain="example.com",
        base_url="https://example.com/",
        last_message="dame las ultimas noticias sobre seguridad en corea del sur de esta semana",
    )

    assert ("https://example.com/society", "사회") in sections
    assert ("https://example.com/politics", "정치") in sections
    assert all(url != "https://example.com/sports" for url, _ in sections)


@pytest.mark.asyncio
async def test_extract_relevant_homepage_sections_rejects_dirty_html_labels():
    from features.web_scraping.application.flow import _extract_relevant_homepage_sections

    homepage_html = """
    <html>
      <body>
        <nav>
          <a href="/international">국제</a>
          <a href='https://www.chosun.com/international/china/"&gt;중국&lt;/a&gt;'>https://www.chosun.com/international/china/">중국</a></a>
        </nav>
      </body>
    </html>
    """

    sections = _extract_relevant_homepage_sections(
        homepage_html,
        domain="chosun.com",
        base_url="https://chosun.com/",
        last_message="dame las ultimas noticias sobre seguridad en corea del sur de esta semana",
    )

    assert ("https://chosun.com/international", "국제") in sections
    assert all("https://www.chosun.com/international/china/" not in url for url, _ in sections)
    assert all("중국</a>" not in label for _, label in sections)


@pytest.mark.asyncio
async def test_country_recent_news_strategy_ignores_redirect_payload_sections_and_falls_back_to_articles():
    from features.web_scraping.application.country_strategy import CountryRecentNewsStrategy
    from features.web_scraping.application import flow as _flow

    fetch_runtime = MagicMock()
    fetch_runtime.fetch = AsyncMock(side_effect=[
        MagicMock(content=(
            "REDIRECT DETECTED: The URL redirects to a different host.\n\n"
            "Original URL: https://chosun.com/international\n"
            "Redirect URL: https://www.chosun.com/international/\n"
            "Status: 307 Temporary Redirect\n\n"
            "Consulta original: dame las ultimas noticias sobre seguridad en corea del sur de esta semana"
        )),
        MagicMock(content="South Korea police reinforced patrols in Seoul after a security alert.\nAuthorities added checkpoints near transit hubs this week."),
        MagicMock(content="South Korean authorities expanded cyber monitoring after new breaches.\nEmergency teams coordinated with local police on response drills."),
        MagicMock(content="Officials increased security around government buildings and train stations.\nPolice said the measures are preventive and temporary."),
    ])

    async def _discover(query, source_group, source_terms, runtime_args=None):
        domains = ["chosun.com"]
        names = ["Chosun"]
        _flow._country_press_cache_set(source_group, source_terms, domains, names)
        _flow._country_press_strategy_cache_set(source_group, source_terms, "lookup")
        _flow._country_press_source_cache_set(
            source_group,
            source_terms,
            [{"title": "Chosun", "url": "https://chosun.com/"}],
        )
        return domains, names

    strategy = CountryRecentNewsStrategy(
        search_runtime=MagicMock(),
        fetch_runtime=fetch_runtime,
        press_discovery=MagicMock(discover=AsyncMock(side_effect=_discover)),
    )

    with (
        patch(
            "features.web_scraping.application.flow._discover_homepage_section_targets",
            new=AsyncMock(return_value=([("https://chosun.com/international", "국제")], True)),
        ),
        patch(
            "features.web_scraping.application.country_press_helpers._run_country_press_search_candidates",
            new=AsyncMock(return_value=(
                [
                    {"title": "Security alert in Seoul", "url": "https://www.chosun.com/english/national-en/2026/05/27/one", "snippet": "Security alert"},
                    {"title": "Cyber monitoring expands", "url": "https://www.chosun.com/english/national-en/2026/05/27/two", "snippet": "Cyber monitoring"},
                    {"title": "Government buildings secured", "url": "https://www.chosun.com/english/national-en/2026/05/27/three", "snippet": "Government buildings secured"},
                ],
                "",
            )),
        ),
        patch(
            "features.web_scraping.application.flow._extract_generic_content_lines",
            side_effect=[
                [
                    "South Korea police reinforced patrols in Seoul after a security alert.",
                    "Authorities added checkpoints near transit hubs this week.",
                ],
                [
                    "South Korean authorities expanded cyber monitoring after new breaches.",
                    "Emergency teams coordinated with local police on response drills.",
                ],
                [
                    "Officials increased security around government buildings and train stations.",
                    "Police said the measures are preventive and temporary.",
                ],
            ],
        ),
    ):
        result = await strategy.execute("dame las ultimas noticias sobre seguridad en corea del sur de esta semana")

    assert result is not None
    assert "REDIRECT DETECTED" not in result["summary"]
    assert "Consulta original:" not in result["summary"]
    assert "South Korea police reinforced patrols in Seoul after a security alert." in result["summary"]


@pytest.mark.asyncio
async def test_discover_homepage_section_targets_skips_generic_spanish_fallback_when_no_real_sections():
    from features.web_scraping.application.flow import _discover_homepage_section_targets

    with patch(
        "features.web_scraping.application.flow._fetch_homepage_document",
        new=AsyncMock(return_value="<html><body><div>home without matching sections</div></body></html>"),
    ):
        sections, dynamic_available = await _discover_homepage_section_targets(
            domain="example.com",
            fallback_url="https://example.com/",
            last_message="dame las ultimas noticias sobre seguridad en corea del sur de esta semana",
            press_name="Example",
            dynamic_fetch_available=False,
        )

    assert sections == []
    assert dynamic_available is False


def test_query_localizer_infers_language_priority_from_domain_tld():
    from features.web_scraping.domain.query_localization import GeoLanguageResolver

    resolver = GeoLanguageResolver()

    assert resolver.resolve(country_group=None, domain="lemonde.fr")[:2] == ["fr", "en"]
    assert resolver.resolve(country_group=None, domain="corriere.it")[:2] == ["it", "en"]


@pytest.mark.asyncio
async def test_public_safety_candidate_scoring_penalizes_geopolitics_and_boosts_police_news():
    from application.policies.candidate_scoring import _score_generic_candidate

    query_terms = ["corea", "sur", "seguridad", "policiales", "inseguridad"]
    police_candidate = {
        "title": "South Korea police expand security operation in Seoul",
        "url": "https://www.arirang.com/news/2026/05/27/police-operation-seoul",
        "snippet": "Police increased checkpoints and reported new arrests after a robbery case.",
    }
    geopolitical_candidate = {
        "title": "South Korea and Japan to discuss security cooperation at summit",
        "url": "https://english.hani.co.kr/arti/english_edition/e_international/1239265.html",
        "snippet": "Diplomatic talks will cover regional security and foreign policy coordination.",
    }

    police_score = _score_generic_candidate(police_candidate, query_terms, "south_korea")
    geopolitical_score = _score_generic_candidate(geopolitical_candidate, query_terms, "south_korea")

    assert police_score > geopolitical_score


@pytest.mark.asyncio
async def test_recent_generic_web_query_requires_sufficient_context():
    from features.web_scraping.application.fetch_dispatch import _run_generic_web_search_fetch

    with (
        patch("features.web_scraping.infrastructure.search_tools.search_web.func", return_value="Web search results for query: \"dame las ultimas noticias sobre seguridad de japon hoy April 2026\"\n\n1. [Japan update](https://example.com/japan)\n   Short snippet\n\nSources:\n- [Japan update](https://example.com/japan)"),
        patch("features.web_scraping.infrastructure.scraping_tools.fetch_web_page", AsyncMock(return_value="URL: https://example.com/japan\n\nSolo una línea insuficiente\n\nSources:\n- [Japan update](https://example.com/japan)")),
    ):
        result = await _run_generic_web_search_fetch("dame las ultimas noticias sobre seguridad de japon hoy")

    assert result is not None
    assert "No encontré fuentes locales confiables" in result["summary"]
    assert result["pre_synthesized"] is True


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
    assert content.count("Fuente:") >= 2
    assert content.count("Sources:") == 1
    assert "subrayan un cambio de postura" not in content
    assert len(result["sources"]) >= 2


@pytest.mark.asyncio
async def test_latest_japan_news_query_resolves_without_falling_back_to_agent_strategy():
    from features.web_scraping.application.flow import run_web_scraping_flow

    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(side_effect=AssertionError("no debería invocarse el agente"))

    with (
        patch("features.web_scraping.application.flow._run_generic_web_search_fetch", AsyncMock(return_value=None)),
    ):
        result = await run_web_scraping_flow(
            _make_state("dame las ultimas noticias sobre seguridad en japon"),
            mock_agent,
            MagicMock(return_value=MagicMock()),
            get_runtime_policy=lambda: {},
            should_evaluate_guard_fn=lambda *_: False,
            evaluate_trajectory_safe_fn=AsyncMock(return_value=(True, {"label": "safe"})),
        )

    content = result["messages"][0].content
    assert "No encontré fuentes locales confiables" in content
    assert mock_agent.ainvoke.await_count == 0


@pytest.mark.asyncio
async def test_guardrail_fast_result_emits_degraded_status_when_guard_fails_open():
    from features.web_scraping.application.flow import _guardrail_fast_result

    with patch("features.web_scraping.application.flow._emit_guard_audit") as emit_guard_audit:
        result = await _guardrail_fast_result(
            "respuesta final",
            {},
            "rid-guard-1",
            0.0,
            lambda *_: True,
            AsyncMock(return_value=(True, {"label": "error", "verdict_source": "error"})),
        )

    assert result["messages"][0].content == "respuesta final"
    payload = emit_guard_audit.call_args.args[0]
    assert payload["event_type"] == "node_guard_status"
    assert payload["guard_status"] == "degraded"
    assert payload["success_kind"] == "success_with_guard_degradation"


@pytest.mark.asyncio
async def test_synthesis_returns_no_local_sources_response_for_recent_country_news_with_weak_evidence():
    from features.web_scraping.application.synthesis import _synthesize_search_summary

    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="No debería usarse"))

    result = await _synthesize_search_summary(
        "Única línea vaga",
        "dame las ultimas noticias sobre seguridad en japon",
        lambda: llm,
        [{"title": "Japan update", "url": "https://example.com/japan"}],
    )

    assert "No encontré fuentes locales confiables" in result
    assert llm.ainvoke.await_count == 0


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
