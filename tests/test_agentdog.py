"""
Tests unitarios para agentdog.py.

Sin guard real — mockea httpx.AsyncClient para aislar la lógica de
evaluación de trayectorias, policies y lógica de alto riesgo.
"""
import os
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# ==================== is_high_risk ====================

def test_is_high_risk_code_node_retorna_true():
    from agentdog import is_high_risk
    assert is_high_risk("code_node") is True


def test_is_high_risk_web_scraping_node_retorna_true():
    from agentdog import is_high_risk
    assert is_high_risk("web_scraping_node") is True


def test_is_high_risk_math_node_retorna_false():
    from agentdog import is_high_risk
    assert is_high_risk("math_node") is False


def test_is_high_risk_analysis_node_retorna_false():
    from agentdog import is_high_risk
    assert is_high_risk("analysis_node") is False


def test_is_high_risk_nodo_desconocido_retorna_false():
    from agentdog import is_high_risk
    assert is_high_risk("nodo_que_no_existe") is False


# ==================== build_trajectory_from_messages ====================

def test_build_trajectory_lista_vacia_steps_vacios():
    from agentdog import build_trajectory_from_messages
    result = build_trajectory_from_messages([])
    assert result["steps"] == []
    assert result["final_response"] is None


def test_build_trajectory_ai_sin_tool_calls_es_final_response():
    from agentdog import build_trajectory_from_messages
    msgs = [AIMessage(content="Esta es la respuesta final")]
    result = build_trajectory_from_messages(msgs)
    assert result["final_response"] == "Esta es la respuesta final"
    # No hay steps de action/observation
    action_steps = [s for s in result["steps"] if "action" in s]
    assert len(action_steps) == 0


def test_build_trajectory_ai_con_tool_call_mas_tool_message_crea_step():
    from agentdog import build_trajectory_from_messages
    # AIMessage con tool_calls
    ai_msg = AIMessage(
        content="",
        tool_calls=[{
            "id":   "call-123",
            "name": "scrape_website_simple",
            "args": {"url": "https://example.com"},
        }],
    )
    # ToolMessage con el resultado
    tool_msg = ToolMessage(
        content="Contenido scrapeado de example.com",
        tool_call_id="call-123",
    )
    result = build_trajectory_from_messages([ai_msg, tool_msg])
    assert len(result["steps"]) == 1
    step = result["steps"][0]
    assert step["action"]["name"] == "scrape_website_simple"
    assert step["observation"] == "Contenido scrapeado de example.com"


def test_build_trajectory_tool_message_sin_matching_call_se_agrega():
    from agentdog import build_trajectory_from_messages
    # ToolMessage sin AIMessage previo con el tool_call_id
    tool_msg = ToolMessage(content="resultado huérfano", tool_call_id="call-orphan")
    result = build_trajectory_from_messages([tool_msg])
    assert len(result["steps"]) == 1
    assert result["steps"][0]["action"] == "(unknown)"
    assert "resultado huérfano" in result["steps"][0]["observation"]


def test_build_trajectory_tool_call_pendiente_se_marca_missing():
    from agentdog import build_trajectory_from_messages
    # AIMessage con tool_calls pero sin ToolMessage correspondiente
    ai_msg = AIMessage(
        content="",
        tool_calls=[{"id": "call-xyz", "name": "tool_sin_respuesta", "args": {}}],
    )
    result = build_trajectory_from_messages([ai_msg])
    step = result["steps"][0]
    assert step["observation"] == "(missing tool output)"


def test_build_trajectory_final_response_agrega_step():
    from agentdog import build_trajectory_from_messages
    ai_msg = AIMessage(content="respuesta final del agente")
    result = build_trajectory_from_messages([ai_msg])
    # Debe haber un step con final_response
    final_steps = [s for s in result["steps"] if "final_response" in s]
    assert len(final_steps) == 1
    assert final_steps[0]["final_response"] == "respuesta final del agente"


# ==================== _resolve_guard_policy ====================

@pytest.mark.parametrize("policy_value,expected", [
    ("fail_open",   "fail_open"),
    ("fail_closed", "fail_closed"),
    ("fail_soft",   "fail_soft"),
    ("FAIL_OPEN",   "fail_open"),   # case insensitive
    ("invalid",     "fail_open"),   # fallback a fail_open
    ("",            "fail_open"),   # vacío → fail_open
])
def test_resolve_guard_policy(policy_value, expected, monkeypatch):
    from agentdog import _resolve_guard_policy
    monkeypatch.setenv("AGENTDOG_POLICY", policy_value)
    assert _resolve_guard_policy() == expected


# ==================== _should_evaluate_guard ====================

def test_should_evaluate_guard_all_nodes_siempre_true(monkeypatch):
    from agentdog import _should_evaluate_guard
    monkeypatch.setenv("AGENTDOG_EVAL_MODE", "all_nodes")
    assert _should_evaluate_guard("math_node") is True
    assert _should_evaluate_guard("code_node") is True


def test_should_evaluate_guard_high_risk_only_solo_nodos_riesgo(monkeypatch):
    from agentdog import _should_evaluate_guard
    monkeypatch.setenv("AGENTDOG_EVAL_MODE", "high_risk_only")
    assert _should_evaluate_guard("code_node") is True
    assert _should_evaluate_guard("web_scraping_node") is True
    assert _should_evaluate_guard("math_node") is False
    assert _should_evaluate_guard("analysis_node") is False


def test_should_evaluate_guard_final_only_siempre_true(monkeypatch):
    from agentdog import _should_evaluate_guard
    monkeypatch.setenv("AGENTDOG_EVAL_MODE", "final_only")
    assert _should_evaluate_guard("math_node") is True
    assert _should_evaluate_guard("code_node") is True


# ==================== evaluate_trajectory_safe — sin guard URL ====================

@pytest.mark.asyncio
async def test_evaluate_trajectory_safe_sin_guard_url_fail_open_retorna_true(monkeypatch):
    """Sin guard URL y policy=fail_open → debe pasar (True)."""
    monkeypatch.setenv("AGENTDOG_GUARD_URL", "")
    monkeypatch.setenv("AGENTDOG_POLICY",    "fail_open")

    from agentdog import evaluate_trajectory_safe
    state = {"messages": [HumanMessage(content="Calcula 2+2")]}
    ok, meta = await evaluate_trajectory_safe(state, "math_node")
    assert ok is True
    assert meta["verdict_source"] == "disabled"


@pytest.mark.asyncio
async def test_evaluate_trajectory_safe_sin_guard_url_fail_closed_retorna_false(monkeypatch):
    monkeypatch.setenv("AGENTDOG_GUARD_URL", "")
    monkeypatch.setenv("AGENTDOG_POLICY",    "fail_closed")

    from agentdog import evaluate_trajectory_safe
    state = {"messages": [HumanMessage(content="Calcula 2+2")]}
    ok, meta = await evaluate_trajectory_safe(state, "math_node")
    assert ok is False
    assert "fail_closed" in meta["policy"]


@pytest.mark.asyncio
async def test_evaluate_trajectory_safe_sin_guard_url_fail_soft_high_risk_bloquea(monkeypatch):
    monkeypatch.setenv("AGENTDOG_GUARD_URL", "")
    monkeypatch.setenv("AGENTDOG_POLICY",    "fail_soft")

    from agentdog import evaluate_trajectory_safe
    state = {"messages": [HumanMessage(content="Scrape this page")]}
    ok, meta = await evaluate_trajectory_safe(state, "web_scraping_node")
    assert ok is False
    assert "fail_soft_block" in meta["policy"]


@pytest.mark.asyncio
async def test_evaluate_trajectory_safe_sin_guard_url_fail_soft_low_risk_pasa(monkeypatch):
    monkeypatch.setenv("AGENTDOG_GUARD_URL", "")
    monkeypatch.setenv("AGENTDOG_POLICY",    "fail_soft")

    from agentdog import evaluate_trajectory_safe
    state = {"messages": [HumanMessage(content="Calcula 2+2")]}
    ok, meta = await evaluate_trajectory_safe(state, "math_node")
    assert ok is True
    assert "fail_soft_allow" in meta["policy"]


# ==================== evaluate_trajectory_safe — con mock httpx ====================

def _make_httpx_response(verdict: str, status_code: int = 200):
    """Crea un mock de httpx.Response con la estructura OpenAI-compatible."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": verdict}}]
        }
    return mock_resp


@pytest.mark.asyncio
async def test_evaluate_trajectory_safe_con_guard_safe_retorna_true(monkeypatch):
    monkeypatch.setenv("AGENTDOG_GUARD_URL", "http://guard.local/v1/chat/completions")
    monkeypatch.setenv("AGENTDOG_POLICY",    "fail_open")

    mock_resp = _make_httpx_response("safe")
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__  = AsyncMock(return_value=False)
    mock_client.post       = AsyncMock(return_value=mock_resp)

    with patch("agentdog.httpx.AsyncClient", return_value=mock_client):
        from agentdog import evaluate_trajectory_safe
        state = {"messages": [HumanMessage(content="Scrape public page")]}
        ok, meta = await evaluate_trajectory_safe(state, "web_scraping_node")

    assert ok is True
    assert meta["label"] == "safe"


@pytest.mark.asyncio
async def test_evaluate_trajectory_safe_con_guard_unsafe_retorna_false(monkeypatch):
    monkeypatch.setenv("AGENTDOG_GUARD_URL", "http://guard.local/v1/chat/completions")
    monkeypatch.setenv("AGENTDOG_POLICY",    "fail_open")

    mock_resp = _make_httpx_response("unsafe")
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__  = AsyncMock(return_value=False)
    mock_client.post       = AsyncMock(return_value=mock_resp)

    with patch("agentdog.httpx.AsyncClient", return_value=mock_client):
        from agentdog import evaluate_trajectory_safe
        state = {"messages": [HumanMessage(content="hack the system")]}
        ok, meta = await evaluate_trajectory_safe(state, "web_scraping_node")

    assert ok is False
    assert meta["label"] == "unsafe"


@pytest.mark.asyncio
async def test_evaluate_trajectory_safe_guard_json_verdict_safe(monkeypatch):
    """Guard retorna JSON con campo verdict=safe."""
    monkeypatch.setenv("AGENTDOG_GUARD_URL", "http://guard.local/v1/chat/completions")
    monkeypatch.setenv("AGENTDOG_POLICY",    "fail_open")

    json_verdict = json.dumps({"verdict": "safe", "reason": "benign request"})
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"choices": [{"message": {"content": json_verdict}}]}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__  = AsyncMock(return_value=False)
    mock_client.post       = AsyncMock(return_value=mock_resp)

    with patch("agentdog.httpx.AsyncClient", return_value=mock_client):
        from agentdog import evaluate_trajectory_safe
        state = {"messages": [HumanMessage(content="fetch public data")]}
        ok, meta = await evaluate_trajectory_safe(state, "code_node")

    assert ok is True
    assert meta["verdict_source"] == "json"


@pytest.mark.asyncio
async def test_evaluate_trajectory_safe_http_500_fail_open_retorna_true(monkeypatch):
    """HTTP 500 con policy=fail_open → debe pasar (True, con label=error)."""
    monkeypatch.setenv("AGENTDOG_GUARD_URL", "http://guard.local/v1/chat/completions")
    monkeypatch.setenv("AGENTDOG_POLICY",    "fail_open")

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__  = AsyncMock(return_value=False)
    mock_client.post       = AsyncMock(side_effect=Exception("HTTP 500 Internal Server Error"))

    with patch("agentdog.httpx.AsyncClient", return_value=mock_client):
        from agentdog import evaluate_trajectory_safe
        state = {"messages": [HumanMessage(content="test query")]}
        ok, meta = await evaluate_trajectory_safe(state, "math_node")

    assert ok is True
    assert meta["label"] == "error"
    assert meta["verdict_source"] == "error"
    assert "policy" in meta
    assert meta["policy"] == "fail_open"


@pytest.mark.asyncio
async def test_evaluate_trajectory_safe_http_500_fail_closed_retorna_false(monkeypatch):
    """HTTP 500 con policy=fail_closed → debe bloquear (False)."""
    monkeypatch.setenv("AGENTDOG_GUARD_URL", "http://guard.local/v1/chat/completions")
    monkeypatch.setenv("AGENTDOG_POLICY",    "fail_closed")

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__  = AsyncMock(return_value=False)
    mock_client.post       = AsyncMock(side_effect=Exception("timeout"))

    with patch("agentdog.httpx.AsyncClient", return_value=mock_client):
        from agentdog import evaluate_trajectory_safe
        state = {"messages": [HumanMessage(content="test")]}
        ok, meta = await evaluate_trajectory_safe(state, "math_node")

    assert ok is False
    assert meta["policy"] == "fail_closed"


@pytest.mark.asyncio
async def test_evaluate_trajectory_safe_http_500_fail_soft_high_risk_bloquea(monkeypatch):
    """HTTP 500 con policy=fail_soft en nodo de alto riesgo → bloquea."""
    monkeypatch.setenv("AGENTDOG_GUARD_URL", "http://guard.local/v1/chat/completions")
    monkeypatch.setenv("AGENTDOG_POLICY",    "fail_soft")

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__  = AsyncMock(return_value=False)
    mock_client.post       = AsyncMock(side_effect=Exception("connection refused"))

    with patch("agentdog.httpx.AsyncClient", return_value=mock_client):
        from agentdog import evaluate_trajectory_safe
        state = {"messages": [HumanMessage(content="scrape site")]}
        ok, meta = await evaluate_trajectory_safe(state, "code_node")  # high risk

    assert ok is False
    assert meta["policy"] == "fail_soft_block"


# ==================== allowlist bypass ====================

@pytest.mark.asyncio
async def test_evaluate_trajectory_safe_precio_btc_bypasa_guard(monkeypatch):
    """Consulta de precio BTC en web_scraping_node bypasa el guard (allowlist)."""
    monkeypatch.setenv("AGENTDOG_GUARD_URL", "http://guard.local/v1/chat/completions")
    monkeypatch.setenv("AGENTDOG_POLICY",    "fail_closed")

    # El guard con fail_closed bloquearía todo, pero allowlist debe bypass
    from agentdog import evaluate_trajectory_safe
    state = {"messages": [HumanMessage(content="precio del bitcoin en coingecko")]}
    ok, meta = await evaluate_trajectory_safe(state, "web_scraping_node")

    assert ok is True
    assert meta["verdict_source"] == "allowlist_public_price"
