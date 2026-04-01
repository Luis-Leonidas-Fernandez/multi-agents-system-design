"""
Smoke tests de importación para los módulos nuevos del refactoring.

Verifica que cada módulo:
  1. Se puede importar sin errores
  2. Exporta los símbolos públicos esperados
  3. No introduce imports circulares

No requieren API key ni red — solo comprobación estructural.
"""
import pytest


# ==================== MÓDULOS HOJA ====================

def test_state_imports():
    from state import AgentName, RoutingDecision, AgentState
    from typing import get_args
    assert set(get_args(AgentName)) == {
        "math_agent", "analysis_agent", "code_agent", "web_scraping_agent"
    }
    # Reducer append-only presente
    import inspect
    hints = AgentState.__annotations__
    assert "messages" in hints


def test_audit_imports():
    from audit import (
        _emit_guard_audit, _emit_node_outcome,
        _extract_tokens, _extract_quality, _extract_followup,
        _node_meta, _get_model_name,
        _truncate_text, _truncate_raw_response,
        MODEL_PRICING,
    )
    assert callable(_emit_guard_audit)
    assert callable(_emit_node_outcome)
    assert isinstance(MODEL_PRICING, dict)


def test_scrape_tracker_imports():
    from scrape_tracker import (
        get_runtime_policy, reset_runtime_policy_cache,
        _update_scrape_tracker, _get_strategy, _get_category_score,
        _detect_query_category, _scrape_reliability,
        _STRATEGY_HINTS, _RETRY_ON_RELIABILITY,
        _STRUCTURED_SOURCE_STRATEGIES, _API_VALIDATION_EPSILON,
    )
    assert callable(get_runtime_policy)
    assert isinstance(_STRATEGY_HINTS, dict)
    assert isinstance(_RETRY_ON_RELIABILITY, frozenset)
    assert 0.0 < _API_VALIDATION_EPSILON < 0.1


# ==================== MÓDULOS INTERMEDIOS ====================

def test_security_imports():
    from security import (
        input_guard, _ask_confirmation, _HITL_ENABLED,
        _get_human_history, _BLOCKED_PATTERNS, _RISK_SIGNALS,
        _extract_msg_text, _check_patterns,
    )
    assert callable(input_guard)
    assert len(_BLOCKED_PATTERNS) >= 6
    assert len(_RISK_SIGNALS) >= 6
    assert isinstance(_HITL_ENABLED, bool)


def test_agentdog_imports():
    from agentdog import (
        HIGH_RISK_NODES, is_high_risk, evaluate_trajectory_safe,
        build_trajectory_from_messages, _resolve_guard_policy,
        _should_evaluate_guard, _flatten_messages_text,
    )
    assert HIGH_RISK_NODES == frozenset({"code_node", "web_scraping_node"})
    assert is_high_risk("code_node") is True
    assert is_high_risk("math_node") is False
    assert callable(evaluate_trajectory_safe)


def test_price_helpers_imports():
    from price_helpers import (
        _extract_structured_price, _extract_price_from_messages,
        _detect_coin_from_query, _format_price_response, _QUERY_COIN_MAP,
    )
    assert _detect_coin_from_query("precio del bitcoin") == "bitcoin"
    assert _detect_coin_from_query("eth price") == "ethereum"
    assert _detect_coin_from_query("completely unrelated query") == "bitcoin"  # default


# ==================== NODOS ====================

def test_nodes_package_imports():
    from nodes import (
        make_math_node, make_analysis_node,
        make_code_node, make_web_scraping_node,
    )
    assert callable(make_math_node)
    assert callable(make_analysis_node)
    assert callable(make_code_node)
    assert callable(make_web_scraping_node)


def test_nodes_factories_return_callables():
    """Cada factory debe retornar un callable, no ejecutar nada."""
    from unittest.mock import MagicMock
    from nodes import (
        make_math_node, make_analysis_node,
        make_code_node, make_web_scraping_node,
    )
    mock_agent = MagicMock()
    mock_llm_fn = MagicMock()

    math_fn         = make_math_node(mock_agent)
    analysis_fn     = make_analysis_node(mock_agent)
    code_fn         = make_code_node(mock_agent)
    web_fn          = make_web_scraping_node(mock_agent, mock_llm_fn)

    import asyncio
    assert asyncio.iscoroutinefunction(math_fn)
    assert asyncio.iscoroutinefunction(analysis_fn)
    assert asyncio.iscoroutinefunction(code_fn)
    assert asyncio.iscoroutinefunction(web_fn)


# ==================== GRAFO ====================

def test_graph_imports():
    from graph import (
        create_supervisor_graph, route_agent,
        supervisor_node, input_guard_node, route_after_guard,
    )
    assert callable(create_supervisor_graph)
    assert callable(route_agent)


def test_graph_compiles():
    """create_supervisor_graph() debe compilar sin necesitar API key."""
    from graph import create_supervisor_graph
    graph = create_supervisor_graph()
    assert graph is not None


def test_route_agent_logic():
    from graph import route_agent
    from langgraph.graph import END
    from langchain_core.messages import HumanMessage

    base = {"messages": [HumanMessage(content="test")]}
    assert route_agent({**base, "next_agent": "math_agent"})         == "math_agent"
    assert route_agent({**base, "next_agent": "analysis_agent"})     == "analysis_agent"
    assert route_agent({**base, "next_agent": "code_agent"})         == "code_agent"
    assert route_agent({**base, "next_agent": "web_scraping_agent"}) == "web_scraping_agent"
    assert route_agent({**base, "next_agent": ""})                   == "supervisor"
    assert route_agent({"messages": [], "next_agent": ""})           == END




# ==================== INVARIANTES CRÍTICOS ====================

def test_inv1_messages_reducer_is_append_only():
    """INV-1: AgentState.messages reducer MUST be lambda x, y: x + y."""
    import inspect
    from state import AgentState
    source = inspect.getsource(AgentState)
    assert "lambda x, y: x + y" in source, "Reducer append-only no encontrado en AgentState"


def test_inv4_high_risk_nodes_immutable():
    """INV-4: HIGH_RISK_NODES debe ser frozenset y contener exactamente los 2 nodos."""
    from agentdog import HIGH_RISK_NODES
    assert isinstance(HIGH_RISK_NODES, frozenset)
    assert HIGH_RISK_NODES == frozenset({"code_node", "web_scraping_node"})


def test_inv10_eval_only_in_agents():
    """INV-10: eval() no debe aparecer en ningún módulo nuevo del refactoring."""
    import ast, pathlib
    new_modules = [
        "state.py", "audit.py", "scrape_tracker.py",
        "security.py", "agentdog.py", "price_helpers.py",
        "graph.py", "agents.py",
    ]
    root = pathlib.Path(__file__).parent.parent
    for fname in new_modules:
        source = (root / fname).read_text()
        tree   = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else None
                assert name != "eval", f"eval() encontrado en {fname}"
