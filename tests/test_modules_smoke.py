"""
Smoke tests de importación para los módulos nuevos del refactoring.

Verifica que cada módulo:
  1. Se puede importar sin errores
  2. Exporta los símbolos públicos esperados
  3. No introduce imports circulares

No requieren API key ni red — solo comprobación estructural.
"""
import pytest
from typing import cast


# ==================== MÓDULOS HOJA ====================

def test_state_imports():
    from domain.models import AgentName, RoutingDecision, AgentState
    from application.services.agent_registry import AGENT_NAMES
    from typing import get_args
    assert get_args(AgentName) == AGENT_NAMES
    # Reducer append-only presente
    import inspect
    hints = AgentState.__annotations__
    assert "messages" in hints


def test_domain_models_imports():
    from domain.models import AgentName, RoutingDecision, AgentState
    from typing import get_args
    assert "code_agent" in get_args(AgentName)
    assert RoutingDecision(agent="math_agent", reason="test").agent == "math_agent"
    assert "messages" in AgentState.__annotations__


def test_ports_imports():
    from ports import ConfirmationPort, LLMFactory
    from application.policies.hitl_flow import InputConfirmationHandler
    assert ConfirmationPort is not None
    assert LLMFactory is not None
    assert issubclass(InputConfirmationHandler, ConfirmationPort)


def test_config_helpers_imports():
    from application.helpers.config_flow_helpers import TEMPERATURE, get_llm, validate_env
    assert isinstance(TEMPERATURE, float)
    assert callable(get_llm)
    assert callable(validate_env)


def test_application_use_case_imports():
    from application.use_cases.web_scraping_flow import run_web_scraping_flow
    assert callable(run_web_scraping_flow)


def test_web_search_registry_imports():
    from application.services.web_search_registry import (
        build_web_search_provider_lines,
        get_web_search_provider_kind,
        list_web_search_provider_specs,
        resolve_web_search_provider_name,
    )

    assert callable(build_web_search_provider_lines)
    assert callable(get_web_search_provider_kind)
    assert callable(list_web_search_provider_specs)
    assert callable(resolve_web_search_provider_name)


def test_main_searxng_bootstrap_helpers(monkeypatch):
    from application.services import cli_lifecycle

    monkeypatch.delenv("SEARXNG_AUTO_START", raising=False)
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)

    assert cli_lifecycle._searxng_auto_start_enabled() is True
    assert cli_lifecycle._searxng_base_url() == "http://localhost:8888"
    assert cli_lifecycle._searxng_is_local_url("http://localhost:8888") is True
    assert cli_lifecycle._searxng_is_local_url("https://search.example.com") is False

    monkeypatch.setenv("SEARXNG_AUTO_START", "false")
    monkeypatch.setenv("SEARXNG_BASE_URL", "https://search.example.com")

    assert cli_lifecycle._searxng_auto_start_enabled() is False
    assert cli_lifecycle._searxng_base_url() == "https://search.example.com"


def test_tools_package_imports():
    from tools import (
        calculate,
        analyze_data,
        write_code,
        get_crypto_price,
        extract_price_from_text,
        search_web,
        scrape_website_simple,
        scrape_website_dynamic,
        scrape_website_with_json_capture,
        web_fetch,
    )
    assert calculate is not None
    assert analyze_data is not None
    assert write_code is not None
    assert get_crypto_price is not None
    assert extract_price_from_text is not None
    assert search_web is not None
    assert scrape_website_simple is not None
    assert scrape_website_dynamic is not None
    assert scrape_website_with_json_capture is not None
    assert web_fetch is not None


def test_web_scraping_barrel_imports():
    from application.use_cases.web_scraping_api import run_web_scraping_flow, CountryRecentNewsStrategy

    assert callable(run_web_scraping_flow)
    assert CountryRecentNewsStrategy is not None


def test_application_input_guard_use_case_imports():
    from application.use_cases.input_guard_flow import run_input_guard
    assert callable(run_input_guard)


def test_application_finer_grained_use_case_imports():
    from application.use_cases.guard_decision import decide_after_guard as decide_after_guard_direct
    from application.use_cases.routing_decision import decide_agent_route as decide_agent_route_direct
    from application.use_cases.supervisor_chain import build_supervisor_chain as build_supervisor_chain_direct
    from application.use_cases.supervisor_routing import run_supervisor_routing as run_supervisor_routing_direct
    from application.use_cases.supervisor_shortcuts import should_route_to_web_scraping
    assert callable(decide_after_guard_direct)
    assert callable(decide_agent_route_direct)
    assert callable(build_supervisor_chain_direct)
    assert callable(run_supervisor_routing_direct)
    assert callable(should_route_to_web_scraping)


def test_application_security_use_case_imports():
    from application.policies.security_flow import input_guard
    assert callable(input_guard)


def test_application_hitl_use_case_imports():
    from application.policies.hitl_flow import HITL_ENABLED, ask_confirmation, get_confirmation_handler
    assert isinstance(HITL_ENABLED, bool)
    assert callable(ask_confirmation)
    assert callable(get_confirmation_handler)


def test_application_guard_use_case_imports():
    from application.use_cases.guard_decision import decide_after_guard
    assert callable(decide_after_guard)


def test_application_routing_use_case_imports():
    from application.use_cases.routing_decision import decide_agent_route
    assert callable(decide_agent_route)


def test_application_supervisor_use_case_imports():
    from application.use_cases.supervisor_chain import build_supervisor_chain
    from application.use_cases.supervisor_routing import run_supervisor_routing
    assert callable(build_supervisor_chain)
    assert callable(run_supervisor_routing)


def test_application_message_helpers_imports():
    from application.helpers.message_flow_helpers import (
        get_last_message_text,
        is_btc_price_query,
        extract_final_ai_text,
    )
    assert callable(get_last_message_text)
    assert callable(is_btc_price_query)
    assert callable(extract_final_ai_text)


def test_application_trace_helpers_imports():
    from application.helpers.trace_flow_helpers import get_or_create_request_id
    assert callable(get_or_create_request_id)


def test_audit_imports():
    import application.helpers.audit_flow_helpers as audit_helpers
    from domain.model_pricing import MODEL_PRICING as MODEL_PRICING_EXTERNAL
    from application.helpers.text_truncation import truncate_head_tail, truncate_suffix
    assert callable(audit_helpers._emit_guard_audit)
    assert callable(audit_helpers._emit_node_outcome)
    assert isinstance(audit_helpers.MODEL_PRICING, dict)
    assert audit_helpers.MODEL_PRICING is MODEL_PRICING_EXTERNAL
    assert callable(audit_helpers._extract_tokens)
    assert callable(audit_helpers._extract_quality)
    assert callable(audit_helpers._extract_followup)
    assert callable(audit_helpers._node_meta)
    assert callable(audit_helpers._get_model_name)
    assert callable(audit_helpers._truncate_text)
    assert callable(audit_helpers._truncate_raw_response)
    assert callable(truncate_head_tail)
    assert callable(truncate_suffix)


def test_model_pricing_imports():
    from domain.model_pricing import MODEL_PRICING
    assert "gpt-4o-mini" in MODEL_PRICING


def test_persistence_helpers_imports():
    from application.helpers.persistence_flow_helpers import (
        _role_from_msg,
        _row_to_msg,
        _msg_to_jsonl_dict,
        _jsonl_dict_to_msg,
        _load_jsonl,
        _save_jsonl,
    )
    assert callable(_role_from_msg)
    assert callable(_row_to_msg)
    assert callable(_msg_to_jsonl_dict)
    assert callable(_jsonl_dict_to_msg)
    assert callable(_load_jsonl)
    assert callable(_save_jsonl)


def test_scraping_helpers_imports():
    from application.helpers.scraping_flow_helpers import (
        _cache_key,
        _get_cache,
        _set_cache,
        _validate_url,
        _extract_text,
        _extract_links,
        _build_result,
    )
    assert callable(_cache_key)
    assert callable(_get_cache)
    assert callable(_set_cache)
    assert callable(_validate_url)
    assert callable(_extract_text)
    assert callable(_extract_links)
    assert callable(_build_result)


def test_text_truncation_imports():
    from application.helpers.text_truncation import truncate_head_tail, truncate_suffix
    assert truncate_head_tail("abc", max_chars=10, head_chars=5, tail_chars=3) == "abc"
    assert truncate_suffix("abc", max_chars=10) == "abc"


def test_scrape_tracker_imports():
    from application.policies.scrape_tracker import (
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
    from application.policies.security_flow import input_guard
    from application.policies.hitl_flow import (
        HITL_ENABLED, ask_confirmation, get_confirmation_handler,
    )
    from application.helpers.security_flow_helpers import (
        _BLOCKED_PATTERNS, _RISK_SIGNALS,
        get_blocked_patterns, get_risk_signals, get_human_history,
    )
    from application.policies.hitl_flow import ConfirmationHandler, InputConfirmationHandler
    assert callable(input_guard)
    assert len(_BLOCKED_PATTERNS) >= 6
    assert len(_RISK_SIGNALS) >= 6
    assert isinstance(HITL_ENABLED, bool)
    assert callable(ask_confirmation)
    assert callable(get_blocked_patterns)
    assert callable(get_risk_signals)
    assert callable(get_confirmation_handler)
    assert callable(get_human_history)
    assert ConfirmationHandler is not None
    assert InputConfirmationHandler is not None


def test_agentdog_imports():
    from application.policies.agentdog import (
        HIGH_RISK_NODES, is_high_risk, evaluate_trajectory_safe,
        build_trajectory_from_messages, _resolve_guard_policy,
        _should_evaluate_guard, _flatten_messages_text,
    )
    assert HIGH_RISK_NODES == frozenset({"code_node", "web_scraping_node"})
    assert is_high_risk("code_node") is True
    assert is_high_risk("math_node") is False
    assert callable(evaluate_trajectory_safe)


def test_price_helpers_imports():
    from application.helpers.price_flow_helpers import (
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
        make_generic_agent_node,
    )
    assert callable(make_math_node)
    assert callable(make_analysis_node)
    assert callable(make_code_node)
    assert callable(make_web_scraping_node)
    assert callable(make_generic_agent_node)


def test_hitl_imports():
    from application.policies.hitl_flow import ConfirmationHandler, InputConfirmationHandler, DEFAULT_CONFIRMATION_HANDLER
    assert ConfirmationHandler is not None
    assert isinstance(DEFAULT_CONFIRMATION_HANDLER, InputConfirmationHandler)


def test_agent_registry_imports():
    from application.services.agent_registry import (
        AGENT_NAMES,
        AgentSpec,
        build_supervisor_agent_lines,
        get_agent_spec,
        get_agent_specs,
        get_agent_temperature,
        get_registered_nodes,
    )
    assert AGENT_NAMES == (
        "math_agent", "analysis_agent", "code_agent", "web_scraping_agent"
    )
    assert AgentSpec is not None
    assert callable(build_supervisor_agent_lines)
    assert get_agent_spec("code_agent").risk_level == "high"
    assert get_agent_temperature("code_agent") == 0.0
    assert len(get_agent_specs()) == len(AGENT_NAMES)
    assert set(get_registered_nodes()) == set(AGENT_NAMES)


def test_tool_registry_imports():
    from application.services.tool_registry import (
        ToolSpec,
        build_agent_tool_lines,
        build_agent_permission_lines,
        build_tool_catalog_lines,
        get_tool_spec,
        get_tools_for_agent,
        list_tool_specs,
    )
    assert ToolSpec is not None
    assert callable(build_agent_tool_lines)
    assert callable(build_agent_permission_lines)
    assert callable(build_tool_catalog_lines)
    assert get_tool_spec("calculate").name == "calculate"
    assert len(get_tools_for_agent("math_agent")) == 1
    assert len(list_tool_specs()) >= 4


def test_tool_permissions_imports():
    from application.policies.tool_permissions import (
        ToolPermissionDecision,
        decide_tool_permission,
        get_tool_permission_mode,
    )
    assert ToolPermissionDecision is not None
    assert callable(decide_tool_permission)
    assert callable(get_tool_permission_mode)


def test_tool_execution_imports():
    from application.services.tool_execution import (
        ToolExecutionContext,
        ToolExecutionResult,
        execute_registered_tool,
    )
    assert ToolExecutionContext is not None
    assert ToolExecutionResult is not None
    assert callable(execute_registered_tool)


def test_tool_audit_imports():
    from application.services.tool_audit import ToolAuditService, ToolCallAuditEvent, tool_audit_service
    from application.services.tool_audit_store import ToolAuditStore, tool_audit_store
    assert ToolAuditService is not None
    assert ToolCallAuditEvent is not None
    assert tool_audit_service is not None
    assert ToolAuditStore is not None
    assert tool_audit_store is not None


def test_background_task_imports():
    from application.services.background_tasks import BackgroundTaskRecord, BackgroundTaskService, BackgroundTaskState, BackgroundTaskStore, BackgroundTaskSummary, background_task_service, background_task_store
    assert BackgroundTaskRecord is not None
    assert BackgroundTaskService is not None
    assert BackgroundTaskState is not None
    assert BackgroundTaskSummary is not None
    assert BackgroundTaskStore is not None
    assert background_task_service is not None
    assert background_task_store is not None


def test_command_registry_imports():
    from application.services.command_registry import COMMAND_REGISTRY, SlashCommandRegistry, SlashCommandSpec
    assert COMMAND_REGISTRY is not None
    assert SlashCommandRegistry is not None
    assert SlashCommandSpec is not None


def test_tool_impact_imports():
    from application.services.tool_impact import ToolImpactPreview, ToolImpactService, tool_impact_service
    assert ToolImpactPreview is not None
    assert ToolImpactService is not None
    assert tool_impact_service is not None


def test_context_budget_imports():
    from application.services.context_budget import ContextBudgetItem, SessionContextBudget, SessionContextBudgetService, context_budget_service
    assert ContextBudgetItem is not None
    assert SessionContextBudget is not None
    assert SessionContextBudgetService is not None
    assert context_budget_service is not None


def test_session_bookmark_imports():
    from application.services.session_bookmarks import SessionBookmark, SessionBookmarkService, SessionBookmarkStore, session_bookmark_service, session_bookmark_store
    assert SessionBookmark is not None
    assert SessionBookmarkService is not None
    assert SessionBookmarkStore is not None
    assert session_bookmark_service is not None
    assert session_bookmark_store is not None


def test_runtime_session_closure_imports():
    from application.services.runtime import RuntimeSessionClosure, RuntimeSessionResolution, RuntimeSessionView, SessionLifecycle
    assert RuntimeSessionClosure is not None
    assert RuntimeSessionResolution is not None
    assert RuntimeSessionView is not None
    assert SessionLifecycle is not None


def test_session_persistence_imports():
    from application.services.session_persistence import SessionPersistence, persistence
    assert SessionPersistence is not None
    assert persistence is not None


def test_session_memory_imports():
    from application.services.session_memory import SessionMemory, memory
    assert SessionMemory is not None
    assert memory is not None


def test_memory_retrieval_imports():
    from application.services.memory_retrieval import MemorySearchHit, MemoryRetrievalService, memory_retrieval_service
    assert MemorySearchHit is not None
    assert MemoryRetrievalService is not None
    assert memory_retrieval_service is not None


def test_tool_approval_imports():
    from application.services.tool_approval import ToolApprovalPreview, ToolApprovalService, tool_approval_service
    assert ToolApprovalPreview is not None
    assert ToolApprovalService is not None
    assert tool_approval_service is not None


def test_session_artifacts_imports():
    from application.services.session_artifacts import SessionArtifact, SessionArtifactService, SessionArtifactStore, session_artifact_service, session_artifact_store
    assert SessionArtifact is not None
    assert SessionArtifactService is not None
    assert SessionArtifactStore is not None
    assert session_artifact_service is not None
    assert session_artifact_store is not None


def test_session_inspection_imports():
    from application.services.session_inspection import format_background_task_state, format_background_task_summary, format_bookmark_detail, format_bookmark_list, format_command_detail, format_command_registry, format_context_budget, format_inspection_help, format_memory_search_results, format_prompt_snapshot, format_prompt_snapshot_list, format_replay_timeline, format_session_artifact, format_tool_approval_preview, format_tool_catalog, format_tool_impact_preview
    assert callable(format_background_task_state)
    assert callable(format_background_task_summary)
    assert callable(format_bookmark_detail)
    assert callable(format_bookmark_list)
    assert callable(format_command_detail)
    assert callable(format_command_registry)
    assert callable(format_context_budget)
    assert callable(format_inspection_help)
    assert callable(format_prompt_snapshot)
    assert callable(format_prompt_snapshot_list)
    assert callable(format_replay_timeline)
    assert callable(format_memory_search_results)
    assert callable(format_tool_approval_preview)
    assert callable(format_tool_impact_preview)
    assert callable(format_tool_catalog)
    assert callable(format_session_artifact)


def test_prompt_versioning_imports():
    from application.services.prompt_versioning import PromptSnapshot, PromptSnapshotStore, PromptVersionService, prompt_snapshot_store, prompt_version_service
    assert PromptSnapshot is not None
    assert PromptSnapshotStore is not None
    assert PromptVersionService is not None
    assert prompt_snapshot_store is not None
    assert prompt_version_service is not None


def test_session_replay_imports():
    from application.services.session_replay import ReplayTimelineItem, SessionReplay, SessionReplayService, format_session_replay, session_replay_service
    assert ReplayTimelineItem is not None
    assert SessionReplay is not None
    assert SessionReplayService is not None
    assert callable(format_session_replay)
    assert session_replay_service is not None


def test_prompt_assembly_imports():
    from application.services.prompt_assembly import AgentPromptAssembly, build_agent_prompt_assembly, build_agent_prompt_extra_context
    from application.services.prompt_loader import load_agent_prompt
    from application.services.prompt_versioning import PromptSnapshot, PromptSnapshotStore, PromptVersionService, prompt_snapshot_store, prompt_version_service
    assert AgentPromptAssembly is not None
    assert callable(build_agent_prompt_assembly)
    assert callable(build_agent_prompt_extra_context)
    assert callable(load_agent_prompt)
    assert PromptSnapshot is not None
    assert PromptSnapshotStore is not None
    assert PromptVersionService is not None
    assert prompt_snapshot_store is not None
    assert prompt_version_service is not None


def test_trace_context_imports():
    from application.services.trace_context import TraceContext, TraceContextService, trace_context_service
    assert TraceContext is not None
    assert TraceContextService is not None
    assert trace_context_service is not None


def test_supervisor_prompt_imports():
    from application.services.supervisor_prompt import SupervisorPromptAssembly, build_supervisor_prompt_assembly
    assert SupervisorPromptAssembly is not None
    assert callable(build_supervisor_prompt_assembly)


def test_coordinator_mode_imports():
    from application.services.coordinator_mode import (
        COORDINATOR_MODE_ENV,
        CoordinatorModeContract,
        CoordinatorPromptAssembly,
        build_coordinator_mode_contract,
        build_coordinator_prompt_assembly,
        build_coordinator_worker_lines,
        is_coordinator_mode_enabled,
    )

    assert COORDINATOR_MODE_ENV == "COORDINATOR_MODE"
    assert CoordinatorModeContract is not None
    assert CoordinatorPromptAssembly is not None
    assert callable(build_coordinator_mode_contract)
    assert callable(build_coordinator_prompt_assembly)
    assert callable(build_coordinator_worker_lines)
    assert callable(is_coordinator_mode_enabled)


def test_security_runtime_overrides(monkeypatch):
    from application.policies.security_flow import input_guard
    from application.helpers.security_flow_helpers import get_blocked_patterns, get_risk_signals
    from application.helpers.security_flow_helpers import get_human_history
    from langchain_core.messages import HumanMessage

    monkeypatch.setenv("SECURITY_BLOCKED_PATTERNS", "custom block phrase")
    monkeypatch.setenv("SECURITY_RISK_SIGNALS", "custom risk phrase")

    assert "custom block phrase" in get_blocked_patterns()
    assert "custom risk phrase" in get_risk_signals()

    state = {
        "messages": [HumanMessage(content="please custom block phrase now")],
        "next_agent": "",
        "risk_flag": False,
        "blocked": False,
    }
    result = input_guard(state)
    assert result is not None
    assert result.get("blocked") is True
    assert callable(get_human_history)


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
    from application.composition.graph import (
        create_supervisor_graph, route_agent,
        supervisor_node, input_guard_node, route_after_guard,
    )
    assert callable(create_supervisor_graph)
    assert callable(route_agent)


def test_graph_compiles():
    """create_supervisor_graph() debe compilar sin necesitar API key."""
    from application.composition.graph import create_supervisor_graph
    graph = create_supervisor_graph()
    assert graph is not None


def test_route_agent_logic():
    from application.composition.graph import route_agent
    from langgraph.graph import END
    from langchain_core.messages import HumanMessage
    from domain.models import AgentState

    base = cast(AgentState, {
        "messages": [HumanMessage(content="test")],
        "next_agent": "",
        "risk_flag": False,
        "blocked": False,
        "request_id": "test-request-id",
        "scrape_tracker": {},
    })
    assert route_agent({**base, "next_agent": "math_agent"})         == "math_agent"
    assert route_agent({**base, "next_agent": "analysis_agent"})     == "analysis_agent"
    assert route_agent({**base, "next_agent": "code_agent"})         == "code_agent"
    assert route_agent({**base, "next_agent": "web_scraping_agent"}) == "web_scraping_agent"
    assert route_agent({**base, "next_agent": ""})                   == "supervisor"
    assert route_agent({**base, "messages": []})                      == END




# ==================== INVARIANTES CRÍTICOS ====================

def test_inv1_messages_reducer_is_append_only():
    """INV-1: AgentState.messages reducer MUST be lambda x, y: x + y."""
    import inspect
    from domain.models import AgentState
    source = inspect.getsource(AgentState)
    assert "lambda x, y: x + y" in source, "Reducer append-only no encontrado en AgentState"


def test_inv4_high_risk_nodes_immutable():
    """INV-4: HIGH_RISK_NODES debe ser frozenset y contener exactamente los 2 nodos."""
    from application.policies.agentdog import HIGH_RISK_NODES
    assert isinstance(HIGH_RISK_NODES, frozenset)
    assert HIGH_RISK_NODES == frozenset({"code_node", "web_scraping_node"})


def test_inv10_eval_only_in_agents():
    """INV-10: eval() no debe aparecer en ningún módulo nuevo del refactoring."""
    import ast, pathlib
    new_modules = [
        "domain/models.py", "application/policies/scrape_tracker.py",
        "application/policies/agentdog.py",
        "application/helpers/price_flow_helpers.py",
        "application/helpers/security_flow_helpers.py",
        "application/helpers/audit_flow_helpers.py",
        "application/helpers/config_flow_helpers.py",
        "application/policies/hitl_flow.py",
        "application/policies/security_flow.py",
        "application/helpers/persistence_flow_helpers.py",
        "application/helpers/scraping_flow_helpers.py",
        "application/composition/graph.py", "application/services/agents_factory.py",
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
