"""Caso de uso para el flujo de web scraping.

Coordina HITL, estrategia, guardrails, retry y postcondiciones.
El nodo LangGraph queda como adaptador fino.
"""
import asyncio
import re
import time
import uuid
from urllib.parse import urlparse
from typing import Any, Optional, Callable, Awaitable, Mapping, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig

from application.policies.agentdog import evaluate_trajectory_safe, _should_evaluate_guard, _is_allowed_public_price_request
from application.helpers.audit_flow_helpers import (
    _emit_node_outcome,
    _extract_tokens,
    _extract_quality,
    _extract_followup,
    _node_meta,
    _get_model_name,
)
from ports.confirmation_port import ConfirmationPort
from application.helpers.message_flow_helpers import extract_final_ai_text, get_last_message_text, is_web_information_query
from application.helpers.trace_flow_helpers import get_or_create_request_id
from application.policies.scrape_tracker import (
    _get_category_score,
    _update_scrape_tracker,
    _STRUCTURED_SOURCE_STRATEGIES,
    _RETRY_ON_RELIABILITY,
    _scrape_reliability,
)
from application.policies.web_source_policy import (
    detect_query_source_group,
    detect_recent_query_horizon,
    get_query_source_terms,
    get_recent_query_requirements,
    score_domain_boost,
)
from tools.web_tools import _is_specific_article_hit
from application.helpers.price_flow_helpers import (
    _detect_coin_from_query,
    _format_price_response,
    _extract_price_from_messages,
    _extract_structured_price,
    _get_crypto_price_fn,
)
from domain.models import AgentState


def _extract_urls_from_text(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)\]]+", text or "")
    cleaned: list[str] = []
    seen: set[str] = set()
    for url in urls:
        normalized = url.rstrip(".,;:")
        if normalized and normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)
    return cleaned


def _clean_source_url(url: str) -> str:
    """Strip CITE_THIS artifacts (|domain=xxx>>>) from URLs."""
    return url.split("|")[0].rstrip(">").strip() if url else url


def _format_sources(sources: list[dict[str, str]]) -> str:
    if not sources:
        return ""
    lines = ["Sources:"]
    seen_domains: set[str] = set()
    for source in sources:
        raw_url = source.get("url") or ""
        url = _clean_source_url(raw_url)
        if not url:
            continue
        domain = source.get("domain") or (urlparse(url).hostname or "").replace("www.", "")
        if domain and domain in seen_domains:
            continue
        if domain:
            seen_domains.add(domain)
        title = source.get("title") or url
        # Clean CITE_THIS artifacts from title too
        if "|domain=" in title:
            title = url
        lines.append(f"- [{title}]({url})")
    return "\n".join(lines)


def _build_source_backed_response(summary_lines: list[str], sources: list[dict[str, str]]) -> str:
    body = []
    seen_lines: set[str] = set()
    for line in summary_lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"^[-•\u2022]\s+", "", cleaned)
        dedupe_key = re.sub(r"\s+", " ", cleaned).strip().lower()
        if dedupe_key in seen_lines:
            continue
        seen_lines.add(dedupe_key)
        body.append(cleaned)
    sources_block = _format_sources(sources)
    if sources_block:
        if body:
            body.append("")
        body.append(sources_block)
    return "\n".join(body).strip()


def _web_search_runtime_args(state: Mapping[str, Any]) -> dict[str, Any]:
    selected = str(state.get("web_search_selected_provider") or "").strip().lower()
    configured = str(state.get("web_search_provider_configured") or "").strip().lower()
    args: dict[str, Any] = {}
    if selected:
        args["runtime_selected_provider"] = selected
    if configured:
        args["runtime_provider_configured"] = configured
    return args


def _select_strategy_context(state: AgentState, last_message: str, get_runtime_policy: Callable[[], dict]) -> dict:
    from application.policies.scrape_tracker import (
        _detect_query_category,
        _get_strategy,
        _score_to_reliability,
        _API_VALIDATION_EPSILON,
        _exploration_rate,
    )

    tracker    = state.get("scrape_tracker") or {}
    turn_count = (tracker.get("_turn_count") or 0) + 1
    category   = _detect_query_category(last_message)
    prior_score       = _get_category_score(tracker, category, turn_count)
    prior_reliability = _score_to_reliability(prior_score)

    _rt           = get_runtime_policy().get(category, {})
    _top_promoted = (_rt.get("promoted") or [None])[0]
    ml_recommended: Optional[str] = (
        _top_promoted.get("strategy") if isinstance(_top_promoted, dict) else _top_promoted
    )

    import random

    if _is_allowed_public_price_request(state.get("messages", []), "web_scraping_node"):
        strategy, exploring = "api_price", False
        exp_rate = 0.0
    elif category == "crypto_price":
        if random.random() < _API_VALIDATION_EPSILON:
            strategy, exploring = "force_search", True
        else:
            strategy, exploring = "api_price", False
        exp_rate = _API_VALIDATION_EPSILON
    else:
        exp_rate  = _exploration_rate(prior_score)
        exploring = random.random() < exp_rate
        strategy  = _get_strategy(tracker, category, prior_score, exploring=exploring)

    prediction_match: Optional[bool] = (
        (strategy == ml_recommended) if ml_recommended is not None else None
    )

    return {
        "tracker": tracker, "turn_count": turn_count, "category": category,
        "prior_score": prior_score, "prior_reliability": prior_reliability,
        "ml_recommended": ml_recommended, "strategy": strategy,
        "exploring": exploring, "exp_rate": exp_rate, "prediction_match": prediction_match,
    }


async def _summarize_if_long(
    text: str, rid: str, get_llm_fn: Callable, *, is_retry: bool = False
) -> str:
    if len(text.split()) <= 200:
        return text

    sources_block = ""
    body_text = text
    if "Sources:" in text:
        body_text, sources_block = text.split("Sources:", 1)
        sources_block = "Sources:" + sources_block

    tags = ["web_scraping", "context_quarantine", "summary"]
    if is_retry:
        tags.append("retry")
    llm = get_llm_fn()
    summary_response = await llm.ainvoke(
        [HumanMessage(content=(
            "Resume el siguiente texto en máximo 200 palabras, "
            f"conservando los datos más importantes:\n\n{body_text[:4000]}"
        ))],
        config=RunnableConfig(
            tags=tags,
            metadata={
                "node":              "web_scraping_node",
                "request_id":        rid,
                "raw_words":         len(body_text.split()),
                "summary_triggered": True,
            },
        ),
    )
    summary = cast(str, summary_response.content)
    if sources_block:
        summary = f"{summary.strip()}\n\n{sources_block.strip()}"
    return summary


async def _run_retry_agent(
    agent,
    last_message: str,
    rid: str,
    get_llm_fn: Callable,
) -> tuple[Optional[str], list[str], dict[str, Any], dict[str, Any]]:
    retry_hint = (
        f"[Sistema | auto-retry por bajo rendimiento | estrategia=force_search]\n"
        + "Usa search_web directamente — no intentes scraping de páginas.\n\n"
    )
    retry_result = await agent.ainvoke(
        {"messages": [HumanMessage(content=retry_hint + last_message)]},
        config=RunnableConfig(
            tags=["web_scraping", "agent", "high_risk", "context_quarantine", "retry"],
            metadata={
                "node":       "web_scraping_node",
                "agent":      "web_scraping_agent",
                "request_id": rid,
                "retry":      True,
            },
        ),
    )

    retry_text = extract_final_ai_text(retry_result.get("messages", []))
    if not retry_text:
        return None, [], {}, {}

    summary = await _summarize_if_long(retry_text, rid, get_llm_fn, is_retry=True)
    return (
        summary,
        retry_text.split(),
        _extract_tokens(retry_result),
        _extract_quality(retry_result),
    )


async def _legacy_run_web_scraping_flow(
    state: AgentState,
    agent,
    get_llm_fn: Callable,
    *,
    hitl_enabled: bool,
    confirmation_handler: Optional[ConfirmationPort] = None,
    ask_confirmation_compat: Optional[Callable[[str], Awaitable[bool]]] = None,
    get_runtime_policy: Callable[[], dict],
    evaluate_trajectory_safe_fn=evaluate_trajectory_safe,
    should_evaluate_guard_fn=_should_evaluate_guard,
) -> dict[str, Any]:
    messages     = state["messages"]
    last_message = get_last_message_text(messages)
    state_dict   = cast(dict[str, Any], state)
    web_search_runtime_args = _web_search_runtime_args(state_dict)
    rid          = get_or_create_request_id(state_dict, lambda: "")
    import time
    import uuid
    t0           = time.time()

    if not rid:
        rid = str(uuid.uuid4())

    urls = []
    if hitl_enabled:
        urls     = re.findall(r'https?://\S+', last_message)
        url_info = f" → URLs: {', '.join(urls)}" if urls else ""
        preview  = last_message[:120] + ("..." if len(last_message) > 120 else "")
        needs_confirmation = bool(urls)
        if _is_allowed_public_price_request(messages, "web_scraping_node"):
            needs_confirmation = False

        confirmed = True
        if needs_confirmation:
            confirmed = False
            if confirmation_handler is not None:
                confirmed = await confirmation_handler.confirm(
                    f"\n[HITL] web_scraping_agent va a procesar: \"{preview}\"{url_info}\n¿Confirmar? [s/n]: "
                )
            elif ask_confirmation_compat is not None:
                confirmed = await ask_confirmation_compat(
                    f"\n[HITL] web_scraping_agent va a procesar: \"{preview}\"{url_info}\n¿Confirmar? [s/n]: "
                )
        if not confirmed:
            _emit_node_outcome(
                rid, "web_scraping_node", "blocked", phase="pre_guard",
                agent="web_scraping_agent",
                duration_ms=int((time.time() - t0) * 1000),
                reason="hitl_rejected",
            )
            return {"messages": [AIMessage(content="Operación cancelada por el usuario.")]} 

    try:
        ctx = _select_strategy_context(state, last_message, get_runtime_policy)
        tracker          = ctx["tracker"]
        turn_count       = ctx["turn_count"]
        category         = ctx["category"]
        prior_score      = ctx["prior_score"]
        prior_reliability = ctx["prior_reliability"]
        ml_recommended   = ctx["ml_recommended"]
        strategy         = ctx["strategy"]
        exploring        = ctx["exploring"]
        exp_rate         = ctx["exp_rate"]
        prediction_match = ctx["prediction_match"]

        explicit_urls = _extract_urls_from_text(last_message)
        if explicit_urls:
            fetch_prompt = last_message.strip() or "Extraé la información relevante de esta URL."
            fetch_result = await _fetch_web_page_follow_redirect(explicit_urls[0], fetch_prompt, use_dynamic=True)
            if isinstance(fetch_result, str) and not fetch_result.startswith("Error") and not fetch_result.startswith("URL rechazada"):
                duration_ms = int((time.time() - t0) * 1000)
                words = fetch_result.split()
                source_type = "webfetch"
                reliability = _scrape_reliability(len(words))
                new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _update_scrape_tracker(
                    tracker, category, len(words), turn_count,
                    duration_ms=duration_ms, cost_usd=0.0,
                    source_type=source_type, reliability_override=reliability,
                ))
                analytics = cast(dict[str, Any], analytics)
                new_score = _get_category_score(new_tracker, category, turn_count)

                _emit_node_outcome(
                    rid, "web_scraping_node", "success", phase="agent",
                    agent="web_scraping_agent", duration_ms=duration_ms,
                    category=category, exploring=False, strategy="web_fetch", exp_rate=0.0,
                    scrape_reliability=reliability, prior_reliability=prior_reliability,
                    prior_score=prior_score, scrape_score=new_score,
                    retry_done=False, source_type=source_type,
                    ml_recommended=ml_recommended, prediction_match=prediction_match,
                    ml_would_succeed=(bool(analytics.get("quality_target", 0)) if prediction_match is True else None),
                    **_extract_tokens({"messages": []}), **_extract_quality({"messages": []}), **_extract_followup({"messages": []}, "success"), **analytics, **_node_meta(),
                )
                return {
                    "messages": [AIMessage(content=fetch_result)],
                    "scrape_tracker": new_tracker,
                }

        if strategy == "api_price":
            from domain.tool_responses import PriceToolResponse
            coin     = _detect_coin_from_query(last_message)
            api_json: Optional[str] = None
            price_resp: Optional[Any] = None
            try:
                loop     = asyncio.get_running_loop()
                api_json = await loop.run_in_executor(
                    None, lambda: _get_crypto_price_fn(coin=coin, vs_currency="usd")
                )
            except Exception:
                pass

            if api_json:
                try:
                    price_resp = PriceToolResponse.model_validate_json(api_json)
                except Exception:
                    price_resp = None

            if price_resp and price_resp.is_valid_price():
                formatted   = _format_price_response(price_resp.model_dump())
                duration_ms = int((time.time() - t0) * 1000)
                tokens_fast = {
                    "model": _get_model_name(), "tokens_available": False,
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "total_tokens": 0, "estimated_cost_usd": 0.0,
                }
                quality_fast  = {"output_length": len(formatted), "tool_calls_count": 1}
                followup_fast = {"followup_likely": False}
                meta          = _node_meta()

                fast_path_result = {"messages": [AIMessage(content=formatted)], "next_agent": state.get("next_agent", "")}
                if should_evaluate_guard_fn("web_scraping_node"):
                    is_safe, _ = await evaluate_trajectory_safe_fn(fast_path_result, "web_scraping_node")
                    if not is_safe:
                        _emit_node_outcome(
                            rid, "web_scraping_node", "blocked", phase="post_guard",
                            agent="web_scraping_agent",
                            duration_ms=duration_ms,
                            reason="agentdog",
                            followup_likely=True,
                            **tokens_fast, **quality_fast, **meta,
                        )
                        return {"messages": [AIMessage(content="Respuesta retenida por política de seguridad.")]}

                new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _update_scrape_tracker(
                    tracker, category, 200, turn_count,
                    duration_ms=duration_ms, cost_usd=0.0,
                    source_type="structured", reliability_override="ok_strong",
                ))
                new_score          = _get_category_score(new_tracker, category, turn_count)
                quality_target_val = analytics.get("quality_target", 0)
                ml_would_succeed: Optional[bool] = (
                    bool(quality_target_val) if prediction_match is True else None
                )
                _emit_node_outcome(
                    rid, "web_scraping_node", "success", phase="agent",
                    agent="web_scraping_agent", duration_ms=duration_ms,
                    category=category, strategy=strategy, exploring=exploring,
                    exp_rate=exp_rate, source_type="structured",
                    price_extracted=price_resp.price, parse_success=True,
                    scrape_reliability="ok_strong",
                    prior_reliability=prior_reliability, prior_score=prior_score,
                    scrape_score=new_score, retry_done=False,
                    ml_recommended=ml_recommended, prediction_match=prediction_match,
                    ml_would_succeed=ml_would_succeed,
                    **tokens_fast, **quality_fast, **followup_fast, **analytics, **meta,
                )
                return {
                    "messages": [AIMessage(content=formatted)],
                    "scrape_tracker": new_tracker,
                }

        if category in {"sports", "news"}:
            discovery = await _run_generic_web_search_fetch(last_message, web_search_runtime_args)
            if discovery is not None:
                _disc_raw = cast(str, discovery["summary"])
                _disc_sources = cast(list[dict[str, str]], discovery.get("sources") or [])
                if discovery.get("pre_synthesized"):
                    summary = _disc_raw
                else:
                    summary = await _synthesize_search_summary(_disc_raw, last_message, get_llm_fn, _disc_sources)
                words = cast(list[str], discovery.get("words") or summary.split())
                duration_ms = int((time.time() - t0) * 1000)
                reliability = _scrape_reliability(len(words))
                source_type = cast(str, discovery.get("source_type") or "webfetch")
                new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _update_scrape_tracker(
                    tracker, category, len(words), turn_count,
                    duration_ms=duration_ms, cost_usd=0.0,
                    source_type=source_type, reliability_override=reliability,
                ))
                analytics = cast(dict[str, Any], analytics)
                new_score = _get_category_score(new_tracker, category, turn_count)

                _emit_node_outcome(
                    rid, "web_scraping_node", "success", phase="agent",
                    agent="web_scraping_agent", duration_ms=duration_ms,
                    category=category, exploring=False, strategy="search_web" if source_type == "search" else "web_search_fetch", exp_rate=0.0,
                    scrape_reliability=reliability, prior_reliability=prior_reliability,
                    prior_score=prior_score, scrape_score=new_score,
                    retry_done=False, source_type=source_type,
                    ml_recommended=ml_recommended, prediction_match=prediction_match,
                    ml_would_succeed=(bool(analytics.get("quality_target", 0)) if prediction_match is True else None),
                    **_extract_tokens({"messages": []}), **_extract_quality({"messages": []}), **_extract_followup({"messages": []}, "success"), **analytics, **_node_meta(),
                )
                return {
                    "messages": [AIMessage(content=summary)],
                    "scrape_tracker": new_tracker,
                }

            from tools import search_web

            loop = asyncio.get_running_loop()
            fallback_search = await loop.run_in_executor(
                None,
                lambda: search_web.invoke({"query": last_message, "use_cache": False, **web_search_runtime_args}),
            )
            if not isinstance(fallback_search, str):
                fallback_search = str(fallback_search)
            fallback_terms = _extract_generic_query_terms(last_message)
            fallback_lines = _extract_generic_content_lines(fallback_search, fallback_terms)
            if fallback_lines:
                fallback_sources = _extract_sources_from_text(fallback_search)
                if not fallback_sources:
                    fallback_sources = [{"title": "search result", "url": ""}]
                summary = _build_source_backed_response(fallback_lines[:8], fallback_sources)
                duration_ms = int((time.time() - t0) * 1000)
                words = summary.split()
                reliability = _scrape_reliability(len(words))
                new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _update_scrape_tracker(
                    tracker, category, len(words), turn_count,
                    duration_ms=duration_ms, cost_usd=0.0,
                    source_type="search", reliability_override=reliability,
                ))
                analytics = cast(dict[str, Any], analytics)
                new_score = _get_category_score(new_tracker, category, turn_count)
                _emit_node_outcome(
                    rid, "web_scraping_node", "success", phase="agent",
                    agent="web_scraping_agent", duration_ms=duration_ms,
                    category=category, exploring=False, strategy="search_web", exp_rate=0.0,
                    scrape_reliability=reliability, prior_reliability=prior_reliability,
                    prior_score=prior_score, scrape_score=new_score,
                    retry_done=False, source_type="search",
                    ml_recommended=ml_recommended, prediction_match=prediction_match,
                    ml_would_succeed=(bool(analytics.get("quality_target", 0)) if prediction_match is True else None),
                    **_extract_tokens({"messages": []}), **_extract_quality({"messages": []}), **_extract_followup({"messages": []}, "success"), **analytics, **_node_meta(),
                )
                return {
                    "messages": [AIMessage(content=summary)],
                    "scrape_tracker": new_tracker,
                }

        if is_web_information_query(last_message) or _is_recent_web_information_query(last_message):
            discovery = await _run_generic_web_search_fetch(last_message, web_search_runtime_args)
            if discovery is not None:
                _disc_raw = cast(str, discovery["summary"])
                _disc_sources = cast(list[dict[str, str]], discovery.get("sources") or [])
                if discovery.get("pre_synthesized"):
                    summary = _disc_raw
                else:
                    summary = await _synthesize_search_summary(_disc_raw, last_message, get_llm_fn, _disc_sources)
                words = cast(list[str], discovery.get("words") or summary.split())
                duration_ms = int((time.time() - t0) * 1000)
                reliability = _scrape_reliability(len(words))
                source_type = cast(str, discovery.get("source_type") or "webfetch")
                new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _update_scrape_tracker(
                    tracker, category, len(words), turn_count,
                    duration_ms=duration_ms, cost_usd=0.0,
                    source_type=source_type, reliability_override=reliability,
                ))
                analytics = cast(dict[str, Any], analytics)
                new_score = _get_category_score(new_tracker, category, turn_count)

                _emit_node_outcome(
                    rid, "web_scraping_node", "success", phase="agent",
                    agent="web_scraping_agent", duration_ms=duration_ms,
                    category=category, exploring=False, strategy="search_web" if source_type == "search" else "web_search_fetch", exp_rate=0.0,
                    scrape_reliability=reliability, prior_reliability=prior_reliability,
                    prior_score=prior_score, scrape_score=new_score,
                    retry_done=False, source_type=source_type,
                    ml_recommended=ml_recommended, prediction_match=prediction_match,
                    ml_would_succeed=(bool(analytics.get("quality_target", 0)) if prediction_match is True else None),
                    **_extract_tokens({"messages": []}), **_extract_quality({"messages": []}), **_extract_followup({"messages": []}, "success"), **analytics, **_node_meta(),
                )
                return {
                    "messages": [AIMessage(content=summary)],
                    "scrape_tracker": new_tracker,
                }

        agent_hint = ""
        if strategy == "force_search":
            agent_hint = "[Sistema] Scraping falló repetidamente para este tipo de query. Usa search_web directamente.\n\n"
        elif strategy == "prefer_search":
            agent_hint = "[Sistema] Scraping devolvió contenido insuficiente o lento antes. Intenta search_web primero.\n\n"

        if agent_hint:
            agent_hint = (
                f"[Sistema | categoría={category} score={prior_score:+.2f} "
                f"estrategia={strategy} exploring={exploring} exp_rate={exp_rate:.0%}]\n{agent_hint}"
            )
        agent_message = agent_hint + last_message

        raw_result = await agent.ainvoke(
            {"messages": [HumanMessage(content=agent_message)]},
            config=RunnableConfig(
                tags=["web_scraping", "agent", "high_risk", "context_quarantine"],
                metadata={
                    "node":              "web_scraping_node",
                    "agent":             "web_scraping_agent",
                    "request_id":        rid,
                    "input_chars":       len(last_message),
                    "prior_reliability": prior_reliability,
                },
            ),
        )

        tokens   = _extract_tokens(raw_result)
        quality  = _extract_quality(raw_result)
        followup = _extract_followup(raw_result, "success")
        meta     = _node_meta()

        if should_evaluate_guard_fn("web_scraping_node"):
            is_safe, _ = await evaluate_trajectory_safe_fn(
                {
                    "messages":   raw_result.get("messages", []),
                    "next_agent": state.get("next_agent", ""),
                },
                "web_scraping_node",
            )
            if not is_safe:
                _emit_node_outcome(
                    rid, "web_scraping_node", "blocked", phase="post_guard",
                    agent="web_scraping_agent",
                    duration_ms=int((time.time() - t0) * 1000),
                    reason="agentdog",
                    followup_likely=True,
                    **tokens, **quality, **meta,
                )
                return {"messages": [AIMessage(content="Respuesta retenida por política de seguridad.")]}

        raw_messages = raw_result.get("messages", [])
        raw_text = extract_final_ai_text(raw_messages)

        if not raw_text:
            _emit_node_outcome(
                rid, "web_scraping_node", "error", phase="agent",
                agent="web_scraping_agent",
                duration_ms=int((time.time() - t0) * 1000),
                reason="empty_response",
                followup_likely=True,
                **meta,
            )
            return {"messages": [AIMessage(content="No se pudo extraer información de la página.")]}

        summary           = await _summarize_if_long(raw_text, rid, get_llm_fn)
        words             = raw_text.split()
        summary_triggered = len(words) > 200

        duration_ms = int((time.time() - t0) * 1000)
        reliability = _scrape_reliability(len(words))
        retry_done  = False

        source_type   = "structured" if strategy in _STRUCTURED_SOURCE_STRATEGIES else "unstructured"
        parsed_price: Optional[float] = None
        parse_success: Optional[bool] = None
        price_data: Optional[dict]    = None

        if source_type == "structured":
            price_data = _extract_price_from_messages(raw_result)
            if price_data:
                parsed_price  = price_data["price"]
                parse_success = True
                reliability   = "ok_strong"
            elif raw_text:
                parsed_price  = _extract_structured_price(raw_text)
                parse_success = parsed_price is not None
                reliability   = "ok_strong" if parse_success else "unreliable"
            else:
                parse_success = False
                reliability   = "unreliable"

        if reliability in _RETRY_ON_RELIABILITY and strategy != "force_search":
            _emit_node_outcome(
                rid, "web_scraping_node", "retry", phase="agent",
                agent="web_scraping_agent", duration_ms=duration_ms,
                reason=f"auto_retry:{reliability}",
                scrape_reliability=reliability, strategy=strategy,
                source_type=source_type, category=category, **tokens, **_node_meta(),
            )
            retry_summary, retry_words, retry_tokens, retry_quality = await _run_retry_agent(
                agent, last_message, rid, get_llm_fn,
            )
            if retry_summary is not None:
                summary           = retry_summary
                words             = cast(list[str], retry_words or [])
                summary_triggered = len(words) > 200
                tokens            = cast(dict[str, Any], retry_tokens or {})
                quality           = cast(dict[str, Any], retry_quality or {})

            strategy    = "force_search"
            reliability = _scrape_reliability(len(words))
            retry_done  = True
            duration_ms = int((time.time() - t0) * 1000)

        if reliability == "unreliable":
            new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _update_scrape_tracker(
                tracker, category, len(words), turn_count,
                duration_ms=duration_ms, cost_usd=tokens.get("estimated_cost_usd"),
                source_type=source_type, reliability_override=reliability,
            ))
            analytics = cast(dict[str, Any], analytics)
            _emit_node_outcome(
                rid, "web_scraping_node", "low_confidence", phase="agent",
                agent="web_scraping_agent", duration_ms=duration_ms,
                scrape_reliability=reliability, strategy=strategy,
                retry_done=retry_done, category=category,
                source_type=source_type, price_extracted=parsed_price, parse_success=parse_success,
                ml_recommended=ml_recommended, prediction_match=prediction_match,
                ml_would_succeed=(False if prediction_match is True else None),
                **tokens, **quality, **_node_meta(), **analytics,
            )
            return {
                "messages": [AIMessage(content=(
                    "No pude obtener información confiable para esta consulta. "
                    "Intenta proporcionar una URL específica o reformular la pregunta."
                ))],
                "scrape_tracker": new_tracker,
            }

        cost_usd    = tokens.get("estimated_cost_usd")
        new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _update_scrape_tracker(
            tracker, category, len(words), turn_count,
            duration_ms=duration_ms, cost_usd=cost_usd,
            source_type=source_type, reliability_override=reliability,
        ))
        analytics = cast(dict[str, Any], analytics)
        new_score = _get_category_score(new_tracker, category, turn_count)

        if reliability not in ("ok_weak", "ok_strong"):
            followup = {"followup_likely": True}

        quality_target_val = analytics.get("quality_target", 0)
        ml_would_succeed = (
            bool(quality_target_val) if prediction_match is True else None
        )

        _emit_node_outcome(
            rid, "web_scraping_node", "success", phase="agent",
            agent="web_scraping_agent", duration_ms=duration_ms,
            summary_triggered=summary_triggered, raw_words=len(words),
            category=category, exploring=exploring, strategy=strategy, exp_rate=exp_rate,
            scrape_reliability=reliability, prior_reliability=prior_reliability,
            prior_score=prior_score, scrape_score=new_score,
            retry_done=retry_done,
            source_type=source_type, price_extracted=parsed_price, parse_success=parse_success,
            ml_recommended=ml_recommended, prediction_match=prediction_match,
            ml_would_succeed=ml_would_succeed,
            **tokens, **quality, **followup, **analytics, **meta,
        )
        return {
            "messages": [AIMessage(content=summary)],
            "scrape_tracker": new_tracker,
        }

    except Exception as e:
        _emit_node_outcome(
            rid, "web_scraping_node", "error", phase="agent",
            agent="web_scraping_agent",
            duration_ms=int((time.time() - t0) * 1000),
            reason=str(e),
            followup_likely=True,
            **_node_meta(),
        )
        raise


# ============================================================================
# Generic Claude-style web flow
# ============================================================================

_GENERIC_WEB_STOPWORDS = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "cada", "como", "con",
    "contra", "cual", "cuál", "cuales", "cuáles", "cuando", "cuándo", "de", "del", "desde",
    "donde", "dónde", "durante", "e", "el", "ella", "ellas", "ellos", "en", "entre",
    "era", "eres", "es", "esa", "esas", "ese", "eso", "esta", "está", "están", "este",
    "esto", "estos", "fue", "ha", "han", "hay", "la", "las", "le", "les", "lo", "los",
    "mas", "más", "mi", "mis", "muy", "no", "nos", "nosotros", "o", "o", "para", "pero",
    "por", "que", "qué", "se", "sin", "sobre", "su", "sus", "te", "tu", "tus", "un",
    "una", "uno", "unos", "unas", "y", "ya", "hoy", "ayer", "mañana", "today", "latest",
    "current", "recent", "news", "noticias", "page", "web", "site",
}


def _extract_generic_query_terms(text: str) -> list[str]:
    terms: list[str] = []
    for raw in re.findall(r"[\wáéíóúñÁÉÍÓÚÑ]+", (text or "").lower()):
        if len(raw) < 3 or raw in _GENERIC_WEB_STOPWORDS:
            continue
        if raw not in terms:
            terms.append(raw)
    return terms


def _is_recent_web_information_query(text: str) -> bool:
    lowered = (text or "").lower()
    recent_terms = (
        "today", "hoy", "latest", "recent", "current", "actual", "actuales",
        "última", "últimas", "ultimo", "último", "ultimas", "ultimos",
        "this week", "esta semana", "semana", "week",
    )
    if not any(term in lowered for term in recent_terms):
        return False
    if any(term in lowered for term in ("price", "precio", "cotiza", "cotización", "cotizacion")):
        return False
    return True


_GEOGRAPHY_TERMS: tuple[tuple[str, str], ...] = (
    # Latin America
    ("ecuatoriano", "Ecuador"), ("ecuatoriana", "Ecuador"), ("ecuador", "Ecuador"),
    ("argentino", "Argentina"), ("argentina", "Argentina"),
    ("colombiano", "Colombia"), ("colombia", "Colombia"),
    ("venezolano", "Venezuela"), ("venezuela", "Venezuela"),
    ("chileno", "Chile"), ("chile", "Chile"),
    ("peruano", "Perú"), ("peru", "Perú"),
    ("boliviano", "Bolivia"), ("bolivia", "Bolivia"),
    ("paraguayo", "Paraguay"), ("paraguay", "Paraguay"),
    ("uruguayo", "Uruguay"), ("uruguay", "Uruguay"),
    ("guatemalteco", "Guatemala"), ("guatemala", "Guatemala"),
    ("hondureño", "Honduras"), ("honduras", "Honduras"),
    ("salvadoreño", "El Salvador"), ("el salvador", "El Salvador"),
    ("nicaragüense", "Nicaragua"), ("nicaragua", "Nicaragua"),
    ("costarricense", "Costa Rica"), ("costa rica", "Costa Rica"),
    ("panameño", "Panamá"), ("panama", "Panamá"),
    ("cubano", "Cuba"), ("cuba", "Cuba"),
    ("dominicano", "República Dominicana"), ("república dominicana", "República Dominicana"),
    ("haitiano", "Haití"), ("haiti", "Haití"),
    # North America
    ("mexicano", "México"), ("mexicana", "México"), ("mexico", "México"),
    ("estadounidense", "Estados Unidos"), ("estados unidos", "Estados Unidos"),
    ("usa", "Estados Unidos"), ("eeuu", "Estados Unidos"),
    ("canadiense", "Canadá"), ("canada", "Canadá"),
    # Europe
    ("español", "España"), ("espanol", "España"), ("españa", "España"),
    ("francés", "Francia"), ("frances", "Francia"), ("france", "Francia"), ("francia", "Francia"),
    ("alemán", "Alemania"), ("aleman", "Alemania"), ("alemania", "Alemania"),
    ("italiano", "Italia"), ("italiana", "Italia"), ("italia", "Italia"),
    ("británico", "Reino Unido"), ("britanico", "Reino Unido"), ("reino unido", "Reino Unido"),
    ("inglés", "Reino Unido"), ("ingles", "Reino Unido"),
    ("portugués", "Portugal"), ("portugues", "Portugal"), ("portugal", "Portugal"),
    ("holandés", "Países Bajos"), ("holanda", "Países Bajos"), ("países bajos", "Países Bajos"),
    ("belga", "Bélgica"), ("belgica", "Bélgica"), ("bélgica", "Bélgica"),
    ("suizo", "Suiza"), ("suiza", "Suiza"),
    ("sueco", "Suecia"), ("suecia", "Suecia"),
    ("noruego", "Noruega"), ("noruega", "Noruega"),
    ("danés", "Dinamarca"), ("danes", "Dinamarca"), ("dinamarca", "Dinamarca"),
    ("finlandés", "Finlandia"), ("finlandia", "Finlandia"),
    ("polaco", "Polonia"), ("polonia", "Polonia"),
    ("checo", "República Checa"), ("república checa", "República Checa"),
    ("húngaro", "Hungría"), ("hungria", "Hungría"),
    ("rumano", "Rumanía"), ("rumania", "Rumanía"),
    ("griego", "Grecia"), ("grecia", "Grecia"),
    ("turco", "Turquía"), ("turquia", "Turquía"), ("turquía", "Turquía"),
    ("ruso", "Rusia"), ("rusia", "Rusia"), ("russia", "Rusia"),
    ("ucraniano", "Ucrania"), ("ucrania", "Ucrania"),
    ("serbio", "Serbia"), ("serbia", "Serbia"),
    # Asia-Pacific
    ("japonés", "Japón"), ("japonesa", "Japón"), ("japones", "Japón"),
    ("japón", "Japón"), ("japon", "Japón"), ("japan", "Japón"),
    ("chino", "China"), ("china", "China"),
    ("surcoreano", "Corea del Sur"), ("corea del sur", "Corea del Sur"),
    ("norcoreano", "Corea del Norte"), ("corea del norte", "Corea del Norte"),
    ("coreano", "Corea"), ("corea", "Corea"),
    ("indio", "India"), ("india", "India"),
    ("paquistaní", "Pakistán"), ("pakistan", "Pakistán"),
    ("bangladesí", "Bangladesh"), ("bangladesh", "Bangladesh"),
    ("indonesio", "Indonesia"), ("indonesia", "Indonesia"),
    ("filipino", "Filipinas"), ("filipinas", "Filipinas"),
    ("vietnamita", "Vietnam"), ("vietnam", "Vietnam"),
    ("tailandés", "Tailandia"), ("tailandia", "Tailandia"),
    ("malayo", "Malasia"), ("malasia", "Malasia"),
    ("singapurense", "Singapur"), ("singapur", "Singapur"),
    ("australiano", "Australia"), ("australia", "Australia"),
    ("neozelandés", "Nueva Zelanda"), ("nueva zelanda", "Nueva Zelanda"),
    # Middle East & Africa
    ("israelí", "Israel"), ("israel", "Israel"),
    ("palestino", "Palestina"), ("palestina", "Palestina"),
    ("iraní", "Irán"), ("iran", "Irán"),
    ("iraquí", "Irak"), ("irak", "Irak"),
    ("sirio", "Siria"), ("siria", "Siria"),
    ("libanés", "Líbano"), ("libano", "Líbano"),
    ("saudí", "Arabia Saudita"), ("arabia saudita", "Arabia Saudita"), ("saudi", "Arabia Saudita"),
    ("emiratense", "Emiratos Árabes"), ("emiratos arabes", "Emiratos Árabes"),
    ("egipcio", "Egipto"), ("egipto", "Egipto"),
    ("nigeriano", "Nigeria"), ("nigeria", "Nigeria"),
    ("sudafricano", "Sudáfrica"), ("sudafrica", "Sudáfrica"),
    ("etíope", "Etiopía"), ("etiopia", "Etiopía"),
    ("keniata", "Kenia"), ("kenia", "Kenia"),
    ("marroquí", "Marruecos"), ("marruecos", "Marruecos"),
    # Brazil (no adjective form yet)
    ("brasileño", "Brasil"), ("brasileña", "Brasil"), ("brasil", "Brasil"),
)


def _extract_query_geography(text: str) -> Optional[str]:
    lowered = (text or "").lower()
    # 1. Check known country terms (longest match first to avoid "corea" matching before "corea del sur")
    for term, country in sorted(_GEOGRAPHY_TERMS, key=lambda x: -len(x[0])):
        if term in lowered:
            return country
    # 2. Pattern-based fallback: extract word after "de"/"en"/"sobre" before "de esta"/"semana"/"hoy"
    #    e.g. "noticias de turquía de esta semana" → "Turquía"
    #    e.g. "qué pasa en nigeria" → "Nigeria"
    fallback_patterns = [
        r"\b(?:de|en|sobre)\s+([a-záéíóúüñ]{4,})\s+(?:de\s+esta|esta\s+semana|hoy|del\b|esta\b)",
        r"\b(?:de|en|sobre)\s+([a-záéíóúüñ]{4,})\s*$",
        r"\b(?:de|en|sobre)\s+([a-záéíóúüñ]{4,})\s",
    ]
    _geo_stopwords = _GENERIC_WEB_STOPWORDS | {
        "noticia", "noticias", "semana", "semanas", "ultima", "ultimas",
        "ultimo", "ultimos", "última", "últimas", "último", "últimos",
        "reciente", "recientes", "informacion", "información", "tema",
        "seguridad", "economia", "politica", "deporte", "cultura",
    }
    for pattern in fallback_patterns:
        m = re.search(pattern, lowered)
        if m:
            word = m.group(1).strip()
            if word not in _geo_stopwords and len(word) >= 4:
                return word.capitalize()
    return None


_TOPIC_ANGLES: dict[str, list[str]] = {
    "security": [
        "{geo} crimen delincuencia seguridad interna {year}",
        "{geo} defensa militar despliegue fuerzas {year}",
        "{geo} diplomacia tensiones política exterior {year}",
        "{geo} desastre emergencia seguridad civil {year}",
    ],
    "economy": [
        "{geo} economía mercado inversión {year}",
        "{geo} empleo salario empresa {year}",
        "{geo} inflación precios comercio {year}",
        "{geo} tecnología industria energía {year}",
    ],
    "politics": [
        "{geo} gobierno elecciones política {year}",
        "{geo} congreso ley reforma legislación {year}",
        "{geo} oposición partido liderazgo {year}",
        "{geo} corrupción justicia tribunal {year}",
    ],
    "default": [
        "{geo} {topic} noticias recientes {year}",
        "{geo} {topic} novedades actualidad {year}",
        "{geo} {topic} últimas noticias semana {year}",
        "{geo} {topic} hoy noticia {year}",
    ],
}

# English equivalents used as supplementary search fallback when Spanish angles yield < 4 candidates.
_TOPIC_ANGLES_EN: dict[str, list[str]] = {
    "security": [
        "{geo_en} crime internal security {year}",
        "{geo_en} defense military deployment {year}",
        "{geo_en} diplomacy tensions foreign policy {year}",
        "{geo_en} disaster emergency civil security {year}",
    ],
    "economy": [
        "{geo_en} economy market investment {year}",
        "{geo_en} employment wages companies {year}",
        "{geo_en} inflation prices trade {year}",
        "{geo_en} technology industry energy {year}",
    ],
    "politics": [
        "{geo_en} government elections politics {year}",
        "{geo_en} congress law reform legislation {year}",
        "{geo_en} opposition party leadership {year}",
        "{geo_en} corruption justice tribunal {year}",
    ],
    "default": [
        "{geo_en} {topic} recent news {year}",
        "{geo_en} {topic} latest news this week {year}",
        "{geo_en} {topic} today news {year}",
        "{geo_en} {topic} updates {year}",
    ],
}

_GEO_ENGLISH: dict[str, str] = {
    # Latin America
    "Ecuador": "Ecuador", "Argentina": "Argentina", "Colombia": "Colombia",
    "Venezuela": "Venezuela", "Chile": "Chile", "Perú": "Peru",
    "Bolivia": "Bolivia", "Paraguay": "Paraguay", "Uruguay": "Uruguay",
    "Guatemala": "Guatemala", "Honduras": "Honduras", "El Salvador": "El Salvador",
    "Nicaragua": "Nicaragua", "Costa Rica": "Costa Rica", "Panamá": "Panama",
    "Cuba": "Cuba", "República Dominicana": "Dominican Republic", "Haití": "Haiti",
    # North America
    "México": "Mexico", "Estados Unidos": "United States", "Canadá": "Canada",
    # Europe
    "España": "Spain", "Francia": "France", "Alemania": "Germany",
    "Italia": "Italy", "Reino Unido": "United Kingdom", "Portugal": "Portugal",
    "Países Bajos": "Netherlands", "Bélgica": "Belgium", "Suiza": "Switzerland",
    "Suecia": "Sweden", "Noruega": "Norway", "Dinamarca": "Denmark",
    "Finlandia": "Finland", "Polonia": "Poland", "República Checa": "Czech Republic",
    "Hungría": "Hungary", "Rumanía": "Romania", "Grecia": "Greece",
    "Turquía": "Turkey", "Rusia": "Russia", "Ucrania": "Ukraine", "Serbia": "Serbia",
    # Asia-Pacific
    "Japón": "Japan", "China": "China", "Corea del Sur": "South Korea",
    "Corea del Norte": "North Korea", "Corea": "Korea",
    "India": "India", "Pakistán": "Pakistan", "Bangladesh": "Bangladesh",
    "Indonesia": "Indonesia", "Filipinas": "Philippines", "Vietnam": "Vietnam",
    "Tailandia": "Thailand", "Malasia": "Malaysia", "Singapur": "Singapore",
    "Australia": "Australia", "Nueva Zelanda": "New Zealand",
    # Middle East & Africa
    "Israel": "Israel", "Palestina": "Palestine", "Irán": "Iran",
    "Irak": "Iraq", "Siria": "Syria", "Líbano": "Lebanon",
    "Arabia Saudita": "Saudi Arabia", "Emiratos Árabes": "UAE",
    "Egipto": "Egypt", "Nigeria": "Nigeria", "Sudáfrica": "South Africa",
    "Etiopía": "Ethiopia", "Kenia": "Kenya", "Marruecos": "Morocco",
    # Brazil
    "Brasil": "Brazil",
}


def _detect_news_topic(query: str) -> str:
    lowered = query.lower()
    if any(k in lowered for k in ["seguridad", "security", "crimen", "defensa", "militar", "policía", "policia", "terroris", "ataque", "atentado", "conflicto"]):
        return "security"
    if any(k in lowered for k in ["economía", "economia", "mercado", "bolsa", "precio", "inflacion", "inflación", "pib", "empleo", "comercio", "empresa"]):
        return "economy"
    if any(k in lowered for k in ["política", "politica", "gobierno", "elección", "eleccion", "presidente", "congreso", "partido", "ministro"]):
        return "politics"
    return "default"


def _build_angle_queries(last_message: str, search_age_days: Optional[int]) -> list[dict]:
    """Generates 4 angle-specific search queries for diverse news coverage."""
    import datetime
    geo = _extract_query_geography(last_message)
    topic = _detect_news_topic(last_message)
    year = datetime.date.today().year
    angles = _TOPIC_ANGLES.get(topic, _TOPIC_ANGLES["default"])
    base_geo = geo or " ".join(
        w for w in last_message.split()
        if len(w) > 3 and w.lower() not in _GENERIC_WEB_STOPWORDS
    )[:40]
    queries = []
    for template in angles:
        q = template.format(geo=base_geo, topic=topic, year=year)
        invoke_args: dict = {"query": q, "use_cache": False}
        if search_age_days is not None:
            invoke_args["max_age_days"] = search_age_days
        queries.append(invoke_args)
    return queries


_MONTH_NAMES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}
_MONTH_NAMES_EN = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _candidate_url_is_recent(url: str, days_threshold: int) -> bool:
    """Returns True if the date embedded in the URL is within `days_threshold` days, or no date found.

    Handles separated dates (/2026/04/02/), compact dates (yjj20260402...), and
    month-name slugs in Spanish or English (e.g. julio-2025, march-2025).
    """
    import datetime
    today = datetime.date.today()
    cutoff = today - datetime.timedelta(days=days_threshold)
    lowered = (url or "").lower()
    # Separated numeric: /2026/04/02/ or /2026-04-02
    for match in re.finditer(r"[/\-](\d{4})[/\-](\d{2})[/\-](\d{2})", lowered):
        try:
            article_date = datetime.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            return article_date >= cutoff
        except ValueError:
            pass
    # Compact: YYYYMMDD anywhere in the URL slug (e.g. yjj20260212... or 20260402_news)
    for match in re.finditer(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])", lowered):
        try:
            article_date = datetime.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            return article_date >= cutoff
        except ValueError:
            pass
    # Month-name slug: "julio-2025", "march-2025", "2025-julio", "2025-march"
    all_months = {**_MONTH_NAMES_ES, **_MONTH_NAMES_EN}
    month_pattern = "|".join(re.escape(m) for m in all_months)
    for match in re.finditer(
        rf"(?:({month_pattern})[- ](\d{{4}})|(\d{{4}})[- ]({month_pattern}))", lowered
    ):
        month_name = match.group(1) or match.group(4)
        year_str = match.group(2) or match.group(3)
        try:
            article_date = datetime.date(int(year_str), all_months[month_name], 1)
            return article_date >= cutoff
        except (ValueError, KeyError):
            pass
    return True  # No date in URL → don't filter (assume recent)


_TITLE_STOPWORDS = {
    "de", "la", "el", "en", "a", "los", "las", "del", "que", "un", "una",
    "por", "con", "se", "ha", "al", "es", "su", "y", "e", "o", "the", "of",
    "in", "to", "a", "and", "for", "on", "at", "by", "with", "from",
}


def _text_keywords(text: str) -> set[str]:
    return {w.lower() for w in text.split() if len(w) > 4 and w.lower() not in _TITLE_STOPWORDS}


_NON_NEWS_DOMAINS = {
    "travel", "tourism", "tripadvisor", "lonelyplanet", "fodors", "frommers",
    "wikivoyage", "wikipedia", "wikitravel", "about.com", "tripsavvy",
    "smartertravel", "booking.com", "expedia", "airbnb",
    # statistics / data aggregators — evergreen content, never news
    "numbeo.com", "statista.com", "macrotrends.net", "worldometers.info",
    "tradingeconomics.com", "indexmundi.com", "globaleconomy.com",
    "countrymeters.info", "globalterrorismindex.org", "visionofhumanity.org",
    # think tanks / advocacy orgs — policy analysis, not news reporting
    "brennancenter.org", "aclu.org", "cato.org", "heritage.org",
    "brookings.edu", "cfr.org", "chathamhouse.org", "sipri.org",
    "amnesty.org", "hrw.org", "freedomhouse.org",
    "dialogopolitico.org", "csis.org", "rand.org", "wilsoncenter.org",
    # government travel advisory portals — evergreen safety ratings, not news
    "osac.gov", "travel.state.gov", "smartraveller.gov.au",
    "travel.gc.ca", "gov.uk/foreign-travel-advice",
    # travel/community forums — threads stay active for years, not current news
    "losviajeros.com", "foro.travel", "viajeros.com", "tripadvisor.com",
    "lonelyplanet.com/thorntree", "reddit.com/r/travel",
}

# URL path segments that indicate forums, threads, or community posts — not journalism.
_FORUM_PATH_SEGMENTS = {
    "/foros/", "/forum/", "/forums/", "/thread/", "/threads/",
    "/topic/", "/topics/", "/post/", "/posts/", "/discussion/",
    "/comunidad/", "/community/", "/board/", "/boards/",
}


def _is_non_news_candidate(candidate: dict[str, str]) -> bool:
    """Returns True if the candidate looks like evergreen/travel/wiki/government-PR content rather than news."""
    url = candidate.get("url", "").lower()
    title = candidate.get("title", "").lower()
    snippet = candidate.get("snippet", "").lower()

    if any(domain in url for domain in _NON_NEWS_DOMAINS):
        return True

    # Forum / community thread URLs — content is user-generated and not time-stamped journalism
    if any(seg in url for seg in _FORUM_PATH_SEGMENTS):
        return True

    # Government press-release sections (.gob. / .gov domains with /prensa/ or /comunicado/)
    # These publish project announcements, not journalistic news.
    _GOV_TLD = (".gob.", ".gov.", "/gob.", "/gov.")
    _PR_PATHS = ("/prensa/", "/comunicado", "/nota-de-prensa", "/press-release", "/sala-de-prensa")
    if any(tld in url for tld in _GOV_TLD) and any(path in url for path in _PR_PATHS):
        return True

    # Law firm / legal publisher URLs — client alerts, legal updates, publications
    _LEGAL_PATHS = (
        "/legal-update", "/client-alert", "/client-advisory", "/legal-alert",
        "/publications/", "/publication/", "/insights/", "/knowledge/",
        "/briefing/", "/memorandum/", "/legal-news/",
    )
    if any(seg in url for seg in _LEGAL_PATHS):
        return True

    # Snippets with generic travel-advice patterns (no concrete event, no date)
    evergreen_signals = [
        "se recomienda a los viajeros", "se aconseja a los viajeros",
        "para los turistas", "consejos de seguridad", "guía de viaje",
        "recomendaciones para viajeros", "baja tasa de criminalidad",
        "travel advisory", "safety tips for travelers",
        # travel advisory language
        "ejercer mayor precaución", "ejercer precaución",
        "se desaconseja viajar", "no se recomienda viajar",
        "nivel de alerta de viaje", "travel level", "do not travel",
        "reconsider travel", "exercise increased caution",
        "high threat location", "ubicación de alta amenaza",
    ]
    return any(signal in snippet or signal in title for signal in evergreen_signals)


def _same_event(
    candidate_a: dict[str, str],
    candidate_b: dict[str, str],
    query_terms: Optional[list[str]] = None,
) -> bool:
    """Returns True if two candidates appear to describe the same event.

    Title overlap ≥ 3 shared keywords (excluding query terms) → same event.
    Full-text (title+snippet) overlap ≥ 5 non-query keywords → same event.

    Thresholds intentionally conservative: for topic-rich queries (e.g. Japan security),
    many articles share 2 generic words (china, tensiones) without covering the same story.
    Requiring 3 title-keyword overlap prevents false deduplication across genuinely
    distinct events, which is what produces 4 diverse bullets.
    """
    excluded = set(t.lower() for t in (query_terms or []))
    excluded.update(_TITLE_STOPWORDS)

    def keywords(text: str) -> set[str]:
        return {w.lower() for w in text.split() if len(w) > 4 and w.lower() not in excluded}

    title_kw_a = keywords(candidate_a.get("title", ""))
    title_kw_b = keywords(candidate_b.get("title", ""))
    if len(title_kw_a & title_kw_b) >= 3:
        return True

    full_a = keywords(f"{candidate_a.get('title', '')} {candidate_a.get('snippet', '')}")
    full_b = keywords(f"{candidate_b.get('title', '')} {candidate_b.get('snippet', '')}")
    return len(full_a & full_b) >= 5


def _dedup_candidates_by_event(
    candidates: list[dict[str, str]],
    query_terms: Optional[list[str]] = None,
) -> list[dict[str, str]]:
    """Keep one candidate per event — drop articles that appear to cover the same story."""
    accepted: list[dict[str, str]] = []
    for candidate in candidates:
        if not any(_same_event(candidate, a, query_terms) for a in accepted):
            accepted.append(candidate)
    return accepted


def _extract_generic_search_candidates(search_text: str) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    current: Optional[dict[str, str]] = None

    for line in [line.rstrip() for line in (search_text or "").splitlines() if line.strip()]:
        # Handle both "1. [title](url)" and "1. [article] [title](url)" / "1. [hub] [title](url)"
        item_match = re.match(r"^\d+\. (?:\[(article|hub)\]\s*)?\[(.+?)\]\((https?://[^)]+)\)", line.strip())
        if item_match:
            if current:
                candidates.append(current)
            tag = item_match.group(1) or ""
            current = {
                "title": item_match.group(2).strip(),
                "url": item_match.group(3).strip(),
                "snippet": "",
                "hit_type": tag,
            }
            continue

        if current is not None and not line.startswith("Sources:") and not line.startswith("-") and not line.startswith("Call web_fetch") and not line.startswith("Next step"):
            snippet = line.strip()
            if snippet:
                current["snippet"] = (current.get("snippet", "") + " " + snippet).strip()

    if current:
        candidates.append(current)

    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for candidate in candidates:
        url = candidate.get("url", "")
        if url and url not in seen:
            seen.add(url)
            deduped.append(candidate)
    return deduped


def _score_generic_candidate(candidate: dict[str, str], query_terms: list[str], query_source_group: Optional[str] = None) -> int:
    blob = " ".join([candidate.get("title", ""), candidate.get("snippet", ""), candidate.get("url", "")]).lower()
    score = 0
    for term in query_terms:
        if term in blob:
            score += 3
    if re.search(r"\b\d+\s*-\s*\d+\b", blob):
        score += 2
    url = candidate.get("url", "")
    path = urlparse(url).path.lower()
    segments = [segment for segment in path.split("/") if segment]
    if re.search(r"(19|20)\d{2}[/\-]?\d{2}[/\-]?\d{2}", path) or re.search(r"\d{6,8}", path):
        score += 4
    if len(segments) >= 3:
        score += 2
    if len(segments) <= 2 and not re.search(r"(19|20)\d{2}[/\-]?\d{2}[/\-]?\d{2}", path):
        score -= 3
    if any(seg in {"topic", "topics", "tag", "tags", "category", "categories", "archive", "author"} for seg in segments):
        score -= 4
    if any(noise in blob for noise in ("login", "signin", "cookie", "privacy", "archive", "perfil")):
        score -= 2
    score += score_domain_boost(query_source_group, url)
    return score


def _strip_accents(text: str) -> str:
    """Remove diacritics so 'japon' matches 'japón', 'ultima' matches 'última', etc."""
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _extract_generic_content_lines(text: str, query_terms: list[str]) -> list[str]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    result: list[str] = []
    score_lines_seen = 0
    # Normalize query terms once so accent-bearing queries (e.g. "japon") match
    # article text that uses accented forms ("japón").
    normalized_terms = [_strip_accents(t) for t in (query_terms or [])]
    for idx, line in enumerate(lines):
        lower = line.lower()
        if lower.startswith(("url:", "sources:", "http")):
            continue
        if "http" in lower or "sources" in lower:
            continue
        # Tavily search-result headers ("Web search results for query: ...") are metadata,
        # not useful content — skip them so they don't inflate the body-lines count.
        if lower.startswith("web search results for query"):
            continue
        if len(line) < 3:
            continue
        if not re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ0-9]", line):
            continue
        # Document section headers from legal/academic documents (e.g. "C. Conclusion", "III. Analysis")
        if re.match(r"^(?:[IVXLC]+\.|[A-Z]\.|[1-9]\d?\.|[a-z]\))\s+[A-ZÁÉÍÓÚ]", line):
            continue
        # Meta-wrapper openers — the sentence summarizes what the page says rather than reporting an event.
        # e.g. "La información más reciente sobre X destaca aspectos clave:"
        #      "Las últimas noticias sobre X indican que los viajeros deben..."
        #      "Los últimos datos sobre X señalan que..."
        if re.match(
            r"^(?:la informaci[oó]n|las [uú]ltimas noticias|los [uú]ltimos datos|el [uú]ltimo informe)"
            r".{0,60}(?:destaca|indican?|se[nñ]alan?|muestra|revela|se centra|aborda|trata)",
            lower,
        ):
            continue
        # Mid-paragraph continuation sentences — start with a demonstrative pronoun
        # that refers to a prior sentence we don't have ("Esta situación", "Este problema",
        # "Esto demuestra", "Esa tendencia"). Without the antecedent they're meaningless as bullets.
        # Exclude temporal openers ("Esta semana", "Este año", "Este mes", "Este lunes") — those are valid.
        _TEMPORAL = (
            "semana", "año", "mes", "dia", "día", "lunes", "martes", "miércoles",
            "miercoles", "jueves", "viernes", "sabado", "sábado", "domingo",
            "mañana", "noche", "tarde", "trimestre", "periodo", "período",
        )
        if re.match(r"^(?:esta|este|esto|esa|ese|eso|dicha|dicho|tal)\s+\w+", lower):
            following_word = re.match(r"^(?:esta|este|esto|esa|ese|eso|dicha|dicho|tal)\s+(\w+)", lower)
            if following_word and following_word.group(1) not in _TEMPORAL:
                continue
        lower_norm = _strip_accents(lower)
        if query_terms:
            if any(term in lower_norm for term in normalized_terms):
                result.append(line)
                for look_ahead in range(1, 3):
                    if idx + look_ahead >= len(lines):
                        break
                    next_line = lines[idx + look_ahead].strip()
                    next_lower = next_line.lower()
                    if not next_line or next_lower.startswith(("url:", "sources:", "http")) or "http" in next_lower or "sources" in next_lower:
                        break
                    if not re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ0-9]", next_line):
                        break
                    result.append(next_line)
            elif re.search(r"\b\d+\s*-\s*\d+\b", line) and score_lines_seen == 0:
                result.append(line)
                score_lines_seen += 1
        else:
            result.append(line)
    return result


_NO_INFO_RE = re.compile(
    # Pattern A: "no/sin + verb + noticias/información/news/contenido"
    # Catches: "no proporciona noticias", "no incluye noticias recientes", "no hay noticias"
    r"\b(?:no|sin)\b.{0,50}\b(?:noticias?|informacion|news|contenido relevante)\b"
    # Pattern B: "subject + no + verb" — "la información proporcionada no incluye"
    r"|\b(?:informacion|pagina|sitio|texto|contenido)\b.{0,40}\bno\b.{0,40}"
    r"\b(?:incluye|proporciona|contiene|ofrece|tiene|encontr)\b"
    # Pattern C: explicit "doesn't address this topic" meta-commentary
    r"|sin abordar (?:directamente|este tema|el tema)"
    r"|no aborda (?:directamente|este tema)"
    r"|no trata (?:directamente|este tema)"
    r"|informacion proporcionada se centra en"
    # Pattern D: English equivalents
    r"|does not (?:contain|provide|include) (?:information|news|relevant)"
    r"|no relevant information|no results found|without relevant information",
    re.DOTALL,
)


def _is_no_info_response(text: str) -> bool:
    lowered = _strip_accents((text or "").lower())
    return bool(_NO_INFO_RE.search(lowered))


def _extract_sources_from_text(text: str) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen: set[str] = set()

    # Parse structured CITE_THIS markers: <<<CITE_THIS: title=...|url=...|domain=...>>>
    for match in re.finditer(r"<<<CITE_THIS:\s*title=([^|]+)\|url=([^|>]+)\|domain=([^|>]+)>>>", text or ""):
        article_title, url, domain = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
        if url and url not in seen:
            seen.add(url)
            sources.append({"title": article_title or domain, "url": url, "domain": domain, "snippet": ""})

    if sources:
        return sources

    # Fallback: parse standard markdown links [title](url)
    for title, url in re.findall(r"\[([^\]]+)\]\((https?://[^)]+)\)", text or ""):
        normalized = url.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        domain = urlparse(normalized).hostname or normalized
        sources.append({"title": title.strip() or normalized, "url": normalized, "domain": domain, "snippet": ""})

    if sources:
        return sources

    for url in _extract_urls_from_text(text):
        if url not in seen:
            seen.add(url)
            domain = urlparse(url).hostname or url
            sources.append({"title": url, "url": url, "domain": domain, "snippet": ""})
    return sources


def _build_generic_fetch_prompt(query: str) -> str:
    geography = _extract_query_geography(query)
    geography_line = f"Contexto geográfico: {geography}. " if geography else ""
    return (
        "Extraé únicamente la información relevante para responder la consulta del usuario. "
        f"{geography_line}"
        "Respondé con 4 párrafos breves sobre el mismo tema solicitado si la consulta es de noticias/actualidad, y con 3-5 viñetas solo si el contenido lo pide. "
        "Incluí el contexto inmediato de la noticia y por qué importa, sin inventar datos. "
        "Si la página mezcla varios temas, otros países o fuentes, devolvé solo la sección pertinente a la consulta. "
        "Si no hay datos claros, decilo.\n\n"
        f"Consulta: {query}"
    )


def _extract_web_fetch_redirect_url(result_text: str) -> Optional[str]:
    match = re.search(r"^Redirect URL:\s*(https?://\S+)$", result_text or "", re.MULTILINE)
    if match:
        return match.group(1).strip().rstrip(".,;:")
    return None


async def _run_week_search_candidates(
    last_message: str,
    search_age_days: Optional[int],
    query_terms: list[str],
    query_source_group: Optional[str],
    web_search_runtime_args: Optional[dict[str, Any]] = None,
) -> tuple[list[dict[str, str]], str]:
    """Runs the generic OpenClaw-style web search path.

    The search provider decides the result set; this helper only normalizes, ranks,
    and deduplicates the returned hits.
    """
    from tools import search_web

    loop = asyncio.get_running_loop()
    search_invoke_args: dict = {"query": last_message, "use_cache": False, **(web_search_runtime_args or {})}
    if search_age_days is not None:
        search_invoke_args["max_age_days"] = search_age_days

    search_text = await loop.run_in_executor(None, lambda: search_web.invoke(search_invoke_args))
    if not isinstance(search_text, str):
        search_text = str(search_text)

    url_age_threshold = search_age_days or 14
    candidates = [
        c for c in _extract_generic_search_candidates(search_text)
        if not _is_non_news_candidate(c)
        and _candidate_url_is_recent(c.get("url", ""), url_age_threshold)
    ]
    ranked_candidates = sorted(
        candidates,
        key=lambda c: _score_generic_candidate(c, query_terms, query_source_group),
        reverse=True,
    )
    diverse_candidates = _dedup_candidates_by_event(ranked_candidates, query_terms)[:4]

    return diverse_candidates, search_text


async def _fetch_web_page_follow_redirect(url: str, prompt: str, *, use_dynamic: bool = True) -> str:
    from tools.web_tools import fetch_web_page

    result = await fetch_web_page(url=url, prompt=prompt, use_dynamic=use_dynamic)
    if not isinstance(result, str):
        result = str(result)

    redirect_url = _extract_web_fetch_redirect_url(result)
    if redirect_url and redirect_url != url:
        redirected = await fetch_web_page(url=redirect_url, prompt=prompt, use_dynamic=use_dynamic)
        return redirected if isinstance(redirected, str) else str(redirected)
    return result


async def _run_generic_web_search_fetch(
    last_message: str,
    web_search_runtime_args: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    from tools import search_web
    from tools.web_tools import fetch_web_page

    query_terms = _extract_generic_query_terms(last_message)
    query_source_group = detect_query_source_group(last_message)
    source_terms = list(get_query_source_terms(last_message))
    if source_terms:
        merged_terms: list[str] = []
        for term in query_terms + source_terms:
            if term not in merged_terms:
                merged_terms.append(term)
        query_terms = merged_terms
    query_horizon = detect_recent_query_horizon(last_message) if _is_recent_web_information_query(last_message) else None
    recent_requirements = get_recent_query_requirements(query_horizon)
    recent_min_score = recent_requirements["min_score"]
    recent_min_body_lines = recent_requirements["min_body_lines"]
    recent_min_sources = recent_requirements["min_sources"]
    recent_min_candidates = recent_requirements["min_candidates"]
    recent_candidate_min_score = recent_requirements["candidate_min_score"] or recent_min_score
    recent_candidate_min_body_lines = 1 if query_horizon == "week" else recent_min_body_lines
    recent_candidate_min_sources = recent_requirements["candidate_min_sources"] or 1
    loop = asyncio.get_running_loop()

    # Date filter for search.
    # "esta semana/hoy" → 14 days (not 7 — Tavily returns only hub/portal pages with days=7,
    # no specific articles; 14 still excludes content older than 2 weeks like Dec-2025 events).
    # Any other recent news query → 30 days.
    search_age_days: Optional[int] = None
    if query_horizon == "week":
        search_age_days = 14
    elif _is_recent_web_information_query(last_message):
        search_age_days = 30

    if query_horizon == "week":
        # OpenClaw-style search: one provider-backed result set, then rank/deduplicate
        # the returned candidates before fetching article pages.
        diverse_candidates, search_text = await _run_week_search_candidates(
            last_message, search_age_days, query_terms, query_source_group, web_search_runtime_args
        )
    else:
        search_invoke_args: dict = {"query": last_message, "use_cache": False, **(web_search_runtime_args or {})}
        if search_age_days is not None:
            search_invoke_args["max_age_days"] = search_age_days

        search_text = await loop.run_in_executor(
            None,
            lambda: search_web.invoke(search_invoke_args),
        )
        if not isinstance(search_text, str):
            search_text = str(search_text)

        candidates = _extract_generic_search_candidates(search_text)

        def _candidate_sort_key(c: dict[str, str]) -> tuple:
            score = _score_generic_candidate(c, query_terms, query_source_group)
            return (score,)

        ranked_candidates = sorted(
            [c for c in candidates if not _is_non_news_candidate(c)],
            key=_candidate_sort_key,
            reverse=True,
        )[:8]

        diverse_candidates = _dedup_candidates_by_event(ranked_candidates, query_terms)

        # Run second search if fewer than 3 distinct events found
        if len(diverse_candidates) < 3:
            alt_invoke_args: dict = {"query": last_message + " últimas noticias recientes", "use_cache": False, **(web_search_runtime_args or {})}
            if search_age_days is not None:
                alt_invoke_args["max_age_days"] = search_age_days
            alt_search_text = await loop.run_in_executor(
                None,
                lambda q=alt_invoke_args: search_web.invoke(q),
            )
            if not isinstance(alt_search_text, str):
                alt_search_text = str(alt_search_text)
            alt_candidates = [c for c in _extract_generic_search_candidates(alt_search_text) if not _is_non_news_candidate(c)]
            for c in sorted(alt_candidates, key=lambda x: _score_generic_candidate(x, query_terms, query_source_group), reverse=True):
                if len(diverse_candidates) >= 4:
                    break
                if not any(_same_event(c, d, query_terms) for d in diverse_candidates):
                    diverse_candidates.append(c)
            search_text = search_text + "\n" + alt_search_text

    ranked_candidates = diverse_candidates[:4]

    # For week queries: hybrid approach —
    # 1. Fetch specific article URLs (non-hub) without dynamic JS (faster, avoids hallucination)
    # 2. Fall back to Tavily snippet when fetch fails or returns poor content
    # This is more reliable than full dynamic fetches for paywalled/dynamic pages.
    if query_horizon == "week":
        # Match URLs that look like specific articles: date in path, or slug ≥15 chars
        # 15 chars covers nippon.com IDs like yjj2026040500456 (16 chars)
        _url_date_re = re.compile(
            r"/\d{4}/\d{2}/\d{2}/|/\d{8}[-_]|\d{4}-\d{2}-\d{2}"
            r"|/[a-z0-9-]{15,}/?$"  # article slug (was 30, lowered to 15)
        )
        week_entry_lines: list[str] = []
        week_entry_sources: list[dict[str, str]] = []
        seen_week_urls: set[str] = set()
        fetch_prompt_week = _build_generic_fetch_prompt(last_message)

        async def _week_entry(c: dict[str, str]) -> tuple[str, str]:
            url = c.get("url", "")
            snippet = c.get("snippet", "").strip()
            # Strip markdown from snippet
            snippet = re.sub(r"^#+\s+", "", snippet)
            snippet = re.sub(r"\s+#+\s+", " ", snippet).strip()
            title = re.sub(r"^#+\s+", "", c.get("title", url)).strip()
            # Fetch if URL looks like a specific article: date in path, long slug,
            # or non-trivial path with ≥5-char last segment (catches nippon.com IDs like d01194)
            _path_fetch = urlparse(url).path.rstrip("/")
            _last_seg_fetch = _path_fetch.rsplit("/", 1)[-1] if _path_fetch else ""
            _is_article_url_fetch = bool(_url_date_re.search(url)) or (
                _path_fetch.count("/") >= 2 and len(_last_seg_fetch) >= 5
            )
            if _is_article_url_fetch:
                try:
                    fetched = await _fetch_web_page_follow_redirect(url, fetch_prompt_week, use_dynamic=False)
                    if (
                        isinstance(fetched, str)
                        and not fetched.startswith("Error")
                        and not fetched.startswith("URL rechazada")
                        and not _is_no_info_response(fetched)
                        and len(fetched.split()) >= 20
                    ):
                        lines = _extract_generic_content_lines(fetched, query_terms)
                        if lines:
                            return title, " ".join(lines[:3])
                except Exception:
                    pass
            # Fallback to Tavily snippet
            return title, snippet

        week_results = await asyncio.gather(*[_week_entry(c) for c in diverse_candidates])

        for (title, content), c in zip(week_results, diverse_candidates):
            if not content or len(content.split()) < 8:
                continue
            # Discard entries that are no-info placeholders
            if _is_no_info_response(content):
                continue
            url = c.get("url", "")
            if url in seen_week_urls:
                continue
            seen_week_urls.add(url)
            week_entry_lines.append(f"[{title}] — {content}")
            # Include URL in Sources if it looks like a specific article:
            # has date/long-slug pattern, OR has a non-trivial path (≥2 segments, last ≥5 chars)
            path = urlparse(url).path.rstrip("/")
            last_segment = path.rsplit("/", 1)[-1] if path else ""
            is_article_url = bool(_url_date_re.search(url)) or (
                path.count("/") >= 2 and len(last_segment) >= 5
            )
            if is_article_url:
                week_entry_sources.append({"title": title, "url": url})

        if len(week_entry_lines) >= 3:
            # Format bullets directly — each fetched entry is already LLM-processed.
            # Bypassing _synthesize_search_summary avoids the LLM merging distinct articles.
            import datetime as _dt_week
            _current_year = _dt_week.date.today().year
            _old_year_re = re.compile(r'\b(20\d{2})\b')
            paragraph_parts = []
            for (content_title, content), c in zip(week_results, diverse_candidates):
                title = content_title or c.get("title") or c.get("url") or ""
                # Skip entries whose content ONLY references years before the current year.
                # "18 de noviembre de 2025" is 5 months old — not "this week".
                # Only kept if the content also mentions the current year as context.
                _years = [int(y) for y in _old_year_re.findall(content)]
                if _years and max(_years) <= _current_year - 1:
                    continue
                # Trim to 3 sentences max to avoid duplicated text in long fetches
                sentences = re.split(r"(?<=[.!?])\s+", content)
                trimmed = " ".join(sentences[:3]).strip()
                # Discard truncated snippets (Tavily cuts them with "…" or "...")
                is_truncated = trimmed.endswith(("…", "...")) or re.search(r"\w…$", trimmed)
                if trimmed and not _is_no_info_response(trimmed) and not is_truncated:
                    paragraph_parts.append(f"{title}: {trimmed}")
            bullets_text = "\n\n".join(paragraph_parts)
            sources_block = _format_sources(week_entry_sources)
            summary = f"{bullets_text}\n\n{sources_block}".strip() if sources_block else bullets_text
            return {
                "summary": summary,
                "words": summary.split(),
                "source_type": "search",
                "sources": week_entry_sources,
                "pre_synthesized": True,
            }
        # Not enough content — fall through to page fetch approach

    if not ranked_candidates:
        search_lines = _extract_generic_content_lines(search_text, query_terms)
        if not search_lines:
            return None
        sources = _extract_sources_from_text(search_text)
        if not sources:
            sources = [{"title": "search result", "url": ""}]
        summary = _build_source_backed_response(search_lines[:8], sources)
        return {
            "summary": summary,
            "words": summary.split(),
            "source_type": "search",
            "sources": sources,
            "search_text": search_text,
        }

    fetch_prompt = _build_generic_fetch_prompt(last_message)

    async def _fetch_candidate(candidate: dict[str, str]) -> tuple[dict[str, str], Any]:
        try:
            result = await _fetch_web_page_follow_redirect(candidate["url"], fetch_prompt, use_dynamic=True)
            return candidate, result
        except Exception as exc:  # pragma: no cover - defensive
            return candidate, exc

    fetched_results = await asyncio.gather(
        *(_fetch_candidate(candidate) for candidate in ranked_candidates),
        return_exceptions=False,
    )

    eligible_entries: list[dict[str, Any]] = []
    for candidate, result in fetched_results:
        if isinstance(result, Exception):
            continue
        if not isinstance(result, str):
            result = str(result)
        if result.startswith("Error") or result.startswith("URL rechazada"):
            continue
        if _is_no_info_response(result):
            continue

        body_lines = _extract_generic_content_lines(result, query_terms)
        candidate_score = _score_generic_candidate(candidate, query_terms, query_source_group)
        content_score = len(body_lines) * 2
        if query_terms and not body_lines:
            result_blob = result.lower()
            if any(term in result_blob for term in query_terms):
                content_score = 1
        if not body_lines and content_score <= 1:
            # Fallback: use the Tavily snippet so diverse candidates aren't dropped
            # just because the fetched page returned poor content
            tavily_snippet = candidate.get("snippet", "").strip()
            if tavily_snippet and len(tavily_snippet.split()) >= 6:
                body_lines = [tavily_snippet]
                content_score = 4
            else:
                continue

        score = candidate_score + content_score
        if score <= 0:
            continue
        if _is_recent_web_information_query(last_message):
            if score < recent_min_score or len(body_lines) < recent_candidate_min_body_lines:
                continue

        fallback_lines = [line.strip() for line in result.splitlines() if line.strip() and not line.strip().lower().startswith(("url:", "sources:", "http")) and "http" not in line.lower()]
        summary_lines = body_lines or _extract_generic_content_lines(search_text, query_terms) or fallback_lines[:5]
        if len(summary_lines) < 3 and fallback_lines:
            seen_lines = set(summary_lines)
            for line in fallback_lines:
                if line not in seen_lines:
                    summary_lines.append(line)
                    seen_lines.add(line)
                if len(summary_lines) >= 6:
                    break
        sources = _extract_sources_from_text(result)
        if not sources:
            sources = [{"title": candidate.get("title") or candidate["url"], "url": candidate["url"]}]
        if _is_recent_web_information_query(last_message) and len(sources) < recent_candidate_min_sources:
            continue

        eligible_entries.append({
            "summary_lines": summary_lines[:10],
            "sources": sources,
            "score": score,
            "candidate": candidate,
        })

    if query_horizon == "week" and eligible_entries:
        ordered_entries = sorted(eligible_entries, key=lambda entry: cast(int, entry["score"]), reverse=True)
        max_size = min(len(ordered_entries), max(4, recent_min_candidates))
        for size in range(min(recent_min_candidates, max_size), max_size + 1):
            selected = ordered_entries[:size]
            combined_score = sum(cast(int, entry["score"]) for entry in selected)
            combined_lines: list[str] = []
            seen_lines: set[str] = set()
            combined_sources: list[dict[str, str]] = []
            seen_urls: set[str] = set()

            for entry in selected:
                for line in cast(list[str], entry["summary_lines"]):
                    normalized_line = re.sub(r"\s+", " ", line).strip().lower()
                    if normalized_line and normalized_line not in seen_lines:
                        seen_lines.add(normalized_line)
                        combined_lines.append(line)
                for source in cast(list[dict[str, str]], entry["sources"]):
                    url = str(source.get("url") or "").strip()
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        combined_sources.append(source)

            if combined_score >= recent_min_score and len(combined_lines) >= recent_min_body_lines and len(combined_sources) >= recent_min_sources:
                # Supplement with Tavily article URLs when web_fetch only got few sources
                final_sources = list(combined_sources)
                if len(final_sources) < 4:
                    seen_urls: set[str] = {str(s.get("url") or "") for s in final_sources if s.get("url")}
                    for extra in _extract_sources_from_text(search_text):
                        extra_url = str(extra.get("url") or "")
                        if extra_url and extra_url not in seen_urls and len(final_sources) < 5:
                            final_sources.append(extra)
                            seen_urls.add(extra_url)
                summary = _build_source_backed_response(combined_lines[:20], final_sources)
                return {
                    "summary": summary,
                    "words": summary.split(),
                    "source_type": "webfetch",
                    "sources": final_sources,
                    "score": combined_score,
                }

    # Week fallback: page fetches gave < min_candidates diverse entries —
    # use Tavily snippets from diverse_candidates directly (more reliable for recent paywalled articles)
    if query_horizon == "week" and len(eligible_entries) < recent_min_candidates:
        snippet_lines: list[str] = []
        snippet_sources: list[dict[str, str]] = []
        seen_snippet_urls: set[str] = set()
        for c in diverse_candidates:
            snippet = c.get("snippet", "").strip()
            if not snippet or len(snippet.split()) < 6:
                continue
            url = c.get("url", "")
            if url in seen_snippet_urls:
                continue
            seen_snippet_urls.add(url)
            title = c.get("title", url)
            snippet_lines.append(f"{title} — {snippet}")
            snippet_sources.append({"title": title, "url": url})
        if len(snippet_lines) >= 2:
            summary = _build_source_backed_response(snippet_lines, snippet_sources)
            return {
                "summary": summary,
                "words": summary.split(),
                "source_type": "search",
                "sources": snippet_sources,
            }

    if query_horizon != "week" and eligible_entries:
        best_entry = max(eligible_entries, key=lambda entry: cast(int, entry["score"]))
        summary = _build_source_backed_response(cast(list[str], best_entry["summary_lines"]), cast(list[dict[str, str]], best_entry["sources"]))
        return {
            "summary": summary,
            "words": summary.split(),
            "source_type": "webfetch",
            "sources": cast(list[dict[str, str]], best_entry["sources"]),
            "score": cast(int, best_entry["score"]),
        }

    search_lines = _extract_generic_content_lines(search_text, query_terms)
    if not search_lines:
        return None
    if _is_recent_web_information_query(last_message):
        if len(search_lines) < recent_min_body_lines:
            strongest_candidate = ranked_candidates[0] if ranked_candidates else None
            strongest_score = _score_generic_candidate(strongest_candidate, query_terms, query_source_group) if strongest_candidate else 0
            if not (
                strongest_candidate is not None
                and strongest_score >= recent_min_score
                and _is_specific_article_hit({
                    "title": strongest_candidate.get("title") or "",
                    "link": strongest_candidate.get("url") or strongest_candidate.get("link") or "",
                    "snippet": strongest_candidate.get("snippet") or "",
                })
            ):
                return None
        sources = _extract_sources_from_text(search_text)
        if len(sources) < recent_min_sources:
            return None

    sources = _extract_sources_from_text(search_text)
    if not sources:
        top = ranked_candidates[0]
        sources = [{"title": top.get("title") or top["url"], "url": top["url"]}]

    if len(search_lines) < 3:
        search_fallback_lines = [line.strip() for line in search_text.splitlines() if line.strip() and not line.strip().lower().startswith(("url:", "sources:", "http")) and "http" not in line.lower()]
        for line in search_fallback_lines:
            if line not in search_lines:
                search_lines.append(line)
            if len(search_lines) >= 6:
                break

    summary = _build_source_backed_response(search_lines[:8], sources)
    return {
        "summary": summary,
        "words": summary.split(),
        "source_type": "search",
        "sources": sources,
        "search_text": search_text,
    }


def _enforce_synthesis_format(text: str) -> str:
    """Post-process LLM output to guarantee bullet spacing and strip header artifacts."""
    lines = text.splitlines()
    result: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Remove markdown headers the LLM may have emitted despite instructions
        if re.match(r"^#{1,4}\s", stripped):
            continue
        # Ensure blank line before every bullet (•, -, *) that starts a new point
        if re.match(r"^[•\-\*]\s", stripped) and result and result[-1].strip():
            result.append("")
        result.append(stripped)
        # Ensure blank line after every bullet line (before next non-empty line)
        if re.match(r"^[•\-\*]\s", stripped):
            # Peek ahead: if next non-empty line isn't a blank, we'll add one later
            pass
    # Second pass: ensure blank line after each bullet block
    final: list[str] = []
    for i, line in enumerate(result):
        final.append(line)
        if re.match(r"^[•\-\*]\s", line):
            # Add blank after bullet if next line is non-empty content
            if i + 1 < len(result) and result[i + 1].strip():
                final.append("")
    # Collapse 3+ consecutive blank lines to 2
    collapsed: list[str] = []
    blank_count = 0
    for line in final:
        if not line.strip():
            blank_count += 1
            if blank_count <= 2:
                collapsed.append(line)
        else:
            blank_count = 0
            collapsed.append(line)
    return "\n".join(collapsed).strip()


def _dedup_synthesis_bullets(text: str, query_terms: Optional[list[str]] = None) -> str:
    """Remove duplicate bullets from a synthesized response.

    Two bullets are considered duplicates when their non-query keyword overlap ≥ 3.
    Keeps the longer (more informative) bullet of each duplicate pair.
    """
    excluded = set(t.lower() for t in (query_terms or []))
    excluded.update(_TITLE_STOPWORDS)

    def kw(s: str) -> set[str]:
        words = set()
        for w in s.split():
            w = w.lower()
            if len(w) <= 4 or w in excluded:
                continue
            # Normalize plural: "terremotos" → "terremoto", "desastres" → "desastre"
            if w.endswith("s") and len(w) > 5:
                w = w[:-1]
            words.add(w)
        return words

    # Split into (bullet_block, non_bullet_prefix) sections
    # A bullet block = the • line plus any continuation lines before the next bullet
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if re.match(r"^[•\-\*]\s", line.strip()) and current:
            blocks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))

    # Separate bullet blocks from header/non-bullet prefix
    bullet_blocks: list[str] = []
    prefix_lines: list[str] = []
    for block in blocks:
        if re.match(r"^[•\-\*]\s", block.strip()):
            bullet_blocks.append(block)
        else:
            prefix_lines.append(block)

    _DISASTER_KW = {"terremoto", "sismo", "maremoto", "tsunami", "earthquake", "seismic"}

    # Dedup bullet blocks
    accepted: list[str] = []
    for block in bullet_blocks:
        block_kw = kw(block)
        duplicate = False
        for i, acc in enumerate(accepted):
            acc_kw = kw(acc)
            # Use a lower threshold when both bullets are about natural disasters
            # (earthquake variants share few unique words but are clearly the same topic)
            both_disaster = bool(block_kw & _DISASTER_KW) and bool(acc_kw & _DISASTER_KW)
            dedup_threshold = 2 if both_disaster else 4
            if len(block_kw & acc_kw) >= dedup_threshold:
                # Keep the longer one
                if len(block) > len(acc):
                    accepted[i] = block
                duplicate = True
                break
        if not duplicate:
            accepted.append(block)

    all_parts = prefix_lines + accepted
    return "\n\n".join(p.strip() for p in all_parts if p.strip())


async def _synthesize_search_summary(
    raw_summary: str,
    query: str,
    get_llm_fn: Callable,
    sources: list[dict[str, str]],
) -> str:
    """Passes raw search content through LLM to produce a clean, structured response."""
    try:
        llm = get_llm_fn(temperature=0.2)
        sources_block = _format_sources(sources)
        clean_lines = []
        for line in raw_summary.splitlines():
            stripped = line.strip()
            if not stripped or stripped in ("...", "[...]"):
                continue
            # Markdown headers from raw scraped pages
            if re.match(r"^#{1,3}\s", stripped):
                continue
            # Author bylines: "Name Name · date" or "Name Name - date"
            if re.search(r"\w+\s+\w+\s+[·\-]\s+\d{1,2}\s+de\s+\w+", stripped):
                continue
            # Image slugs and filenames with timestamps
            if re.match(r"^\d{8,14}[_\-]\w", stripped):
                continue
            # URL-path-like slugs without spaces
            if re.match(r"^[\w\-]+(?:[_\-][\w\-]+){3,}$", stripped) and " " not in stripped:
                continue
            # Lines with heavily hyphenated words (image alt text artifacts)
            if any(word.count("-") >= 3 for word in stripped.split()):
                continue
            clean_lines.append(stripped)
        import datetime
        today_str = datetime.date.today().strftime("%d de %B de %Y")
        clean_content = "\n\n".join(clean_lines[:40])
        query_terms_for_dedup = _extract_generic_query_terms(query)
        query_horizon_local = detect_recent_query_horizon(query) if _is_recent_web_information_query(query) else None
        prompt = (
            f"Fecha actual: {today_str}\n"
            f"Consulta del usuario: {query}\n\n"
            f"Información recopilada de la web:\n{clean_content}\n\n"
            "Sintetizá una respuesta clara respondiendo ÚNICAMENTE con lo que está en el texto de arriba. "
            "PROHIBIDO usar conocimiento propio o información que no esté en el texto provisto.\n\n"
            "Reglas de contenido:\n"
            "- Usá el mismo idioma que la consulta\n"
            "- IGNORÁ completamente: pie de fotos, descripciones de imágenes, nombres de personas sin contexto noticioso, títulos de anime/manga, fragmentos sin información útil\n"
            "- PRIORIZÁ: artículos con hechos concretos, cifras, eventos, decisiones o noticias verificables\n"
            "- Si el contenido disponible no responde bien la consulta, indicalo brevemente\n\n"
            "Reglas de formato:\n"
            "- Cada punto DEBE comenzar con '•' seguido de un espacio\n"
            "- Cada artículo/fuente del texto = UN punto separado, pero TODOS deben responder al mismo tema solicitado.\n"
            "  Ejemplo: si la consulta es seguridad japonesa, podés usar un artículo sobre misiles y otro sobre una embajada,\n"
            "  pero no mezcles clima, deportes o política general.\n"
            "- NUNCA combines dos artículos en un solo punto. Cada punto viene de UNA sola fuente y no debe repetir la misma noticia.\n"
            "- Si un artículo tiene información irrelevante para la consulta (noticias de otro país, entretenimiento, deportes sin relación), omitilo.\n"
            "- OBLIGATORIO: dejá UNA línea en blanco entre cada punto\n"
            "- Cada punto tiene 2-3 oraciones con el hecho concreto, quiénes están involucrados y por qué importa\n"
            "- NO uses títulos ni headers (##, ###) dentro de la respuesta\n"
            "- NO incluyas una sección Sources — se agrega automáticamente"
        )
        if query_horizon_local == "week":
            cutoff = (datetime.date.today() - datetime.timedelta(days=30)).strftime('%d/%m/%Y')
            prompt += f"\n- Solo incluí eventos de los últimos 30 días (desde el {cutoff}). Descartá cualquier evento más antiguo aunque esté en el texto."
        elif query_horizon_local:
            prompt += f"\n- Solo incluí eventos ocurridos en los últimos 30 días. Descartá cualquier evento más antiguo."
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        synthesized = getattr(response, "content", str(response)).strip()
        # Strip any LLM-generated Sources section (always unreliable) and replace with real one
        synthesized = re.split(r"\n\s*sources\s*:", synthesized, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        synthesized = _enforce_synthesis_format(synthesized)
        synthesized = _dedup_synthesis_bullets(synthesized, query_terms_for_dedup)
        if sources_block:
            synthesized = f"{synthesized}\n\n{sources_block}"
        return synthesized
    except Exception as _synth_exc:
        import logging
        logging.warning(f"_synthesize_search_summary FAILED: {type(_synth_exc).__name__}: {_synth_exc}")
        return _enforce_synthesis_format(raw_summary)


async def run_web_scraping_flow(
    state: AgentState,
    agent,
    get_llm_fn: Callable,
    *,
    hitl_enabled: bool,
    confirmation_handler: Optional[ConfirmationPort] = None,
    ask_confirmation_compat: Optional[Callable[[str], Awaitable[bool]]] = None,
    get_runtime_policy: Callable[[], dict],
    evaluate_trajectory_safe_fn=evaluate_trajectory_safe,
    should_evaluate_guard_fn=_should_evaluate_guard,
) -> dict[str, Any]:
    messages = state["messages"]
    last_message = get_last_message_text(messages)
    state_dict = cast(dict[str, Any], state)
    web_search_runtime_args = _web_search_runtime_args(state_dict)
    rid = get_or_create_request_id(state_dict, lambda: "")
    t0 = time.time()

    if not rid:
        rid = str(uuid.uuid4())

    explicit_urls = _extract_urls_from_text(last_message)
    if hitl_enabled:
        url_info = f" → URLs: {', '.join(explicit_urls)}" if explicit_urls else ""
        preview = last_message[:120] + ("..." if len(last_message) > 120 else "")
        needs_confirmation = bool(explicit_urls)

        confirmed = True
        if needs_confirmation:
            confirmed = False
            if confirmation_handler is not None:
                confirmed = await confirmation_handler.confirm(
                    f"\n[HITL] web_scraping_agent va a procesar: \"{preview}\"{url_info}\n¿Confirmar? [s/n]: "
                )
            elif ask_confirmation_compat is not None:
                confirmed = await ask_confirmation_compat(
                    f"\n[HITL] web_scraping_agent va a procesar: \"{preview}\"{url_info}\n¿Confirmar? [s/n]: "
                )
        if not confirmed:
            _emit_node_outcome(
                rid, "web_scraping_node", "blocked", phase="pre_guard",
                agent="web_scraping_agent",
                duration_ms=int((time.time() - t0) * 1000),
                reason="hitl_rejected",
            )
            return {"messages": [AIMessage(content="Operación cancelada por el usuario.")]}

    ctx = _select_strategy_context(state, last_message, get_runtime_policy)
    tracker = ctx["tracker"]
    turn_count = ctx["turn_count"]
    category = ctx["category"]
    prior_score = ctx["prior_score"]
    prior_reliability = ctx["prior_reliability"]
    ml_recommended = ctx["ml_recommended"]
    strategy = ctx["strategy"]
    exploring = ctx["exploring"]
    exp_rate = ctx["exp_rate"]
    prediction_match = ctx["prediction_match"]

    try:
        if explicit_urls:
            fetch_prompt = last_message.strip() or "Extraé la información relevante de esta URL."
            fetch_result = await _fetch_web_page_follow_redirect(explicit_urls[0], fetch_prompt, use_dynamic=True)
            if isinstance(fetch_result, str) and not fetch_result.startswith("Error") and not fetch_result.startswith("URL rechazada"):
                summary = fetch_result.strip()
                words = summary.split()
                duration_ms = int((time.time() - t0) * 1000)
                reliability = _scrape_reliability(len(words))
                new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _update_scrape_tracker(
                    tracker, category, len(words), turn_count,
                    duration_ms=duration_ms, cost_usd=0.0,
                    source_type="webfetch", reliability_override=reliability,
                ))
                analytics = cast(dict[str, Any], analytics)
                new_score = _get_category_score(new_tracker, category, turn_count)

                _emit_node_outcome(
                    rid, "web_scraping_node", "success", phase="agent",
                    agent="web_scraping_agent", duration_ms=duration_ms,
                    category=category, exploring=False, strategy="web_fetch", exp_rate=0.0,
                    scrape_reliability=reliability, prior_reliability=prior_reliability,
                    prior_score=prior_score, scrape_score=new_score,
                    retry_done=False, source_type="webfetch",
                    **_extract_tokens({"messages": []}), **_extract_quality({"messages": []}),
                    **_extract_followup({"messages": []}, "success"), **analytics, **_node_meta(),
                )
                return {
                    "messages": [AIMessage(content=summary)],
                    "scrape_tracker": new_tracker,
                }

        if category in {"sports", "news"}:
            discovery = await _run_generic_web_search_fetch(last_message, web_search_runtime_args)
            if discovery is not None:
                _disc_raw = cast(str, discovery["summary"])
                _disc_sources = cast(list[dict[str, str]], discovery.get("sources") or [])
                if discovery.get("pre_synthesized"):
                    summary = _disc_raw
                else:
                    summary = await _synthesize_search_summary(_disc_raw, last_message, get_llm_fn, _disc_sources)
                words = cast(list[str], discovery.get("words") or summary.split())
                duration_ms = int((time.time() - t0) * 1000)
                reliability = _scrape_reliability(len(words))
                source_type = cast(str, discovery.get("source_type") or "webfetch")
                new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _update_scrape_tracker(
                    tracker, category, len(words), turn_count,
                    duration_ms=duration_ms, cost_usd=0.0,
                    source_type=source_type, reliability_override=reliability,
                ))
                analytics = cast(dict[str, Any], analytics)
                new_score = _get_category_score(new_tracker, category, turn_count)

                _emit_node_outcome(
                    rid, "web_scraping_node", "success", phase="agent",
                    agent="web_scraping_agent", duration_ms=duration_ms,
                    category=category, exploring=False, strategy="search_web" if source_type == "search" else "web_search_fetch", exp_rate=0.0,
                    scrape_reliability=reliability, prior_reliability=prior_reliability,
                    prior_score=prior_score, scrape_score=new_score,
                    retry_done=False, source_type=source_type,
                    ml_recommended=ml_recommended, prediction_match=prediction_match,
                    ml_would_succeed=(bool(analytics.get("quality_target", 0)) if prediction_match is True else None),
                    **_extract_tokens({"messages": []}), **_extract_quality({"messages": []}), **_extract_followup({"messages": []}, "success"), **analytics, **_node_meta(),
                )
                return {
                    "messages": [AIMessage(content=summary)],
                    "scrape_tracker": new_tracker,
                }

            from tools import search_web

            loop = asyncio.get_running_loop()
            fallback_search = await loop.run_in_executor(
                None,
                lambda: search_web.invoke({"query": last_message, "use_cache": False, **web_search_runtime_args}),
            )
            if not isinstance(fallback_search, str):
                fallback_search = str(fallback_search)
            fallback_terms = _extract_generic_query_terms(last_message)
            fallback_query_source_group = detect_query_source_group(last_message)
            fallback_lines = _extract_generic_content_lines(fallback_search, fallback_terms)
            if fallback_lines:
                fallback_candidates = _extract_generic_search_candidates(fallback_search)
                fallback_sources = _extract_sources_from_text(fallback_search)
                if fallback_candidates:
                    top_candidate = max(
                        fallback_candidates,
                        key=lambda candidate: _score_generic_candidate(candidate, fallback_terms, fallback_query_source_group),
                    )
                    fallback_sources = [{
                        "title": top_candidate.get("title") or top_candidate.get("url") or "search result",
                        "url": top_candidate.get("url") or "",
                    }]
                elif not fallback_sources:
                    fallback_sources = [{"title": "search result", "url": ""}]
                _fallback_raw = _build_source_backed_response(fallback_lines[:10], fallback_sources)
                summary = await _synthesize_search_summary(_fallback_raw, last_message, get_llm_fn, fallback_sources)
                duration_ms = int((time.time() - t0) * 1000)
                words = summary.split()
                reliability = _scrape_reliability(len(words))
                new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _update_scrape_tracker(
                    tracker, category, len(words), turn_count,
                    duration_ms=duration_ms, cost_usd=0.0,
                    source_type="search", reliability_override=reliability,
                ))
                analytics = cast(dict[str, Any], analytics)
                new_score = _get_category_score(new_tracker, category, turn_count)
                _emit_node_outcome(
                    rid, "web_scraping_node", "success", phase="agent",
                    agent="web_scraping_agent", duration_ms=duration_ms,
                    category=category, exploring=False, strategy="search_web", exp_rate=0.0,
                    scrape_reliability=reliability, prior_reliability=prior_reliability,
                    prior_score=prior_score, scrape_score=new_score,
                    retry_done=False, source_type="search",
                    ml_recommended=ml_recommended, prediction_match=prediction_match,
                    ml_would_succeed=(bool(analytics.get("quality_target", 0)) if prediction_match is True else None),
                    **_extract_tokens({"messages": []}), **_extract_quality({"messages": []}), **_extract_followup({"messages": []}, "success"), **analytics, **_node_meta(),
                )
                return {
                    "messages": [AIMessage(content=summary)],
                    "scrape_tracker": new_tracker,
                }

        if is_web_information_query(last_message) or _is_recent_web_information_query(last_message):
            discovery = await _run_generic_web_search_fetch(last_message, web_search_runtime_args)
            if discovery is not None:
                _disc_raw = cast(str, discovery["summary"])
                _disc_sources = cast(list[dict[str, str]], discovery.get("sources") or [])
                if discovery.get("pre_synthesized"):
                    summary = _disc_raw
                else:
                    summary = await _synthesize_search_summary(_disc_raw, last_message, get_llm_fn, _disc_sources)
                words = cast(list[str], discovery.get("words") or summary.split())
                duration_ms = int((time.time() - t0) * 1000)
                reliability = _scrape_reliability(len(words))
                source_type = cast(str, discovery.get("source_type") or "webfetch")
                new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _update_scrape_tracker(
                    tracker, category, len(words), turn_count,
                    duration_ms=duration_ms, cost_usd=0.0,
                    source_type=source_type, reliability_override=reliability,
                ))
                analytics = cast(dict[str, Any], analytics)
                new_score = _get_category_score(new_tracker, category, turn_count)

                _emit_node_outcome(
                    rid, "web_scraping_node", "success", phase="agent",
                    agent="web_scraping_agent", duration_ms=duration_ms,
                    category=category, exploring=False, strategy="search_web" if source_type == "search" else "web_search_fetch", exp_rate=0.0,
                    scrape_reliability=reliability, prior_reliability=prior_reliability,
                    prior_score=prior_score, scrape_score=new_score,
                    retry_done=False, source_type=source_type,
                    ml_recommended=ml_recommended, prediction_match=prediction_match,
                    ml_would_succeed=(bool(analytics.get("quality_target", 0)) if prediction_match is True else None),
                    **_extract_tokens({"messages": []}), **_extract_quality({"messages": []}), **_extract_followup({"messages": []}, "success"), **analytics, **_node_meta(),
                )
                return {
                    "messages": [AIMessage(content=summary)],
                    "scrape_tracker": new_tracker,
                }

        agent_hint = (
            "[Sistema | web] Usa search_web para descubrir fuentes dinámicamente. "
            "Si la consulta es de información reciente, incluí el año actual en la búsqueda. "
            "Para noticias o información reciente, hacé varias búsquedas antes de responder; search_web puede usarse varias veces. "
            "Después usa web_fetch sobre varias URLs relevantes, no solo la primera. "
            "Si la consulta pide noticias o actualidad, reuní varias fuentes antes de responder. "
            "Si una fuente mezcla temas, países o resultados no relacionados, recházala y vuelve a buscar. "
            "Si web_fetch informa un redirect a otro host, repetí web_fetch con la URL de redirect. "
            "No respondas hasta tener fuentes que apoyen directamente la afirmación; si aparece una noticia vieja, un evento futuro o una página evergreen, descartala. "
            "Si la respuesta es de noticias o actualidad, desarrollala en 4 párrafos breves sobre el mismo tema solicitado, sin repetir noticias. "
            "Tu respuesta final debe incluir un bloque Sources con enlaces markdown.\n\n"
        )
        raw_result = await agent.ainvoke(
            {"messages": [HumanMessage(content=agent_hint + last_message)]},
            config=RunnableConfig(
                tags=["web_scraping", "agent", "high_risk", "context_quarantine"],
                metadata={
                    "node": "web_scraping_node",
                    "agent": "web_scraping_agent",
                    "request_id": rid,
                    "input_chars": len(last_message),
                    "prior_reliability": prior_reliability,
                },
            ),
        )

        tokens = _extract_tokens(raw_result)
        quality = _extract_quality(raw_result)
        followup = _extract_followup(raw_result, "success")
        meta = _node_meta()

        if should_evaluate_guard_fn("web_scraping_node"):
            is_safe, _ = await evaluate_trajectory_safe_fn(
                {
                    "messages": raw_result.get("messages", []),
                    "next_agent": state.get("next_agent", ""),
                },
                "web_scraping_node",
            )
            if not is_safe:
                _emit_node_outcome(
                    rid, "web_scraping_node", "blocked", phase="post_guard",
                    agent="web_scraping_agent",
                    duration_ms=int((time.time() - t0) * 1000),
                    reason="agentdog",
                    followup_likely=True,
                    **tokens, **quality, **meta,
                )
                return {"messages": [AIMessage(content="Respuesta retenida por política de seguridad.")]}

        raw_messages = raw_result.get("messages", [])
        raw_text = extract_final_ai_text(raw_messages)
        if not raw_text:
            _emit_node_outcome(
                rid, "web_scraping_node", "error", phase="agent",
                agent="web_scraping_agent",
                duration_ms=int((time.time() - t0) * 1000),
                reason="empty_response",
                followup_likely=True,
                **meta,
            )
            return {"messages": [AIMessage(content="No se pudo extraer información de la página.")]}

        sources_from_raw = _extract_sources_from_text(raw_text)
        if sources_from_raw or len(raw_text.split()) < 80:
            summary = await _synthesize_search_summary(raw_text, last_message, get_llm_fn, sources_from_raw)
        else:
            summary = await _summarize_if_long(raw_text, rid, get_llm_fn)
        words = raw_text.split()
        summary_triggered = len(words) > 200
        duration_ms = int((time.time() - t0) * 1000)
        reliability = _scrape_reliability(len(words))
        source_type = "agent"
        retry_done = False

        if reliability in {"unreliable"}:
            _emit_node_outcome(
                rid, "web_scraping_node", "retry", phase="agent",
                agent="web_scraping_agent", duration_ms=duration_ms,
                reason=f"auto_retry:{reliability}",
                scrape_reliability=reliability, strategy="web_search_fetch",
                source_type=source_type, category=category, **tokens, **_node_meta(),
            )
            retry_summary, retry_words, retry_tokens, retry_quality = await _run_retry_agent(
                agent, last_message, rid, get_llm_fn,
            )
            if retry_summary is not None:
                summary = retry_summary
                words = cast(list[str], retry_words or [])
                tokens = cast(dict[str, Any], retry_tokens or {})
                quality = cast(dict[str, Any], retry_quality or {})
            retry_done = True
            duration_ms = int((time.time() - t0) * 1000)
            reliability = _scrape_reliability(len(words))

        cost_usd = tokens.get("estimated_cost_usd")
        new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _update_scrape_tracker(
            tracker, category, len(words), turn_count,
            duration_ms=duration_ms, cost_usd=cost_usd,
            source_type=source_type, reliability_override=reliability,
        ))
        analytics = cast(dict[str, Any], analytics)
        new_score = _get_category_score(new_tracker, category, turn_count)
        if reliability not in ("ok_weak", "ok_strong"):
            followup = {"followup_likely": True}

        _emit_node_outcome(
            rid, "web_scraping_node", "success", phase="agent",
            agent="web_scraping_agent", duration_ms=duration_ms,
            summary_triggered=summary_triggered, raw_words=len(words),
            category=category, exploring=False, strategy="web_search_fetch", exp_rate=0.0,
            scrape_reliability=reliability, prior_reliability=prior_reliability,
            prior_score=prior_score, scrape_score=new_score,
            retry_done=retry_done,
            source_type=source_type,
            **tokens, **quality, **followup, **analytics, **meta,
        )
        return {
            "messages": [AIMessage(content=summary)],
            "scrape_tracker": new_tracker,
        }

    except Exception as e:
        _emit_node_outcome(
            rid, "web_scraping_node", "error", phase="agent",
            agent="web_scraping_agent",
            duration_ms=int((time.time() - t0) * 1000),
            reason=str(e),
            followup_likely=True,
            **_node_meta(),
        )
        raise
