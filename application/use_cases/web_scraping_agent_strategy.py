"""Agent execution branch for web scraping."""
from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, Optional, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig

from application.services.prompt_loader import load_agent_prompt


async def _run_web_scraping_agent_strategy(
    *,
    state,
    agent,
    get_llm_fn: Callable,
    last_message: str,
    category: str,
    tracker: dict[str, Any],
    turn_count: int,
    prior_score: Any,
    prior_reliability: Any,
    ml_recommended: Any,
    prediction_match: Any,
    rid: str,
    t0: float,
    web_search_runtime_args: Optional[dict[str, Any]],
    should_evaluate_guard_fn: Callable,
    evaluate_trajectory_safe_fn: Callable,
) -> dict[str, Any]:
    from application.use_cases import web_scraping_flow as _flow

    agent_prompt = load_agent_prompt("web_scraping_agent")
    try:
        raw_result = await agent.ainvoke(
            {"messages": [HumanMessage(content=f"{agent_prompt}\n\n{last_message}")]},
            config=RunnableConfig(
                tags=["web_scraping", "agent", "high_risk", "context_quarantine"],
                metadata={
                    "node": "web_scraping_node",
                    "agent": "web_scraping_agent",
                    "request_id": rid,
                    "input_chars": len(last_message),
                    "prior_reliability": prior_reliability,
                },
                recursion_limit=16,
            ),
        )
    except Exception as exc:
        _flow._web_debug("run_web_scraping_flow.agent_exception", error=repr(exc))
        fallback_discovery = await _flow._run_generic_web_search_fetch(last_message, web_search_runtime_args)
        if fallback_discovery is not None:
            _flow._web_debug(
                "run_web_scraping_flow.agent_exception_recovered",
                source_type=fallback_discovery.get("source_type"),
                source_count=len(cast(list[dict[str, str]], fallback_discovery.get("sources") or [])),
            )
            summary = cast(str, fallback_discovery["summary"])
            words = cast(list[str], fallback_discovery.get("words") or summary.split())
            duration_ms = int((time.time() - t0) * 1000)
            reliability = _flow._scrape_reliability(len(words))
            source_type = cast(str, fallback_discovery.get("source_type") or "search")
            new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _flow._update_scrape_tracker(
                tracker, category, len(words), turn_count,
                duration_ms=duration_ms, cost_usd=0.0,
                source_type=source_type, reliability_override=reliability,
            ))
            analytics = cast(dict[str, Any], analytics)
            new_score = _flow._get_category_score(new_tracker, category, turn_count)
            _flow._emit_node_outcome(
                rid, "web_scraping_node", "success", phase="agent",
                agent="web_scraping_agent", duration_ms=duration_ms,
                category=category, exploring=False, strategy="search_web" if source_type == "search" else "web_search_fetch", exp_rate=0.0,
                scrape_reliability=reliability, prior_reliability=prior_reliability,
                prior_score=prior_score, scrape_score=new_score,
                retry_done=False, source_type=source_type,
                ml_recommended=ml_recommended, prediction_match=prediction_match,
                ml_would_succeed=(bool(analytics.get("quality_target", 0)) if prediction_match is True else None),
                **_flow._extract_tokens({"messages": []}), **_flow._extract_quality({"messages": []}), **_flow._extract_followup({"messages": []}, "success"), **analytics, **_flow._node_meta(),
            )
            return await _flow._guardrail_fast_result(
                summary, new_tracker, rid, t0,
                should_evaluate_guard_fn, evaluate_trajectory_safe_fn,
            )
        _flow._emit_node_outcome(
            rid, "web_scraping_node", "error", phase="agent",
            agent="web_scraping_agent",
            duration_ms=int((time.time() - t0) * 1000),
            reason=str(exc),
            followup_likely=True,
            **_flow._node_meta(),
        )
        return {"messages": [AIMessage(content="No pude conectar con el motor de síntesis, pero no pude recuperar fuentes útiles tampoco. Probá de nuevo en unos minutos.")]}

    tokens = _flow._extract_tokens(raw_result)
    quality = _flow._extract_quality(raw_result)
    followup = _flow._extract_followup(raw_result, "success")
    meta = _flow._node_meta()

    if should_evaluate_guard_fn("web_scraping_node"):
        is_safe, _ = await evaluate_trajectory_safe_fn(
            {
                "messages": raw_result.get("messages", []),
                "next_agent": state.get("next_agent", ""),
            },
            "web_scraping_node",
        )
        if not is_safe:
            _flow._emit_node_outcome(
                rid, "web_scraping_node", "blocked", phase="post_guard",
                agent="web_scraping_agent",
                duration_ms=int((time.time() - t0) * 1000),
                reason="agentdog",
                followup_likely=True,
                **tokens, **quality, **meta,
            )
            return {"messages": [AIMessage(content="Respuesta retenida por política de seguridad.")]}

    raw_messages = raw_result.get("messages", [])
    raw_text = _flow.extract_final_ai_text(raw_messages)
    _flow._web_debug(
        "run_web_scraping_flow.agent_final",
        raw_text_preview=raw_text[:500],
        message_count=len(cast(list[Any], raw_messages)),
    )
    if not raw_text:
        _flow._emit_node_outcome(
            rid, "web_scraping_node", "error", phase="agent",
            agent="web_scraping_agent",
            duration_ms=int((time.time() - t0) * 1000),
            reason="empty_response",
            followup_likely=True,
            **meta,
        )
        return {"messages": [AIMessage(content="No se pudo extraer información de la página.")]}

    sources_from_raw = _flow._extract_sources_from_text(raw_text)
    if sources_from_raw or len(raw_text.split()) < 80:
        summary = await _flow._synthesize_search_summary(raw_text, last_message, get_llm_fn, sources_from_raw)
    else:
        summary = await _flow._summarize_if_long(raw_text, rid, get_llm_fn)
    summary, _, _ = _flow._finalize_web_user_summary(summary, last_message, sources_from_raw or None)
    words = raw_text.split()
    summary_triggered = len(words) > 200
    duration_ms = int((time.time() - t0) * 1000)
    reliability = _flow._scrape_reliability(len(words))
    source_type = "agent"
    retry_done = False

    if reliability in {"unreliable"}:
        _flow._emit_node_outcome(
            rid, "web_scraping_node", "retry", phase="agent",
            agent="web_scraping_agent", duration_ms=duration_ms,
            reason=f"auto_retry:{reliability}",
            scrape_reliability=reliability, strategy="web_search_fetch",
            source_type=source_type, category=category, **tokens, **_flow._node_meta(),
        )
        from application.use_cases.web_scraping_retry_helpers import handle_unreliable_retry

        return await handle_unreliable_retry(
            agent=agent,
            last_message=last_message,
            rid=rid,
            get_llm_fn=get_llm_fn,
            summary=summary,
            words=cast(list[str], words),
            tokens=cast(dict[str, Any], tokens),
            quality=cast(dict[str, Any], quality),
            tracker=tracker,
            category=category,
            turn_count=turn_count,
            prior_reliability=prior_reliability,
            prior_score=prior_score,
            source_type=source_type,
            t0=t0,
            should_evaluate_guard_fn=should_evaluate_guard_fn,
            evaluate_trajectory_safe_fn=evaluate_trajectory_safe_fn,
            update_scrape_tracker_fn=_flow._update_scrape_tracker,
            get_category_score_fn=_flow._get_category_score,
            emit_node_outcome_fn=_flow._emit_node_outcome,
            extract_tokens_fn=_flow._extract_tokens,
            extract_quality_fn=_flow._extract_quality,
            extract_followup_fn=_flow._extract_followup,
            node_meta_fn=_flow._node_meta,
            guardrail_fast_result_fn=_flow._guardrail_fast_result,
            run_retry_agent_fn=_flow._run_retry_agent,
        )

    cost_usd = tokens.get("estimated_cost_usd")
    new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _flow._update_scrape_tracker(
        tracker, category, len(words), turn_count,
        duration_ms=duration_ms, cost_usd=cost_usd,
        source_type=source_type, reliability_override=reliability,
    ))
    analytics = cast(dict[str, Any], analytics)
    new_score = _flow._get_category_score(new_tracker, category, turn_count)
    if reliability not in ("ok_weak", "ok_strong"):
        followup = {"followup_likely": True}

    _flow._emit_node_outcome(
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
    if retry_done:
        return await _flow._guardrail_fast_result(
            summary, new_tracker, rid, t0,
            should_evaluate_guard_fn, evaluate_trajectory_safe_fn,
        )
    return {
        "messages": [AIMessage(content=summary)],
        "scrape_tracker": new_tracker,
    }


__all__ = ["_run_web_scraping_agent_strategy"]
