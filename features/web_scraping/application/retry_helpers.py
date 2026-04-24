"""Helpers para el camino de retry del web scraping."""
from __future__ import annotations

from typing import Any, Callable, cast


async def handle_unreliable_retry(
    *,
    agent,
    last_message: str,
    rid: str,
    get_llm_fn: Callable,
    summary: str,
    words: list[str],
    tokens: dict[str, Any],
    quality: dict[str, Any],
    tracker: dict[str, Any],
    category: str,
    turn_count: int,
    prior_reliability: Any,
    prior_score: Any,
    source_type: str,
    t0: float,
    should_evaluate_guard_fn: Callable,
    evaluate_trajectory_safe_fn: Callable,
    update_scrape_tracker_fn: Callable,
    get_category_score_fn: Callable,
    emit_node_outcome_fn: Callable,
    extract_tokens_fn: Callable,
    extract_quality_fn: Callable,
    extract_followup_fn: Callable,
    node_meta_fn: Callable,
    guardrail_fast_result_fn: Callable,
    run_retry_agent_fn: Callable,
) -> dict[str, Any]:
    from features.web_scraping.application import flow as _flow

    retry_summary, retry_words, retry_tokens, retry_quality = await run_retry_agent_fn(agent, last_message, rid, get_llm_fn)
    if retry_summary is not None:
        summary = retry_summary
        words = cast(list[str], retry_words or [])
        tokens = cast(dict[str, Any], retry_tokens or {})
        quality = cast(dict[str, Any], retry_quality or {})
    duration_ms = int((__import__("time").time() - t0) * 1000)
    reliability = _flow._scrape_reliability(len(words))
    cost_usd = tokens.get("estimated_cost_usd")
    new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], update_scrape_tracker_fn(
        tracker, category, len(words), turn_count,
        duration_ms=duration_ms, cost_usd=cost_usd,
        source_type=source_type, reliability_override=reliability,
    ))
    analytics = cast(dict[str, Any], analytics)
    new_score = get_category_score_fn(new_tracker, category, turn_count)
    if reliability not in ("ok_weak", "ok_strong"):
        followup = {"followup_likely": True}
    else:
        followup = {}
    emit_node_outcome_fn(
        rid, "web_scraping_node", "success", phase="agent",
        agent="web_scraping_agent", duration_ms=duration_ms,
        summary_triggered=len(words) > 200, raw_words=len(words),
        category=category, exploring=False, strategy="web_search_fetch", exp_rate=0.0,
        scrape_reliability=reliability, prior_reliability=prior_reliability,
        prior_score=prior_score, scrape_score=new_score,
        retry_done=True,
        source_type=source_type,
        **tokens, **quality, **followup, **analytics, **node_meta_fn(),
    )
    return await guardrail_fast_result_fn(summary, new_tracker, rid, t0, should_evaluate_guard_fn, evaluate_trajectory_safe_fn)


__all__ = ["handle_unreliable_retry"]
