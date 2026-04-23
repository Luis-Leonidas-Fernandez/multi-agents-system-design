"""Pipelines de búsqueda/fetch para web scraping."""
from __future__ import annotations

import asyncio
import re
from typing import Any, Optional

from application.policies.web_search_context import QueryContext, RecentPolicy


async def _fetch_and_score_entries(
    ranked_candidates: list[dict[str, str]],
    last_message: str,
    ctx: QueryContext,
    policy: RecentPolicy,
    search_text: str,
) -> list[dict[str, Any]]:
    from application.use_cases import web_scraping_flow as _flow

    fetch_prompt = _flow._build_generic_fetch_prompt(last_message)
    query_terms = ctx.query_terms
    query_source_group = ctx.query_source_group

    async def _fetch_candidate(candidate: dict[str, str]) -> tuple[dict[str, str], Any]:
        try:
            result = await _flow._fetch_web_page_follow_redirect(candidate["url"], fetch_prompt, use_dynamic=False)
            return candidate, result
        except Exception as exc:  # pragma: no cover - defensive
            return candidate, exc

    fetched_results = await asyncio.gather(*(_fetch_candidate(candidate) for candidate in ranked_candidates), return_exceptions=False)

    eligible_entries: list[dict[str, Any]] = []
    for candidate, result in fetched_results:
        if _flow._is_invalid_news_candidate(candidate, last_message):
            _flow._web_debug("generic_fetch.entry_rejected_invalid_structure", url=candidate.get("url", ""), title=candidate.get("title", ""))
            continue
        if isinstance(result, Exception):
            _flow._web_debug("generic_fetch.fetch_exception", url=candidate.get("url", ""), error=repr(result))
            snippet_lines = _flow._candidate_snippet_lines(candidate)
            if not snippet_lines:
                continue
            result = "\n".join(snippet_lines)
        if not isinstance(result, str):
            result = str(result)
        if result.startswith("Error") or result.startswith("URL rechazada") or _flow._is_no_info_response(result):
            _flow._web_debug("generic_fetch.fetch_bad_result", url=candidate.get("url", ""), result_preview=result[:300])
            snippet_lines = _flow._candidate_snippet_lines(candidate)
            if not snippet_lines:
                continue
            result = "\n".join(snippet_lines)

        body_lines = _flow._extract_generic_content_lines(result, query_terms)
        candidate_score = _flow._score_generic_candidate(candidate, query_terms, query_source_group)
        content_score = len(body_lines) * 2
        if query_terms and not body_lines:
            result_blob = result.lower()
            if any(term in result_blob for term in query_terms):
                content_score = 1
        if not body_lines and content_score <= 1:
            snippet_lines = _flow._candidate_snippet_lines(candidate)
            if snippet_lines:
                body_lines = snippet_lines
                content_score = 3
            else:
                continue

        score = candidate_score + content_score
        if score <= 0:
            continue
        if _flow._is_recent_web_information_query(last_message):
            if score < policy.min_score or len(body_lines) < policy.candidate_min_body_lines:
                _flow._web_debug("generic_fetch.entry_rejected_recent_threshold", url=candidate.get("url", ""), score=score, min_score=policy.min_score, body_lines_count=len(body_lines), min_body_lines=policy.candidate_min_body_lines)
                continue

        fallback_lines = [line.strip() for line in result.splitlines() if line.strip() and not line.strip().lower().startswith(("url:", "sources:", "http")) and "http" not in line.lower()]
        summary_lines = body_lines or _flow._extract_generic_content_lines(search_text, query_terms) or fallback_lines[:5]
        if len(summary_lines) < 3 and fallback_lines:
            seen_lines = set(summary_lines)
            for line in fallback_lines:
                if line not in seen_lines:
                    summary_lines.append(line)
                    seen_lines.add(line)
                if len(summary_lines) >= 6:
                    break
        sources = _flow._extract_sources_from_text(result)
        if not sources:
            sources = [{"title": candidate.get("title") or candidate["url"], "url": candidate["url"]}]
        if _flow._is_recent_web_information_query(last_message) and len(sources) < policy.candidate_min_sources:
            _flow._web_debug("generic_fetch.entry_rejected_sources", url=candidate.get("url", ""), source_count=len(sources), min_sources=policy.candidate_min_sources)
            continue

        eligible_entries.append({"summary_lines": summary_lines[:10], "sources": sources, "score": score, "candidate": candidate})

    _flow._web_debug("generic_fetch.eligible_entries", eligible_count=len(eligible_entries), urls=[entry["candidate"].get("url", "") for entry in eligible_entries])
    return eligible_entries


async def _run_week_search_pipeline(
    last_message: str,
    ctx: QueryContext,
    web_search_runtime_args: Optional[dict[str, Any]],
) -> tuple[list[dict[str, str]], str, Optional[dict[str, Any]]]:
    from application.use_cases import web_scraping_flow as _flow

    diverse_candidates, search_text = await _flow._run_week_search_candidates(last_message, ctx.search_age_days, ctx.query_terms, ctx.query_source_group, web_search_runtime_args)
    local_source_strategy = _flow._country_press_strategy_cache_get(ctx.query_source_group, ctx.source_terms)
    _flow._web_debug("generic_fetch.week_candidates", query=last_message, search_age_days=ctx.search_age_days, candidate_count=len(diverse_candidates), local_source_strategy=local_source_strategy, urls=[c.get("url", "") for c in diverse_candidates])
    if ctx.query_source_group and not diverse_candidates and local_source_strategy in {"cache", "directory", "policy", "lookup"}:
        return [], search_text, _flow._build_no_local_sources_response(last_message)
    return diverse_candidates, search_text, None


__all__ = ["_fetch_and_score_entries", "_run_week_search_pipeline"]
