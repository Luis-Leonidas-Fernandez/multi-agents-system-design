"""Query planning y helpers de búsqueda para web scraping."""
from __future__ import annotations

import asyncio
from typing import Any, Mapping, Optional

from application.helpers.url_helpers import _extract_web_fetch_redirect_url
from application.policies.web_source_policy import (
    detect_query_source_group,
    detect_recent_query_horizon,
    get_query_source_terms,
    get_recent_query_requirements,
    is_recent_web_information_query,
)
from application.policies.web_search_context import QueryContext, RecentPolicy
from domain.country_resolver import extract_query_geography


def _web_search_runtime_args(state: Mapping[str, Any]) -> dict[str, Any]:
    selected = str(state.get("web_search_selected_provider") or "").strip().lower()
    configured = str(state.get("web_search_provider_configured") or "").strip().lower()
    args: dict[str, Any] = {}
    if selected:
        args["runtime_selected_provider"] = selected
    if configured:
        args["runtime_provider_configured"] = configured
    return args


def _build_generic_fetch_prompt(query: str) -> str:
    geography = extract_query_geography(query)
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


async def _fetch_web_page_follow_redirect(url: str, prompt: str, *, use_dynamic: bool = True) -> str:
    from tools.scraping_tools import fetch_web_page
    from tools.web_fetch_helpers import _check_fetch_input_guard

    blocked = _check_fetch_input_guard(url, prompt)
    if blocked:
        return blocked

    result = await fetch_web_page(url=url, prompt=prompt, use_dynamic=use_dynamic)
    if not isinstance(result, str):
        result = str(result)

    redirect_url = _extract_web_fetch_redirect_url(result)
    if redirect_url and redirect_url != url:
        redirected = await fetch_web_page(url=redirect_url, prompt=prompt, use_dynamic=use_dynamic)
        return redirected if isinstance(redirected, str) else str(redirected)
    return result


def _build_query_context(last_message: str) -> tuple[QueryContext, RecentPolicy]:
    from application.use_cases import web_scraping_flow as _flow

    query_terms = _flow._extract_generic_query_terms(last_message)
    query_source_group = detect_query_source_group(last_message)
    source_terms = list(get_query_source_terms(last_message))
    if source_terms:
        merged: list[str] = []
        for term in query_terms + source_terms:
            if term not in merged:
                merged.append(term)
        query_terms = merged

    query_horizon = (
        detect_recent_query_horizon(last_message)
        if is_recent_web_information_query(last_message)
        else None
    )
    reqs = get_recent_query_requirements(query_horizon)

    search_age_days: Optional[int] = None
    if query_horizon == "week":
        search_age_days = 14
    elif query_horizon == "month":
        search_age_days = 45
    elif query_horizon:
        search_age_days = 30

    ctx = QueryContext(
        query_terms=query_terms,
        query_source_group=query_source_group,
        source_terms=source_terms,
        query_horizon=query_horizon,
        search_age_days=search_age_days,
    )
    policy = RecentPolicy(
        min_score=reqs["min_score"],
        min_body_lines=reqs["min_body_lines"],
        min_sources=reqs["min_sources"],
        min_candidates=reqs["min_candidates"],
        candidate_min_body_lines=1 if query_horizon == "week" else reqs["min_body_lines"],
        candidate_min_sources=reqs["candidate_min_sources"] or 1,
    )
    return ctx, policy


async def _run_week_search_candidates(
    last_message: str,
    search_age_days: Optional[int],
    query_terms: list[str],
    query_source_group: Optional[str],
    web_search_runtime_args: Optional[dict[str, Any]] = None,
) -> tuple[list[dict[str, str]], str]:
    from application.use_cases import web_scraping_flow as _flow
    from tools.search_tools import search_web

    source_terms = list(get_query_source_terms(last_message))
    loop = asyncio.get_running_loop()
    search_invoke_args: dict = {"query": last_message, "use_cache": False, **(web_search_runtime_args or {})}
    if search_age_days is not None:
        search_invoke_args["max_age_days"] = search_age_days
    search_invoke_args["topic"] = "news"
    search_invoke_args["time_range"] = "week"

    search_text = await loop.run_in_executor(None, lambda: search_web.invoke(search_invoke_args))
    if not isinstance(search_text, str):
        search_text = str(search_text)

    url_age_threshold = search_age_days or 14
    candidates = [
        c for c in _flow._extract_generic_search_candidates(search_text)
        if not _flow._is_non_news_candidate(c)
        and _flow._candidate_url_is_recent(c.get("url", ""), url_age_threshold)
        and not _flow._is_invalid_news_candidate(c, last_message)
    ]
    ranked_candidates = _flow._rank_candidates_by_source_policy(candidates, query_terms, query_source_group)
    diverse_candidates = _flow._dedup_candidates_by_event(ranked_candidates, query_terms)[:8]
    _flow._web_debug(
        "week_search.generic",
        invoke_args=search_invoke_args,
        url_age_threshold=url_age_threshold,
        extracted_candidate_count=len(candidates),
        ranked_candidate_count=len(ranked_candidates),
        diverse_candidate_count=len(diverse_candidates),
        search_preview=search_text[:500],
        diverse_urls=[c.get("url", "") for c in diverse_candidates],
    )

    if query_source_group == "japan":
        return diverse_candidates, search_text

    if len(diverse_candidates) >= 2:
        return diverse_candidates, search_text

    country_press_candidates, country_press_search_text = [], ""
    if query_source_group != "japan":
        country_press_candidates, country_press_search_text = await _flow._run_country_press_search_candidates(
            last_message,
            search_age_days,
            query_terms,
            query_source_group,
            source_terms,
            web_search_runtime_args,
            query_horizon="week",
        )
    if country_press_candidates:
        filtered_candidates = [
            c for c in country_press_candidates
            if _flow._candidate_url_is_recent(c.get("url", ""), url_age_threshold)
        ]
        _flow._web_debug(
            "week_search.country_press",
            candidate_count=len(country_press_candidates),
            filtered_candidate_count=len(filtered_candidates),
            url_age_threshold=url_age_threshold,
            urls=[c.get("url", "") for c in filtered_candidates[:8]],
        )
        if len(filtered_candidates) >= 2:
            return filtered_candidates[:8], country_press_search_text

    local_source_strategy = _flow._country_press_strategy_cache_get(query_source_group, source_terms)
    if query_source_group and local_source_strategy in {"cache", "directory", "policy", "lookup"}:
        _flow._web_debug(
            "week_search.global_skipped_no_local_sources",
            query=last_message,
            query_source_group=query_source_group,
            local_source_strategy=local_source_strategy,
        )
        return [], country_press_search_text

    return diverse_candidates, search_text


async def _run_general_search_pipeline(
    last_message: str,
    ctx: QueryContext,
    loop: asyncio.AbstractEventLoop,
    web_search_runtime_args: Optional[dict[str, Any]],
) -> tuple[list[dict[str, str]], str]:
    from application.use_cases import web_scraping_flow as _flow
    from tools.search_tools import search_web

    query_terms = ctx.query_terms
    query_source_group = ctx.query_source_group
    source_terms = ctx.source_terms
    query_horizon = ctx.query_horizon
    search_age_days = ctx.search_age_days

    country_press_domains, country_press_names = await _flow._discover_country_press_sources(
        last_message,
        query_source_group,
        source_terms,
        web_search_runtime_args,
    )
    search_invoke_args: dict = {"query": last_message, "use_cache": False, **(web_search_runtime_args or {})}
    if search_age_days is not None:
        search_invoke_args["max_age_days"] = search_age_days
    if query_horizon:
        search_invoke_args["topic"] = "news"
        if query_horizon == "today":
            search_invoke_args["time_range"] = "day"
    if country_press_domains and not search_invoke_args.get("allowed_domains"):
        search_invoke_args["allowed_domains"] = country_press_domains
    if country_press_names:
        search_invoke_args["query"] = f"{last_message} {' '.join(country_press_names[:4])}".strip()

    search_text = await loop.run_in_executor(None, lambda: search_web.invoke(search_invoke_args))
    if not isinstance(search_text, str):
        search_text = str(search_text)

    candidates = _flow._extract_generic_search_candidates(search_text)
    ranked_candidates = _flow._rank_candidates_by_source_policy(
        [c for c in candidates if not _flow._is_non_news_candidate(c) and not _flow._is_invalid_news_candidate(c, last_message)],
        query_terms,
        query_source_group,
    )[:8]
    diverse_candidates = _flow._dedup_candidates_by_event(ranked_candidates, query_terms)
    _flow._web_debug(
        "generic_fetch.initial_search",
        query=last_message,
        invoke_args=search_invoke_args,
        candidate_count=len(candidates),
        ranked_candidate_count=len(ranked_candidates),
        diverse_candidate_count=len(diverse_candidates),
        search_preview=search_text[:500],
    )
    return diverse_candidates, search_text


__all__ = [
    "_web_search_runtime_args",
    "_build_generic_fetch_prompt",
    "_fetch_web_page_follow_redirect",
    "_build_query_context",
    "_run_week_search_candidates",
    "_run_general_search_pipeline",
]
