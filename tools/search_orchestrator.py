"""Planificación y ejecución de búsqueda web."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from infra.web_cache import _get_search_cache, _set_search_cache
from infra.web_circuit_breaker import _circuit_open, _circuit_trip, _is_non_retryable_provider_error

from tools.search_core import _format_search_results, _web_search_debug
from tools.search_providers import _query_web_search_provider


@dataclass(frozen=True)
class WebSearchProviderSpec:
    name: str
    kind: str


@dataclass(frozen=True)
class WebSearchResolution:
    selected_provider: str
    provider_candidates: tuple[WebSearchProviderSpec, ...]
    provider_explicit: bool


def _has_tavily() -> bool:
    return bool((os.getenv("TAVILY_API_KEY") or "").strip())


def _has_searxng() -> bool:
    return bool((os.getenv("SEARXNG_BASE_URL") or "").strip())


def _news_provider_ready() -> bool:
    try:
        import socket
        socket.getaddrinfo("news.google.com", 443)
        return True
    except Exception:
        return False


def _resolve_web_search_plan(
    provider: Optional[str],
    runtime_selected_provider: Optional[str],
    runtime_provider_configured: Optional[str],
    topic: Optional[str] = None,
    time_range: Optional[str] = None,
) -> WebSearchResolution:
    explicit_provider = (provider or "").strip().lower() or None
    runtime_selected_provider = (runtime_selected_provider or "").strip().lower() or None
    runtime_provider_configured = (runtime_provider_configured or "").strip().lower() or None

    preferred_provider = runtime_selected_provider or runtime_provider_configured or (os.getenv("WEB_SEARCH_PROVIDER") or "").strip().lower() or None
    provider_explicit = bool(explicit_provider)

    if explicit_provider:
        if explicit_provider not in {"tavily", "searxng", "google_news_rss"}:
            raise ValueError(f"Proveedor de web search desconocido: {explicit_provider}")
        provider_candidates = (WebSearchProviderSpec(explicit_provider, explicit_provider),)
    else:
        news_related = topic == "news" or bool(time_range)
        candidates: list[WebSearchProviderSpec] = []
        if news_related and _news_provider_ready():
            candidates.append(WebSearchProviderSpec("google_news_rss", "google_news_rss"))
        if preferred_provider == "tavily" or _has_tavily() or not candidates:
            candidates.append(WebSearchProviderSpec("tavily", "tavily"))
        if _has_searxng():
            candidates.append(WebSearchProviderSpec("searxng", "searxng"))
        if not candidates:
            candidates.append(WebSearchProviderSpec("tavily", "tavily"))
        provider_candidates = tuple(dict.fromkeys(candidates))

    selected_provider = provider_candidates[0].name
    return WebSearchResolution(selected_provider=selected_provider, provider_candidates=provider_candidates, provider_explicit=provider_explicit)


def _execute_web_search_plan(
    plan: WebSearchResolution,
    query: str,
    allowed_domains: Optional[list[str]],
    blocked_domains: Optional[list[str]],
    num_results: int,
    max_age_days: Optional[int],
    use_cache: bool,
    topic: Optional[str] = None,
    time_range: Optional[str] = None,
) -> str:
    try:
        last_error: Exception | None = None
        provider_chain = ",".join(spec.name for spec in plan.provider_candidates)
        _web_search_debug(
            "search_plan.start",
            selected_provider=plan.selected_provider,
            provider_chain=provider_chain,
            provider_explicit=plan.provider_explicit,
            query=query,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
            num_results=num_results,
            max_age_days=max_age_days,
            topic=topic,
            time_range=time_range,
        )
        for index, provider_spec in enumerate(plan.provider_candidates):
            if _circuit_open(provider_spec.name):
                _web_search_debug("search_plan.provider_skipped_circuit", provider=provider_spec.name)
                continue

            cache_key = (
                f"provider={provider_spec.name}|selected={plan.selected_provider}|chain={provider_chain}|explicit={plan.provider_explicit}|{query}"
                f"|allowed={sorted(allowed_domains or [])}|blocked={sorted(blocked_domains or [])}"
                f"|n={num_results}|days={max_age_days}|topic={topic}|time_range={time_range}"
            )
            if use_cache:
                cached = _get_search_cache(cache_key)
                if cached:
                    return cached

            try:
                hits = _query_web_search_provider(
                    provider_spec.kind,
                    query,
                    allowed_domains,
                    blocked_domains,
                    num_results,
                    max_age_days,
                    topic=topic,
                    time_range=time_range,
                )
                result = _format_search_results(query, hits)
                _web_search_debug(
                    "search_plan.provider_success",
                    provider=provider_spec.name,
                    hit_count=len(hits),
                    sample_urls=[hit.get("url", "") for hit in hits[:5]],
                    result_preview=result[:500],
                )
                if use_cache:
                    _set_search_cache(cache_key, result)
                return result
            except Exception as error:
                last_error = error if isinstance(error, Exception) else Exception(str(error))
                if _is_non_retryable_provider_error(last_error):
                    _circuit_trip(provider_spec.name)
                _web_search_debug("search_plan.provider_error", provider=provider_spec.name, error=repr(last_error))
                if plan.provider_explicit or index == len(plan.provider_candidates) - 1:
                    break

        if last_error is not None:
            _web_search_debug("search_plan.failed", error=repr(last_error))
            return f"Error en búsqueda: {str(last_error)}"
        return "Error en búsqueda: no hay providers disponibles"
    except Exception as e:
        return f"Error en búsqueda: {str(e)}"


__all__ = ["WebSearchProviderSpec", "WebSearchResolution", "_resolve_web_search_plan", "_execute_web_search_plan"]
