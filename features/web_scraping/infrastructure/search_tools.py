"""Herramientas de búsqueda web para el sistema multi-agentes."""

import re
from typing import Annotated, Optional, Any

from langchain_core.tools import tool
from pydantic import Field

try:  # pragma: no cover - optional provider dependency
    from tavily import TavilyClient
except Exception:  # pragma: no cover - tavily package may be unavailable in test env
    TavilyClient = None  # type: ignore[assignment]

from features.web_scraping.infrastructure.scraping_core import _filter_search_hits_by_domains
from features.web_scraping.infrastructure.search_core import _format_search_results, _web_search_debug, _web_search_debug_enabled
from features.web_scraping.infrastructure.search_orchestrator import WebSearchResolution, _execute_web_search_plan, _resolve_web_search_plan
from features.web_scraping.infrastructure.search_providers import (
    _normalize_search_hits as _providers_normalize_search_hits,
    _query_google_news_rss_provider as _providers_query_google_news_rss_provider,
    _query_searxng_provider as _providers_query_searxng_provider,
    _query_tavily_provider as _providers_query_tavily_provider,
    _query_web_search_provider as _providers_query_web_search_provider,
)


def _extract_search_candidates_from_text(text: str) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for title, url in re.findall(r"\[([^\]]+)\]\((https?://[^)]+)\)", text or ""):
        normalized = url.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append({"title": title.strip() or normalized, "url": normalized, "content": "", "snippet": ""})
    return candidates


def _normalize_search_hits(raw_hits: Any) -> list[dict[str, str]]:
    if isinstance(raw_hits, str):
        return _extract_search_candidates_from_text(raw_hits)
    return _providers_normalize_search_hits(raw_hits)


def _query_tavily_provider(
    query: str,
    allowed_domains: Optional[list[str]],
    blocked_domains: Optional[list[str]],
    num_results: int,
    max_age_days: Optional[int],
    topic: Optional[str] = None,
    time_range: Optional[str] = None,
) -> list[dict[str, str]]:
    return _providers_query_tavily_provider(query, allowed_domains, blocked_domains, num_results, max_age_days, topic=topic, time_range=time_range, tavily_client_cls=TavilyClient)


def _query_searxng_provider(
    query: str,
    allowed_domains: Optional[list[str]],
    blocked_domains: Optional[list[str]],
    num_results: int,
    max_age_days: Optional[int],
    topic: Optional[str] = None,
    time_range: Optional[str] = None,
) -> list[dict[str, str]]:
    return _providers_query_searxng_provider(query, allowed_domains, blocked_domains, num_results, max_age_days, topic=topic, time_range=time_range)


def _query_google_news_rss_provider(
    query: str,
    allowed_domains: Optional[list[str]],
    blocked_domains: Optional[list[str]],
    num_results: int,
    max_age_days: Optional[int],
    topic: Optional[str] = None,
    time_range: Optional[str] = None,
) -> list[dict[str, str]]:
    return _providers_query_google_news_rss_provider(query, allowed_domains, blocked_domains, num_results, max_age_days, topic=topic, time_range=time_range)


def _query_web_search_provider(
    provider_kind: str,
    query: str,
    allowed_domains: Optional[list[str]],
    blocked_domains: Optional[list[str]],
    num_results: int,
    max_age_days: Optional[int],
    topic: Optional[str] = None,
    time_range: Optional[str] = None,
) -> list[dict[str, str]]:
    return _providers_query_web_search_provider(provider_kind, query, allowed_domains, blocked_domains, num_results, max_age_days, topic=topic, time_range=time_range, tavily_client_cls=TavilyClient)


@tool
def search_web(
    query: Annotated[str, Field(description="Consulta de búsqueda en lenguaje natural, ej: 'bitcoin price usd today', 'precio actual ethereum'")],
    provider: Annotated[Optional[str], Field(description="Proveedor de búsqueda a usar explícitamente")] = None,
    runtime_selected_provider: Annotated[Optional[str], Field(description="Provider seleccionado por el runtime")] = None,
    runtime_provider_configured: Annotated[Optional[str], Field(description="Provider configurado por el runtime")] = None,
    allowed_domains: Annotated[Optional[list[str]], Field(description="Dominios permitidos para filtrar resultados")] = None,
    blocked_domains: Annotated[Optional[list[str]], Field(description="Dominios bloqueados para filtrar resultados")] = None,
    num_results: Annotated[int, Field(description="Cantidad máxima de resultados", ge=1, le=10)] = 8,
    max_age_days: Annotated[Optional[int], Field(description="Limitar resultados a los últimos N días. Usá 7 para 'esta semana/hoy', 30 para noticias recientes. None para sin límite.", ge=1, le=365)] = None,
    topic: Annotated[Optional[str], Field(description="Tipo de búsqueda: 'news' para noticias recientes, 'general' para búsqueda general, 'finance' para finanzas. Usá 'news' cuando busques noticias.")] = None,
    time_range: Annotated[Optional[str], Field(description="Rango temporal para Tavily: 'day' (hoy), 'week' (esta semana), 'month' (este mes), 'year'. Tiene precedencia sobre max_age_days.")] = None,
    use_cache: Annotated[bool, Field(description="Si True, usa caché de resultados por 15 minutos")] = True,
) -> str:
    """Busca información en internet usando el provider configurado. No requiere URL."""
    try:
        if allowed_domains and blocked_domains:
            return "Error: Cannot specify both allowed_domains and blocked_domains in the same request"

        plan = _resolve_web_search_plan(
            provider,
            runtime_selected_provider,
            runtime_provider_configured,
            topic=topic,
            time_range=time_range,
        )
        return _execute_web_search_plan(
            plan,
            query,
            allowed_domains,
            blocked_domains,
            num_results,
            max_age_days,
            use_cache,
            topic=topic,
            time_range=time_range,
        )
    except Exception as e:
        return f"Error en búsqueda: {str(e)}"


__all__ = [
    "WebSearchResolution",
    "_web_search_debug_enabled",
    "_web_search_debug",
    "_format_search_results",
    "_extract_search_candidates_from_text",
    "_normalize_search_hits",
    "_filter_search_hits_by_domains",
    "_query_web_search_provider",
    "_resolve_web_search_plan",
    "_execute_web_search_plan",
    "search_web",
]
