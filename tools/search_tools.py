"""Herramientas de búsqueda web para el sistema multi-agentes."""

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Annotated, Optional, Any
from urllib.parse import urljoin

from infra.web_circuit_breaker import (
    _is_non_retryable_provider_error,
    _circuit_trip,
    _circuit_open,
)
from infra.web_cache import (
    _get_search_cache,
    _set_search_cache,
)

from langchain_core.tools import tool
from pydantic import Field

try:  # pragma: no cover - optional provider dependency
    from tavily import TavilyClient
except Exception:  # pragma: no cover - tavily package may be unavailable in test env
    TavilyClient = None  # type: ignore[assignment]

from application.services.web_search_registry import (
    get_web_search_provider_spec,
    resolve_web_search_provider_candidates,
    WebSearchProviderSpec,
)
from domain.web_classifier import _is_specific_article_hit
from tools.scraping_tools import _domain_allowed, _filter_search_hits_by_domains


def _web_search_debug_enabled() -> bool:
    return (os.getenv("WEB_DEBUG") or "").strip().lower() in {"1", "true", "yes", "on"}


def _web_search_debug(label: str, **data: Any) -> None:
    if not _web_search_debug_enabled():
        return
    payload = " ".join(f"{key}={repr(value)}" for key, value in data.items())
    print(f"[WEB_DEBUG] {label}{(' ' + payload) if payload else ''}", flush=True)


@dataclass(frozen=True)
class WebSearchResolution:
    selected_provider: str
    provider_candidates: tuple[Any, ...]
    provider_explicit: bool


def _format_search_results(query: str, hits: list[dict[str, str]]) -> str:
    if not hits:
        return "No results found."

    lines = []
    for idx, hit in enumerate(hits[:8], start=1):
        title = hit.get("title") or hit.get("url") or "result"
        link = hit.get("url") or ""
        snippet = (hit.get("content") or "").strip()
        tag = "[article]" if _is_specific_article_hit(hit) else "[hub]"
        if link:
            lines.append(f"{idx}. {tag} [{title}]({link})")
        else:
            lines.append(f"{idx}. {tag} {title}")
        if snippet:
            lines.append(f"   {snippet[:120].rstrip()}{'…' if len(snippet) > 120 else ''}")

    lines.append("")
    lines.append("Call web_fetch on [article] URLs (not [hub]) to read full content before writing your summary.")
    return "\n".join(lines).strip()


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
    hits: list[dict[str, str]] = []
    if isinstance(raw_hits, str):
        return _extract_search_candidates_from_text(raw_hits)
    if not isinstance(raw_hits, list):
        return hits

    for hit in raw_hits:
        if not isinstance(hit, dict):
            continue
        title = str(hit.get("title") or hit.get("name") or "").strip()
        link = str(hit.get("url") or hit.get("link") or "").strip()
        content = str(hit.get("content") or hit.get("snippet") or "").strip()
        if not title and not link:
            continue
        hits.append({
            "title": title or link or "result",
            "url": link,
            "link": link,
            "content": content,
            "snippet": content,
        })
    return hits


def _query_tavily_provider(
    query: str,
    allowed_domains: Optional[list[str]],
    blocked_domains: Optional[list[str]],
    num_results: int,
    max_age_days: Optional[int],
    topic: Optional[str] = None,
    time_range: Optional[str] = None,
) -> list[dict[str, str]]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY no configurada. Agregá la variable de entorno para usar Tavily.")

    if TavilyClient is None:
        raise ValueError("tavily no está instalado en este entorno.")

    client = TavilyClient(api_key=api_key)
    search_kwargs: dict[str, Any] = {
        "query": query,
        "max_results": num_results,
        "include_domains": allowed_domains or [],
        "exclude_domains": blocked_domains or [],
        "search_depth": "advanced",
    }
    if topic:
        search_kwargs["topic"] = topic
    if time_range:
        search_kwargs["time_range"] = time_range
    elif max_age_days is not None:
        search_kwargs["days"] = max_age_days
    response = client.search(**search_kwargs)
    hits = _normalize_search_hits(response.get("results") or [])
    hits = _filter_search_hits_by_domains(hits, allowed_domains, blocked_domains)
    return hits[:num_results]


def _query_searxng_provider(
    query: str,
    allowed_domains: Optional[list[str]],
    blocked_domains: Optional[list[str]],
    num_results: int,
    max_age_days: Optional[int],
    topic: Optional[str] = None,
    time_range: Optional[str] = None,
) -> list[dict[str, str]]:
    base_url = (os.getenv("SEARXNG_BASE_URL") or "").strip()
    if not base_url:
        raise ValueError("SEARXNG_BASE_URL no configurada. Agregá la URL base de tu instancia SearXNG.")

    try:
        import requests
    except Exception as exc:  # pragma: no cover - optional provider dependency
        raise ValueError("requests no está instalado en este entorno.") from exc

    search_url = urljoin(base_url.rstrip("/") + "/", "search")
    categories = (os.getenv("SEARXNG_CATEGORIES") or "").strip() or ("news" if topic == "news" else "general")
    params: dict[str, Any] = {
        "q": query,
        "format": "json",
        "categories": categories,
    }
    language = (os.getenv("SEARXNG_LANGUAGE") or "").strip()
    if language:
        params["language"] = language
    if time_range:
        params["time_range"] = time_range

    headers = {
        "User-Agent": os.getenv("SEARXNG_USER_AGENT") or "Mozilla/5.0 (Multi-Agents)",
        "Accept": os.getenv("SEARXNG_ACCEPT") or "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": os.getenv("SEARXNG_ACCEPT_LANGUAGE") or "es-AR,es;q=0.9,en;q=0.8",
        "Accept-Encoding": os.getenv("SEARXNG_ACCEPT_ENCODING") or "gzip, deflate, br",
        "Connection": "keep-alive",
        "X-Forwarded-For": os.getenv("SEARXNG_FORWARDED_FOR") or "127.0.0.1",
        "X-Real-IP": os.getenv("SEARXNG_REAL_IP") or "127.0.0.1",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Site": "same-origin",
    }
    _web_search_debug(
        "searxng.request",
        query=query,
        search_url=search_url,
        params=params,
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
        num_results=num_results,
        max_age_days=max_age_days,
    )
    response = requests.get(search_url, params=params, headers=headers, timeout=20)
    response.raise_for_status()
    payload = response.json()
    hits = _normalize_search_hits(payload.get("results") or [])
    hits = _filter_search_hits_by_domains(hits, allowed_domains, blocked_domains)
    _web_search_debug(
        "searxng.response",
        status_code=response.status_code,
        payload_result_count=len(payload.get("results") or []),
        filtered_hit_count=len(hits),
        sample_urls=[hit.get("url", "") for hit in hits[:5]],
    )
    return hits[:num_results]


def _query_google_news_rss_provider(
    query: str,
    allowed_domains: Optional[list[str]],
    blocked_domains: Optional[list[str]],
    num_results: int,
    max_age_days: Optional[int],
    topic: Optional[str] = None,
    time_range: Optional[str] = None,
) -> list[dict[str, str]]:
    try:
        import requests
    except Exception as exc:  # pragma: no cover - optional provider dependency
        raise ValueError("requests no está instalado en este entorno.") from exc

    try:
        import socket
        socket.getaddrinfo("news.google.com", 443)
    except Exception as exc:
        raise ValueError("news.google.com no responde en este entorno.") from exc

    hl = (os.getenv("GOOGLE_NEWS_HL") or "es-419").strip()
    gl = (os.getenv("GOOGLE_NEWS_GL") or "US").strip()
    ceid = (os.getenv("GOOGLE_NEWS_CEID") or "US:es-419").strip()
    response = requests.get(
        "https://news.google.com/rss/search",
        params={
            "q": query,
            "hl": hl,
            "gl": gl,
            "ceid": ceid,
        },
        timeout=10,
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)

    hits: list[dict[str, str]] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        description = (item.findtext("description") or "").strip()
        source_elem = item.find("source")
        source_text = (source_elem.text or "").strip() if source_elem is not None else ""
        source_url = (source_elem.attrib.get("url") or "").strip() if source_elem is not None else ""
        url = link or source_url
        if not title and not url:
            continue
        hits.append({
            "title": title or url or "result",
            "url": url,
            "link": url,
            "content": description or source_text,
            "snippet": description or source_text,
        })

    hits = _normalize_search_hits(hits)
    hits = _filter_search_hits_by_domains(hits, allowed_domains, blocked_domains)
    return hits[:num_results]


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
    if provider_kind == "tavily":
        return _query_tavily_provider(query, allowed_domains, blocked_domains, num_results, max_age_days, topic=topic, time_range=time_range)
    if provider_kind == "google_news_rss":
        return _query_google_news_rss_provider(query, allowed_domains, blocked_domains, num_results, max_age_days, topic=topic, time_range=time_range)
    if provider_kind == "searxng":
        return _query_searxng_provider(query, allowed_domains, blocked_domains, num_results, max_age_days, topic=topic, time_range=time_range)
    raise ValueError(f"Proveedor de web search desconocido: {provider_kind}")


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
    provider_candidates = resolve_web_search_provider_candidates(
        explicit_provider=explicit_provider,
        runtime_selected_provider=runtime_selected_provider,
        runtime_provider_configured=runtime_provider_configured,
    )
    provider_explicit = bool(explicit_provider)

    if not provider_explicit:
        news_related = (topic == "news" or time_range)
        if news_related:
            provider_map = {spec.name: spec for spec in provider_candidates}
            news_provider = provider_map.get("google_news_rss")
            if news_provider is not None:
                prioritized = [spec for spec in provider_candidates if spec.name != "google_news_rss"]
                reordered: list[WebSearchProviderSpec] = []
                inserted = False
                for spec in prioritized:
                    if spec.name == "searxng" and not inserted:
                        reordered.append(news_provider)
                        inserted = True
                    reordered.append(spec)
                if not inserted:
                    reordered.append(news_provider)
                provider_candidates = tuple(reordered)
        else:
            filtered = tuple(spec for spec in provider_candidates if spec.name != "google_news_rss")
            if filtered:
                provider_candidates = filtered
            else:
                provider_candidates = (get_web_search_provider_spec("tavily"),)

    return WebSearchResolution(
        selected_provider=provider_candidates[0].name,
        provider_candidates=provider_candidates,
        provider_explicit=provider_explicit,
    )


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
                _web_search_debug(
                    "search_plan.provider_skipped_circuit",
                    provider=provider_spec.name,
                )
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
                _web_search_debug(
                    "search_plan.provider_error",
                    provider=provider_spec.name,
                    error=repr(last_error),
                )
                if plan.provider_explicit or index == len(plan.provider_candidates) - 1:
                    break

        if last_error is not None:
            _web_search_debug("search_plan.failed", error=repr(last_error))
            return f"Error en búsqueda: {str(last_error)}"
        return "Error en búsqueda: no hay providers disponibles"
    except Exception as e:
        return f"Error en búsqueda: {str(e)}"


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


__all__ = [
    "WebSearchResolution",
    "_web_search_debug_enabled",
    "_web_search_debug",
    "_format_search_results",
    "_extract_search_candidates_from_text",
    "_normalize_search_hits",
    "_filter_search_hits_by_domains",
    "_query_tavily_provider",
    "_query_searxng_provider",
    "_query_google_news_rss_provider",
    "_query_web_search_provider",
    "_resolve_web_search_plan",
    "_execute_web_search_plan",
    "search_web",
]
