"""Providers concretos de búsqueda web."""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Any, Optional
from urllib.parse import urljoin

from features.web_scraping.infrastructure.scraping_core import _filter_search_hits_by_domains


def _normalize_search_hits(raw_hits: Any) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    if isinstance(raw_hits, str):
        return []
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
        hits.append({"title": title or link or "result", "url": link, "link": link, "content": content, "snippet": content})
    return hits


def _query_tavily_provider(
    query: str,
    allowed_domains: Optional[list[str]],
    blocked_domains: Optional[list[str]],
    num_results: int,
    max_age_days: Optional[int],
    topic: Optional[str] = None,
    time_range: Optional[str] = None,
    tavily_client_cls: Any = None,
) -> list[dict[str, str]]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY no configurada. Agregá la variable de entorno para usar Tavily.")

    if tavily_client_cls is None:
        try:
            from tavily import TavilyClient as tavily_client_cls  # type: ignore[no-redef]
        except Exception:
            tavily_client_cls = None
    if tavily_client_cls is None:
        raise ValueError("tavily no está instalado en este entorno.")

    client = tavily_client_cls(api_key=api_key)
    search_kwargs: dict[str, Any] = {"query": query, "max_results": num_results, "include_domains": allowed_domains or [], "exclude_domains": blocked_domains or [], "search_depth": "advanced"}
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
    except Exception as exc:
        raise ValueError("requests no está instalado en este entorno.") from exc

    search_url = urljoin(base_url.rstrip("/") + "/", "search")
    categories = (os.getenv("SEARXNG_CATEGORIES") or "").strip() or ("news" if topic == "news" else "general")
    params: dict[str, Any] = {"q": query, "format": "json", "categories": categories}
    language = (os.getenv("SEARXNG_LANGUAGE") or "").strip()
    if language:
        params["language"] = language
    if time_range:
        params["time_range"] = time_range

    headers = {"User-Agent": os.getenv("SEARXNG_USER_AGENT") or "Mozilla/5.0 (Multi-Agents)", "Accept": os.getenv("SEARXNG_ACCEPT") or "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "Accept-Language": os.getenv("SEARXNG_ACCEPT_LANGUAGE") or "es-AR,es;q=0.9,en;q=0.8", "Accept-Encoding": os.getenv("SEARXNG_ACCEPT_ENCODING") or "gzip, deflate, br", "Connection": "keep-alive", "X-Forwarded-For": os.getenv("SEARXNG_FORWARDED_FOR") or "127.0.0.1", "X-Real-IP": os.getenv("SEARXNG_REAL_IP") or "127.0.0.1", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document", "Sec-Fetch-Site": "same-origin"}
    response = requests.get(search_url, params=params, headers=headers, timeout=20)
    response.raise_for_status()
    payload = response.json()
    hits = _normalize_search_hits(payload.get("results") or [])
    hits = _filter_search_hits_by_domains(hits, allowed_domains, blocked_domains)
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
    except Exception as exc:
        raise ValueError("requests no está instalado en este entorno.") from exc

    try:
        import socket
        socket.getaddrinfo("news.google.com", 443)
    except Exception as exc:
        raise ValueError("news.google.com no responde en este entorno.") from exc

    hl = (os.getenv("GOOGLE_NEWS_HL") or "es-419").strip()
    gl = (os.getenv("GOOGLE_NEWS_GL") or "US").strip()
    ceid = (os.getenv("GOOGLE_NEWS_CEID") or "US:es-419").strip()
    response = requests.get("https://news.google.com/rss/search", params={"q": query, "hl": hl, "gl": gl, "ceid": ceid}, timeout=10)
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
        hits.append({"title": title or url or "result", "url": url, "link": url, "content": description or source_text, "snippet": description or source_text})

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
    tavily_client_cls: Any = None,
) -> list[dict[str, str]]:
    if provider_kind == "tavily":
        return _query_tavily_provider(query, allowed_domains, blocked_domains, num_results, max_age_days, topic=topic, time_range=time_range, tavily_client_cls=tavily_client_cls)
    if provider_kind == "google_news_rss":
        return _query_google_news_rss_provider(query, allowed_domains, blocked_domains, num_results, max_age_days, topic=topic, time_range=time_range)
    if provider_kind == "searxng":
        return _query_searxng_provider(query, allowed_domains, blocked_domains, num_results, max_age_days, topic=topic, time_range=time_range)
    raise ValueError(f"Proveedor de web search desconocido: {provider_kind}")


__all__ = ["_normalize_search_hits", "_query_tavily_provider", "_query_searxng_provider", "_query_google_news_rss_provider", "_query_web_search_provider"]
