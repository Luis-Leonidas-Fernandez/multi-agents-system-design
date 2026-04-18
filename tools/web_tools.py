"""Tools web del sistema multi-agentes."""

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Annotated, Optional, Any
from urllib.parse import urlparse, urljoin

from infra.web_circuit_breaker import (
    _is_non_retryable_provider_error,
    _circuit_trip,
    _circuit_open,
)
from infra.web_cache import (
    _get_search_cache,
    _set_search_cache,
    _get_web_fetch_cache,
    _set_web_fetch_cache,
    _web_fetch_cache_key,
)

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import Field

try:  # pragma: no cover - optional provider dependency
    from tavily import TavilyClient
except Exception:  # pragma: no cover - tavily package may be unavailable in test env
    TavilyClient = None  # type: ignore[assignment]

from application.helpers.scraping_flow_helpers import (
    _validate_url, _cache_key, _get_cache, _set_cache,
    _build_result, _extract_text, _extract_links,
)
from application.services.web_search_registry import (
    get_web_search_provider_spec,
    resolve_web_search_provider_candidates,
)
from infra import scraping_infra



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

_PREAPPROVED_HOSTS = {
    "platform.claude.com",
    "code.claude.com",
    "modelcontextprotocol.io",
    "github.com/anthropics",
    "agentskills.io",
    "docs.python.org",
    "en.cppreference.com",
    "docs.oracle.com",
    "learn.microsoft.com",
    "developer.mozilla.org",
    "go.dev",
    "pkg.go.dev",
    "www.php.net",
    "docs.swift.org",
    "kotlinlang.org",
    "ruby-doc.org",
    "doc.rust-lang.org",
    "www.typescriptlang.org",
    "react.dev",
    "angular.io",
    "vuejs.org",
    "nextjs.org",
    "expressjs.com",
    "nodejs.org",
    "bun.sh",
    "jquery.com",
    "getbootstrap.com",
    "tailwindcss.com",
    "d3js.org",
    "threejs.org",
    "redux.js.org",
    "webpack.js.org",
    "jestjs.io",
    "reactrouter.com",
    "docs.djangoproject.com",
    "flask.palletsprojects.com",
    "fastapi.tiangolo.com",
    "pandas.pydata.org",
    "numpy.org",
    "www.tensorflow.org",
    "pytorch.org",
    "scikit-learn.org",
    "matplotlib.org",
    "requests.readthedocs.io",
    "jupyter.org",
    "laravel.com",
    "symfony.com",
    "wordpress.org",
    "docs.spring.io",
    "hibernate.org",
    "tomcat.apache.org",
    "gradle.org",
    "maven.apache.org",
    "asp.net",
    "dotnet.microsoft.com",
    "nuget.org",
    "blazor.net",
    "reactnative.dev",
    "docs.flutter.dev",
    "developer.apple.com",
    "developer.android.com",
    "keras.io",
    "spark.apache.org",
    "huggingface.co",
    "www.kaggle.com",
    "www.mongodb.com",
    "redis.io",
    "www.postgresql.org",
    "dev.mysql.com",
    "www.sqlite.org",
    "graphql.org",
    "prisma.io",
    "docs.aws.amazon.com",
    "cloud.google.com",
    "kubernetes.io",
    "www.docker.com",
    "www.terraform.io",
    "www.ansible.com",
    "vercel.com/docs",
    "docs.netlify.com",
    "devcenter.heroku.com",
    "cypress.io",
    "selenium.dev",
    "docs.unity.com",
    "docs.unrealengine.com",
    "git-scm.com",
    "nginx.org",
    "httpd.apache.org",
}

_PREAPPROVED_HOSTNAME_ONLY: set[str] = set()
_PREAPPROVED_PATH_PREFIXES: dict[str, list[str]] = {}

for _entry in _PREAPPROVED_HOSTS:
    _slash = _entry.find("/")
    if _slash == -1:
        _PREAPPROVED_HOSTNAME_ONLY.add(_entry)
    else:
        _host = _entry[:_slash]
        _path = _entry[_slash:]
        _PREAPPROVED_PATH_PREFIXES.setdefault(_host, []).append(_path)


def _split_domain_list(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _domain_matches(domain: str, pattern: str) -> bool:
    domain = domain.lower().lstrip("www.")
    pattern = pattern.lower().lstrip("www.")
    return domain == pattern or domain.endswith("." + pattern)


def _domain_allowed(url: str, *, allowed: Optional[list[str]] = None, blocked: Optional[list[str]] = None) -> bool:
    host = urlparse(url).hostname or ""
    if not host:
        return False

    env_allowed = _split_domain_list(os.getenv("WEB_ALLOWED_DOMAINS"))
    env_blocked = _split_domain_list(os.getenv("WEB_BLOCKED_DOMAINS"))
    allowed = (allowed or []) + env_allowed
    blocked = (blocked or []) + env_blocked

    if blocked and any(_domain_matches(host, pattern) for pattern in blocked):
        return False
    if allowed:
        return any(_domain_matches(host, pattern) for pattern in allowed)
    return True



def _is_preapproved_host(hostname: str, pathname: str) -> bool:
    host = hostname.lower().lstrip("www.")
    if host in _PREAPPROVED_HOSTNAME_ONLY:
        return True
    prefixes = _PREAPPROVED_PATH_PREFIXES.get(host, [])
    for prefix in prefixes:
        if pathname == prefix or pathname.startswith(prefix + "/"):
            return True
    return False


def _is_preapproved_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return bool(host) and _is_preapproved_host(host, parsed.path or "/")


def _hit_path_segments(link: str) -> list[str]:
    path = urlparse(link or "").path.strip("/")
    if not path:
        return []
    return [segment for segment in path.split("/") if segment]


def _hit_path_tokens(link: str) -> list[str]:
    tokens: list[str] = []
    for segment in _hit_path_segments(link):
        for token in re.split(r"[-_]+", segment):
            token = token.strip().lower()
            if token:
                tokens.append(token)
    return tokens


def _path_has_date_or_slug(link: str) -> bool:
    path = urlparse(link or "").path.lower()
    segments = _hit_path_segments(link)
    return bool(
        re.search(r"(19|20)\d{2}[/\-]?\d{2}[/\-]?\d{2}", path)
        or re.search(r"\d{6,8}", path)
        or any("-" in seg or "_" in seg or re.search(r"\d", seg) for seg in segments)
    )


def _blob_has_article_signal(blob: str) -> bool:
    article_signals = (
        "security", "safety", "economy", "economic", "politics", "political", "election",
        "breaking", "update", "reported", "report", "announced", "announcement", "court",
        "attack", "disaster", "conflict", "strike", "result", "results", "resultado", "resultados",
        "match", "game", "policy", "strategy", "minister", "president", "prime minister",
    )
    return any(term in blob for term in article_signals)


def _is_topic_or_hub_hit(hit: dict[str, str]) -> bool:
    link = str(hit.get("url") or hit.get("link") or "")
    if not link:
        return False
    segments = _hit_path_segments(link)
    blob = " ".join(str(hit.get(field) or "") for field in ("title", "url", "link", "content", "snippet")).lower()
    tokens = _hit_path_tokens(link)
    hub_terms = {"topic", "topics", "tag", "tags", "category", "categories", "archive", "author", "world", "mundo", "index", "home", "partidos", "resultados", "ultima-ora"}
    if "/t/" in urlparse(link).path.lower():
        return True
    if any(tok in hub_terms for tok in tokens):
        return True
    if len(segments) >= 1 and segments[0] in {"news", "noticias", "world", "mundo", "partidos", "resultados", "home", "index"} and not _path_has_date_or_slug(link):
        if segments[0] in {"news", "noticias"} and _blob_has_article_signal(blob):
            return False
        return True
    has_slug = any("-" in seg or "_" in seg or re.search(r"\d", seg) for seg in segments)
    if len(segments) <= 2 and not _path_has_date_or_slug(link) and not has_slug and not _blob_has_article_signal(blob):
        return True
    return False


def _is_specific_article_hit(hit: dict[str, str]) -> bool:
    link = str(hit.get("url") or hit.get("link") or "")
    if not link:
        return False
    path = urlparse(link).path.lower()
    segments = _hit_path_segments(link)
    tokens = _hit_path_tokens(link)
    blob = " ".join(str(hit.get(field) or "") for field in ("title", "url", "link", "content", "snippet")).lower()
    if re.search(r"(19|20)\d{2}[/\-]?\d{2}[/\-]?\d{2}", path) or re.search(r"\d{6,8}", path):
        return True
    if _is_topic_or_hub_hit(hit):
        return False
    if any(tok in {"news", "noticias"} for tok in tokens) and _blob_has_article_signal(blob):
        return True
    if any(tok in {"topic", "topics", "tag", "tags", "category", "categories", "archive", "author", "world", "mundo", "index", "home", "partidos", "resultados"} for tok in tokens):
        return False
    if len(segments) >= 2 and _blob_has_article_signal(blob):
        return True
    if len(segments) == 1 and any("-" in seg or "_" in seg for seg in segments) and _blob_has_article_signal(blob):
        return True
    if any("-" in seg or "_" in seg or re.search(r"\d", seg) for seg in segments):
        return True
    return len(segments) >= 3


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


def _filter_search_hits_by_domains(
    hits: list[dict[str, str]],
    allowed_domains: Optional[list[str]],
    blocked_domains: Optional[list[str]],
) -> list[dict[str, str]]:
    filtered_hits: list[dict[str, str]] = []
    for hit in hits:
        url = str(hit.get("url") or hit.get("link") or "").strip()
        if not url:
            continue
        if not _domain_allowed(url, allowed=allowed_domains, blocked=blocked_domains):
            continue
        filtered_hits.append(hit)
    return filtered_hits


def _run_tavily_search(
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


def _run_searxng_search(
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


def _run_google_news_rss_search(
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


def _run_web_search_provider(
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
        return _run_tavily_search(query, allowed_domains, blocked_domains, num_results, max_age_days, topic=topic, time_range=time_range)
    if provider_kind == "google_news_rss":
        return _run_google_news_rss_search(query, allowed_domains, blocked_domains, num_results, max_age_days, topic=topic, time_range=time_range)
    if provider_kind == "searxng":
        return _run_searxng_search(query, allowed_domains, blocked_domains, num_results, max_age_days, topic=topic, time_range=time_range)
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
                hits = _run_web_search_provider(
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




def _html_to_markdownish_text(url: str, title: str, text: str, links: list[dict[str, str]]) -> str:
    parts = [f"# {title or url}", ""]
    if text.strip():
        parts.append(text.strip())
        parts.append("")
    if links:
        parts.append("## Links")
        for link in links[:12]:
            link_text = (link.get("text") or link.get("href") or "link").strip()
            href = (link.get("href") or "").strip()
            if href:
                parts.append(f"- [{link_text}]({href})")
        parts.append("")
    parts.append(f"Source URL: {url}")
    return "\n".join(parts).strip()


def _build_web_fetch_prompt(markdown_content: str, prompt: str, is_preapproved_domain: bool) -> str:
    guidelines = (
        "Provide a moderately detailed response based on the content above. Include relevant details, code examples, and documentation excerpts as needed."
        if is_preapproved_domain
        else "Provide a moderately detailed response based only on the content above. In your response:\n"
        " - Enforce a strict 125-character maximum for quotes from any source document. Open Source Software is ok as long as we respect the license.\n"
        " - Use quotation marks for exact language from articles; any language outside of the quotation should never be word-for-word the same.\n"
        " - You are not a lawyer and never comment on the legality of your own prompts and responses.\n"
        " - Never produce or reproduce exact song lyrics."
    )

    return f"""
Web page content:
---
{markdown_content}
---

{prompt}

{guidelines}
""".strip()


def _build_redirect_message(original_url: str, redirect_url: str, status_code: int, prompt: str) -> str:
    status_text = {
        301: "Moved Permanently",
        307: "Temporary Redirect",
        308: "Permanent Redirect",
    }.get(status_code, "Found")
    return (
        "REDIRECT DETECTED: The URL redirects to a different host.\n\n"
        f"Original URL: {original_url}\n"
        f"Redirect URL: {redirect_url}\n"
        f"Status: {status_code} {status_text}\n\n"
        "To complete your request, I need to fetch content from the redirected URL. "
        "Please use WebFetch again with these parameters:\n"
        f'- url: "{redirect_url}"\n'
        f'- prompt: "{prompt}"'
    )


@tool
def extract_price_from_text(
    text: Annotated[str, Field(description="Texto crudo del que extraer un precio numérico")],
) -> str:
    """Extrae un número tipo precio desde un texto y devuelve un valor normalizado."""
    import re

    if not text:
        return "No hay texto para extraer precio."

    m = re.search(r'([0-9]{1,3}(?:[,\.\s][0-9]{3})*(?:[,\.\s][0-9]{2,8})|[0-9]+(?:[,\.\s][0-9]{2,8})?)', text)
    if not m:
        return "No encontré un número de precio en el texto."

    raw = m.group(1).strip()
    if "." in raw and "," in raw:
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    else:
        raw = raw.replace(" ", "")
        if raw.count(",") == 1 and raw.count(".") == 0:
            parts = raw.split(",")
            if len(parts[-1]) in (2, 3, 4, 5, 6, 7, 8):
                raw = raw.replace(",", ".")

    return f"Precio detectado: {raw}"


@tool
def search_web(
    query: Annotated[str, Field(description="Consulta de búsqueda en lenguaje natural, ej: 'bitcoin price usd today', 'precio actual ethereum'")],
    provider: Annotated[Optional[str], Field(description="Proveedor de búsqueda a usar explícitamente")]=None,
    runtime_selected_provider: Annotated[Optional[str], Field(description="Provider seleccionado por el runtime")]=None,
    runtime_provider_configured: Annotated[Optional[str], Field(description="Provider configurado por el runtime")]=None,
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


@tool
def scrape_website_simple(
    url: Annotated[str, Field(description="URL completa incluyendo https://, para páginas estáticas (blogs, docs, noticias)")],
    extract_text: Annotated[bool, Field(description="Si True, extrae el texto principal de la página")] = True,
    extract_links: Annotated[bool, Field(description="Si True, extrae los enlaces encontrados en la página")] = False,
    max_chars: Annotated[int, Field(description="Límite de caracteres del texto extraído", ge=100, le=10000)] = 2000,
) -> str:
    """Extrae información de una página web estática usando requests + BeautifulSoup."""
    url_error = _validate_url(url)
    if url_error:
        return f"URL rechazada: {url_error}"
    if not _domain_allowed(url):
        return f"URL rechazada: dominio no permitido por la configuración actual ({urlparse(url).hostname or 'desconocido'})"
    try:
        from bs4 import BeautifulSoup

        html = scraping_infra._fetch_html(url)
        soup = BeautifulSoup(html, "html.parser")
        text = None
        links_text = None
        total_links = 0

        if extract_text:
            text = _extract_text(soup, max_chars)

        if extract_links:
            total_links, links_text = _extract_links(soup, url)

        return _build_result(url, text, links_text, total_links)
    except Exception as e:
        return f"Error al procesar la pagina web: {str(e)}"


@tool
def scrape_website_dynamic(
    url: Annotated[str, Field(description="URL completa incluyendo https://, para páginas con JavaScript (precios, dashboards, SPAs)")],
    wait_for_selector: Annotated[Optional[str], Field(description="Selector CSS a esperar antes de extraer, ej: '.price', '#content'")] = None,
    extract_selector: Annotated[Optional[str], Field(description="Selector CSS del bloque específico a extraer, ej: 'main', '.article-body'")] = None,
    extract_text: Annotated[bool, Field(description="Si True, extrae el texto principal de la página")] = True,
    extract_links: Annotated[bool, Field(description="Si True, extrae los enlaces encontrados")] = False,
    max_chars: Annotated[int, Field(description="Límite de caracteres del texto extraído", ge=100, le=10000)] = 2000,
    block_resources: Annotated[bool, Field(description="Si True, bloquea imágenes y fonts para mayor velocidad")] = True,
    use_cache: Annotated[bool, Field(description="Si True, usa caché de 60s por URL para evitar requests repetidos")] = True,
) -> str:
    """Extrae información de páginas web con JavaScript usando Playwright (sync)."""
    url_error = _validate_url(url)
    if url_error:
        return f"URL rechazada: {url_error}"
    if not _domain_allowed(url):
        return f"URL rechazada: dominio no permitido por la configuración actual ({urlparse(url).hostname or 'desconocido'})"
    cache_params = {
        "wait_for_selector": wait_for_selector,
        "extract_selector": extract_selector,
        "extract_text": extract_text,
        "extract_links": extract_links,
        "max_chars": max_chars,
        "block_resources": block_resources,
    }
    cache_key = _cache_key(url, cache_params)
    if use_cache:
        cached = _get_cache(cache_key)
        if cached:
            return cached

    try:
        from bs4 import BeautifulSoup

        browser = scraping_infra._get_browser()
        page = browser.new_page()
        scraping_infra._configure_page(page, block_resources=block_resources)

        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        if wait_for_selector:
            page.wait_for_selector(wait_for_selector, timeout=30000)

        html = page.content()
        page.close()

        soup = BeautifulSoup(html, "html.parser")
        text = None
        links_text = None
        total_links = 0

        if extract_text:
            text = _extract_text(soup, max_chars, extract_selector=extract_selector)

        if extract_links:
            total_links, links_text = _extract_links(soup, url)

        result = _build_result(url, text, links_text, total_links)
        if use_cache:
            _set_cache(cache_key, result)
        return result
    except Exception as e:
        return f"Error al procesar la pagina web: {str(e)}"


@tool
async def scrape_website_with_json_capture(
    url: Annotated[str, Field(description="URL completa incluyendo https://, ideal para páginas con APIs/endpoints JSON (trading, precios, datos en tiempo real)")],
    wait_for_selector: Annotated[Optional[str], Field(description="Selector CSS a esperar antes de extraer, ej: '.price', '#ticker'")] = None,
    extract_selector: Annotated[Optional[str], Field(description="Selector CSS del bloque específico a extraer")] = None,
    max_chars: Annotated[int, Field(description="Límite de caracteres del texto extraído", ge=100, le=10000)] = 2000,
    capture_json: Annotated[bool, Field(description="Si True, intercepta y guarda respuestas JSON de APIs en data_trading/")] = True,
) -> str:
    """Extrae información de páginas con JS y captura endpoints JSON automáticamente."""
    url_error = _validate_url(url)
    if url_error:
        return f"URL rechazada: {url_error}"
    if not _domain_allowed(url):
        return f"URL rechazada: dominio no permitido por la configuración actual ({urlparse(url).hostname or 'desconocido'})"
    try:
        result = await scraping_infra._scrape_dynamic_async(
            url=url,
            wait_for_selector=wait_for_selector,
            extract_selector=extract_selector,
            text_limit=max_chars,
            capture_json=capture_json,
        )

        parts = [f"URL: {result['url']}"]
        if result.get("title"):
            parts.append(f"Titulo: {result['title']}")
        parts.append(f"\nTexto extraido:\n{result['main_text']}")

        if result.get("links"):
            links_str = "\n".join([f"- {l['text']}: {l['href']}" for l in result["links"][:20]])
            parts.append(f"\n\nEnlaces encontrados ({len(result['links'])} total):\n{links_str}")

        if result.get("json_bundle_path"):
            parts.append(f"\n\n[JSON Capturado]")
            parts.append(f"Archivo: {result['json_bundle_path']}")
            parts.append(f"Respuestas capturadas: {result['json_captured_count']}")
            parts.append(f"Total bytes JSON: {result['json_total_bytes']}")

        return "\n".join(parts)
    except Exception as e:
        return f"Error al procesar la pagina web: {str(e)}"


@tool
async def web_fetch(
    url: Annotated[str, Field(description="URL completa incluyendo https://, para recuperar y sintetizar el contenido de una página web")],
    prompt: Annotated[str, Field(description="Prompt que define qué información querés extraer o sintetizar de la página")],
    use_dynamic: Annotated[bool, Field(description="Si True, usa Playwright para páginas con JavaScript")] = True,
    wait_for_selector: Annotated[Optional[str], Field(description="Selector CSS a esperar antes de extraer")] = None,
    extract_selector: Annotated[Optional[str], Field(description="Selector CSS del bloque específico a extraer")] = None,
    max_chars: Annotated[int, Field(description="Límite de caracteres del contenido base", ge=100, le=20000)] = 8000,
    block_resources: Annotated[bool, Field(description="Si True, bloquea imágenes y fonts para mayor velocidad")] = True,
    use_cache: Annotated[bool, Field(description="Si True, cachea resultados por URL+prompt por 15 minutos")] = True,
) -> str:
    """Recupera una página web, la convierte a texto estilo markdown y la sintetiza con un modelo chico."""
    url_error = _validate_url(url)
    if url_error:
        return f"URL rechazada: {url_error}"
    if not _domain_allowed(url):
        return f"URL rechazada: dominio no permitido por la configuración actual ({urlparse(url).hostname or 'desconocido'})"

    cache_key = _web_fetch_cache_key(url, prompt, use_dynamic, wait_for_selector, extract_selector, max_chars)
    if use_cache:
        cached = _get_web_fetch_cache(cache_key)
        if cached:
            return cached

    try:
        final_url = url
        result: dict[str, Any] = {}
        title_tag = urlparse(url).hostname or url
        if use_dynamic:
            import asyncio as _asyncio
            _fetch_url = url
            _fetch_wait = wait_for_selector
            _fetch_extract = extract_selector
            _fetch_limit = max_chars
            _fetch_block = block_resources
            result = await _asyncio.get_event_loop().run_in_executor(
                None,
                lambda: scraping_infra._scrape_page_sync(
                    url=_fetch_url,
                    wait_for_selector=_fetch_wait,
                    extract_selector=_fetch_extract,
                    text_limit=_fetch_limit,
                    block_resources=_fetch_block,
                ),
            )
            fetched_url = str(result.get("url") or url)
            parsed_original = urlparse(url)
            parsed_fetched = urlparse(fetched_url)
            if parsed_original.hostname and parsed_fetched.hostname and parsed_original.hostname != parsed_fetched.hostname:
                return _build_redirect_message(url, fetched_url, 307, prompt)
            final_url = fetched_url
            title = str(result.get("title") or fetched_url)
            text = str(result.get("main_text") or "")
            links = list(result.get("links") or [])
            markdown_content = _html_to_markdownish_text(fetched_url, title, text, links)
        else:
            import requests
            from bs4 import BeautifulSoup

            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            response.raise_for_status()
            fetched_url = str(response.url or url)
            parsed_original = urlparse(url)
            parsed_fetched = urlparse(fetched_url)
            if parsed_original.hostname and parsed_fetched.hostname and parsed_original.hostname != parsed_fetched.hostname:
                status_code = int(getattr(response, "status_code", 302) or 302)
                return _build_redirect_message(url, fetched_url, status_code, prompt)
            final_url = fetched_url
            html = response.content
            soup = BeautifulSoup(html, "html.parser")
            title_tag = soup.title.get_text(strip=True) if soup.title else fetched_url
            text = _extract_text(soup, max_chars, extract_selector=extract_selector)
            total_links, links_text = _extract_links(soup, fetched_url)
            links: list[dict[str, str]] = []
            if links_text:
                for line in links_text.splitlines():
                    if ": " in line:
                        link_text, href = line[2:].split(": ", 1) if line.startswith("- ") else line.split(": ", 1)
                        links.append({"text": link_text.strip(), "href": href.strip()})
            markdown_content = _html_to_markdownish_text(fetched_url, title_tag, text, links)

        from application.helpers.config_flow_helpers import get_llm

        is_preapproved = _is_preapproved_url(final_url)

        llm = get_llm()
        synthesized = await llm.ainvoke([
            HumanMessage(content=_build_web_fetch_prompt(markdown_content, prompt, is_preapproved))
        ])
        summary = getattr(synthesized, "content", str(synthesized)).strip()
        # Use real article title; fall back to hostname if missing
        if use_dynamic:
            article_title = str(result.get("title") or "").strip() or (urlparse(final_url).hostname or final_url)
        else:
            article_title = title_tag.strip() or (urlparse(final_url).hostname or final_url)
        domain = urlparse(final_url).hostname or final_url
        result_text = f"{summary}\n\n<<<CITE_THIS: title={article_title}|url={final_url}|domain={domain}>>>"

        if use_cache:
            _set_web_fetch_cache(cache_key, result_text)
        return result_text
    except Exception as e:
        return f"Error al procesar la pagina web: {str(e)}"


async def fetch_web_page(**kwargs) -> str:
    """Helper async directo para reutilizar la lógica WebFetch sin invocar el wrapper LangChain."""
    return await web_fetch.coroutine(**kwargs)  # pyright: ignore[reportAttributeAccessIssue]


__all__ = [
    "extract_price_from_text",
    "search_web",
    "scrape_website_simple",
    "scrape_website_dynamic",
    "scrape_website_with_json_capture",
    "web_fetch",
    "fetch_web_page",
]
