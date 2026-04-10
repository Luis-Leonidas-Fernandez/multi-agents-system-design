"""
Tests for search_web and scrape helpers in tools/web_tools.py.

search_web now resolves providers through the registry and can fall back from a
configured provider to the next available one when no explicit provider was selected.
Mocks patch tools.web_tools.TavilyClient and requests.get so the underlying
network calls are controlled. Hits use provider format:
  {"title": "...", "url": "...", "content": "..."}
"""
from unittest.mock import MagicMock, patch

import pytest

from tools.web_tools import _resolve_web_search_plan, search_web, scrape_website_simple


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_tavily_client(results: list[dict]) -> MagicMock:
    """Return a mock TavilyClient class whose .search() yields the given results."""
    client = MagicMock()
    client.search.return_value = {"results": results}
    cls = MagicMock(return_value=client)
    return cls


def _tavily_article(title: str, url: str, content: str = "") -> dict:
    return {"title": title, "url": url, "content": content}


def _searxng_article(title: str, url: str, content: str = "") -> dict:
    return {"title": title, "url": url, "content": content}


def _make_searxng_response(results: list[dict]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"results": results}
    return response


def _make_google_news_response(xml_text: str) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.text = xml_text
    return response


# ── validation ────────────────────────────────────────────────────────────────

def test_search_web_rejects_allowed_and_blocked_together():
    result = search_web.invoke({
        "query": "example query",
        "allowed_domains": ["allowed.example.com"],
        "blocked_domains": ["blocked.example.com"],
    })

    assert "Cannot specify both allowed_domains and blocked_domains" in result


def test_search_web_requires_tavily_key_when_no_provider_override(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)

    result = search_web.invoke({"query": "test", "use_cache": False})

    assert "TAVILY_API_KEY no configurada" in result


def test_search_web_honors_explicit_tavily_provider(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    hits = [_tavily_article("Explicit provider", "https://example.com/explicit", "Explicit content")]

    with patch("tools.web_tools.TavilyClient", _make_tavily_client(hits)) as tavily_mock:
        result = search_web.invoke({"query": "explicit provider", "provider": "tavily", "use_cache": False})

    assert tavily_mock.return_value.search.call_count == 1
    assert "Explicit provider" in result


def test_search_web_honors_runtime_selected_provider_and_keeps_fallback(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://localhost:8888")

    hits = [_tavily_article("Runtime provider", "https://example.com/runtime", "Runtime content")]

    with patch("tools.web_tools.TavilyClient", _make_tavily_client(hits)) as tavily_mock:
        result = search_web.invoke({
            "query": "runtime provider",
            "runtime_selected_provider": "tavily",
            "use_cache": False,
        })

    assert tavily_mock.return_value.search.call_count == 1
    assert "Runtime provider" in result
    plan = _resolve_web_search_plan(provider=None, runtime_selected_provider="tavily", runtime_provider_configured="tavily")
    assert [spec.name for spec in plan.provider_candidates] == ["tavily", "searxng"]
    assert plan.provider_explicit is False


def test_resolve_web_search_plan_prefers_runtime_selected_provider(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://localhost:8888")

    plan = _resolve_web_search_plan(
        provider=None,
        runtime_selected_provider="tavily",
        runtime_provider_configured="tavily",
    )

    assert plan.selected_provider == "tavily"
    assert plan.provider_explicit is False
    assert plan.provider_candidates[0].name == "tavily"
    assert plan.provider_candidates[1].name == "searxng"


def test_resolve_web_search_plan_precedence_chain(monkeypatch, tmp_path):
    from application.helpers.config_flow_helpers import get_web_search_runtime_config

    config_path = tmp_path / "web-search.json"
    config_path.write_text("{\"provider_configured\": \"tavily\"}", encoding="utf-8")
    monkeypatch.setenv("WEB_SEARCH_CONFIG", str(config_path))
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://localhost:8888")
    get_web_search_runtime_config.cache_clear()

    explicit = _resolve_web_search_plan(
        provider="tavily",
        runtime_selected_provider="tavily",
        runtime_provider_configured="tavily",
    )
    runtime_selected = _resolve_web_search_plan(
        provider=None,
        runtime_selected_provider="tavily",
        runtime_provider_configured="tavily",
    )
    runtime_configured = _resolve_web_search_plan(
        provider=None,
        runtime_selected_provider=None,
        runtime_provider_configured="tavily",
    )
    auto_detected = _resolve_web_search_plan(provider=None, runtime_selected_provider=None, runtime_provider_configured=None)

    assert explicit.selected_provider == "tavily"
    assert runtime_selected.selected_provider == "tavily"
    assert runtime_configured.selected_provider == "tavily"
    assert auto_detected.selected_provider == "tavily"
    assert [spec.name for spec in runtime_selected.provider_candidates] == ["tavily", "searxng"]
    assert [spec.name for spec in runtime_configured.provider_candidates] == ["tavily", "searxng"]
    assert [spec.name for spec in auto_detected.provider_candidates] == ["tavily", "searxng"]


def test_resolve_web_search_plan_uses_env_config_when_runtime_absent(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://localhost:8888")

    plan = _resolve_web_search_plan(provider=None, runtime_selected_provider=None, runtime_provider_configured=None)

    assert plan.selected_provider == "tavily"
    assert plan.provider_explicit is False
    assert [spec.name for spec in plan.provider_candidates] == ["tavily", "searxng"]


def test_search_web_falls_back_to_searxng_when_tavily_fails(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://localhost:8888")

    mock_cls = _make_tavily_client([])
    mock_cls.return_value.search.side_effect = RuntimeError("tavily down")
    searxng_results = [_searxng_article("SearXNG fallback", "https://example.org/searxng", "Fallback content")]
    searxng_response = _make_searxng_response(searxng_results)

    with patch("tools.web_tools.TavilyClient", mock_cls) as tavily_cls, patch(
        "requests.get", return_value=searxng_response
    ) as searxng_get:
        result = search_web.invoke({"query": "fallback", "use_cache": False})

    assert tavily_cls.return_value.search.call_count == 1
    assert searxng_get.call_count == 1
    assert "SearXNG fallback" in result


def test_search_web_prefers_google_news_rss_for_news_queries(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://localhost:8888")

    rss_xml = """<?xml version='1.0' encoding='UTF-8'?>
    <rss><channel>
      <item>
        <title>Italia refuerza la seguridad esta semana</title>
        <link>https://news.example.com/italia-seguridad</link>
        <description>La cobertura local describe medidas de seguridad en Italia.</description>
        <source url="https://news.example.com">Example News</source>
      </item>
    </channel></rss>"""

    with patch("requests.get", return_value=_make_google_news_response(rss_xml)) as rss_get, patch(
        "socket.getaddrinfo", return_value=[(None, None, None, None, None)]
    ):
        result = search_web.invoke({
            "query": "seguridad en italia esta semana",
            "topic": "news",
            "time_range": "week",
            "use_cache": False,
        })

    assert rss_get.call_count == 1
    assert "Italia refuerza la seguridad esta semana" in result
    assert "news.example.com" in result


def test_search_web_uses_searxng_without_tavily_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://localhost:8888")

    searxng_results = [_searxng_article("SearXNG only", "https://example.org/searxng-only", "SearXNG content")]
    searxng_response = _make_searxng_response(searxng_results)

    with patch("requests.get", return_value=searxng_response) as searxng_get:
        result = search_web.invoke({"query": "only searxng", "use_cache": False})

    assert searxng_get.call_count == 1
    assert "SearXNG only" in result


def test_search_web_honors_explicit_searxng_provider(monkeypatch):
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://localhost:8888")

    searxng_results = [_searxng_article("Explicit SearXNG", "https://example.org/explicit", "Explicit content")]
    searxng_response = _make_searxng_response(searxng_results)

    with patch("requests.get", return_value=searxng_response) as searxng_get:
        result = search_web.invoke({"query": "explicit searxng", "provider": "searxng", "use_cache": False})

    assert searxng_get.call_count == 1
    assert "Explicit SearXNG" in result


def test_search_web_filters_searxng_results_by_domain(monkeypatch):
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://localhost:8888")

    searxng_results = [
        _searxng_article("Allowed", "https://news.example.com/article", "Allowed content"),
        _searxng_article("Blocked", "https://blocked.example.com/article", "Blocked content"),
    ]
    searxng_response = _make_searxng_response(searxng_results)

    with patch("requests.get", return_value=searxng_response):
        result = search_web.invoke({
            "query": "domain filtering",
            "allowed_domains": ["news.example.com"],
            "use_cache": False,
        })

    assert "Allowed" in result
    assert "Blocked" not in result


# ── output format ─────────────────────────────────────────────────────────────

def test_search_web_tags_article_url_as_article(monkeypatch):
    """URL with date slug → [article] tag."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    hits = [_tavily_article(
        "Japan security update",
        "https://www3.nhk.or.jp/nhkworld/es/news/20260404_05/",
        "Tokio anuncia medidas de seguridad",
    )]

    with patch("tools.web_tools.TavilyClient", _make_tavily_client(hits)):
        result = search_web.invoke({"query": "japan security", "use_cache": False})

    assert "[article]" in result
    assert "nhk.or.jp" in result


def test_search_web_tags_hub_url_as_hub(monkeypatch):
    """Short URL with no date/slug → [hub] tag."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    hits = [_tavily_article(
        "CNN Breaking News",
        "https://cnn.com/",
        "Latest headlines",
    )]

    with patch("tools.web_tools.TavilyClient", _make_tavily_client(hits)):
        result = search_web.invoke({"query": "news today", "use_cache": False})

    assert "[hub]" in result


def test_search_web_includes_call_web_fetch_hint(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    hits = [_tavily_article("Some Article", "https://example.com/article-slug", "Content here")]

    with patch("tools.web_tools.TavilyClient", _make_tavily_client(hits)):
        result = search_web.invoke({"query": "something", "use_cache": False})

    assert "Call web_fetch" in result


def test_search_web_returns_no_results_when_empty(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    with patch("tools.web_tools.TavilyClient", _make_tavily_client([])):
        result = search_web.invoke({"query": "something", "use_cache": False})

    assert "No results found" in result


# ── Tavily parameter forwarding ───────────────────────────────────────────────

def test_search_web_passes_blocked_domains_to_tavily(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    mock_cls = _make_tavily_client([])

    with patch("tools.web_tools.TavilyClient", mock_cls):
        search_web.invoke({
            "query": "test",
            "blocked_domains": ["bad.example.com"],
            "use_cache": False,
        })

    call_kwargs = mock_cls.return_value.search.call_args.kwargs
    assert "bad.example.com" in call_kwargs.get("exclude_domains", [])


def test_search_web_passes_allowed_domains_to_tavily(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    mock_cls = _make_tavily_client([])

    with patch("tools.web_tools.TavilyClient", mock_cls):
        search_web.invoke({
            "query": "test",
            "allowed_domains": ["trusted.example.com"],
            "use_cache": False,
        })

    call_kwargs = mock_cls.return_value.search.call_args.kwargs
    assert "trusted.example.com" in call_kwargs.get("include_domains", [])


def test_search_web_passes_max_age_days_to_tavily(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    mock_cls = _make_tavily_client([])

    with patch("tools.web_tools.TavilyClient", mock_cls):
        search_web.invoke({"query": "recent news", "max_age_days": 7, "use_cache": False})

    call_kwargs = mock_cls.return_value.search.call_args.kwargs
    assert call_kwargs.get("days") == 7


def test_search_web_omits_days_when_max_age_days_is_none(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    mock_cls = _make_tavily_client([])

    with patch("tools.web_tools.TavilyClient", mock_cls):
        search_web.invoke({"query": "general query", "max_age_days": None, "use_cache": False})

    call_kwargs = mock_cls.return_value.search.call_args.kwargs
    assert "days" not in call_kwargs


# ── caching ──────────────────────────────────────────────────────────────────

def test_search_web_caches_result_for_same_query(monkeypatch):
    """Second identical call must not hit Tavily again."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    hits = [_tavily_article("Cached Article", "https://example.com/cached-2026-04-08", "Body")]
    mock_cls = _make_tavily_client(hits)

    cache_key_suffix = "caching_test_unique_xyz"
    with patch("tools.web_tools.TavilyClient", mock_cls):
        first = search_web.invoke({"query": cache_key_suffix, "use_cache": True})
        second = search_web.invoke({"query": cache_key_suffix, "use_cache": True})

    assert first == second
    assert mock_cls.return_value.search.call_count == 1


# ── scrape helpers ────────────────────────────────────────────────────────────

def test_scrape_website_simple_blocks_disallowed_domains(monkeypatch):
    monkeypatch.setenv("WEB_BLOCKED_DOMAINS", "example.com")

    result = scrape_website_simple.invoke({"url": "https://example.com"})

    assert "dominio no permitido" in result
