from unittest.mock import AsyncMock, patch

import pytest

from application.services.web_runtime import (
    WebFetchRequest,
    WebFetchRuntime,
    WebSearchRequest,
    WebSearchRuntime,
)


@pytest.mark.asyncio
async def test_web_search_runtime_returns_structured_hits(monkeypatch):
    monkeypatch.setenv("SEARXNG_BASE_URL", "http://localhost:8888")

    runtime = WebSearchRuntime()
    raw_result = (
        "1. [article] [Titular](https://example.com/article)\n"
        "   Resumen corto del artículo\n"
    )

    with patch("tools.search_tools.search_web.func", return_value=raw_result):
        response = await runtime.search(
            WebSearchRequest(
                query="seguridad en italia",
                provider="searxng",
                use_cache=False,
            )
        )

    assert response.provider_name == "searxng"
    assert response.hits
    assert response.hits[0].title == "Titular"
    assert response.hits[0].url == "https://example.com/article"


@pytest.mark.asyncio
async def test_web_fetch_runtime_reports_fetch_status():
    runtime = WebFetchRuntime()

    with patch("tools.scraping_tools.fetch_web_page", new=AsyncMock(return_value="REDIRECT DETECTED\nRedirect URL: https://example.org")):
        response = await runtime.fetch(
            WebFetchRequest(
                url="https://example.com/article",
                prompt="Resumí",
                mode="dynamic",
                use_cache=False,
            )
        )

    assert response.provider_name == "default"
    assert response.status == "redirect"
    assert response.fetch_kind == "dynamic"
