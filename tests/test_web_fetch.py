import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from features.web_scraping.infrastructure import scraping_infra
from features.web_scraping.infrastructure.scraping_tools import fetch_web_page


@pytest.mark.asyncio
async def test_web_fetch_synthesizes_markdown_content():
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="Resumen conciso")))

    with (
        patch("features.web_scraping.infrastructure.scraping_infra._scrape_page_sync", MagicMock(return_value={
            "url": "https://example.com/article",
            "title": "Example Article",
            "main_text": "Contenido principal de la pagina",
            "links": [{"text": "More", "href": "https://example.com/more"}],
        })),
        patch("core.helpers.config_flow_helpers.get_llm", return_value=llm),
    ):
        result = await fetch_web_page(
            url="https://example.com/article",
            prompt="Resumi lo importante",
            use_dynamic=True,
            use_cache=False,
        )

    assert "Resumen conciso" in result
    # Citation is emitted as a <<<CITE_THIS:...>>> marker (processed downstream into Sources block)
    assert "<<<CITE_THIS:" in result
    assert "https://example.com/article" in result
    assert llm.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_web_fetch_blocks_disallowed_domains(monkeypatch):
    monkeypatch.setenv("WEB_BLOCKED_DOMAINS", "example.com")

    result = await fetch_web_page(url="https://example.com/article", prompt="Resumi lo importante")

    assert "dominio no permitido" in result


@pytest.mark.asyncio
async def test_web_fetch_reports_cross_host_redirect():
    llm = SimpleNamespace(ainvoke=AsyncMock())

    with (
        patch("features.web_scraping.infrastructure.scraping_infra._scrape_page_sync", MagicMock(return_value={
            "url": "https://redirect.example.org/article",
            "title": "Redirected",
            "main_text": "Contenido movido",
            "links": [],
        })),
        patch("core.helpers.config_flow_helpers.get_llm", return_value=llm),
    ):
        result = await fetch_web_page(
            url="https://example.com/article",
            prompt="Resumi lo importante",
            use_dynamic=True,
        )

    assert "REDIRECT DETECTED" in result
    assert "Redirect URL: https://redirect.example.org/article" in result
    assert llm.ainvoke.await_count == 0


def test_sync_playwright_browser_is_scoped_per_thread():
    scraping_infra._shutdown_playwright()

    launched = []
    stopped = []

    class FakeBrowser:
        def __init__(self, owner: int):
            self.owner = owner

        def close(self):
            pass

    class FakeChromium:
        def launch(self, headless=True):
            browser = FakeBrowser(threading.get_ident())
            launched.append(browser)
            return browser

    class FakePlaywright:
        def __init__(self, owner: int):
            self.owner = owner
            self.chromium = FakeChromium()

        def stop(self):
            stopped.append(self.owner)

    class FakeSyncPlaywright:
        def start(self):
            return FakePlaywright(threading.get_ident())

    browser_by_thread: dict[int, object] = {}

    with patch("playwright.sync_api.sync_playwright", return_value=FakeSyncPlaywright()):
        def worker():
            browser_by_thread[threading.get_ident()] = scraping_infra._get_browser()

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(browser_by_thread) == 2
        assert len({id(browser) for browser in browser_by_thread.values()}) == 2
        assert len(launched) == 2

        scraping_infra._shutdown_playwright()

    assert len(stopped) == 2
