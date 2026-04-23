"""Internal helpers for web fetch execution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage

from application.helpers.scraping_flow_helpers import _extract_links, _extract_text
from application.policies.security_flow import input_guard
from tools.scraping_core import _build_redirect_message, _html_to_markdownish_text, _is_preapproved_url
from infra import scraping_infra


@dataclass(frozen=True)
class WebFetchDraft:
    final_url: str
    title: str
    markdown_content: str
    is_preapproved: bool


def _check_fetch_input_guard(url: str, prompt: str) -> Optional[str]:
    guard_state = {"messages": [HumanMessage(content=f"URL: {url}\n\nPrompt: {prompt}")]}
    guard_result = input_guard(guard_state)
    if not guard_result:
        return None
    messages = guard_result.get("messages") if isinstance(guard_result, dict) else None
    if messages:
        first = messages[0]
        return getattr(first, "content", str(first))
    return "Solicitud bloqueada por política de seguridad."


async def _build_dynamic_web_fetch_draft(
    url: str,
    prompt: str,
    wait_for_selector: Optional[str],
    extract_selector: Optional[str],
    max_chars: int,
    block_resources: bool,
) -> Union[WebFetchDraft, str]:
    blocked = _check_fetch_input_guard(url, prompt)
    if blocked:
        return blocked

    import asyncio as _asyncio

    result = await _asyncio.get_event_loop().run_in_executor(
        None,
        lambda: scraping_infra._scrape_page_sync(
            url=url,
            wait_for_selector=wait_for_selector,
            extract_selector=extract_selector,
            text_limit=max_chars,
            block_resources=block_resources,
        ),
    )
    fetched_url = str(result.get("url") or url)
    parsed_original = urlparse(url)
    parsed_fetched = urlparse(fetched_url)
    if parsed_original.hostname and parsed_fetched.hostname and parsed_original.hostname != parsed_fetched.hostname:
        return _build_redirect_message(url, fetched_url, 307, prompt)

    title = str(result.get("title") or fetched_url)
    text = str(result.get("main_text") or "")
    links = list(result.get("links") or [])
    return WebFetchDraft(
        final_url=fetched_url,
        title=title,
        markdown_content=_html_to_markdownish_text(fetched_url, title, text, links),
        is_preapproved=_is_preapproved_url(fetched_url),
    )


def _build_static_web_fetch_draft(
    url: str,
    prompt: str,
    max_chars: int,
    extract_selector: Optional[str],
) -> Union[WebFetchDraft, str]:
    blocked = _check_fetch_input_guard(url, prompt)
    if blocked:
        return blocked

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

    soup = BeautifulSoup(response.content, "html.parser")
    title_tag = soup.title.get_text(strip=True) if soup.title else fetched_url
    text = _extract_text(soup, max_chars, extract_selector=extract_selector)
    total_links, links_text = _extract_links(soup, fetched_url)
    links: list[dict[str, str]] = []
    if links_text:
        for line in links_text.splitlines():
            if ": " in line:
                link_text, href = line[2:].split(": ", 1) if line.startswith("- ") else line.split(": ", 1)
                links.append({"text": link_text.strip(), "href": href.strip()})

    _ = total_links
    return WebFetchDraft(
        final_url=fetched_url,
        title=title_tag,
        markdown_content=_html_to_markdownish_text(fetched_url, title_tag, text, links),
        is_preapproved=_is_preapproved_url(fetched_url),
    )


async def build_web_fetch_draft(
    url: str,
    prompt: str,
    use_dynamic: bool,
    wait_for_selector: Optional[str],
    extract_selector: Optional[str],
    max_chars: int,
    block_resources: bool,
) -> Union[WebFetchDraft, str]:
    if use_dynamic:
        return await _build_dynamic_web_fetch_draft(url, prompt, wait_for_selector, extract_selector, max_chars, block_resources)
    return _build_static_web_fetch_draft(url, prompt, max_chars, extract_selector)
