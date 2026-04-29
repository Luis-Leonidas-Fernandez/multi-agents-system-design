"""Despacho de estrategias de web scraping."""
from __future__ import annotations

from typing import Any, Optional, cast

from features.web_scraping.infrastructure.runtime import WebFetchRuntime, WebSearchRuntime


async def _run_generic_web_search_fetch(
    last_message: str,
    web_search_runtime_args: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    from features.web_scraping.application.country_strategy import CountryRecentNewsStrategy
    from features.web_scraping.application.generic_strategy import GenericWebSearchStrategy
    from features.web_scraping.application import flow as _flow

    search_runtime = WebSearchRuntime()
    fetch_runtime = WebFetchRuntime()
    if _flow.detect_query_source_group(last_message) == "japan" and _flow.detect_recent_query_horizon(last_message) == "week":
        from features.web_scraping.infrastructure.search_tools import search_web
        query = last_message
        search_text = search_web.invoke({"query": query, "use_cache": False, **(web_search_runtime_args or {}), "topic": "news", "time_range": "week"})
        if not isinstance(search_text, str):
            search_text = str(search_text)
        candidates = _flow._extract_generic_search_candidates(search_text)
        if candidates:
            lines = []
            sources = []
            seen_urls: set[str] = set()
            for candidate in candidates[:3]:
                title = candidate.get("title") or candidate.get("url") or ""
                snippet = (candidate.get("snippet") or "").strip()
                if title and snippet:
                    lines.append(f"{title} — {snippet}")
                url = candidate.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({"title": title or url, "url": url})
            if lines:
                digest_contract = _flow._build_web_digest_contract(lines, sources)
                summary = _flow._format_web_digest_contract(digest_contract)
                return {"summary": summary, "words": summary.split(), "source_type": "search", "sources": sources, "pre_synthesized": True, "digest_contract": digest_contract}

    if _flow.detect_query_source_group(last_message) == "japan" and _flow.detect_recent_query_horizon(last_message) == "week":
        generic_strategy = GenericWebSearchStrategy(search_runtime=search_runtime, fetch_runtime=fetch_runtime)
        result = await generic_strategy.execute(last_message, web_search_runtime_args)
        if result is not None:
            _flow._web_debug(
                "generic_fetch.strategy_selected",
                strategy="generic_web_search",
                query=last_message,
                source_count=len(cast(list[dict[str, str]], result.get("sources") or [])),
            )
        return result

    country_strategy = CountryRecentNewsStrategy(search_runtime=search_runtime, fetch_runtime=fetch_runtime)
    local_result = await country_strategy.execute(last_message, web_search_runtime_args)
    if local_result is not None:
        _flow._web_debug(
            "generic_fetch.strategy_selected",
            strategy="country_recent_news",
            query=last_message,
            source_count=len(cast(list[dict[str, str]], local_result.get("sources") or [])),
        )
        return local_result

    generic_strategy = GenericWebSearchStrategy(search_runtime=search_runtime, fetch_runtime=fetch_runtime)
    result = await generic_strategy.execute(last_message, web_search_runtime_args)
    if result is not None:
        _flow._web_debug(
            "generic_fetch.strategy_selected",
            strategy="generic_web_search",
            query=last_message,
            source_count=len(cast(list[dict[str, str]], result.get("sources") or [])),
        )
    return result


__all__ = ["_run_generic_web_search_fetch"]
