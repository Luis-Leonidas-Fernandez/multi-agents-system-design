"""Despacho de estrategias de web scraping."""
from __future__ import annotations

import re
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
    query_source_group = _flow.detect_query_source_group(last_message)
    query_horizon = _flow.detect_recent_query_horizon(last_message) if _flow._is_recent_web_information_query(last_message) else None

    def _has_min_recent_news_evidence(lines: list[str], sources: list[dict[str, str]]) -> bool:
        useful_lines = [line for line in lines if len(re.sub(r"\s+", " ", line).split()) >= 5]
        useful_sources = [source for source in sources if (source.get("url") or "").strip()]
        return len(useful_lines) >= 2 and len(useful_sources) >= 2

    def _result_has_min_recent_news_evidence(result: Optional[dict[str, Any]]) -> bool:
        if not result:
            return False
        sources = cast(list[dict[str, str]], result.get("sources") or [])
        if len([source for source in sources if (source.get("url") or "").strip()]) < 2:
            return False
        summary = str(result.get("summary") or "")
        return summary.count("Fuente:") >= 2 or summary.count("• ") >= 2

    if query_source_group == "japan" and query_horizon in {"today", "week"}:
        from features.web_scraping.infrastructure.search_tools import search_web
        query = last_message
        search_text = search_web.invoke({
            "query": query,
            "use_cache": False,
            **(web_search_runtime_args or {}),
            "topic": "news",
            "time_range": "week",
            "max_age_days": 14,
        })
        if not isinstance(search_text, str):
            search_text = str(search_text)
        candidates = _flow._extract_generic_search_candidates(search_text)
        _flow._web_debug(
            "fetch_dispatch.japan.search_results",
            query=last_message,
            query_horizon=query_horizon,
            candidate_count=len(candidates),
            search_preview=search_text[:500],
        )
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
            if _has_min_recent_news_evidence(lines, sources):
                _flow._web_debug(
                    "fetch_dispatch.japan.direct_digest_return",
                    line_count=len(lines),
                    source_count=len(sources),
                    urls=[source.get("url", "") for source in sources],
                )
                digest_contract = _flow._build_web_digest_contract(lines, sources)
                summary = _flow._format_web_digest_contract(digest_contract)
                return {"summary": summary, "words": summary.split(), "source_type": "search", "sources": sources, "pre_synthesized": True, "digest_contract": digest_contract}

    if query_source_group == "japan" and query_horizon in {"today", "week"}:
        generic_strategy = GenericWebSearchStrategy(search_runtime=search_runtime, fetch_runtime=fetch_runtime)
        result = await generic_strategy.execute(last_message, web_search_runtime_args)
        if _result_has_min_recent_news_evidence(result):
            _flow._web_debug(
                "fetch_dispatch.japan.generic_strategy_return",
                source_count=len(cast(list[dict[str, str]], result.get("sources") or [])),
                summary_preview=str(result.get("summary") or "")[:300],
            )
            _flow._web_debug(
                "generic_fetch.strategy_selected",
                strategy="generic_web_search",
                query=last_message,
                source_count=len(cast(list[dict[str, str]], result.get("sources") or [])),
            )
            return result
        _flow._web_debug(
            "fetch_dispatch.japan.no_local_evidence",
            result_is_none=result is None,
            source_count=len(cast(list[dict[str, str]], (result or {}).get("sources") or [])),
            summary_preview=str((result or {}).get("summary") or "")[:300],
        )
        return _flow._build_no_local_sources_response(last_message)

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
