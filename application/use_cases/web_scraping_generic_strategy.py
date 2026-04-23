"""Generic web search strategy orchestration."""
from __future__ import annotations

import asyncio
import re
from typing import Any, Optional, cast


async def _run_generic_web_search_strategy_impl(
    last_message: str,
    web_search_runtime_args: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    from application.use_cases import web_scraping_flow as _flow
    from domain.web_classifier import _is_specific_article_hit

    ctx, policy = _flow._build_query_context(last_message)
    query_terms = ctx.query_terms
    query_source_group = ctx.query_source_group
    query_horizon = ctx.query_horizon
    search_age_days = ctx.search_age_days
    recent_min_score = policy.min_score
    recent_min_body_lines = policy.min_body_lines
    recent_min_sources = policy.min_sources
    recent_min_candidates = policy.min_candidates
    loop = asyncio.get_running_loop()

    if query_horizon == "week":
        diverse_candidates, search_text, early = await _flow._run_week_search_pipeline(last_message, ctx, web_search_runtime_args)
        if early is not None:
            return early
    else:
        diverse_candidates, search_text = await _flow._run_general_search_pipeline(last_message, ctx, loop, web_search_runtime_args)

    ranked_candidates = diverse_candidates[:4]
    _flow._web_debug(
        "generic_fetch.ranked_candidates",
        query=last_message,
        query_horizon=query_horizon,
        ranked_candidate_count=len(ranked_candidates),
        urls=[c.get("url", "") for c in ranked_candidates],
    )

    if query_horizon == "week":
        direct_candidates = _flow._extract_generic_search_candidates(search_text)
        if len(direct_candidates) >= 2:
            week_direct_lines = []
            week_direct_sources: list[dict[str, str]] = []
            seen_urls: set[str] = set()
            for candidate in direct_candidates[:3]:
                title = candidate.get("title") or candidate.get("url") or ""
                snippet = (candidate.get("snippet") or "").strip()
                if title and snippet:
                    week_direct_lines.append(f"{title} — {snippet}")
                url = candidate.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    week_direct_sources.append({"title": title or url, "url": url})
            if week_direct_lines:
                summary = _flow._build_source_backed_response(week_direct_lines, week_direct_sources)
                if week_direct_sources and "Sources:" not in summary:
                    summary = summary + "\n\nSources:\n" + "\n".join(
                        f"- [{source['title']}]({source['url']})" for source in week_direct_sources if source.get("url")
                    )
                return {
                    "summary": summary,
                    "words": summary.split(),
                    "source_type": "search",
                    "sources": week_direct_sources,
                    "pre_synthesized": True,
                }

    if query_horizon == "week" and len(ranked_candidates) >= 2:
        week_direct_lines = []
        week_direct_sources: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for candidate in ranked_candidates[:3]:
            title = candidate.get("title") or candidate.get("url") or ""
            snippet = (candidate.get("snippet") or "").strip()
            if title and snippet:
                week_direct_lines.append(f"{title} — {snippet}")
            url = candidate.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                week_direct_sources.append({"title": title or url, "url": url})
        if week_direct_lines:
            summary = _flow._build_source_backed_response(week_direct_lines, week_direct_sources)
            if week_direct_sources and "Sources:" not in summary:
                summary = summary + "\n\nSources:\n" + "\n".join(
                    f"- [{source['title']}]({source['url']})" for source in week_direct_sources if source.get("url")
                )
            return {
                "summary": summary,
                "words": summary.split(),
                "source_type": "search",
                "sources": week_direct_sources,
                "pre_synthesized": True,
            }

    if query_horizon == "week":
        week_entry_lines: list[str] = []
        week_snippet_lines: list[str] = []
        week_entry_sources: list[dict[str, str]] = []
        week_snippet_sources: list[dict[str, str]] = []
        seen_week_urls: set[str] = set()
        fetch_prompt_week = _flow._build_generic_fetch_prompt(last_message)

        async def _week_entry(c: dict[str, str]) -> tuple[str, str, bool]:
            url = c.get("url", "")
            snippet = c.get("snippet", "").strip()
            snippet = re.sub(r"^#+\s+", "", snippet)
            snippet = re.sub(r"\s+#+\s+", " ", snippet).strip()
            title = re.sub(r"^#+\s+", "", c.get("title", url)).strip()
            if _flow._is_article_url(url):
                try:
                    fetched = await _flow._fetch_web_page_follow_redirect(url, fetch_prompt_week, use_dynamic=False)
                    if (
                        isinstance(fetched, str)
                        and not fetched.startswith("Error")
                        and not fetched.startswith("URL rechazada")
                        and not _flow._is_no_info_response(fetched)
                        and len(fetched.split()) >= 20
                    ):
                        lines = _flow._extract_generic_content_lines(fetched, query_terms)
                        if lines:
                            return title, " ".join(lines[:3]), False
                except Exception:
                    pass
            return title, snippet, True

        week_results = await asyncio.gather(*[_week_entry(c) for c in diverse_candidates], return_exceptions=True)

        for result, c in zip(week_results, diverse_candidates):
            if isinstance(result, Exception):
                title = c.get("title") or c.get("url") or ""
                content = c.get("snippet", "").strip()
                from_snippet = True
            else:
                title, content, from_snippet = result
            min_words = 4 if from_snippet else 8
            if not content or len(content.split()) < min_words:
                continue
            if _flow._is_no_info_response(content):
                continue
            url = c.get("url", "")
            if url in seen_week_urls:
                continue
            seen_week_urls.add(url)
            week_entry_lines.append(f"[{title}] — {content}")
            if from_snippet:
                week_snippet_lines.append(f"[{title}] — {content}")
            if _flow._is_article_url(url):
                week_entry_sources.append({"title": title, "url": url})
                if from_snippet:
                    week_snippet_sources.append({"title": title, "url": url})

        _flow._web_debug(
            "generic_fetch.week_entries",
            entry_count=len(week_entry_lines),
            snippet_count=len(week_snippet_lines),
            entry_sources=week_entry_sources,
            snippet_sources=week_snippet_sources,
        )

        if len(week_entry_lines) >= 2:
            import datetime as _dt_week
            _current_year = _dt_week.date.today().year
            _old_year_re = re.compile(r'\b(20\d{2})\b')
            paragraph_parts = []
            for (content_title, content, _from_snippet), c in zip(week_results, diverse_candidates):
                title = content_title or c.get("title") or c.get("url") or ""
                _years = [int(y) for y in _old_year_re.findall(content)]
                if _years and max(_years) <= _current_year - 1:
                    continue
                sentences = re.split(r"(?<=[.!?])\s+", content)
                trimmed = " ".join(sentences[:3]).strip()
                is_truncated = trimmed.endswith(("…", "...")) or re.search(r"\w…$", trimmed)
                if trimmed and not _flow._is_no_info_response(trimmed) and not is_truncated:
                    url = _flow._clean_source_url(c.get("url", ""))
                    src_line = f"Fuente: [{title}]({url})" if url else (f"Fuente: {title}" if title else "")
                    entry = f"{title}: {trimmed}"
                    if src_line:
                        entry = f"{entry}\n\n{src_line}"
                    paragraph_parts.append(entry)
            summary = "\n\n".join(paragraph_parts)
            if week_entry_sources and "Sources:" not in summary:
                summary = summary + "\n\nSources:\n" + "\n".join(
                    f"- [{source['title']}]({source['url']})" for source in week_entry_sources if source.get("url")
                )
            return {
                "summary": summary,
                "words": summary.split(),
                "source_type": "search",
                "sources": week_entry_sources,
                "pre_synthesized": True,
            }
        if week_snippet_lines:
            snippet_sources = week_snippet_sources or week_entry_sources or [{"title": "search result", "url": ""}]
            snippet_summary = _flow._build_source_backed_response(
                week_snippet_lines[:8],
                snippet_sources,
            )
            if snippet_sources and "Sources:" not in snippet_summary:
                snippet_summary = snippet_summary + "\n\nSources:\n" + "\n".join(
                    f"- [{source['title']}]({source['url']})" for source in snippet_sources if source.get("url")
                )
            return {
                "summary": snippet_summary,
                "words": snippet_summary.split(),
                "source_type": "search",
                "sources": snippet_sources,
                "pre_synthesized": True,
            }

    if not ranked_candidates:
        search_lines = _flow._extract_generic_content_lines(search_text, query_terms)
        if not search_lines:
            _flow._web_debug(
                "generic_fetch.no_ranked_candidates",
                query=last_message,
                search_preview=search_text[:500],
                search_lines_count=0,
            )
            return None
        sources = _flow._extract_sources_from_text(search_text)
        if not sources:
            sources = [{"title": "search result", "url": ""}]
        summary = _flow._build_source_backed_response(search_lines[:8], sources)
        _flow._web_debug(
            "generic_fetch.search_only_fallback",
            query=last_message,
            search_lines_count=len(search_lines),
            source_count=len(sources),
        )
        return {
            "summary": summary,
            "words": summary.split(),
            "source_type": "search",
            "sources": sources,
            "search_text": search_text,
        }

    eligible_entries = await _flow._fetch_and_score_entries(ranked_candidates, last_message, ctx, policy, search_text)

    if query_horizon == "week" and eligible_entries:
        recent_article_entries = [
            entry for entry in eligible_entries
            if _flow._candidate_url_is_recent(cast(dict[str, str], entry["candidate"]).get("url", ""), search_age_days or 14)
            and _is_specific_article_hit(cast(dict[str, str], entry["candidate"]))
            and not _flow._is_invalid_news_candidate(cast(dict[str, str], entry["candidate"]), last_message)
        ]
        if recent_article_entries:
            best_recent_entry = sorted(
                recent_article_entries,
                key=lambda entry: (
                    _flow._candidate_source_priority(cast(dict[str, str], entry["candidate"]), query_source_group),
                    -cast(int, entry["score"]),
                ),
            )[0]
            summary = _flow._build_source_backed_response(
                cast(list[str], best_recent_entry["summary_lines"]),
                cast(list[dict[str, str]], best_recent_entry["sources"]),
            )
            _flow._web_debug(
                "generic_fetch.week_single_recent_article",
                url=cast(dict[str, str], best_recent_entry["candidate"]).get("url", ""),
                score=cast(int, best_recent_entry["score"]),
                source_count=len(cast(list[dict[str, str]], best_recent_entry["sources"])),
            )
            return {
                "summary": summary,
                "words": summary.split(),
                "source_type": "webfetch",
                "sources": cast(list[dict[str, str]], best_recent_entry["sources"]),
                "score": cast(int, best_recent_entry["score"]),
            }

    if query_horizon == "week" and eligible_entries:
        eligible_entries = [
            entry for entry in eligible_entries
            if not _flow._is_invalid_news_candidate(cast(dict[str, str], entry["candidate"]), last_message)
        ]
        if not eligible_entries:
            return None
        ordered_entries = sorted(
            eligible_entries,
            key=lambda entry: (
                _flow._candidate_source_priority(cast(dict[str, str], entry["candidate"]), query_source_group),
                -cast(int, entry["score"]),
            ),
        )
        max_size = min(len(ordered_entries), max(4, recent_min_candidates))
        for size in range(min(recent_min_candidates, max_size), max_size + 1):
            selected = ordered_entries[:size]
            combined_score = sum(cast(int, entry["score"]) for entry in selected)
            combined_lines: list[str] = []
            seen_lines: set[str] = set()
            combined_sources: list[dict[str, str]] = []
            seen_urls: set[str] = set()

            for entry in selected:
                for line in cast(list[str], entry["summary_lines"]):
                    normalized_line = re.sub(r"\s+", " ", line).strip().lower()
                    if normalized_line and normalized_line not in seen_lines:
                        seen_lines.add(normalized_line)
                        combined_lines.append(line)
                for source in cast(list[dict[str, str]], entry["sources"]):
                    url = str(source.get("url") or "").strip()
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        combined_sources.append(source)

            if combined_score >= recent_min_score and len(combined_lines) >= recent_min_body_lines and len(combined_sources) >= recent_min_sources:
                final_sources = list(combined_sources)
                if len(final_sources) < 4:
                    seen_urls = {str(s.get("url") or "") for s in final_sources if s.get("url")}
                    for extra in _flow._extract_sources_from_text(search_text):
                        extra_url = str(extra.get("url") or "")
                        if extra_url and extra_url not in seen_urls and len(final_sources) < 5:
                            final_sources.append(extra)
                            seen_urls.add(extra_url)
                summary = _flow._build_source_backed_response(combined_lines[:20], final_sources)
                _flow._web_debug(
                    "generic_fetch.week_success",
                    selected_size=size,
                    combined_score=combined_score,
                    combined_lines_count=len(combined_lines),
                    combined_sources_count=len(combined_sources),
                    final_sources_count=len(final_sources),
                )
                return {
                    "summary": summary,
                    "words": summary.split(),
                    "source_type": "webfetch",
                    "sources": final_sources,
                    "score": combined_score,
                }

    if query_horizon == "week" and len(eligible_entries) < recent_min_candidates:
        snippet_lines: list[str] = []
        snippet_sources: list[dict[str, str]] = []
        seen_snippet_urls: set[str] = set()
        for c in diverse_candidates:
            snippet = c.get("snippet", "").strip()
            if not snippet or len(snippet.split()) < 6:
                continue
            url = c.get("url", "")
            if url in seen_snippet_urls:
                continue
            seen_snippet_urls.add(url)
            title = c.get("title", url)
            snippet_lines.append(f"{title} — {snippet}")
            snippet_sources.append({"title": title, "url": url})
        if snippet_lines:
            summary = _flow._build_source_backed_response(snippet_lines, snippet_sources)
            _flow._web_debug(
                "generic_fetch.week_snippet_fallback",
                snippet_count=len(snippet_lines),
                source_count=len(snippet_sources),
                urls=[source.get("url", "") for source in snippet_sources],
            )
            return {
                "summary": summary,
                "words": summary.split(),
                "source_type": "search",
                "sources": snippet_sources,
                "pre_synthesized": True,
            }

    if query_horizon != "week" and eligible_entries:
        valid_entries = [
            entry for entry in eligible_entries
            if not _flow._is_invalid_news_candidate(cast(dict[str, str], entry["candidate"]), last_message)
        ]
        if valid_entries:
            eligible_entries = valid_entries
        best_entry = sorted(
            eligible_entries,
            key=lambda entry: (
                _flow._candidate_source_priority(cast(dict[str, str], entry["candidate"]), query_source_group),
                -cast(int, entry["score"]),
            ),
        )[0]
        summary = _flow._build_source_backed_response(cast(list[str], best_entry["summary_lines"]), cast(list[dict[str, str]], best_entry["sources"]))
        _flow._web_debug(
            "generic_fetch.non_week_success",
            url=cast(dict[str, str], best_entry["candidate"]).get("url", ""),
            score=cast(int, best_entry["score"]),
            source_count=len(cast(list[dict[str, str]], best_entry["sources"])),
        )
        return {
            "summary": summary,
            "words": summary.split(),
            "source_type": "webfetch",
            "sources": cast(list[dict[str, str]], best_entry["sources"]),
            "score": cast(int, best_entry["score"]),
        }

    search_lines = _flow._extract_generic_content_lines(search_text, query_terms)
    if not search_lines:
        _flow._web_debug("generic_fetch.search_lines_empty", query=last_message, search_preview=search_text[:500])
        return None
    if _flow._is_recent_web_information_query(last_message):
        if len(search_lines) < recent_min_body_lines:
            strongest_candidate = ranked_candidates[0] if ranked_candidates else None
            strongest_score = _flow._score_generic_candidate(strongest_candidate, query_terms, query_source_group) if strongest_candidate else 0
            if not (
                strongest_candidate is not None
                and strongest_score >= recent_min_score
                and _is_specific_article_hit({
                    "title": strongest_candidate.get("title") or "",
                    "link": strongest_candidate.get("url") or strongest_candidate.get("link") or "",
                    "snippet": strongest_candidate.get("snippet") or "",
                })
            ):
                _flow._web_debug(
                    "generic_fetch.search_lines_rejected_recent",
                    search_lines_count=len(search_lines),
                    strongest_url=(strongest_candidate or {}).get("url", "") if strongest_candidate else "",
                    strongest_score=strongest_score,
                    min_score=recent_min_score,
                    min_body_lines=recent_min_body_lines,
                )
                return None
        sources = _flow._extract_sources_from_text(search_text)
        if len(sources) < recent_min_sources:
            _flow._web_debug(
                "generic_fetch.search_sources_rejected_recent",
                source_count=len(sources),
                min_sources=recent_min_sources,
            )
            return None

    sources = _flow._extract_sources_from_text(search_text)
    if not sources:
        top = ranked_candidates[0]
        sources = [{"title": top.get("title") or top["url"], "url": top["url"]}]

    if len(search_lines) < 3:
        search_fallback_lines = [line.strip() for line in search_text.splitlines() if line.strip() and not line.strip().lower().startswith(("url:", "sources:", "http")) and "http" not in line.lower()]
        for line in search_fallback_lines:
            if line not in search_lines:
                search_lines.append(line)
            if len(search_lines) >= 6:
                break

    summary = _flow._build_source_backed_response(search_lines[:8], sources)
    _flow._web_debug(
        "generic_fetch.final_search_summary",
        search_lines_count=len(search_lines),
        source_count=len(sources),
        urls=[source.get("url", "") for source in sources],
    )
    return {
        "summary": summary,
        "words": summary.split(),
        "source_type": "search",
        "sources": sources,
        "search_text": search_text,
    }


class GenericWebSearchStrategy:
    """Compatibility wrapper around the generic web search strategy function."""

    def __init__(self, search_runtime=None, fetch_runtime=None):
        self.search_runtime = search_runtime
        self.fetch_runtime = fetch_runtime

    async def execute(self, last_message: str, web_search_runtime_args: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
        return await _run_generic_web_search_strategy_impl(last_message, web_search_runtime_args)


__all__ = ["_run_generic_web_search_strategy_impl", "GenericWebSearchStrategy"]
