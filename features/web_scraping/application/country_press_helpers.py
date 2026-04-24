"""Helpers for country-press discovery and search."""
from __future__ import annotations

import asyncio
from typing import Any, Optional
from urllib.parse import urlparse


async def _run_country_press_search_candidates(
    last_message: str,
    search_age_days: Optional[int],
    query_terms: list[str],
    query_source_group: Optional[str],
    source_terms: list[str],
    web_search_runtime_args: Optional[dict[str, Any]] = None,
    query_horizon: Optional[str] = None,
) -> tuple[list[dict[str, str]], str]:
    from features.web_scraping.domain.classifier import _is_specific_article_hit
    from features.web_scraping.infrastructure.search_tools import search_web
    from features.web_scraping.infrastructure.scraping_tools import fetch_web_page
    from features.web_scraping.application import flow as _flow

    country_press_domains, country_press_names = await _flow._discover_country_press_sources(
        last_message,
        query_source_group,
        source_terms,
        web_search_runtime_args,
    )
    if not country_press_domains:
        _flow._web_debug(
            "country_press.search.no_domains",
            query=last_message,
            query_source_group=query_source_group,
            source_terms=source_terms,
        )
        return [], ""

    loop = asyncio.get_running_loop()
    combined_search_text: list[str] = []
    raw_candidates: list[dict[str, str]] = []
    dynamic_fetch_available: Optional[bool] = None
    country_press_sources = _flow._country_press_source_cache_get(query_source_group, source_terms)
    sources_by_domain: dict[str, dict[str, str]] = {}
    for source in country_press_sources:
        url = source.get("url", "")
        hostname = (urlparse(url).hostname or "").lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        if hostname and hostname not in sources_by_domain:
            sources_by_domain[hostname] = source

    relevant_targets: list[tuple[str, str]] = []
    for idx, domain in enumerate(country_press_domains):
        press_name = country_press_names[idx] if idx < len(country_press_names) else domain
        source_meta = sources_by_domain.get(domain, {"title": press_name, "url": _flow._default_press_homepage_url(domain)})
        if not _flow._is_press_source_relevant_for_query(source_meta, last_message):
            _flow._web_debug("country_press.search.source_skipped", domain=domain, press_name=press_name, reason="irrelevant_for_query")
            continue
        relevant_targets.append((domain, press_name))
        if len(relevant_targets) >= 8:
            break

    for domain, press_name in relevant_targets:
        all_diary_candidates: list[dict[str, str]] = []
        domain_search_texts: list[str] = []
        queries = _flow._build_country_press_search_queries(last_message, domain, press_name)
        for query in queries:
            query_attempts = [
                ("news", _flow._build_country_press_search_invoke_args(
                    query,
                    domain,
                    search_age_days=search_age_days,
                    query_horizon=query_horizon,
                    web_search_runtime_args=web_search_runtime_args,
                    broad=False,
                )),
                ("general", _flow._build_country_press_search_invoke_args(
                    query,
                    domain,
                    search_age_days=search_age_days,
                    query_horizon=query_horizon,
                    web_search_runtime_args=web_search_runtime_args,
                    broad=True,
                )),
            ]
            for attempt_label, invoke_args in query_attempts:
                try:
                    search_text = await loop.run_in_executor(None, lambda q=invoke_args: search_web.invoke(q))
                except Exception:
                    _flow._web_debug("country_press.search.exception", domain=domain, query=query, attempt=attempt_label)
                    continue
                if not isinstance(search_text, str):
                    search_text = str(search_text)
                domain_search_texts.append(search_text)
                combined_search_text.append(search_text)
                diary_candidates = [c for c in _flow._extract_generic_search_candidates(search_text) if not _flow._is_non_news_candidate(c)]
                all_diary_candidates.extend(diary_candidates)
                article_candidates = [c for c in diary_candidates if _is_specific_article_hit(c)]
                _flow._web_debug(
                    "country_press.search.domain_result",
                    domain=domain,
                    press_name=press_name,
                    query=query,
                    attempt=attempt_label,
                    invoke_args=invoke_args,
                    candidate_count=len(diary_candidates),
                    article_candidate_count=len(article_candidates),
                    search_preview=search_text[:500],
                )
                if article_candidates or diary_candidates:
                    break
            if any(_is_specific_article_hit(c) for c in all_diary_candidates):
                break

        diary_candidates = _flow._dedup_candidates_by_event(all_diary_candidates, query_terms) if all_diary_candidates else []
        diary_candidates = [c for c in diary_candidates if not _flow._is_invalid_news_candidate(c, last_message)]
        if query_horizon == "week":
            url_age_threshold = search_age_days or 14
            recent_diary_candidates = [c for c in diary_candidates if _flow._candidate_url_is_recent(c.get("url", ""), url_age_threshold)]
            strict_recent_candidates = [
                c for c in recent_diary_candidates
                if _flow._candidate_url_has_date(c.get("url", "")) or not _flow._is_invalid_news_candidate(c, last_message)
            ]
            article_recent_candidates = [
                c for c in strict_recent_candidates
                if _is_specific_article_hit(c) and not _flow._is_invalid_news_candidate(c, last_message)
            ]
            _flow._web_debug(
                "country_press.search.week_filter",
                domain=domain,
                deduped_candidate_count=len(diary_candidates),
                recent_candidate_count=len(recent_diary_candidates),
                strict_recent_candidate_count=len(strict_recent_candidates),
                article_recent_candidate_count=len(article_recent_candidates),
                recent_urls=[c.get("url", "") for c in strict_recent_candidates],
            )
            if article_recent_candidates:
                diary_candidates = article_recent_candidates
            else:
                diary_candidates = [c for c in strict_recent_candidates if not _flow._is_invalid_news_candidate(c, last_message)]
        article_candidates = [c for c in diary_candidates if _is_specific_article_hit(c)]
        if article_candidates:
            raw_candidates.extend(article_candidates)
        else:
            raw_candidates.extend(diary_candidates)
            combined_domain_text = "\n".join(domain_search_texts)
            if not diary_candidates or combined_domain_text.startswith("Error en búsqueda:") or "No results found." in combined_domain_text:
                fallback_source = sources_by_domain.get(domain, {"title": press_name, "url": _flow._default_press_homepage_url(domain)})
                fallback_url = (fallback_source or {}).get("url", "").strip()
                if fallback_url:
                    homepage_prompt = _flow._build_newspaper_homepage_fetch_prompt(last_message, fallback_source.get("title") or domain)
                    try:
                        fetched_home = await fetch_web_page(url=fallback_url, prompt=homepage_prompt, use_dynamic=False)
                    except Exception:
                        fetched_home = ""
                    if not isinstance(fetched_home, str):
                        fetched_home = str(fetched_home)
                    homepage_lines = _flow._filter_homepage_lines_for_query(
                        _flow._extract_generic_content_lines(fetched_home, query_terms),
                        last_message,
                        query_terms,
                    )
                    homepage_lines = _flow._dedupe_homepage_lines(homepage_lines)
                    if not homepage_lines and dynamic_fetch_available is not False:
                        try:
                            _flow._web_debug("country_press.search.homepage_retry_dynamic", domain=domain, fallback_url=fallback_url)
                            fetched_home_dynamic = await fetch_web_page(url=fallback_url, prompt=homepage_prompt, use_dynamic=True)
                        except Exception:
                            fetched_home_dynamic = ""
                        if not isinstance(fetched_home_dynamic, str):
                            fetched_home_dynamic = str(fetched_home_dynamic)
                        dynamic_issue = _flow._classify_fetch_error(fetched_home_dynamic)
                        if dynamic_issue == "missing_playwright":
                            dynamic_fetch_available = False
                            _flow._web_debug("country_press.search.dynamic_unavailable", reason="missing_playwright", domain=domain)
                        homepage_lines = _flow._filter_homepage_lines_for_query(
                            _flow._extract_generic_content_lines(fetched_home_dynamic, query_terms),
                            last_message,
                            query_terms,
                        )
                        homepage_lines = _flow._dedupe_homepage_lines(homepage_lines)
                    section_candidates: list[dict[str, str]] = []
                    for section_url, section_label in _flow._build_country_press_section_targets(domain, fallback_url, last_message):
                        section_prompt = _flow._build_newspaper_section_fetch_prompt(
                            last_message,
                            fallback_source.get("title") or domain,
                            section_label,
                        )
                        _flow._web_debug(
                            "country_press.search.section_fetch_start",
                            domain=domain,
                            section_label=section_label,
                            section_url=section_url,
                        )
                        fetched_section = ""
                        dynamic_modes = (False, True) if dynamic_fetch_available is not False else (False,)
                        for use_dynamic in dynamic_modes:
                            try:
                                fetched_section = await fetch_web_page(
                                    url=section_url,
                                    prompt=section_prompt,
                                    use_dynamic=use_dynamic,
                                )
                            except Exception:
                                fetched_section = ""
                            if not isinstance(fetched_section, str):
                                fetched_section = str(fetched_section)
                            _flow._web_debug(
                                "country_press.search.section_fetch_result",
                                domain=domain,
                                section_label=section_label,
                                section_url=section_url,
                                use_dynamic=use_dynamic,
                                content_length=len(fetched_section or ""),
                                preview=(fetched_section or "")[:500],
                            )
                            fetch_issue = _flow._classify_fetch_error(fetched_section)
                            if fetch_issue == "missing_playwright":
                                dynamic_fetch_available = False
                                _flow._web_debug("country_press.search.dynamic_unavailable", reason="missing_playwright", domain=domain)
                            elif fetch_issue in {"not_found", "dns", "fetch_error"}:
                                _flow._web_debug(
                                    "country_press.search.section_unavailable",
                                    domain=domain,
                                    section_label=section_label,
                                    section_url=section_url,
                                    use_dynamic=use_dynamic,
                                    reason=fetch_issue,
                                )
                            elif fetch_issue == "blocked":
                                _flow._web_debug(
                                    "country_press.search.section_blocked",
                                    domain=domain,
                                    section_label=section_label,
                                    section_url=section_url,
                                    use_dynamic=use_dynamic,
                                    reason=fetch_issue,
                                )
                            extracted_section_lines = _flow._extract_section_content_lines(
                                fetched_section,
                                last_message,
                                section_label,
                            )
                            _flow._web_debug(
                                "country_press.search.section_lines_extracted",
                                domain=domain,
                                section_label=section_label,
                                section_url=section_url,
                                use_dynamic=use_dynamic,
                                raw_line_count=len(extracted_section_lines),
                                raw_lines=extracted_section_lines[:5],
                            )
                            section_lines = _flow._filter_section_lines_for_query(
                                extracted_section_lines,
                                last_message,
                                section_label,
                            )
                            section_lines = _flow._dedupe_homepage_lines(section_lines)
                            _flow._web_debug(
                                "country_press.search.section_lines_filtered",
                                domain=domain,
                                section_label=section_label,
                                section_url=section_url,
                                use_dynamic=use_dynamic,
                                filtered_count=len(section_lines),
                                filtered_lines=section_lines[:5],
                            )
                            if not fetched_section.strip():
                                _flow._web_debug(
                                    "country_press.search.section_empty",
                                    domain=domain,
                                    section_label=section_label,
                                    section_url=section_url,
                                    use_dynamic=use_dynamic,
                                    reason="empty_fetch",
                                )
                            elif _flow._is_no_info_response(fetched_section):
                                _flow._web_debug(
                                    "country_press.search.section_empty",
                                    domain=domain,
                                    section_label=section_label,
                                    section_url=section_url,
                                    use_dynamic=use_dynamic,
                                    reason="no_info_response",
                                )
                            if any(_flow._is_homepage_meta_line(line) for line in section_lines):
                                _flow._web_debug(
                                    "country_press.search.section_rejected_meta",
                                    domain=domain,
                                    section_label=section_label,
                                    section_url=section_url,
                                    use_dynamic=use_dynamic,
                                    section_lines=section_lines[:5],
                                )
                                section_lines = []
                            else:
                                section_lines = [line for line in section_lines if _flow._is_concrete_homepage_line(line)]
                                if extracted_section_lines and not section_lines:
                                    _flow._web_debug(
                                        "country_press.search.section_rejected_non_concrete",
                                        domain=domain,
                                        section_label=section_label,
                                        section_url=section_url,
                                        use_dynamic=use_dynamic,
                                        raw_lines=extracted_section_lines[:5],
                                    )
                            if section_lines:
                                _flow._web_debug(
                                    "country_press.search.section_fallback",
                                    domain=domain,
                                    section_label=section_label,
                                    section_url=section_url,
                                    use_dynamic=use_dynamic,
                                    section_lines=section_lines[:5],
                                )
                                section_candidates.append({
                                    "title": f"{fallback_source.get('title') or domain} — {section_label}",
                                    "url": section_url,
                                    "snippet": " ".join(section_lines[:3]),
                                    "source_kind": "section_fallback",
                                })
                                break
                        if len(section_candidates) >= 2:
                            break
                    if section_candidates:
                        raw_candidates.extend(section_candidates)
                    elif homepage_lines:
                        if any(_flow._is_homepage_meta_line(line) for line in homepage_lines):
                            _flow._web_debug(
                                "country_press.search.homepage_rejected_meta",
                                domain=domain,
                                fallback_url=fallback_url,
                                homepage_lines=homepage_lines[:5],
                            )
                            homepage_lines = []
                        else:
                            homepage_lines = [line for line in homepage_lines if _flow._is_concrete_homepage_line(line)]
                            if not homepage_lines:
                                _flow._web_debug(
                                    "country_press.search.homepage_rejected_non_concrete",
                                    domain=domain,
                                    fallback_url=fallback_url,
                                )
                    if not section_candidates and homepage_lines:
                        _flow._web_debug(
                            "country_press.search.homepage_fallback",
                            domain=domain,
                            fallback_url=fallback_url,
                            homepage_lines=homepage_lines[:5],
                        )
                        raw_candidates.append({
                            "title": fallback_source.get("title") or domain,
                            "url": fallback_url,
                            "snippet": " ".join(homepage_lines[:3]),
                            "source_kind": "homepage_fallback",
                        })

    ranked_candidates = _flow._rank_candidates_by_source_policy(raw_candidates, query_terms, query_source_group)
    ranked_candidates = [
        c for c in ranked_candidates
        if (
            c.get("source_kind") in {"homepage_fallback", "section_fallback"}
            or not _flow._is_invalid_news_candidate(c, last_message)
        )
    ]
    diverse_candidates = _flow._dedup_candidates_by_event(ranked_candidates, query_terms)[:8]
    _flow._web_debug(
        "country_press.search.final",
        raw_candidate_count=len(raw_candidates),
        ranked_candidate_count=len(ranked_candidates),
        diverse_candidate_count=len(diverse_candidates),
        diverse_urls=[c.get("url", "") for c in diverse_candidates],
    )
    combined_search_entries: list[str] = []
    seen_combined_entries: set[str] = set()
    for entry in combined_search_text:
        normalized_entry = (entry or "").strip()
        if not normalized_entry or normalized_entry in seen_combined_entries:
            continue
        seen_combined_entries.add(normalized_entry)
        combined_search_entries.append(normalized_entry)
    return diverse_candidates, "\n".join(combined_search_entries)


__all__ = ["_run_country_press_search_candidates"]
