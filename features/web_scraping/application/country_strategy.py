"""Estrategia de noticias locales recientes por país."""
from __future__ import annotations

import asyncio
import re
from time import perf_counter
from typing import TYPE_CHECKING, Any, Optional

from application.policies.candidate_scoring import (
    _build_relevance_context,
    _is_relevant_candidate_for_query,
    _score_candidate_relevance,
)
from core.helpers.url_helpers import _safe_hostname
from features.web_scraping.infrastructure.runtime import WebFetchRuntime, WebSearchRuntime

if TYPE_CHECKING:
    from core.ports.country_news_ports import (
        ICountryProfileRepository,
        ICountryResolver,
        IDynamicPressSourceDiscovery,
        IPressSourceDiscovery,
        ISectionPathResolver,
    )


class CountryRecentNewsStrategy:
    """Estrategia principal para noticias locales recientes basadas en secciones."""

    _SECTION_FETCH_WORKERS = 2
    _ARTICLE_FETCH_WORKERS = 3
    _MIN_FINAL_NEWS = 4
    _MAX_ARTICLE_FETCH_CANDIDATES = 4

    def __init__(
        self,
        *,
        search_runtime: WebSearchRuntime,
        fetch_runtime: WebFetchRuntime,
        country_resolver: Optional["ICountryResolver"] = None,
        profile_repo: Optional["ICountryProfileRepository"] = None,
        section_path_resolver: Optional["ISectionPathResolver"] = None,
        press_discovery: Optional["IPressSourceDiscovery"] = None,
        dynamic_discovery: Optional["IDynamicPressSourceDiscovery"] = None,
    ) -> None:
        from features.web_scraping.infrastructure.country_news_adapters import (
            DefaultCountryResolver,
            DefaultCountryProfileRepository,
            DefaultSectionPathResolver,
            DefaultPressSourceDiscovery,
        )
        from features.web_scraping.infrastructure.dynamic_press_discovery import DefaultDynamicPressDiscovery

        self._search_runtime = search_runtime
        self._fetch_runtime = fetch_runtime
        self._country_resolver = country_resolver or DefaultCountryResolver()
        self._profile_repo = profile_repo or DefaultCountryProfileRepository()
        self._section_path_resolver = section_path_resolver or DefaultSectionPathResolver()
        self._press_discovery = press_discovery or DefaultPressSourceDiscovery()
        self._dynamic_discovery = dynamic_discovery or DefaultDynamicPressDiscovery()

    async def execute(self, last_message: str, web_search_runtime_args: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
        from features.web_scraping.application import flow as _flow
        from features.web_scraping.application.country_press_helpers import _run_country_press_search_candidates
        from features.web_scraping.domain.classifier import _is_specific_article_hit

        started_at = perf_counter()
        query_source_group = _flow.detect_query_source_group(last_message)
        query_horizon = _flow.detect_recent_query_horizon(last_message) if _flow._is_recent_web_information_query(last_message) else None
        use_bootstrap = _flow._should_use_country_recent_news_strategy(last_message, query_source_group, query_horizon)

        if query_source_group == "japan":
            return None

        if use_bootstrap:
            source_terms = list(_flow.get_query_source_terms(last_message))
            country_press_domains, country_press_names = await self._press_discovery.discover(last_message, query_source_group, source_terms, web_search_runtime_args)
            if not country_press_domains:
                _flow._emit_country_news_metrics(geography=query_source_group, resolution_path="none", domains_found=0)
                return None
            _flow._emit_country_news_metrics(geography=query_source_group, resolution_path="bootstrap", domains_found=len(country_press_domains))
        elif query_source_group is None:
            inferred_geo = self._country_resolver.resolve(last_message)
            lowered_msg = (last_message or "").lower()
            _has_news = any(t in lowered_msg for t in ("noticia", "noticias", "news", "headline", "headlines"))
            _has_topic = _flow._detect_news_topic(last_message) in {"security", "economy", "politics"}
            _valid_horizon = query_horizon in {"today", "week", "month"}
            if not (inferred_geo and _valid_horizon and (_has_news or _has_topic)):
                return None
            _flow._web_debug("country_press.dynamic.attempt", geography=inferred_geo, horizon=query_horizon, reason="source_group_missing")
            country_press_domains, country_press_names = await self._dynamic_discovery.discover_for_unknown_country(last_message, inferred_geo, web_search_runtime_args)
            if not country_press_domains:
                _flow._web_debug("country_press.dynamic.no_sources", geography=inferred_geo)
                _flow._emit_country_news_metrics(geography=inferred_geo, resolution_path="none", domains_found=0)
                return None
            query_source_group = f"dynamic:{inferred_geo.lower()}"
            source_terms = [inferred_geo.lower()]
            _flow._country_press_cache_set(query_source_group, source_terms, country_press_domains, country_press_names)
            _flow._country_press_strategy_cache_set(query_source_group, source_terms, "dynamic")
            _flow._web_debug("country_press.dynamic.success", geography=inferred_geo, domains=country_press_domains)
            _flow._emit_country_news_metrics(geography=inferred_geo, resolution_path="dynamic", domains_found=len(country_press_domains))
        else:
            return None

        query_terms = _flow._extract_generic_query_terms(last_message)
        for term in source_terms:
            if term not in query_terms:
                query_terms.append(term)

        country_press_sources = _flow._country_press_source_cache_get(query_source_group, source_terms)
        discovery_strategy = _flow._country_press_strategy_cache_get(query_source_group, source_terms)
        if discovery_strategy == "none" and not country_press_sources:
            return None
        sources_by_domain: dict[str, dict[str, str]] = {}
        for source in country_press_sources:
            url = source.get("url", "")
            hostname = _safe_hostname(url).lower().removeprefix("www.")
            if hostname and hostname not in sources_by_domain:
                sources_by_domain[hostname] = source

        structured_candidates: list[Any] = []
        seen_urls: set[str] = set()
        dynamic_fetch_available = True
        sec_topic = _flow._detect_news_topic(last_message)
        topic_terms_for_filter: dict[str, set[str]] = {
            "security": {"seguridad", "sicurezza", "crime", "crimen", "cronaca", "polizia", "policia", "policiales", "detenid", "arrestad", "operativo", "homicidio", "asesin", "robo", "narco", "violencia", "sucesos", "delito", "fiscal", "tribunal", "ertzaintza", "mossos", "guardia civil", "omicid", "arrest", "blitz"},
            "economy": {"econom", "mercad", "mercato", "finanz", "inflac", "presupuesto", "negocios", "empresa", "bolsa", "pib", "deuda"},
            "politics": {"politic", "govern", "parlament", "elecci", "presidente", "ministro", "congreso", "senado", "partido", "decreto"},
        }
        filter_terms_for_section = topic_terms_for_filter.get(sec_topic, set())

        def _line_looks_concrete(text: str) -> bool:
            normalized = re.sub(r"\s+", " ", (text or "").strip())
            if len(normalized.split()) < 6:
                return False
            if _flow._is_no_info_response(normalized) or _flow._is_homepage_meta_line(normalized):
                return False
            return _flow._is_concrete_homepage_line(normalized) or any(mark in normalized for mark in (".", ":", "—"))

        def _is_invalid_section_payload(text: str) -> bool:
            normalized = _flow._strip_accents((text or "").lower()).strip()
            if not normalized:
                return True
            if normalized.startswith(("url:", "sources:", "http", "<<<cite_this:")):
                return True
            if "<<<cite_this:" in normalized:
                return True
            if "aspxerrorpath=" in normalized:
                return True
            if any(token in normalized for token in ("|url=", "|domain=", "title=")):
                return True
            return False

        def _candidate_alternative_urls(candidate_url: str) -> list[str]:
            urls: list[str] = []
            if "://www." in candidate_url:
                urls.append(candidate_url.replace("://www.", "://m.", 1))
            if "/news/view" in candidate_url.lower() and "m." not in candidate_url:
                urls.append(candidate_url.replace("://www.", "://m.", 1))
            if "?" not in candidate_url:
                urls.append(f"{candidate_url}?output=amp")
            return [url for url in urls if url != candidate_url]

        def _snippet_is_render_ready(snippet: str) -> bool:
            stripped = (snippet or "").strip()
            if not stripped:
                return False
            non_latin_chars = re.findall(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", stripped)
            if not non_latin_chars:
                return True
            latin_chars = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñÜü]", stripped)
            return len(non_latin_chars) <= 12 and len(non_latin_chars) <= max(4, len(latin_chars) // 8)

        def _safe_qsize(queue: asyncio.Queue[Any]) -> int:
            try:
                return queue.qsize()
            except NotImplementedError:
                return -1

        def _is_concrete_renderable_candidate(candidate: Any) -> bool:
            title = getattr(candidate, "title", "") or ""
            snippet = getattr(candidate, "snippet", "") or ""
            url = getattr(candidate, "url", "") or ""
            blob = f"{title}\n{snippet}"
            if _flow._is_redirect_payload(blob):
                return False
            if _flow._is_prompt_echo_line(title) or _flow._is_prompt_echo_line(snippet):
                return False
            if _flow._is_no_info_response(blob):
                return False
            if len(snippet.strip()) < 40:
                return False
            if not url:
                return False
            return True

        relevance_context = _build_relevance_context(last_message, query_source_group)

        def _is_relevant_concrete_candidate(candidate: Any) -> bool:
            if not _is_concrete_renderable_candidate(candidate):
                return False
            return _is_relevant_candidate_for_query(candidate.as_candidate(), last_message, query_source_group)

        async def _build_section_candidates() -> list[Any]:
            stage_started_at = perf_counter()
            section_candidates: list[Any] = []
            local_seen_urls: set[str] = set()
            local_dynamic_fetch_available = True
            seen_urls_lock = asyncio.Lock()
            for idx, domain in enumerate(country_press_domains):
                if (perf_counter() - stage_started_at) >= _flow._SECTION_DISCOVERY_TIME_BUDGET_SECONDS:
                    _flow._web_debug(
                        "country_strategy.section_budget_exhausted",
                        elapsed_ms=round((perf_counter() - stage_started_at) * 1000, 1),
                        processed_domains=idx,
                        remaining_domains=country_press_domains[idx:],
                    )
                    break
                domain_started_at = perf_counter()
                press_name = country_press_names[idx] if idx < len(country_press_names) else domain
                source_meta = sources_by_domain.get(domain, {"title": press_name, "url": _flow._default_press_homepage_url(domain)})
                if not _flow._is_press_source_relevant_for_query(source_meta, last_message):
                    _flow._web_debug(
                        "country_strategy.domain_skipped",
                        domain=domain,
                        press_name=press_name,
                        reason="irrelevant_for_query",
                    )
                    continue
                fallback_url = (source_meta.get("url") or "").strip()
                if not fallback_url:
                    _flow._web_debug(
                        "country_strategy.domain_skipped",
                        domain=domain,
                        press_name=press_name,
                        reason="missing_fallback_url",
                    )
                    continue
                section_targets, local_dynamic_fetch_available = await _flow._discover_homepage_section_targets(
                    domain=domain,
                    fallback_url=fallback_url,
                    last_message=last_message,
                    press_name=source_meta.get("title") or press_name,
                    dynamic_fetch_available=local_dynamic_fetch_available,
                )
                domain_candidate_count = 0

                async def _process_section_task(
                    *,
                    worker_id: int,
                    task_idx: int,
                    section_url: str,
                    section_label: str,
                ) -> tuple[list[Any], bool]:
                    section_started_at = perf_counter()
                    section_prompt = _flow._build_newspaper_section_fetch_prompt(last_message, source_meta.get("title") or press_name, section_label)
                    _flow._web_debug(
                        "country_strategy.section_task_started",
                        domain=domain,
                        press_name=press_name,
                        worker_id=worker_id,
                        task_idx=task_idx,
                        section_label=section_label,
                        section_url=section_url,
                    )
                    try:
                        fetch_response = await self._fetch_runtime.fetch(_flow.WebFetchRequest(url=section_url, prompt=section_prompt, mode="static", use_cache=False))
                    except Exception:
                        _flow._web_debug(
                            "country_strategy.section_task_completed",
                            domain=domain,
                            press_name=press_name,
                            worker_id=worker_id,
                            task_idx=task_idx,
                            section_label=section_label,
                            section_url=section_url,
                            status="exception",
                            produced_candidates=0,
                            elapsed_ms=round((perf_counter() - section_started_at) * 1000, 1),
                        )
                        return [], False
                    section_text = fetch_response.content
                    redirect_url = _flow._extract_web_fetch_redirect_url(section_text)
                    if redirect_url and _flow._is_same_site_redirect(section_url, redirect_url):
                        try:
                            fetch_response = await self._fetch_runtime.fetch(
                                _flow.WebFetchRequest(url=redirect_url, prompt=section_prompt, mode="static", use_cache=False)
                            )
                            section_text = fetch_response.content
                            _flow._web_debug(
                                "country_strategy.section_redirect_followed",
                                domain=domain,
                                section_label=section_label,
                                original_url=section_url,
                                redirect_url=redirect_url,
                            )
                        except Exception:
                            pass
                    issue = _flow._classify_fetch_error(section_text)
                    missing_playwright = issue == "missing_playwright"
                    if issue in {"not_found", "blocked", "dns", "fetch_error"}:
                        _flow._web_debug(
                            "country_strategy.section_fetch_rejected",
                            domain=domain,
                            section_label=section_label,
                            section_url=section_url,
                            reason=issue,
                            elapsed_ms=round((perf_counter() - section_started_at) * 1000, 1),
                        )
                        _flow._web_debug(
                            "country_strategy.section_task_completed",
                            domain=domain,
                            press_name=press_name,
                            worker_id=worker_id,
                            task_idx=task_idx,
                            section_label=section_label,
                            section_url=section_url,
                            status="rejected",
                            produced_candidates=0,
                            elapsed_ms=round((perf_counter() - section_started_at) * 1000, 1),
                        )
                        return [], missing_playwright
                    lines = _flow._filter_section_lines_for_query(_flow._extract_section_content_lines(section_text, last_message, section_label), last_message, section_label)
                    lines = _flow._dedupe_homepage_lines(lines)
                    fallback_lines = [__import__("re").sub(r"^\s*(?:[-*•]\s+|\d+\.\s+)", "", line).strip() for line in (section_text or "").splitlines() if line.strip()]
                    fallback_lines = [
                        line for line in fallback_lines
                        if len(line) > 20
                        and not _flow._is_homepage_meta_line(line)
                        and not _flow._is_no_info_response(line)
                        and not _flow._is_redirect_payload(line)
                        and not _flow._is_prompt_echo_line(line)
                        and not _is_invalid_section_payload(line)
                    ]
                    _flow._web_debug(
                        "country_strategy.section_lines",
                        domain=domain,
                        section_label=section_label,
                        section_url=section_url,
                        filtered_count=len(lines),
                        fallback_count=len(fallback_lines),
                        filtered_preview=lines[:3],
                        fallback_preview=fallback_lines[:3],
                    )
                    if not lines:
                        lines = _flow._dedupe_homepage_lines(fallback_lines)
                    if not lines and not fallback_lines and local_dynamic_fetch_available and not (section_text or "").strip():
                        try:
                            dynamic_response = await self._fetch_runtime.fetch(_flow.WebFetchRequest(url=section_url, prompt=section_prompt, mode="dynamic", use_cache=False))
                        except Exception:
                            _flow._web_debug(
                                "country_strategy.section_task_completed",
                                domain=domain,
                                press_name=press_name,
                                worker_id=worker_id,
                                task_idx=task_idx,
                                section_label=section_label,
                                section_url=section_url,
                                status="dynamic_exception",
                                produced_candidates=0,
                                elapsed_ms=round((perf_counter() - section_started_at) * 1000, 1),
                            )
                            return [], missing_playwright
                        dynamic_issue = _flow._classify_fetch_error(dynamic_response.content)
                        if dynamic_issue == "missing_playwright":
                            missing_playwright = True
                        if dynamic_issue not in {"not_found", "blocked", "dns", "fetch_error", "missing_playwright"}:
                            lines = _flow._filter_section_lines_for_query(_flow._extract_section_content_lines(dynamic_response.content, last_message, section_label), last_message, section_label)
                            lines = _flow._dedupe_homepage_lines(lines)
                            section_text = dynamic_response.content
                            _flow._web_debug(
                                "country_strategy.section_lines_dynamic",
                                domain=domain,
                                section_label=section_label,
                                section_url=section_url,
                                dynamic_issue=dynamic_issue,
                                filtered_count=len(lines),
                                filtered_preview=lines[:3],
                            )
                    if not lines:
                        _flow._web_debug(
                            "country_strategy.section_skipped",
                            domain=domain,
                            section_label=section_label,
                            section_url=section_url,
                            reason="no_lines_after_filter",
                            elapsed_ms=round((perf_counter() - section_started_at) * 1000, 1),
                        )
                        _flow._web_debug(
                            "country_strategy.section_task_completed",
                            domain=domain,
                            press_name=press_name,
                            worker_id=worker_id,
                            task_idx=task_idx,
                            section_label=section_label,
                            section_url=section_url,
                            status="empty",
                            produced_candidates=0,
                            elapsed_ms=round((perf_counter() - section_started_at) * 1000, 1),
                        )
                        return [], missing_playwright
                    raw_blocks = [" ".join(ln.strip() for ln in block.splitlines() if ln.strip()) for block in (section_text or "").split("\n\n") if block.strip()]
                    section_items: list[str] = []
                    seen_block_prefixes: set[str] = set()
                    for block_text in raw_blocks:
                        if (
                            len(block_text) < 20
                            or _flow._is_no_info_response(block_text)
                            or _flow._is_redirect_payload(block_text)
                            or _flow._is_prompt_echo_line(block_text)
                            or _is_invalid_section_payload(block_text)
                        ):
                            continue
                        block_norm = _flow._strip_accents(block_text.lower())
                        if not filter_terms_for_section or any(term in block_norm for term in filter_terms_for_section):
                            prefix = " ".join(block_norm.split()[:4])
                            if prefix and prefix not in seen_block_prefixes:
                                seen_block_prefixes.add(prefix)
                                section_items.append(block_text)
                    if not section_items:
                        section_items = [
                            ln for ln in lines
                            if len(ln) > 20
                            and not _flow._is_redirect_payload(ln)
                            and not _flow._is_prompt_echo_line(ln)
                            and not _is_invalid_section_payload(ln)
                        ]
                    _flow._web_debug(
                        "country_strategy.section_items",
                        domain=domain,
                        section_label=section_label,
                        section_url=section_url,
                        item_count=len(section_items),
                        item_preview=section_items[:2],
                    )
                    if not section_items:
                        _flow._web_debug(
                            "country_strategy.section_skipped",
                            domain=domain,
                            section_label=section_label,
                            section_url=section_url,
                            reason="invalid_or_placeholder_content",
                            elapsed_ms=round((perf_counter() - section_started_at) * 1000, 1),
                        )
                        _flow._web_debug(
                            "country_strategy.section_task_completed",
                            domain=domain,
                            press_name=press_name,
                            worker_id=worker_id,
                            task_idx=task_idx,
                            section_label=section_label,
                            section_url=section_url,
                            status="invalid",
                            produced_candidates=0,
                            elapsed_ms=round((perf_counter() - section_started_at) * 1000, 1),
                        )
                        return [], missing_playwright
                    built_section_candidates: list[Any] = []
                    for item_idx, item_text in enumerate(section_items):
                        candidate_url = section_url if item_idx == 0 else f"{section_url}#n{item_idx}"
                        async with seen_urls_lock:
                            if candidate_url in local_seen_urls:
                                continue
                            local_seen_urls.add(candidate_url)
                        built_section_candidates.append(_flow.WebCandidate(title=f"{source_meta.get('title') or press_name} — {section_label}", url=candidate_url, snippet=item_text, source_kind=_flow.SourceKind.SECTION, evidence_kind=_flow.EvidenceKind.SECTION_LINES, recency=_flow.Recency.DATED_RECENT, specificity=_flow.Specificity.CONCRETE, source_label=section_label))
                        _flow._web_debug(
                            "country_strategy.candidate_added",
                            domain=domain,
                            section_label=section_label,
                            candidate_url=candidate_url,
                            snippet_preview=item_text[:220],
                            elapsed_ms=round((perf_counter() - section_started_at) * 1000, 1),
                        )
                    _flow._web_debug(
                        "country_strategy.section_task_completed",
                        domain=domain,
                        press_name=press_name,
                        worker_id=worker_id,
                        task_idx=task_idx,
                        section_label=section_label,
                        section_url=section_url,
                        status="ok",
                        produced_candidates=len(built_section_candidates),
                        elapsed_ms=round((perf_counter() - section_started_at) * 1000, 1),
                    )
                    return built_section_candidates, missing_playwright

                if section_targets:
                    section_queue: asyncio.Queue[tuple[int, str, str] | None] = asyncio.Queue()
                    for task_idx, (section_url, section_label) in enumerate(section_targets, start=1):
                        section_queue.put_nowait((task_idx, section_url, section_label))
                    worker_count = min(self._SECTION_FETCH_WORKERS, len(section_targets))
                    for _ in range(worker_count):
                        section_queue.put_nowait(None)

                    async def _section_worker(worker_id: int) -> tuple[list[Any], int, bool]:
                        local_candidates: list[Any] = []
                        produced_count = 0
                        missing_playwright_seen = False
                        while True:
                            job = await section_queue.get()
                            if job is None:
                                section_queue.task_done()
                                break
                            task_idx, section_url, section_label = job
                            built, missing_playwright = await _process_section_task(
                                worker_id=worker_id,
                                task_idx=task_idx,
                                section_url=section_url,
                                section_label=section_label,
                            )
                            if missing_playwright:
                                missing_playwright_seen = True
                            local_candidates.extend(built)
                            produced_count += len(built)
                            _flow._web_debug(
                                "country_strategy.section_worker_progress",
                                domain=domain,
                                press_name=press_name,
                                worker_id=worker_id,
                                task_idx=task_idx,
                                pending_tasks=_safe_qsize(section_queue),
                                produced_candidates=produced_count,
                            )
                            section_queue.task_done()
                        return local_candidates, produced_count, missing_playwright_seen

                    worker_results = await asyncio.gather(*[
                        _section_worker(worker_id)
                        for worker_id in range(1, worker_count + 1)
                    ])
                    for built, produced_count, missing_playwright_seen in worker_results:
                        if built:
                            section_candidates.extend(built)
                        domain_candidate_count += produced_count
                        if missing_playwright_seen:
                            local_dynamic_fetch_available = False
                concrete_candidates = [candidate for candidate in section_candidates if _is_concrete_renderable_candidate(candidate)]
                relevant_concrete_candidates = [candidate for candidate in concrete_candidates if _is_relevant_candidate_for_query(candidate.as_candidate(), last_message, query_source_group)]
                _flow._web_debug(
                    "country_strategy.domain_section_summary",
                    domain=domain,
                    press_name=press_name,
                    section_labels=[label for _, label in section_targets],
                    section_target_count=len(section_targets),
                    candidate_count=domain_candidate_count,
                    concrete_candidate_count=len([candidate for candidate in concrete_candidates if getattr(candidate, "url", "").startswith(f"https://{domain}") or getattr(candidate, "url", "").startswith(f"http://{domain}") or domain in getattr(candidate, "url", "")]),
                    relevant_candidate_count=len([candidate for candidate in relevant_concrete_candidates if getattr(candidate, "url", "").startswith(f"https://{domain}") or getattr(candidate, "url", "").startswith(f"http://{domain}") or domain in getattr(candidate, "url", "")]),
                    elapsed_ms=round((perf_counter() - domain_started_at) * 1000, 1),
                )
            concrete_candidates = [candidate for candidate in section_candidates if _is_concrete_renderable_candidate(candidate)]
            relevant_concrete_candidates = [candidate for candidate in concrete_candidates if _is_relevant_candidate_for_query(candidate.as_candidate(), last_message, query_source_group)]
            rejected_redirect = len([candidate for candidate in section_candidates if _flow._is_redirect_payload(f"{candidate.title}\n{candidate.snippet}")])
            rejected_prompt_echo = len([candidate for candidate in section_candidates if _flow._is_prompt_echo_line(candidate.title) or _flow._is_prompt_echo_line(candidate.snippet)])
            rejected_short_snippet = len([candidate for candidate in section_candidates if len((candidate.snippet or "").strip()) < 40])
            rejected_low_relevance = len(concrete_candidates) - len(relevant_concrete_candidates)
            _flow._web_debug(
                "country_strategy.section_candidate_quality",
                total=len(section_candidates),
                concrete=len(concrete_candidates),
                relevant=len(relevant_concrete_candidates),
                rejected_redirect=rejected_redirect,
                rejected_prompt_echo=rejected_prompt_echo,
                rejected_short_snippet=rejected_short_snippet,
                rejected_low_relevance=rejected_low_relevance,
                intent_subtype=relevance_context.intent_subtype,
                min_relevance_score=relevance_context.min_relevance_score,
            )
            _flow._web_debug(
                "country_strategy.section_candidates_ready",
                query=last_message,
                query_source_group=query_source_group,
                candidate_count=len(section_candidates),
                candidate_urls=[candidate.url for candidate in section_candidates[:8]],
                candidate_titles=[candidate.title for candidate in section_candidates[:8]],
                elapsed_ms=round((perf_counter() - stage_started_at) * 1000, 1),
            )
            return section_candidates

        async def _build_article_candidates() -> list[Any]:
            stage_started_at = perf_counter()
            search_age_days = 14 if query_horizon in {"today", "week"} else 30 if query_horizon == "month" else None
            search_candidates, _combined_search = await _run_country_press_search_candidates(
                last_message,
                search_age_days,
                query_terms,
                query_source_group,
                source_terms,
                web_search_runtime_args,
                query_horizon=query_horizon,
            )
            article_candidates = [
                candidate for candidate in search_candidates
                if _is_specific_article_hit(candidate)
            ]
            _flow._web_debug(
                "country_strategy.article_search_candidates",
                query=last_message,
                query_source_group=query_source_group,
                total_candidate_count=len(search_candidates),
                article_candidate_count=len(article_candidates),
                article_urls=[candidate.get("url", "") for candidate in article_candidates[: self._MAX_ARTICLE_FETCH_CANDIDATES]],
            )
            if not article_candidates:
                return []

            fetch_prompt = _flow._build_generic_fetch_prompt(last_message)
            built_candidates: list[Any] = []
            seen_article_urls: set[str] = set()
            max_fetch_attempts_per_target = 2
            built_candidates_lock = asyncio.Lock()
            stats_lock = asyncio.Lock()
            stop_article_workers = asyncio.Event()
            stats = {
                "searched": len(article_candidates[: self._MAX_ARTICLE_FETCH_CANDIDATES]),
                "diverse": len(search_candidates[:8]),
                "fetch_ok": 0,
                "fetch_error": 0,
                "empty_content": 0,
                "final": 0,
            }

            async def _process_article_candidate(
                *,
                worker_id: int,
                task_idx: int,
                candidate: dict[str, Any],
            ) -> None:
                candidate_url = (candidate.get("url") or "").strip()
                if not candidate_url:
                    return
                article_text = ""
                article_issue = ""
                article_lines: list[str] = []
                article_started_at = perf_counter()
                fetch_targets = [(candidate_url, "static")]
                fetch_targets.extend((alt_url, "static") for alt_url in _candidate_alternative_urls(candidate_url))
                fetch_targets.append((candidate_url, "dynamic"))
                _flow._web_debug(
                    "country_strategy.article_task_started",
                    worker_id=worker_id,
                    task_idx=task_idx,
                    candidate_url=candidate_url,
                    title=candidate.get("title", ""),
                )
                for fetch_url, fetch_mode in fetch_targets:
                    for attempt in range(1, max_fetch_attempts_per_target + 1):
                        try:
                            article_response = await self._fetch_runtime.fetch(
                                _flow.WebFetchRequest(
                                    url=fetch_url,
                                    prompt=fetch_prompt,
                                    mode=fetch_mode,
                                    use_cache=False,
                                )
                            )
                        except Exception as exc:
                            article_issue = "fetch_error"
                            _flow._web_debug(
                                "country_strategy.article_fetch_exception",
                                candidate_url=candidate_url,
                                fetch_url=fetch_url,
                                fetch_mode=fetch_mode,
                                title=candidate.get("title", ""),
                                attempt=attempt,
                                max_attempts=max_fetch_attempts_per_target,
                                error=repr(exc),
                            )
                            if attempt < max_fetch_attempts_per_target:
                                _flow._web_debug(
                                    "country_strategy.article_fetch_retry",
                                    candidate_url=candidate_url,
                                    fetch_url=fetch_url,
                                    fetch_mode=fetch_mode,
                                    title=candidate.get("title", ""),
                                    next_attempt=attempt + 1,
                                )
                                continue
                            break
                        article_text = article_response.content
                        article_issue = _flow._classify_fetch_error(article_text)
                        if article_issue in {"not_found", "blocked", "dns", "fetch_error", "missing_playwright"}:
                            _flow._web_debug(
                                "country_strategy.article_fetch_rejected",
                                candidate_url=candidate_url,
                                fetch_url=fetch_url,
                                fetch_mode=fetch_mode,
                                title=candidate.get("title", ""),
                                attempt=attempt,
                                reason=article_issue,
                            )
                            if article_issue == "fetch_error" and attempt < max_fetch_attempts_per_target:
                                _flow._web_debug(
                                    "country_strategy.article_fetch_retry",
                                    candidate_url=candidate_url,
                                    fetch_url=fetch_url,
                                    fetch_mode=fetch_mode,
                                    title=candidate.get("title", ""),
                                    next_attempt=attempt + 1,
                                )
                                continue
                            break
                        article_lines = _flow._extract_generic_content_lines(article_text, query_terms)
                        article_lines = _flow._dedupe_homepage_lines(article_lines)
                        article_lines = [line for line in article_lines if _line_looks_concrete(line)]
                        if article_lines:
                            break
                    if article_lines:
                        break
                _flow._web_debug(
                    "country_strategy.article_fetch_lines",
                    candidate_url=candidate_url,
                    title=candidate.get("title", ""),
                    line_count=len(article_lines),
                    line_preview=article_lines[:3],
                )
                if not article_lines:
                    async with stats_lock:
                        if article_issue:
                            stats["fetch_error"] += 1
                        else:
                            stats["empty_content"] += 1
                    _flow._web_debug(
                        "country_strategy.article_task_completed",
                        worker_id=worker_id,
                        task_idx=task_idx,
                        candidate_url=candidate_url,
                        title=candidate.get("title", ""),
                        status="empty",
                        produced_candidates=0,
                        elapsed_ms=round((perf_counter() - article_started_at) * 1000, 1),
                    )
                    return
                snippet = " ".join(article_lines[:3]).strip()
                if not snippet:
                    async with stats_lock:
                        stats["empty_content"] += 1
                    _flow._web_debug(
                        "country_strategy.article_task_completed",
                        worker_id=worker_id,
                        task_idx=task_idx,
                        candidate_url=candidate_url,
                        title=candidate.get("title", ""),
                        status="blank_snippet",
                        produced_candidates=0,
                        elapsed_ms=round((perf_counter() - article_started_at) * 1000, 1),
                    )
                    return
                async with built_candidates_lock:
                    if len(built_candidates) < 4:
                        built_candidates.append(
                            _flow.WebCandidate(
                                title=(candidate.get("title") or candidate_url).strip(),
                                url=candidate_url,
                                snippet=snippet,
                                source_kind=_flow.SourceKind.ARTICLE,
                                evidence_kind=_flow.EvidenceKind.FETCHED_ARTICLE,
                                recency=_flow.Recency.DATED_RECENT if _flow._candidate_url_has_date(candidate_url) else _flow.Recency.UNDATED,
                                specificity=_flow.Specificity.CONCRETE,
                                source_label="article",
                            )
                        )
                        should_stop = len(built_candidates) >= 4
                    else:
                        should_stop = True
                async with stats_lock:
                    stats["fetch_ok"] += 1
                _flow._web_debug(
                    "country_strategy.article_candidate_added",
                    candidate_url=candidate_url,
                    title=candidate.get("title", ""),
                    snippet_preview=snippet[:220],
                )
                _flow._web_debug(
                    "country_strategy.article_task_completed",
                    worker_id=worker_id,
                    task_idx=task_idx,
                    candidate_url=candidate_url,
                    title=candidate.get("title", ""),
                    status="ok",
                    produced_candidates=1,
                    elapsed_ms=round((perf_counter() - article_started_at) * 1000, 1),
                )
                if should_stop:
                    stop_article_workers.set()

            article_queue: asyncio.Queue[tuple[int, dict[str, Any]] | None] = asyncio.Queue()
            for task_idx, candidate in enumerate(article_candidates[: self._MAX_ARTICLE_FETCH_CANDIDATES], start=1):
                candidate_url = (candidate.get("url") or "").strip()
                if not candidate_url or candidate_url in seen_article_urls:
                    continue
                seen_article_urls.add(candidate_url)
                article_queue.put_nowait((task_idx, candidate))
            worker_count = min(self._ARTICLE_FETCH_WORKERS, article_queue.qsize())
            if worker_count == 0:
                return []
            for _ in range(worker_count):
                article_queue.put_nowait(None)

            async def _article_worker(worker_id: int) -> None:
                while True:
                    job = await article_queue.get()
                    if job is None:
                        article_queue.task_done()
                        break
                    task_idx, candidate = job
                    if stop_article_workers.is_set():
                        article_queue.task_done()
                        continue
                    await _process_article_candidate(
                        worker_id=worker_id,
                        task_idx=task_idx,
                        candidate=candidate,
                    )
                    _flow._web_debug(
                        "country_strategy.article_worker_progress",
                        worker_id=worker_id,
                        task_idx=task_idx,
                        pending_tasks=_safe_qsize(article_queue),
                        built_candidates=len(built_candidates),
                        stop_requested=stop_article_workers.is_set(),
                    )
                    article_queue.task_done()

            await asyncio.gather(*[
                _article_worker(worker_id)
                for worker_id in range(1, worker_count + 1)
            ])
            stats["final"] = len(built_candidates)
            _flow._web_debug(
                "country_strategy.article_candidate_stats",
                elapsed_ms=round((perf_counter() - stage_started_at) * 1000, 1),
                **stats,
            )
            return built_candidates

        section_structured_candidates = await _build_section_candidates()
        structured_candidates.extend(section_structured_candidates)
        concrete_section_candidates = [candidate for candidate in section_structured_candidates if _is_concrete_renderable_candidate(candidate)]
        relevant_section_candidates = [candidate for candidate in concrete_section_candidates if _is_relevant_concrete_candidate(candidate)]
        _flow._web_debug(
            "country_strategy.stage_timing",
            stage="section_candidates",
            elapsed_ms=round((perf_counter() - started_at) * 1000, 1),
            candidate_count=len(section_structured_candidates),
        )
        if len(relevant_section_candidates) < self._MIN_FINAL_NEWS:
            article_structured_candidates = await _build_article_candidates()
            structured_candidates.extend(article_structured_candidates)
            _flow._web_debug(
                "country_strategy.article_candidates_supplemented",
                query=last_message,
                query_source_group=query_source_group,
                section_candidate_count=len(section_structured_candidates),
                article_candidate_count=len(article_structured_candidates),
                elapsed_ms=round((perf_counter() - started_at) * 1000, 1),
            )
        else:
            article_structured_candidates = []
            _flow._web_debug(
                "country_strategy.section_first_short_circuit",
                query=last_message,
                query_source_group=query_source_group,
                reason="enough_relevant_candidates",
                section_candidate_count=len(section_structured_candidates),
                concrete_count=len(concrete_section_candidates),
                relevant_count=len(relevant_section_candidates),
                min_required=self._MIN_FINAL_NEWS,
            )

        if not structured_candidates:
            _flow._web_debug(
                "country_strategy.no_structured_candidates",
                query=last_message,
                query_source_group=query_source_group,
                domains_checked=country_press_domains,
                elapsed_ms=round((perf_counter() - started_at) * 1000, 1),
            )
            return None

        if not section_structured_candidates and not article_structured_candidates:
            _flow._web_debug(
                "country_strategy.article_candidates_insufficient",
                query=last_message,
                query_source_group=query_source_group,
                article_candidate_count=0,
                next_step="no_candidates",
            )
            return None

        if structured_candidates:
            ranking_started_at = perf_counter()
            ordered = sorted(
                structured_candidates,
                key=lambda candidate: (
                    0 if _score_candidate_relevance(candidate.as_candidate(), last_message, query_source_group) >= relevance_context.min_relevance_score else 1,
                    _flow._candidate_strategy_priority(candidate.as_candidate(), query=last_message, query_horizon=query_horizon),
                    -_score_candidate_relevance(candidate.as_candidate(), last_message, query_source_group),
                    -len(candidate.snippet.split()),
                ),
            )
            def _para_key(text: str) -> str:
                words = __import__("re").sub(r"[^\w\s]", "", _flow._strip_accents(text.lower())).split()
                return " ".join(words[:6])
            seen_para_keys: set[str] = set()
            deduped_candidates: list[Any] = []
            for c in ordered:
                key = _para_key(c.snippet)
                if key and key not in seen_para_keys:
                    seen_para_keys.add(key)
                    deduped_candidates.append(c)
                if len(deduped_candidates) >= 4:
                    break
            top = deduped_candidates
            article_top = [
                candidate for candidate in top
                if candidate.source_kind == _flow.SourceKind.ARTICLE
                and candidate.evidence_kind == _flow.EvidenceKind.FETCHED_ARTICLE
            ]
            _flow._web_debug(
                "country_strategy.final_candidate_mix",
                top_count=len(top),
                article_top_count=len(article_top),
                section_top_count=len([candidate for candidate in top if candidate.source_kind == _flow.SourceKind.SECTION]),
                top_urls=[candidate.url for candidate in top],
                top_source_kinds=[candidate.source_kind.value for candidate in top],
                top_relevance_scores=[_score_candidate_relevance(candidate.as_candidate(), last_message, query_source_group) for candidate in top],
            )
            sources = [{"title": candidate.title, "url": candidate.url.split("#")[0]} for candidate in top]
            if top and sources:
                source_group_name = _flow.detect_query_source_group(last_message)
                group_lang = _flow.get_group_language(source_group_name)
                needs_translation = group_lang not in (None, "es", "en")
                _flow._web_debug(
                    "country_strategy.return_ready",
                    query=last_message,
                    query_source_group=query_source_group,
                    source_group_name=source_group_name,
                    top_count=len(top),
                    source_count=len(sources),
                    group_lang=group_lang,
                    needs_translation=needs_translation,
                    top_urls=[candidate.url for candidate in top],
                    top_titles=[candidate.title for candidate in top],
                )
                if len(top) >= 3 and not needs_translation:
                    summary_lines = [candidate.snippet for candidate in top if candidate.snippet]
                    digest_contract = _flow._build_web_digest_contract(summary_lines, sources)
                    summary = _flow._format_web_digest_contract(digest_contract)
                    _flow._web_debug(
                        "country_strategy.return_article_digest_min3",
                        rendered_items=len([candidate for candidate in top if candidate.snippet]),
                        summary_preview=summary[:400],
                        elapsed_ms=round((perf_counter() - ranking_started_at) * 1000, 1),
                    )
                    return {
                        "summary": summary,
                        "words": summary.split(),
                        "source_type": "search",
                        "sources": sources,
                        "pre_synthesized": True,
                        "digest_contract": digest_contract,
                    }
                render_ready_count = len([candidate for candidate in top if _snippet_is_render_ready(candidate.snippet)])
                if len(top) >= 3 and render_ready_count >= 3:
                    summary_lines = [candidate.snippet for candidate in top if candidate.snippet]
                    digest_contract = _flow._build_web_digest_contract(summary_lines, sources)
                    summary = _flow._format_web_digest_contract(digest_contract)
                    _flow._web_debug(
                        "country_strategy.return_article_digest_render_ready",
                        rendered_items=len(summary_lines),
                        render_ready_count=render_ready_count,
                        summary_preview=summary[:400],
                        elapsed_ms=round((perf_counter() - ranking_started_at) * 1000, 1),
                    )
                    return {
                        "summary": summary,
                        "words": summary.split(),
                        "source_type": "search",
                        "sources": sources,
                        "pre_synthesized": True,
                        "digest_contract": digest_contract,
                    }
                summary_lines = [candidate.snippet for candidate in top if candidate.snippet]
                digest_contract = _flow._build_web_digest_contract(summary_lines, sources)
                summary = _flow._format_web_digest_contract(digest_contract)
                _flow._web_debug(
                    "country_strategy.return_digest_with_translation",
                    summary_preview=summary[:400],
                    rendered_items=len(summary_lines),
                    elapsed_ms=round((perf_counter() - ranking_started_at) * 1000, 1),
                    total_elapsed_ms=round((perf_counter() - started_at) * 1000, 1),
                )
                return {
                    "summary": summary,
                    "words": summary.split(),
                    "source_type": "search",
                    "sources": sources,
                    "pre_synthesized": True,
                    "digest_contract": digest_contract,
                }
        _flow._web_debug(
            "country_strategy.no_structured_candidates",
            query=last_message,
            query_source_group=query_source_group,
            domains_checked=country_press_domains,
            elapsed_ms=round((perf_counter() - started_at) * 1000, 1),
        )
        return None
