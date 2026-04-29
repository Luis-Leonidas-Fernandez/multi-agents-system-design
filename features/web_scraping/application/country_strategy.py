"""Estrategia de noticias locales recientes por país."""
from __future__ import annotations

from typing import Any, Optional

from features.web_scraping.infrastructure.runtime import WebFetchRuntime, WebSearchRuntime


class CountryRecentNewsStrategy:
    """Estrategia principal para noticias locales recientes basadas en secciones."""

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
            hostname = (_flow.urlparse(url).hostname or "").lower().removeprefix("www.")
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

        for idx, domain in enumerate(country_press_domains):
            press_name = country_press_names[idx] if idx < len(country_press_names) else domain
            source_meta = sources_by_domain.get(domain, {"title": press_name, "url": _flow._default_press_homepage_url(domain)})
            if not _flow._is_press_source_relevant_for_query(source_meta, last_message):
                continue
            fallback_url = (source_meta.get("url") or "").strip()
            if not fallback_url:
                continue
            for section_url, section_label in _flow._build_country_press_section_targets(domain, fallback_url, last_message):
                section_prompt = _flow._build_newspaper_section_fetch_prompt(last_message, source_meta.get("title") or press_name, section_label)
                try:
                    fetch_response = await self._fetch_runtime.fetch(_flow.WebFetchRequest(url=section_url, prompt=section_prompt, mode="static", use_cache=False))
                except Exception:
                    continue
                section_text = fetch_response.content
                issue = _flow._classify_fetch_error(section_text)
                if issue == "missing_playwright":
                    dynamic_fetch_available = False
                if issue in {"not_found", "blocked", "dns", "fetch_error"}:
                    continue
                lines = _flow._filter_section_lines_for_query(_flow._extract_section_content_lines(section_text, last_message, section_label), last_message, section_label)
                lines = _flow._dedupe_homepage_lines(lines)
                fallback_lines = [__import__("re").sub(r"^\s*(?:[-*•]\s+|\d+\.\s+)", "", line).strip() for line in (section_text or "").splitlines() if line.strip()]
                fallback_lines = [line for line in fallback_lines if len(line) > 20 and not _flow._is_homepage_meta_line(line) and not _flow._is_no_info_response(line)]
                if not lines:
                    lines = _flow._dedupe_homepage_lines(fallback_lines)
                if not lines and not fallback_lines and dynamic_fetch_available and not (section_text or "").strip():
                    try:
                        dynamic_response = await self._fetch_runtime.fetch(_flow.WebFetchRequest(url=section_url, prompt=section_prompt, mode="dynamic", use_cache=False))
                    except Exception:
                        continue
                    dynamic_issue = _flow._classify_fetch_error(dynamic_response.content)
                    if dynamic_issue == "missing_playwright":
                        dynamic_fetch_available = False
                    if dynamic_issue not in {"not_found", "blocked", "dns", "fetch_error", "missing_playwright"}:
                        lines = _flow._filter_section_lines_for_query(_flow._extract_section_content_lines(dynamic_response.content, last_message, section_label), last_message, section_label)
                        lines = _flow._dedupe_homepage_lines(lines)
                        section_text = dynamic_response.content
                if not lines:
                    continue
                raw_blocks = [" ".join(ln.strip() for ln in block.splitlines() if ln.strip()) for block in (section_text or "").split("\n\n") if block.strip()]
                section_items: list[str] = []
                seen_block_prefixes: set[str] = set()
                for block_text in raw_blocks:
                    if len(block_text) < 20 or _flow._is_no_info_response(block_text):
                        continue
                    block_norm = _flow._strip_accents(block_text.lower())
                    if not filter_terms_for_section or any(term in block_norm for term in filter_terms_for_section):
                        prefix = " ".join(block_norm.split()[:4])
                        if prefix and prefix not in seen_block_prefixes:
                            seen_block_prefixes.add(prefix)
                            section_items.append(block_text)
                if not section_items:
                    section_items = [ln for ln in lines if len(ln) > 20]
                for item_idx, item_text in enumerate(section_items):
                    candidate_url = section_url if item_idx == 0 else f"{section_url}#n{item_idx}"
                    if candidate_url in seen_urls:
                        continue
                    seen_urls.add(candidate_url)
                    structured_candidates.append(_flow.WebCandidate(title=f"{source_meta.get('title') or press_name} — {section_label}", url=candidate_url, snippet=item_text, source_kind=_flow.SourceKind.SECTION, evidence_kind=_flow.EvidenceKind.SECTION_LINES, recency=_flow.Recency.DATED_RECENT, specificity=_flow.Specificity.CONCRETE, source_label=section_label))

        if structured_candidates:
            ordered = sorted(structured_candidates, key=lambda candidate: _flow._candidate_strategy_priority(candidate.as_candidate(), query=last_message, query_horizon=query_horizon))
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
            sources = [{"title": candidate.title, "url": candidate.url.split("#")[0]} for candidate in top]
            if top and sources:
                geography = self._country_resolver.resolve(last_message) or ""
                topic = _flow._detect_news_topic(last_message)
                topic_label = {"security": "Seguridad", "politics": "Política", "economy": "Economía"}.get(topic, "Noticias")
                header = f"**{topic_label} en {geography}**" if geography else f"**{topic_label}**"
                source_group_name = _flow.detect_query_source_group(last_message)
                group_lang = _flow.get_group_language(source_group_name)
                section_hint = any("/news/" in candidate.url or "section" in candidate.source_kind for candidate in top)
                needs_translation = group_lang not in (None, "es", "en") and not section_hint
                if not needs_translation:
                    summary_lines = [candidate.snippet for candidate in top if candidate.snippet]
                    digest_contract = _flow._build_web_digest_contract(summary_lines, sources)
                    summary = _flow._format_web_digest_contract(digest_contract)
                    return {"summary": summary, "words": summary.split(), "source_type": "search", "sources": sources, "pre_synthesized": True, "digest_contract": digest_contract}
                summary_lines = [f"{candidate.title} — {candidate.snippet}" for candidate in top if candidate.snippet]
                digest_contract = _flow._build_web_digest_contract(summary_lines, sources, intro=header.replace("**", ""))
                summary = _flow._format_web_digest_contract(digest_contract)
                return {"summary": summary, "words": summary.split(), "source_type": "search", "sources": sources, "pre_synthesized": True, "digest_contract": digest_contract}
        return None
