"""Caso de uso para el flujo de web scraping.

Coordina HITL, estrategia, guardrails, retry y postcondiciones.
El nodo LangGraph queda como adaptador fino.
"""
import asyncio
import os
import re
import time
import uuid
from urllib.parse import urljoin, urlparse
from typing import Any, Optional, Callable, Awaitable, Mapping, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig

from application.policies.agentdog import evaluate_trajectory_safe, _should_evaluate_guard, _is_allowed_public_price_request
from application.helpers.audit_flow_helpers import (
    _emit_node_outcome,
    _emit_country_news_metrics,
    _extract_tokens,
    _extract_quality,
    _extract_followup,
    _node_meta,
    _get_model_name,
)
from ports.confirmation_port import ConfirmationPort
from application.helpers.message_flow_helpers import extract_final_ai_text, get_last_message_text, is_web_information_query
from application.helpers.trace_flow_helpers import get_or_create_request_id
from application.policies.scrape_tracker import (
    _get_category_score,
    _update_scrape_tracker,
    _STRUCTURED_SOURCE_STRATEGIES,
    _RETRY_ON_RELIABILITY,
    _scrape_reliability,
)
from application.policies.web_source_policy import (
    detect_query_source_group,
    detect_recent_query_horizon,
    get_group_language,
    get_preferred_domains_for_group,
    get_query_source_terms,
    get_recent_query_requirements,
    get_source_domain_priority,
    is_recent_web_information_query,
    score_domain_boost,
)
from application.policies.candidate_scoring import (
    _score_generic_candidate,
    _candidate_source_priority,
    _rank_candidates_by_source_policy,
)
from application.policies.web_search_context import QueryContext, RecentPolicy
from domain.country_profile import GEO_ENGLISH
from domain.country_resolver import GENERIC_WEB_STOPWORDS, GEOGRAPHY_TERMS, extract_query_geography
from domain.topic_detector import TOPIC_ANGLES, TOPIC_ANGLES_EN, detect_news_topic
from domain.section_path_resolver import (
    COUNTRY_PRESS_SECTION_PATHS,
    GENERIC_SECTION_PATHS,
    build_country_press_section_targets,
)
from infra.country_profile_repo import PERIODICOS_CONTINENT_SLUG_BY_COUNTRY
from ports.country_news_ports import (
    ICountryResolver,
    ICountryProfileRepository,
    ISectionPathResolver,
    IPressSourceDiscovery,
    IDynamicPressSourceDiscovery,
)
from application.helpers.url_helpers import _is_article_url, _extract_web_fetch_redirect_url
from domain.web_models import (
    CandidateDict,
    EvidenceKind,
    Recency,
    SourceDict,
    SourceKind,
    Specificity,
    WebCandidate,
)
from domain.web_text_utils import (
    _TITLE_STOPWORDS,
    _MONTH_NAMES_ES,
    _MONTH_NAMES_EN,
    _NO_INFO_RE,
    _text_keywords,
    _extract_urls_from_text,
    _clean_source_url,
    _format_sources,
    _build_source_backed_response,
    _strip_accents,
    _slugify_periodicos_label,
    _is_no_info_response,
    _enforce_synthesis_format,
    _dedup_synthesis_bullets,
    _candidate_url_has_date,
    _candidate_url_is_recent,
)
from domain.web_classifier import (
    _NON_NEWS_DOMAINS,
    _FORUM_PATH_SEGMENTS,
    _is_non_news_candidate,
    _same_event,
    _dedup_candidates_by_event,
    _extract_generic_search_candidates,
    _candidate_snippet_lines,
    _is_hub_like_candidate,
    _query_targets_public_safety,
    _is_tangential_vertical_candidate,
    _is_invalid_news_candidate,
    _candidate_record_from_dict,
    _classify_candidate_source_kind,
    _classify_candidate_recency,
    _classify_candidate_specificity,
    _candidate_strategy_priority,
)
from application.helpers.price_flow_helpers import (
    _detect_coin_from_query,
    _format_price_response,
    _extract_price_from_messages,
    _extract_structured_price,
    _get_crypto_price_fn,
)
from application.services.web_runtime import (
    WebFetchRequest,
    WebFetchRuntime,
    WebSearchRequest,
    WebSearchRuntime,
)
from application.services.web_response_post_filter import apply_web_response_post_filter
from domain.models import AgentState


def _web_debug_enabled() -> bool:
    return (os.getenv("WEB_DEBUG") or "").strip().lower() in {"1", "true", "yes", "on"}


def _web_debug(label: str, **data: Any) -> None:
    if not _web_debug_enabled():
        return
    payload = " ".join(
        f"{key}={repr(value)}"
        for key, value in data.items()
    )
    print(f"[WEB_DEBUG] {label}{(' ' + payload) if payload else ''}", flush=True)



def _finalize_web_user_summary(
    summary: str,
    last_message: str,
    sources: Optional[list[dict[str, str]]] = None,
) -> tuple[str, Optional[list[dict[str, str]]], list[str]]:
    filtered_summary, filtered_sources = apply_web_response_post_filter(
        summary=summary,
        query=last_message,
        sources=sources,
    )
    final_words = filtered_summary.split()
    return filtered_summary, filtered_sources, final_words


WebCandidateRecord = WebCandidate



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
        from infra.country_news_adapters import (
            DefaultCountryResolver,
            DefaultCountryProfileRepository,
            DefaultSectionPathResolver,
            DefaultPressSourceDiscovery,
        )
        from infra.dynamic_press_discovery import DefaultDynamicPressDiscovery
        self._search_runtime = search_runtime
        self._fetch_runtime = fetch_runtime
        self._country_resolver = country_resolver or DefaultCountryResolver()
        self._profile_repo = profile_repo or DefaultCountryProfileRepository()
        self._section_path_resolver = section_path_resolver or DefaultSectionPathResolver()
        self._press_discovery = press_discovery or DefaultPressSourceDiscovery()
        self._dynamic_discovery = dynamic_discovery or DefaultDynamicPressDiscovery()

    async def execute(
        self,
        last_message: str,
        web_search_runtime_args: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        query_source_group = detect_query_source_group(last_message)
        query_horizon = detect_recent_query_horizon(last_message) if _is_recent_web_information_query(last_message) else None
        use_bootstrap = _should_use_country_recent_news_strategy(last_message, query_source_group, query_horizon)

        # ── Bootstrap path ───────────────────────────────────────────────────
        if use_bootstrap:
            source_terms = list(get_query_source_terms(last_message))
            country_press_domains, country_press_names = await self._press_discovery.discover(
                last_message,
                query_source_group,
                source_terms,
                web_search_runtime_args,
            )
            if not country_press_domains:
                _emit_country_news_metrics(
                    geography=query_source_group,
                    resolution_path="none",
                    domains_found=0,
                )
                return None
            _emit_country_news_metrics(
                geography=query_source_group,
                resolution_path="bootstrap",
                domains_found=len(country_press_domains),
            )

        # ── Fase 4: soft gate — país no registrado pero detectable ───────────
        elif query_source_group is None:
            inferred_geo = self._country_resolver.resolve(last_message)
            lowered_msg = (last_message or "").lower()
            _has_news = any(t in lowered_msg for t in ("noticia", "noticias", "news", "headline", "headlines"))
            _has_topic = _detect_news_topic(last_message) in {"security", "economy", "politics"}
            _valid_horizon = query_horizon in {"today", "week", "month"}
            if not (inferred_geo and _valid_horizon and (_has_news or _has_topic)):
                return None
            _web_debug(
                "country_press.dynamic.attempt",
                geography=inferred_geo,
                horizon=query_horizon,
                reason="source_group_missing",
            )
            country_press_domains, country_press_names = await self._dynamic_discovery.discover_for_unknown_country(
                last_message,
                inferred_geo,
                web_search_runtime_args,
            )
            if not country_press_domains:
                _web_debug("country_press.dynamic.no_sources", geography=inferred_geo)
                _emit_country_news_metrics(
                    geography=inferred_geo,
                    resolution_path="none",
                    domains_found=0,
                )
                return None
            # Fabricar un source_group sintético para que el caché funcione igual.
            query_source_group = f"dynamic:{inferred_geo.lower()}"
            source_terms = [inferred_geo.lower()]
            _country_press_cache_set(query_source_group, source_terms, country_press_domains, country_press_names)
            _country_press_strategy_cache_set(query_source_group, source_terms, "dynamic")
            _web_debug("country_press.dynamic.success", geography=inferred_geo, domains=country_press_domains)
            _emit_country_news_metrics(
                geography=inferred_geo,
                resolution_path="dynamic",
                domains_found=len(country_press_domains),
            )

        else:
            # Otro motivo para que el gate falle (deporte, horizonte inválido, etc.)
            return None

        # ── Continuación común (bootstrap + dynamic) ─────────────────────────
        query_terms = _extract_generic_query_terms(last_message)
        for term in source_terms:
            if term not in query_terms:
                query_terms.append(term)

        country_press_sources = _country_press_source_cache_get(query_source_group, source_terms)
        discovery_strategy = _country_press_strategy_cache_get(query_source_group, source_terms)
        if discovery_strategy == "none" and not country_press_sources:
            return None
        sources_by_domain: dict[str, dict[str, str]] = {}
        for source in country_press_sources:
            url = source.get("url", "")
            hostname = (urlparse(url).hostname or "").lower().removeprefix("www.")
            if hostname and hostname not in sources_by_domain:
                sources_by_domain[hostname] = source

        structured_candidates: list[WebCandidateRecord] = []
        seen_urls: set[str] = set()
        dynamic_fetch_available = True
        _sec_topic = _detect_news_topic(last_message)
        _topic_terms_for_filter: dict[str, set[str]] = {
            "security": {
                "seguridad", "sicurezza", "crime", "crimen", "cronaca", "polizia", "policia",
                "policiales", "detenid", "arrestad", "operativo", "homicidio", "asesin",
                "robo", "narco", "violencia", "sucesos", "delito", "fiscal", "tribunal",
                "ertzaintza", "mossos", "guardia civil", "omicid", "arrest", "blitz",
            },
            "economy": {
                "econom", "mercad", "mercato", "finanz", "inflac", "presupuesto",
                "negocios", "empresa", "bolsa", "pib", "deuda",
            },
            "politics": {
                "politic", "govern", "parlament", "elecci", "presidente", "ministro",
                "congreso", "senado", "partido", "decreto",
            },
        }
        _filter_terms_for_section = _topic_terms_for_filter.get(_sec_topic, set())

        for idx, domain in enumerate(country_press_domains):
            press_name = country_press_names[idx] if idx < len(country_press_names) else domain
            source_meta = sources_by_domain.get(domain, {"title": press_name, "url": _default_press_homepage_url(domain)})
            if not _is_press_source_relevant_for_query(source_meta, last_message):
                continue
            fallback_url = (source_meta.get("url") or "").strip()
            if not fallback_url:
                continue
            for section_url, section_label in self._section_path_resolver.resolve_targets(domain, fallback_url, last_message):
                section_prompt = _build_newspaper_section_fetch_prompt(
                    last_message,
                    source_meta.get("title") or press_name,
                    section_label,
                )
                try:
                    fetch_response = await self._fetch_runtime.fetch(
                        WebFetchRequest(
                            url=section_url,
                            prompt=section_prompt,
                            mode="static",
                            use_cache=False,
                        )
                    )
                except Exception:
                    continue
                section_text = fetch_response.content
                issue = _classify_fetch_error(section_text)
                if issue == "missing_playwright":
                    dynamic_fetch_available = False
                if issue in {"not_found", "blocked", "dns", "fetch_error"}:
                    continue
                lines = _filter_section_lines_for_query(
                    _extract_section_content_lines(section_text, last_message, section_label),
                    last_message,
                    section_label,
                )
                lines = _dedupe_homepage_lines(lines)
                if not lines and dynamic_fetch_available:
                    try:
                        dynamic_response = await self._fetch_runtime.fetch(
                            WebFetchRequest(
                                url=section_url,
                                prompt=section_prompt,
                                mode="dynamic",
                                use_cache=False,
                            )
                        )
                    except Exception:
                        continue
                    dynamic_issue = _classify_fetch_error(dynamic_response.content)
                    if dynamic_issue == "missing_playwright":
                        dynamic_fetch_available = False
                    if dynamic_issue not in {"not_found", "blocked", "dns", "fetch_error", "missing_playwright"}:
                        lines = _filter_section_lines_for_query(
                            _extract_section_content_lines(dynamic_response.content, last_message, section_label),
                            last_message,
                            section_label,
                        )
                        lines = _dedupe_homepage_lines(lines)
                        section_text = dynamic_response.content
                if not lines:
                    continue
                # Extraemos TODOS los párrafos válidos de la sección (separados por \n\n).
                # Cada párrafo es una noticia distinta reportada por el LLM.
                # Luego el dedup cross-fuente al final selecciona las 4 mejores globalmente.
                raw_blocks = [
                    " ".join(ln.strip() for ln in block.splitlines() if ln.strip())
                    for block in (section_text or "").split("\n\n")
                    if block.strip()
                ]
                section_items: list[str] = []
                seen_block_prefixes: set[str] = set()
                for block_text in raw_blocks:
                    if len(block_text) < 20:
                        continue
                    if _is_no_info_response(block_text):
                        continue
                    block_norm = _strip_accents(block_text.lower())
                    if not _filter_terms_for_section or any(term in block_norm for term in _filter_terms_for_section):
                        prefix = " ".join(block_norm.split()[:4])
                        if prefix and prefix not in seen_block_prefixes:
                            seen_block_prefixes.add(prefix)
                            section_items.append(block_text)
                # Fallback: si el LLM no usó párrafos separados, usamos las líneas filtradas
                if not section_items:
                    section_items = [ln for ln in lines if len(ln) > 20]
                for item_idx, item_text in enumerate(section_items):
                    candidate_url = section_url if item_idx == 0 else f"{section_url}#n{item_idx}"
                    if candidate_url in seen_urls:
                        continue
                    seen_urls.add(candidate_url)
                    structured_candidates.append(WebCandidate(
                        title=f"{source_meta.get('title') or press_name} — {section_label}",
                        url=candidate_url,
                        snippet=item_text,
                        source_kind=SourceKind.SECTION,
                        evidence_kind=EvidenceKind.SECTION_LINES,
                        recency=Recency.DATED_RECENT,
                        specificity=Specificity.CONCRETE,
                        source_label=section_label,
                    ))

        if structured_candidates:
            ordered = sorted(
                structured_candidates,
                key=lambda candidate: _candidate_strategy_priority(
                    candidate.as_candidate(),
                    query=last_message,
                    query_horizon=query_horizon,
                ),
            )
            # Dedup cross-candidatos: descartamos párrafos que comparten el núcleo
            # de la misma noticia (primeras 6 palabras normalizadas en común)
            def _para_key(text: str) -> str:
                words = re.sub(r"[^\w\s]", "", _strip_accents(text.lower())).split()
                return " ".join(words[:6])

            seen_para_keys: set[str] = set()
            deduped_candidates: list[WebCandidateRecord] = []
            for c in ordered:
                key = _para_key(c.snippet)
                if key and key not in seen_para_keys:
                    seen_para_keys.add(key)
                    deduped_candidates.append(c)
                if len(deduped_candidates) >= 4:
                    break

            top = deduped_candidates
            sources = [
                {"title": candidate.title, "url": candidate.url.split("#")[0]}
                for candidate in top
            ]
            if top and sources:
                geography = self._country_resolver.resolve(last_message) or ""
                topic = _detect_news_topic(last_message)
                topic_label = {
                    "security": "Seguridad",
                    "politics": "Política",
                    "economy": "Economía",
                }.get(topic, "Noticias")
                header = f"**{topic_label} en {geography}**" if geography else f"**{topic_label}**"

                # Detect if source language needs translation to query language.
                # es/en sources → fast path (inline Fuente:, no LLM call).
                # Other languages (it, ja, ko, etc.) → synthesis path so the LLM
                # translates to the query language and attributes sources per bullet.
                source_group_name = detect_query_source_group(last_message)
                group_lang = get_group_language(source_group_name)
                needs_translation = group_lang not in (None, "es", "en")

                if not needs_translation:
                    # Fast path: build each candidate with its source inline (1:1 guaranteed).
                    blocks: list[str] = []
                    for i, candidate in enumerate(top):
                        if not candidate.snippet:
                            continue
                        url = candidate.url.split("#")[0]
                        src_line = f"Fuente: [{candidate.title}]({url})"
                        snippet = candidate.snippet
                        if i == 0:
                            snippet = f"{header}\n\n{snippet}"
                        blocks.append(f"{snippet}\n\n{src_line}")
                    summary = "\n\n".join(blocks)
                    return {
                        "summary": summary,
                        "words": summary.split(),
                        "source_type": "search",
                        "sources": sources,
                        "pre_synthesized": True,
                    }
                else:
                    # Translation path: build labeled content so _synthesize_search_summary
                    # can translate AND attribute each bullet to its source.
                    labeled_parts = [
                        f"[{candidate.title}]: {candidate.snippet}"
                        for candidate in top if candidate.snippet
                    ]
                    raw_for_synthesis = f"{header}\n\n" + "\n\n".join(labeled_parts)
                    return {
                        "summary": raw_for_synthesis,
                        "words": raw_for_synthesis.split(),
                        "source_type": "search",
                        "sources": sources,
                        "pre_synthesized": False,
                        "has_labeled_content": True,
                    }
        return None


class GenericWebSearchStrategy:
    """Estrategia generalista search+fetch con runtime encapsulado."""

    def __init__(self, *, search_runtime: WebSearchRuntime, fetch_runtime: WebFetchRuntime) -> None:
        self._search_runtime = search_runtime
        self._fetch_runtime = fetch_runtime

    async def execute(
        self,
        last_message: str,
        web_search_runtime_args: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        return await _run_generic_web_search_strategy_impl(last_message, web_search_runtime_args)


_COUNTRY_PRESS_CACHE: dict[str, tuple[float, tuple[list[str], list[str]]]] = {}
_COUNTRY_PRESS_CACHE_TTL_SECONDS = 60 * 60 * 24
_COUNTRY_PRESS_SOURCE_CACHE: dict[str, tuple[float, list[dict[str, str]]]] = {}
_COUNTRY_PRESS_DISCOVERY_STRATEGY_CACHE: dict[str, tuple[float, str]] = {}


def _country_press_cache_key(query_source_group: Optional[str], source_terms: list[str]) -> str:
    normalized_terms = tuple(sorted({term.strip().lower() for term in source_terms if term.strip()}))
    return f"{query_source_group or ''}|{'|'.join(normalized_terms)}"


def _country_press_cache_get(query_source_group: Optional[str], source_terms: list[str]) -> Optional[tuple[list[str], list[str]]]:
    cache_key = _country_press_cache_key(query_source_group, source_terms)
    cached = _COUNTRY_PRESS_CACHE.get(cache_key)
    if not cached:
        return None
    cached_at, value = cached
    if (time.time() - cached_at) > _COUNTRY_PRESS_CACHE_TTL_SECONDS:
        _COUNTRY_PRESS_CACHE.pop(cache_key, None)
        return None
    domains, titles = value
    return list(domains), list(titles)


def _country_press_cache_set(query_source_group: Optional[str], source_terms: list[str], domains: list[str], titles: list[str]) -> None:
    cache_key = _country_press_cache_key(query_source_group, source_terms)
    _COUNTRY_PRESS_CACHE[cache_key] = (time.time(), (list(domains), list(titles)))


def _country_press_source_cache_get(query_source_group: Optional[str], source_terms: list[str]) -> list[dict[str, str]]:
    cache_key = _country_press_cache_key(query_source_group, source_terms)
    cached = _COUNTRY_PRESS_SOURCE_CACHE.get(cache_key)
    if not cached:
        return []
    cached_at, value = cached
    if (time.time() - cached_at) > _COUNTRY_PRESS_CACHE_TTL_SECONDS:
        _COUNTRY_PRESS_SOURCE_CACHE.pop(cache_key, None)
        return []
    return [dict(source) for source in value]


def _country_press_source_cache_set(query_source_group: Optional[str], source_terms: list[str], sources: list[dict[str, str]]) -> None:
    cache_key = _country_press_cache_key(query_source_group, source_terms)
    _COUNTRY_PRESS_SOURCE_CACHE[cache_key] = (time.time(), [dict(source) for source in sources])


def _country_press_strategy_cache_get(query_source_group: Optional[str], source_terms: list[str]) -> str:
    cache_key = _country_press_cache_key(query_source_group, source_terms)
    cached = _COUNTRY_PRESS_DISCOVERY_STRATEGY_CACHE.get(cache_key)
    if not cached:
        return "none"
    cached_at, value = cached
    if (time.time() - cached_at) > _COUNTRY_PRESS_CACHE_TTL_SECONDS:
        _COUNTRY_PRESS_DISCOVERY_STRATEGY_CACHE.pop(cache_key, None)
        return "none"
    return value


def _country_press_strategy_cache_set(query_source_group: Optional[str], source_terms: list[str], strategy: str) -> None:
    cache_key = _country_press_cache_key(query_source_group, source_terms)
    _COUNTRY_PRESS_DISCOVERY_STRATEGY_CACHE[cache_key] = (time.time(), strategy)


def _build_policy_country_press_sources(query_source_group: Optional[str]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    for domain in get_preferred_domains_for_group(query_source_group):
        hostname = domain.strip().lower()
        if not hostname:
            continue
        sources.append({
            "title": hostname,
            "url": _default_press_homepage_url(hostname),
            "domain": hostname,
        })
    return sources


def _default_press_homepage_url(domain: str) -> str:
    hostname = (domain or "").strip().lower().removeprefix("www.")
    if not hostname:
        return ""
    return f"https://www.{hostname}/"


def _build_no_local_sources_response(last_message: str) -> dict[str, Any]:
    geography = _extract_query_geography(last_message) or "ese país"
    summary = (
        f"No encontré fuentes locales confiables de {geography} para esta semana. "
        "Prefiero no mezclar resultados globales ruidosos o tangenciales."
    )
    return {
        "summary": summary,
        "words": summary.split(),
        "source_type": "search",
        "sources": [],
        "pre_synthesized": True,
    }


def _debug_periodicos_fetch(url: str, stage: str) -> bytes:
    from infra import scraping_infra

    _web_debug("country_press.directory.fetch_start", stage=stage, url=url)
    html = scraping_infra._fetch_html(url)
    _web_debug("country_press.directory.fetch_success", stage=stage, url=url, bytes=len(html))
    return html


def _web_search_runtime_args(state: Mapping[str, Any]) -> dict[str, Any]:
    selected = str(state.get("web_search_selected_provider") or "").strip().lower()
    configured = str(state.get("web_search_provider_configured") or "").strip().lower()
    args: dict[str, Any] = {}
    if selected:
        args["runtime_selected_provider"] = selected
    if configured:
        args["runtime_provider_configured"] = configured
    return args


def _select_strategy_context(state: AgentState, last_message: str, get_runtime_policy: Callable[[], dict]) -> dict:
    from application.policies.scrape_tracker import (
        _detect_query_category,
        _get_strategy,
        _score_to_reliability,
        _API_VALIDATION_EPSILON,
        _exploration_rate,
    )

    tracker    = state.get("scrape_tracker") or {}
    turn_count = (tracker.get("_turn_count") or 0) + 1
    category   = _detect_query_category(last_message)
    prior_score       = _get_category_score(tracker, category, turn_count)
    prior_reliability = _score_to_reliability(prior_score)

    _rt           = get_runtime_policy().get(category, {})
    _top_promoted = (_rt.get("promoted") or [None])[0]
    ml_recommended: Optional[str] = (
        _top_promoted.get("strategy") if isinstance(_top_promoted, dict) else _top_promoted
    )

    import random

    if _is_allowed_public_price_request(state.get("messages", []), "web_scraping_node"):
        strategy, exploring = "api_price", False
        exp_rate = 0.0
    elif category == "crypto_price":
        if random.random() < _API_VALIDATION_EPSILON:
            strategy, exploring = "force_search", True
        else:
            strategy, exploring = "api_price", False
        exp_rate = _API_VALIDATION_EPSILON
    else:
        exp_rate  = _exploration_rate(prior_score)
        exploring = random.random() < exp_rate
        strategy  = _get_strategy(tracker, category, prior_score, exploring=exploring)

    prediction_match: Optional[bool] = (
        (strategy == ml_recommended) if ml_recommended is not None else None
    )

    return {
        "tracker": tracker, "turn_count": turn_count, "category": category,
        "prior_score": prior_score, "prior_reliability": prior_reliability,
        "ml_recommended": ml_recommended, "strategy": strategy,
        "exploring": exploring, "exp_rate": exp_rate, "prediction_match": prediction_match,
    }


async def _summarize_if_long(
    text: str, rid: str, get_llm_fn: Callable, *, is_retry: bool = False
) -> str:
    if len(text.split()) <= 200:
        return text

    sources_block = ""
    body_text = text
    if "Sources:" in text:
        body_text, sources_block = text.split("Sources:", 1)
        sources_block = "Sources:" + sources_block

    tags = ["web_scraping", "context_quarantine", "summary"]
    if is_retry:
        tags.append("retry")
    try:
        llm = get_llm_fn()
        summary_response = await llm.ainvoke(
            [HumanMessage(content=(
                "Resume el siguiente texto en máximo 200 palabras, "
                f"conservando los datos más importantes:\n\n{body_text[:4000]}"
            ))],
            config=RunnableConfig(
                tags=tags,
                metadata={
                    "node":              "web_scraping_node",
                    "request_id":        rid,
                    "raw_words":         len(body_text.split()),
                    "summary_triggered": True,
                },
            ),
        )
        summary = cast(str, summary_response.content)
        if sources_block:
            summary = f"{summary.strip()}\n\n{sources_block.strip()}"
        return summary
    except Exception:
        # If the model backend is unavailable, truncate to 200 words to preserve context quarantine.
        truncated = " ".join(body_text.split()[:200])
        if sources_block:
            return f"{truncated}\n\n{sources_block.strip()}"
        return truncated


async def _run_retry_agent(
    agent,
    last_message: str,
    rid: str,
    get_llm_fn: Callable,
) -> tuple[Optional[str], list[str], dict[str, Any], dict[str, Any]]:
    retry_hint = (
        f"[Sistema | auto-retry por bajo rendimiento | estrategia=force_search]\n"
        + "Usa search_web directamente — no intentes scraping de páginas.\n\n"
    )
    retry_result = await agent.ainvoke(
        {"messages": [HumanMessage(content=retry_hint + last_message)]},
        config=RunnableConfig(
            tags=["web_scraping", "agent", "high_risk", "context_quarantine", "retry"],
            metadata={
                "node":       "web_scraping_node",
                "agent":      "web_scraping_agent",
                "request_id": rid,
                "retry":      True,
            },
        ),
    )

    retry_text = extract_final_ai_text(retry_result.get("messages", []))
    if not retry_text:
        return None, [], {}, {}

    summary = await _summarize_if_long(retry_text, rid, get_llm_fn, is_retry=True)
    return (
        summary,
        retry_text.split(),
        _extract_tokens(retry_result),
        _extract_quality(retry_result),
    )


# ============================================================================
# Generic Claude-style web flow
# ============================================================================

# Alias para compatibilidad con referencias internas al módulo.
_GENERIC_WEB_STOPWORDS = GENERIC_WEB_STOPWORDS


def _extract_generic_query_terms(text: str) -> list[str]:
    terms: list[str] = []
    for raw in re.findall(r"[\wáéíóúñÁÉÍÓÚÑ]+", (text or "").lower()):
        if len(raw) < 3 or raw in _GENERIC_WEB_STOPWORDS:
            continue
        if raw not in terms:
            terms.append(raw)
    return terms


_is_recent_web_information_query = is_recent_web_information_query


def _should_use_country_recent_news_strategy(
    text: str,
    query_source_group: Optional[str],
    query_horizon: Optional[str],
) -> bool:
    if not query_source_group or query_horizon not in {"today", "week", "month"}:
        return False
    lowered = (text or "").lower()
    if any(term in lowered for term in ("resultado", "resultados", "partido", "partidos", "futbol", "football", "soccer", "nba", "nfl")):
        return False
    if not _is_recent_web_information_query(text):
        return False
    topic = _detect_news_topic(text)
    has_news_word = any(term in lowered for term in ("noticia", "noticias", "news", "headline", "headlines"))
    return topic in {"security", "economy", "politics"} or has_news_word


# Alias para compatibilidad con referencias internas al módulo.
_GEOGRAPHY_TERMS = GEOGRAPHY_TERMS
_extract_query_geography = extract_query_geography

# Alias para compatibilidad con referencias internas al módulo.
_TOPIC_ANGLES = TOPIC_ANGLES
_TOPIC_ANGLES_EN = TOPIC_ANGLES_EN
_GEO_ENGLISH = GEO_ENGLISH
_PERIODICOS_CONTINENT_SLUG_BY_COUNTRY = PERIODICOS_CONTINENT_SLUG_BY_COUNTRY
_detect_news_topic = detect_news_topic


def _country_press_query_terms(last_message: str) -> list[str]:
    geography = _extract_query_geography(last_message) or ""
    geo_en = _GEO_ENGLISH.get(geography, geography)
    topic = _detect_news_topic(last_message)
    horizon = detect_recent_query_horizon(last_message)

    topic_terms_map = {
        "security": ["seguridad", "sicurezza", "cronaca", "polizia"],
        "economy": ["economia", "mercato", "finanza"],
        "politics": ["politica", "governo", "parlamento"],
        "default": ["noticias", "attualita"],
    }
    terms: list[str] = []
    for value in [geography, geo_en, *topic_terms_map.get(topic, topic_terms_map["default"])]:
        cleaned = str(value or "").strip()
        if cleaned and cleaned.lower() not in {term.lower() for term in terms}:
            terms.append(cleaned)
    if horizon == "week":
        for value in ["esta semana", "week"]:
            if value.lower() not in {term.lower() for term in terms}:
                terms.append(value)
    return terms


def _build_country_press_search_query(last_message: str, domain: str, press_name: str) -> str:
    query_terms = _country_press_query_terms(last_message)
    query = " ".join([f"site:{domain}", *query_terms]).strip()
    normalized_press = _strip_accents((press_name or "").lower())
    if (
        press_name
        and len(press_name.split()) <= 4
        and not any(noise in normalized_press for noise in ("deportivo", "sport", "stadio"))
    ):
        query = f"{query} {press_name.strip()}".strip()
    return query


def _build_country_press_search_queries(last_message: str, domain: str, press_name: str) -> list[str]:
    geography = _extract_query_geography(last_message) or ""
    geo_en = _GEO_ENGLISH.get(geography, geography)
    topic = _detect_news_topic(last_message)
    horizon = detect_recent_query_horizon(last_message)

    variants_by_topic = {
        "security": [
            [geography, "sicurezza"],
            [geography, "cronaca"],
            [geo_en, "security"],
            [geography, "polizia"],
        ],
        "economy": [
            [geography, "economia"],
            [geo_en, "economy"],
            [geography, "mercato"],
        ],
        "politics": [
            [geography, "politica"],
            [geo_en, "politics"],
            [geography, "governo"],
        ],
        "default": [
            [geography, "noticias"],
            [geo_en, "news"],
        ],
    }
    variants = variants_by_topic.get(topic, variants_by_topic["default"])
    queries: list[str] = []
    seen: set[str] = set()
    normalized_press = _strip_accents((press_name or "").lower())
    short_press_name = (
        press_name.strip()
        if press_name
        and len(press_name.split()) <= 4
        and not any(noise in normalized_press for noise in ("deportivo", "sport", "stadio"))
        else ""
    )
    for variant in variants:
        parts = [f"site:{domain}", *[part for part in variant if part]]
        if horizon == "week":
            parts.append("week")
        if short_press_name:
            parts.append(short_press_name)
        query = " ".join(str(part).strip() for part in parts if str(part).strip())
        if query and query not in seen:
            seen.add(query)
            queries.append(query)
    if not queries:
        queries.append(_build_country_press_search_query(last_message, domain, press_name))
    return queries


def _build_country_press_search_invoke_args(
    query: str,
    domain: str,
    *,
    search_age_days: Optional[int],
    query_horizon: Optional[str],
    web_search_runtime_args: Optional[dict[str, Any]],
    broad: bool = False,
) -> dict[str, Any]:
    invoke_args: dict[str, Any] = {
        "query": query,
        "use_cache": False,
        **(web_search_runtime_args or {}),
    }
    invoke_args["allowed_domains"] = [domain]
    if search_age_days is not None:
        invoke_args["max_age_days"] = search_age_days
    if not broad:
        invoke_args["topic"] = "news"
        if query_horizon == "today":
            invoke_args["time_range"] = "day"
        elif query_horizon == "week":
            invoke_args["time_range"] = "week"
    else:
        invoke_args.pop("topic", None)
        invoke_args.pop("time_range", None)
    return invoke_args


def _is_press_source_relevant_for_query(source: dict[str, str], last_message: str) -> bool:
    topic = _detect_news_topic(last_message)
    title_blob = _strip_accents(f"{source.get('title', '')} {source.get('url', '')}".lower())
    if topic != "security":
        return True
    disallowed = ("deportivo", "sport", "calcio", "football", "futbol", "stadio")
    return not any(token in title_blob for token in disallowed)


def _filter_homepage_lines_for_query(lines: list[str], last_message: str, query_terms: list[str]) -> list[str]:
    if not lines:
        return []
    topic = _detect_news_topic(last_message)
    normalized_terms = {_strip_accents(term.lower()) for term in query_terms if len(term) >= 4}
    topical_terms_map = {
        "security": {
            "seguridad", "sicurezza", "crime", "crimen", "cronaca", "polizia", "policia", "ciber", "cyber",
            "difesa", "defensa", "migr", "attacco", "ataque", "policiales", "detenid", "arrestad",
            "operativo", "homicidio", "asesinato", "robo", "hurto", "narco", "violencia", "sucesos",
        },
        "economy": {
            "economia", "mercato", "finanza", "inflacion", "inflazione", "presupuesto",
            "negocios", "mercados", "finanzas", "empresa", "bolsa",
        },
        "politics": {
            "politica", "governo", "parlamento", "elezioni", "gobierno", "elecciones",
            "presidente", "ministro", "congreso", "senado", "decreto", "partido",
        },
        "default": set(),
    }
    topical_terms = topical_terms_map.get(topic, set())
    geography = _extract_query_geography(last_message) or ""
    geo_en = _GEO_ENGLISH.get(geography, geography)
    geo_terms = {
        _strip_accents(term.lower())
        for term in (geography, geo_en)
        if term
    }
    geo_norm = _strip_accents(geography.lower()) if geography else ""
    _city_map: dict[str, set[str]] = {
        "italia": {"roma", "milan", "milano", "napoli", "palermo", "torino", "firenze", "bologna", "genova", "venezia", "sicilia"},
        "espana": {"madrid", "barcelona", "valencia", "sevilla", "bilbao", "malaga", "zaragoza"},
        "argentina": {"buenos aires", "cordoba", "rosario", "mendoza", "tucuman", "salta"},
        "chile": {"santiago", "valparaiso", "concepcion"},
        "mexico": {"ciudad de mexico", "guadalajara", "monterrey", "puebla"},
    }
    italian_city_terms = _city_map.get(geo_norm, set())
    foreign_noise = {
        "islamabad", "iran", "pakistan", "gaza", "ucrania", "ukraine", "russia",
        "washington", "trump", "hormuz",
    }
    meta_phrases = (
        "estos titulares reflejan",
        "estos titulares destacan",
        "estos temas reflejan",
        "estas notas reflejan",
        "situacion actual",
        "temas relevantes",
        "abordando temas",
        "destacando eventos recientes",
        "temas de preocupacion",
        "en el ambito de",
        "en italia y en el extranjero",
    )
    filtered: list[str] = []
    for line in lines:
        normalized = _strip_accents(line.lower())
        if normalized.startswith(("aqui tienes", "el contenido proporcionado", "no se encontraron", "however,", "sin embargo", "lo siento")):
            continue
        if "puedes visitar el sitio web" in normalized:
            continue
        if any(phrase in normalized for phrase in meta_phrases):
            continue
        if _is_no_info_response(normalized):
            continue
        if topical_terms and not any(term in normalized for term in topical_terms):
            continue
        if foreign_noise and any(term in normalized for term in foreign_noise):
            continue
        if geo_terms and not any(term in normalized for term in geo_terms.union(italian_city_terms)):
            continue
        if normalized_terms and not any(term in normalized for term in normalized_terms.union(topical_terms)):
            continue
        filtered.append(line)
    return filtered


def _is_homepage_meta_line(line: str) -> bool:
    normalized = _strip_accents((line or "").lower())
    meta_patterns = (
        "estos titulares",
        "estos temas",
        "estas notas",
        "temas de preocupacion",
        "en el ambito de",
        "situacion actual",
        "temas relevantes",
        "actualidad del pais",
        "actualidad y la cronica",
        "en italia y en el extranjero",
    )
    if any(pattern in normalized for pattern in meta_patterns):
        return True
    if any(verb in normalized for verb in ("destacan", "reflejan", "abordan")) and any(
        token in normalized for token in ("temas", "titulares", "notas", "ambito", "actualidad")
    ):
        return True
    return False


def _is_concrete_homepage_line(line: str) -> bool:
    normalized = _strip_accents((line or "").lower()).strip()
    if not normalized or _is_homepage_meta_line(normalized):
        return False
    if _is_no_info_response(normalized):
        return False
    vague_buckets = (
        "seguridad y politica",
        "politica y seguridad",
        "cronica y seguridad",
        "actualidad del pais",
        "actualidad y la cronica",
        "en italia y en el extranjero",
    )
    if any(bucket in normalized for bucket in vague_buckets):
        return False
    if "se discute" in normalized and not re.search(r"\b\d+\b", normalized):
        return False
    if '"' in line or ":" in line:
        return True
    if re.search(r"\b\d+\b", normalized):
        return True
    if re.search(r"\b(?:roma|milano|napoli|palermo|torino|firenze|bologna|genova|venezia|sicilia)\b", normalized):
        return True
    if re.search(
        r"\b(?:detenido|detenida|murio|murieron|accidente|ataque|operativo|decreto|arresto|investigacion|polizia|policia|ciberataque|explosion|incendio|tribunal|condena|allanamiento|control(?:es)?|medida(?:s)?|refuerza|reporta|novedades)\b",
        normalized,
    ):
        return True
    return False


def _normalize_homepage_line(line: str) -> str:
    cleaned = (line or "").strip()
    cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _dedupe_homepage_lines(lines: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = _normalize_homepage_line(line)
        if not normalized:
            continue
        key = _strip_accents(normalized.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


# Alias para compatibilidad con referencias internas al módulo.
_COUNTRY_PRESS_SECTION_PATHS = COUNTRY_PRESS_SECTION_PATHS
_GENERIC_SECTION_PATHS = GENERIC_SECTION_PATHS
_build_country_press_section_targets = build_country_press_section_targets


_SECTION_LOCAL_LABELS = {
    "cronaca", "italia", "roma", "politica", "interni", "economia", "mercati",
    "seguridad", "policiales", "sociedad", "sucesos", "españa", "nacional",
    "política", "noticias", "actualidad", "último momento",
}


def _build_newspaper_homepage_fetch_prompt(last_message: str, press_name: str) -> str:
    topic = _detect_news_topic(last_message)
    geography = _extract_query_geography(last_message) or ""
    geo_line = f"País objetivo: {geography}. " if geography else ""
    topic_line = {
        "security": "Tema objetivo: seguridad, crimen, policía, ciberseguridad, migración, defensa.",
        "politics": "Tema objetivo: política, gobierno, parlamento, elecciones, decretos.",
        "economy": "Tema objetivo: economía, finanzas, inflación, mercado, empresas.",
    }.get(topic, "Tema objetivo: noticias y actualidad.")
    return (
        f"Leé la homepage del diario {press_name}. "
        f"{geo_line}{topic_line} "
        "Extraé SOLO titulares o notas concretas y recientes que respondan la consulta. "
        "Devolvé una línea por noticia, sin introducciones, sin resúmenes editoriales, sin frases meta, sin repetir líneas. "
        "Conservá nombres propios, ciudades, fechas, números y hechos verificables. "
        "No escribas frases como 'estos titulares destacan' o 'estos temas reflejan'. "
        "Si no hay noticias concretas relevantes, devolvé exactamente: 'No hay noticias concretas relevantes.'\n\n"
        f"Consulta original: {last_message}"
    )


def _build_newspaper_section_fetch_prompt(last_message: str, press_name: str, section_label: str) -> str:
    topic = _detect_news_topic(last_message)
    geography = _extract_query_geography(last_message) or ""
    geo_line = f"País objetivo: {geography}. " if geography else ""
    topic_line = {
        "security": "Tema objetivo: seguridad, crimen, policía, policiales, cronaca, ciberseguridad, migración, defensa.",
        "politics": "Tema objetivo: política, gobierno, parlamento, elecciones, decretos, coaliciones.",
        "economy": "Tema objetivo: economía, finanzas, inflación, mercado, empresas, presupuesto.",
    }.get(topic, "Tema objetivo: noticias y actualidad.")
    return (
        f"Leé la sección {section_label} del diario {press_name}. "
        f"{geo_line}{topic_line} "
        "Identificá TODAS las noticias distintas que encuentres en la sección (pueden ser 1, 2, 3 o más). "
        "Por cada noticia escribí UN PÁRRAFO separado. Cada párrafo debe tener entre 2 y 5 oraciones que expliquen "
        "claramente: qué ocurrió, quiénes están involucrados, cuándo y dónde. "
        "Usá el suficiente detalle para que alguien que no leyó la nota original entienda qué pasó. "
        "Separá CADA párrafo con UNA LÍNEA EN BLANCO (línea vacía entre párrafos). "
        "No escribas títulos, subtítulos, numeración ni introducciones editoriales antes de los párrafos. "
        "No uses frases como 'La noticia trata sobre...' o 'Este artículo informa...'. "
        "Arrancá cada párrafo directamente con el hecho: quién hizo qué. "
        "Preservá nombres propios, ciudades, fechas, números, cargos y datos verificables. "
        "Si la sección no tiene noticias concretas sobre el tema, devolvé exactamente: 'No hay noticias concretas relevantes.'\n\n"
        f"Consulta original: {last_message}"
    )




def _classify_fetch_error(fetch_text: str) -> Optional[str]:
    normalized = _strip_accents((fetch_text or "").lower())
    if "no module named 'playwright'" in normalized:
        return "missing_playwright"
    if "404 client error" in normalized or "not found" in normalized:
        return "not_found"
    if "403 client error" in normalized or "forbidden" in normalized:
        return "blocked"
    if "nameresolutionerror" in normalized or "failed to resolve" in normalized:
        return "dns"
    if normalized.startswith("error al procesar la pagina web:"):
        return "fetch_error"
    return None


def _build_angle_queries(last_message: str, search_age_days: Optional[int]) -> list[dict]:
    """Generates 4 angle-specific search queries for diverse news coverage."""
    import datetime
    geo = _extract_query_geography(last_message)
    topic = _detect_news_topic(last_message)
    year = datetime.date.today().year
    angles = _TOPIC_ANGLES.get(topic, _TOPIC_ANGLES["default"])
    base_geo = geo or " ".join(
        w for w in last_message.split()
        if len(w) > 3 and w.lower() not in _GENERIC_WEB_STOPWORDS
    )[:40]
    queries = []
    for template in angles:
        q = template.format(geo=base_geo, topic=topic, year=year)
        invoke_args: dict = {"query": q, "use_cache": False}
        if search_age_days is not None:
            invoke_args["max_age_days"] = search_age_days
        queries.append(invoke_args)
    return queries







def _extract_generic_content_lines(text: str, query_terms: list[str]) -> list[str]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    result: list[str] = []
    score_lines_seen = 0
    # Normalize query terms once so accent-bearing queries (e.g. "japon") match
    # article text that uses accented forms ("japón").
    normalized_terms = [_strip_accents(t) for t in (query_terms or [])]
    for idx, line in enumerate(lines):
        lower = line.lower()
        if lower.startswith(("url:", "sources:", "http")):
            continue
        if "http" in lower or "sources" in lower:
            continue
        # Tavily search-result headers ("Web search results for query: ...") are metadata,
        # not useful content — skip them so they don't inflate the body-lines count.
        if lower.startswith("web search results for query"):
            continue
        if len(line) < 3:
            continue
        if not re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ0-9]", line):
            continue
        # Document section headers from legal/academic documents (e.g. "C. Conclusion", "III. Analysis")
        if re.match(r"^(?:[IVXLC]+\.|[A-Z]\.|[1-9]\d?\.|[a-z]\))\s+[A-ZÁÉÍÓÚ]", line):
            continue
        # Meta-wrapper openers — the sentence summarizes what the page says rather than reporting an event.
        # e.g. "La información más reciente sobre X destaca aspectos clave:"
        #      "Las últimas noticias sobre X indican que los viajeros deben..."
        #      "Los últimos datos sobre X señalan que..."
        if re.match(
            r"^(?:la informaci[oó]n|las [uú]ltimas noticias|los [uú]ltimos datos|el [uú]ltimo informe)"
            r".{0,60}(?:destaca|indican?|se[nñ]alan?|muestra|revela|se centra|aborda|trata)",
            lower,
        ):
            continue
        # Mid-paragraph continuation sentences — start with a demonstrative pronoun
        # that refers to a prior sentence we don't have ("Esta situación", "Este problema",
        # "Esto demuestra", "Esa tendencia"). Without the antecedent they're meaningless as bullets.
        # Exclude temporal openers ("Esta semana", "Este año", "Este mes", "Este lunes") — those are valid.
        _TEMPORAL = (
            "semana", "año", "mes", "dia", "día", "lunes", "martes", "miércoles",
            "miercoles", "jueves", "viernes", "sabado", "sábado", "domingo",
            "mañana", "noche", "tarde", "trimestre", "periodo", "período",
        )
        if re.match(r"^(?:esta|este|esto|esa|ese|eso|dicha|dicho|tal)\s+\w+", lower):
            following_word = re.match(r"^(?:esta|este|esto|esa|ese|eso|dicha|dicho|tal)\s+(\w+)", lower)
            if following_word and following_word.group(1) not in _TEMPORAL:
                continue
        lower_norm = _strip_accents(lower)
        if query_terms:
            if any(term in lower_norm for term in normalized_terms):
                result.append(line)
                for look_ahead in range(1, 3):
                    if idx + look_ahead >= len(lines):
                        break
                    next_line = lines[idx + look_ahead].strip()
                    next_lower = next_line.lower()
                    if not next_line or next_lower.startswith(("url:", "sources:", "http")) or "http" in next_lower or "sources" in next_lower:
                        break
                    if not re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ0-9]", next_line):
                        break
                    result.append(next_line)
            elif re.search(r"\b\d+\s*-\s*\d+\b", line) and score_lines_seen == 0:
                result.append(line)
                score_lines_seen += 1
        else:
            result.append(line)
    return result


def _extract_section_content_lines(text: str, last_message: str, section_label: str) -> list[str]:
    if not text:
        return []
    issue = _classify_fetch_error(text)
    if issue or _is_no_info_response(text):
        return []
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    result: list[str] = []
    for line in lines:
        normalized = _strip_accents(line.lower())
        if normalized.startswith(("url:", "sources:", "http", "<<<cite_this:")):
            continue
        if "<<<cite_this:" in normalized:
            continue
        if normalized.startswith("error al procesar la pagina web"):
            continue
        cleaned = re.sub(r"^\s*(?:[-*•]\s+|\d+\.\s+)", "", line).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if len(cleaned) < 12:
            continue
        if not re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]", cleaned):
            continue
        if _is_homepage_meta_line(cleaned) or _is_no_info_response(cleaned):
            continue
        if re.match(r'^[\"“].+[\"”]$', cleaned) or re.match(r'^[\"“].+[\"”]\s*[-:]\s*.+$', cleaned):
            result.append(cleaned)
            continue
        if re.search(
            r"\b(?:"
            # italiano
            r"accoltell|omicid|arrest|morti|morto|uccis|esplos|condann|indagat|rapin|furto|blitz|nas|polizia|carabini|droga|sparator|violenza|tribunal|decreto|parlament|governo"
            r"|"
            # español
            r"detenid|arrestad|operativo|homicidio|asesin|robo|hurto|narco|policial|fiscal|juzgado|imputad|sentencia|condena|presidio|carcel|prision|ministro|presidente|congreso|senado|partido|eleccion|gobierno|decreto|presupuesto|inflacion|mercado|empresa|bolsa"
            r")\w*\b",
            normalized,
        ):
            result.append(cleaned)
            continue
        if '"' in cleaned or ":" in cleaned:
            result.append(cleaned)
            continue
    if result:
        return _dedupe_homepage_lines(result)
    compact = " ".join(lines)
    extracted = []
    for match in re.finditer(r'(?:^|\s)(?:[-*•]|\d+\.)\s+([^-\n].{12,220}?)(?=(?:\s(?:[-*•]|\d+\.)\s+)|$)', compact):
        cleaned = match.group(1).strip()
        if cleaned and not _is_homepage_meta_line(cleaned) and not _is_no_info_response(cleaned):
            extracted.append(cleaned)
    return _dedupe_homepage_lines(extracted)


def _filter_section_lines_for_query(lines: list[str], last_message: str, section_label: str) -> list[str]:
    if not lines:
        return []
    topic = _detect_news_topic(last_message)
    topical_terms_map = {
        "security": {
            "seguridad", "sicurezza", "crime", "crimen", "cronaca", "polizia", "policia", "ciber", "cyber",
            "difesa", "defensa", "migr", "attacco", "ataque", "omicid", "arrest", "esplos", "accoltell",
            "policiales", "detenid", "arrestad", "operativo", "homicidio", "asesinato", "robo", "narco",
            "violencia", "sucesos", "delito", "fiscal", "tribunal",
        },
        "economy": {
            "economia", "mercato", "finanza", "inflacion", "inflazione", "presupuesto",
            "negocios", "mercados", "finanzas", "empresa", "bolsa", "pib", "deuda",
        },
        "politics": {
            "politica", "governo", "parlamento", "elezioni", "coalizion", "decreto",
            "gobierno", "elecciones", "presidente", "ministro", "congreso", "senado", "partido",
        },
        "default": set(),
    }
    topical_terms = topical_terms_map.get(topic, set())
    geography = _extract_query_geography(last_message) or ""
    geography_normalized = _strip_accents(geography.lower()) if geography else ""
    _city_map: dict[str, set[str]] = {
        "italia": {"roma", "milan", "milano", "napoli", "palermo", "torino", "firenze", "bologna", "genova", "venezia", "sicilia"},
        "espana": {"madrid", "barcelona", "valencia", "sevilla", "bilbao", "malaga", "zaragoza"},
        "argentina": {"buenos aires", "cordoba", "rosario", "mendoza", "tucuman"},
        "chile": {"santiago", "valparaiso", "concepcion"},
        "mexico": {"ciudad de mexico", "guadalajara", "monterrey", "puebla"},
    }
    italian_city_terms = _city_map.get(geography_normalized, set())
    section_label_normalized = _strip_accents(section_label.lower())
    is_local_section = section_label_normalized in _SECTION_LOCAL_LABELS or "homepage" in section_label_normalized
    filtered: list[str] = []
    for line in lines:
        normalized = _strip_accents(line.lower())
        if _is_homepage_meta_line(normalized) or _is_no_info_response(normalized):
            continue
        if any(term in normalized for term in ("islamabad", "iran", "pakistan", "gaza", "ukraine", "ucrania", "russia", "washington", "trump")):
            continue
        if topic == "security":
            has_security_signal = any(term in normalized for term in topical_terms) or _is_concrete_homepage_line(line)
            if not has_security_signal:
                continue
        elif topical_terms and not any(term in normalized for term in topical_terms):
            continue
        if geography_normalized == "italia" and not is_local_section:
            if not any(term in normalized for term in italian_city_terms.union({"italia", "italy", "italiano", "italiana"})):
                continue
        filtered.append(line)
    return _dedupe_homepage_lines(filtered)



def _extract_sources_from_text(text: str) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen: set[str] = set()

    # Parse structured CITE_THIS markers: <<<CITE_THIS: title=...|url=...|domain=...>>>
    for match in re.finditer(r"<<<CITE_THIS:\s*title=([^|]+)\|url=([^|>]+)\|domain=([^|>]+)>>>", text or ""):
        article_title, url, domain = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
        if url and url not in seen:
            seen.add(url)
            sources.append({"title": article_title or domain, "url": url, "domain": domain, "snippet": ""})

    if sources:
        return sources

    # Fallback: parse standard markdown links [title](url)
    for title, url in re.findall(r"\[([^\]]+)\]\((https?://[^)]+)\)", text or ""):
        normalized = url.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        domain = urlparse(normalized).hostname or normalized
        sources.append({"title": title.strip() or normalized, "url": normalized, "domain": domain, "snippet": ""})

    if sources:
        return sources

    for url in _extract_urls_from_text(text):
        if url not in seen:
            seen.add(url)
            domain = urlparse(url).hostname or url
            sources.append({"title": url, "url": url, "domain": domain, "snippet": ""})
    return sources



def _extract_periodicos_directory_links(html: str, *, base_url: str) -> list[dict[str, str]]:
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return []

    soup = BeautifulSoup(html or "", "html.parser")
    links: list[dict[str, str]] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href") or "").strip()
        if not href:
            continue
        absolute = href if href.startswith("http") else f"{base_url.rstrip('/')}/{href.lstrip('/')}"
        absolute = re.sub(r"(?<!:)/{2,}", "/", absolute.replace(":/", "://"))
        title = " ".join(anchor.get_text(" ", strip=True).split())
        if absolute in seen:
            continue
        seen.add(absolute)
        links.append({"title": title, "url": absolute})
    return links


def _match_periodicos_directory_url(
    links: list[dict[str, str]],
    *,
    expected_slug: str,
    must_contain_slug: Optional[str] = None,
) -> Optional[str]:
    normalized_expected = _slugify_periodicos_label(expected_slug)
    normalized_must = _slugify_periodicos_label(must_contain_slug or "")

    for link in links:
        url = link.get("url", "").strip()
        title = link.get("title", "").strip()
        if "periodicos.com.ar/periodicos/" not in url:
            continue
        normalized_url = _slugify_periodicos_label(urlparse(url).path)
        normalized_title = _slugify_periodicos_label(title)
        haystack = f"{normalized_url} {normalized_title}".strip()
        if normalized_expected and normalized_expected not in haystack:
            continue
        if normalized_must and normalized_must not in normalized_url:
            continue
        return url
    return None


async def _discover_country_press_sources_via_directory(
    geography: str,
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    continent_slug = _PERIODICOS_CONTINENT_SLUG_BY_COUNTRY.get(geography)
    if not continent_slug:
        _web_debug("country_press.directory.skip", geography=geography, reason="missing_continent_slug")
        return [], [], []

    country_slug = _slugify_periodicos_label(geography)
    directory_root_url = "https://periodicos.com.ar/periodicos/"
    current_url = directory_root_url
    current_stage = "root"

    try:
        root_html = _debug_periodicos_fetch(directory_root_url, stage=current_stage).decode("utf-8", errors="ignore")
        root_links = _extract_periodicos_directory_links(root_html, base_url="https://periodicos.com.ar")
        continent_url = _match_periodicos_directory_url(root_links, expected_slug=continent_slug)
        if not continent_url:
            continent_url = f"{directory_root_url}{continent_slug}/"

        current_url = continent_url
        current_stage = "continent"
        continent_html = _debug_periodicos_fetch(continent_url, stage=current_stage).decode("utf-8", errors="ignore")
        continent_links = _extract_periodicos_directory_links(continent_html, base_url="https://periodicos.com.ar")
        country_url = _match_periodicos_directory_url(
            continent_links,
            expected_slug=country_slug,
            must_contain_slug=continent_slug,
        )
        if not country_url:
            country_url = f"{continent_url.rstrip('/')}/{country_slug}/"

        current_url = country_url
        current_stage = "country"
        country_html = _debug_periodicos_fetch(country_url, stage=current_stage).decode("utf-8", errors="ignore")
    except Exception as exc:
        _web_debug(
            "country_press.directory.exception",
            geography=geography,
            stage=current_stage,
            url=current_url,
            error=repr(exc),
        )
        return [], [], []

    country_links = _extract_periodicos_directory_links(country_html, base_url="https://periodicos.com.ar")
    sources: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for source in country_links:
        url = source.get("url", "").strip()
        if not url or "periodicos.com.ar" in url:
            continue
        hostname = (urlparse(url).hostname or "").lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        if not hostname or url in seen_urls:
            continue
        seen_urls.add(url)
        sources.append({
            "title": source.get("title") or hostname,
            "url": url,
            "domain": hostname,
        })

    domains: list[str] = []
    titles: list[str] = []
    seen_domains: set[str] = set()
    for source in sources:
        hostname = source.get("domain", "")
        if hostname and hostname not in seen_domains:
            seen_domains.add(hostname)
            domains.append(hostname)
            titles.append(source.get("title") or hostname)
        if len(domains) >= 10:
            break

    _web_debug(
        "country_press.directory.result",
        geography=geography,
        continent_slug=continent_slug,
        country_slug=country_slug,
        source_count=len(sources),
        domains=domains,
    )
    return domains, titles, sources


def _extract_country_press_sources(text: str) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for source in _extract_sources_from_text(text):
        url = source.get("url", "")
        if not url:
            continue
        if "periodicos.com.ar" in url:
            continue
        if url in seen:
            continue
        seen.add(url)
        sources.append(source)
    return sources


async def _discover_country_press_sources(
    last_message: str,
    query_source_group: Optional[str],
    source_terms: list[str],
    web_search_runtime_args: Optional[dict[str, Any]] = None,
) -> tuple[list[str], list[str]]:
    if not query_source_group or not source_terms:
        _web_debug("country_press.discovery.skip", query_source_group=query_source_group, source_terms=source_terms)
        _country_press_strategy_cache_set(query_source_group, source_terms, "none")
        return [], []

    cached = _country_press_cache_get(query_source_group, source_terms)
    if cached is not None:
        _country_press_strategy_cache_set(query_source_group, source_terms, "cache")
        _web_debug("country_press.discovery.cache_hit", query_source_group=query_source_group, source_terms=source_terms, domains=cached[0], titles=cached[1])
        _web_debug("country_press.discovery.local_strategy_selected", query_source_group=query_source_group, strategy="cache", domains=cached[0])
        return cached

    from tools import search_web
    from tools.web_tools import fetch_web_page

    geography = _extract_query_geography(last_message)
    if geography:
        directory_domains, directory_titles, directory_sources = await _discover_country_press_sources_via_directory(geography)
        if directory_domains:
            _country_press_source_cache_set(query_source_group, source_terms, directory_sources)
            _country_press_cache_set(query_source_group, source_terms, directory_domains, directory_titles)
            _country_press_strategy_cache_set(query_source_group, source_terms, "directory")
            _web_debug("country_press.discovery.local_strategy_selected", query_source_group=query_source_group, strategy="directory", domains=directory_domains)
            return directory_domains, directory_titles

    lookup_terms = [term for term in source_terms if len(term) >= 3][:4]
    if not lookup_terms:
        lookup_terms = [query_source_group]

    lookup_query = " ".join([
        'site:periodicos.com.ar',
        *lookup_terms,
        "periódicos",
        "diarios",
        "medios",
    ])
    lookup_args: dict[str, Any] = {
        "query": lookup_query,
        "use_cache": False,
        "allowed_domains": ["periodicos.com.ar"],
        "num_results": 5,
    }
    if web_search_runtime_args:
        lookup_args["blocked_domains"] = web_search_runtime_args.get("blocked_domains") or None

    lookup_text = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: search_web.invoke(lookup_args),
    )
    if not isinstance(lookup_text, str):
        lookup_text = str(lookup_text)
    _web_debug(
        "country_press.discovery.lookup",
        query=lookup_query,
        lookup_args=lookup_args,
        lookup_preview=lookup_text[:500],
    )

    directory_urls = [
        source.get("url", "")
        for source in _extract_sources_from_text(lookup_text)
        if "periodicos.com.ar" in (source.get("url") or "")
    ]

    discovered_sources: list[dict[str, str]] = _extract_country_press_sources(lookup_text)
    seen_urls = {source.get("url", "") for source in discovered_sources if source.get("url")}

    if not discovered_sources:
        try:
            homepage = await fetch_web_page(
                url="https://periodicos.com.ar/",
                prompt=(
                    "Extraé únicamente la lista de periódicos, diarios y medios del país solicitado, "
                    "con sus nombres y enlaces si están disponibles."
                ),
                use_dynamic=False,
            )
        except Exception:
            homepage = ""
        if not isinstance(homepage, str):
            homepage = str(homepage)
        homepage_sources = _extract_country_press_sources(homepage)
        if homepage_sources:
            match_terms = [term.lower() for term in lookup_terms + [query_source_group or ""] if term]
            if match_terms:
                filtered_homepage_sources = [
                    source for source in homepage_sources
                    if any(
                        term in f"{(source.get('title') or '').lower()} {(source.get('url') or '').lower()}"
                        for term in match_terms
                    )
                ]
                if filtered_homepage_sources:
                    homepage_sources = filtered_homepage_sources
            for source in homepage_sources:
                url = source.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    discovered_sources.append(source)

    if len(discovered_sources) >= 2:
        directory_urls = []

    for directory_url in directory_urls[:2]:
        try:
            fetched = await fetch_web_page(
                url=directory_url,
                prompt=(
                    "Extraé únicamente la lista de periódicos, diarios y medios del país solicitado, "
                    "con sus nombres y enlaces si están disponibles."
                ),
                use_dynamic=False,
            )
        except Exception:
            continue
        if not isinstance(fetched, str):
            fetched = str(fetched)
        for source in _extract_country_press_sources(fetched):
            url = source.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                discovered_sources.append(source)

    domains: list[str] = []
    titles: list[str] = []
    seen_domains: set[str] = set()
    seen_titles: set[str] = set()
    for source in discovered_sources:
        url = source.get("url", "")
        title = (source.get("title") or "").strip()
        hostname = (urlparse(url).hostname or "").lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        if hostname and hostname not in seen_domains:
            seen_domains.add(hostname)
            domains.append(hostname)
        if title and title not in seen_titles:
            seen_titles.add(title)
            titles.append(title)

    domains = domains[:10]
    titles = titles[:10]
    if not domains:
        policy_sources = _build_policy_country_press_sources(query_source_group)
        if policy_sources:
            for source in policy_sources:
                hostname = source.get("domain", "").strip().lower()
                title = (source.get("title") or hostname).strip()
                if hostname and hostname not in seen_domains:
                    seen_domains.add(hostname)
                    domains.append(hostname)
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    titles.append(title)
                if source.get("url") and source.get("url") not in seen_urls:
                    seen_urls.add(source["url"])
                    discovered_sources.append(source)
            domains = domains[:10]
            titles = titles[:10]
            _web_debug(
                "country_press.discovery.policy_fallback",
                query_source_group=query_source_group,
                geography=geography,
                domains=domains,
            )
    _web_debug(
        "country_press.discovery.result",
        query_source_group=query_source_group,
        source_terms=source_terms,
        domains=domains,
        titles=titles,
        discovered_count=len(discovered_sources),
    )
    _country_press_source_cache_set(query_source_group, source_terms, discovered_sources)
    _country_press_cache_set(query_source_group, source_terms, domains, titles)
    _country_press_strategy_cache_set(query_source_group, source_terms, "lookup" if domains else "none")
    if domains:
        _web_debug("country_press.discovery.local_strategy_selected", query_source_group=query_source_group, strategy="lookup", domains=domains)
    return domains, titles


async def _run_country_press_search_candidates(
    last_message: str,
    search_age_days: Optional[int],
    query_terms: list[str],
    query_source_group: Optional[str],
    source_terms: list[str],
    web_search_runtime_args: Optional[dict[str, Any]] = None,
    query_horizon: Optional[str] = None,
) -> tuple[list[dict[str, str]], str]:
    from tools import search_web
    from tools.web_tools import fetch_web_page, _is_specific_article_hit

    country_press_domains, country_press_names = await _discover_country_press_sources(
        last_message,
        query_source_group,
        source_terms,
        web_search_runtime_args,
    )
    if not country_press_domains:
        _web_debug(
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
    country_press_sources = _country_press_source_cache_get(query_source_group, source_terms)
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
        source_meta = sources_by_domain.get(domain, {"title": press_name, "url": _default_press_homepage_url(domain)})
        if not _is_press_source_relevant_for_query(source_meta, last_message):
            _web_debug("country_press.search.source_skipped", domain=domain, press_name=press_name, reason="irrelevant_for_query")
            continue
        relevant_targets.append((domain, press_name))
        if len(relevant_targets) >= 8:
            break

    for domain, press_name in relevant_targets:
        all_diary_candidates: list[dict[str, str]] = []
        domain_search_texts: list[str] = []
        queries = _build_country_press_search_queries(last_message, domain, press_name)
        for query in queries:
            query_attempts = [
                ("news", _build_country_press_search_invoke_args(
                    query,
                    domain,
                    search_age_days=search_age_days,
                    query_horizon=query_horizon,
                    web_search_runtime_args=web_search_runtime_args,
                    broad=False,
                )),
                ("general", _build_country_press_search_invoke_args(
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
                    search_text = await loop.run_in_executor(
                        None,
                        lambda q=invoke_args: search_web.invoke(q),
                    )
                except Exception:
                    _web_debug("country_press.search.exception", domain=domain, query=query, attempt=attempt_label)
                    continue
                if not isinstance(search_text, str):
                    search_text = str(search_text)
                domain_search_texts.append(search_text)
                combined_search_text.append(search_text)
                diary_candidates = [
                    c for c in _extract_generic_search_candidates(search_text)
                    if not _is_non_news_candidate(c)
                ]
                all_diary_candidates.extend(diary_candidates)
                article_candidates = [c for c in diary_candidates if _is_specific_article_hit(c)]
                _web_debug(
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

        diary_candidates = _dedup_candidates_by_event(all_diary_candidates, query_terms) if all_diary_candidates else []
        diary_candidates = [c for c in diary_candidates if not _is_invalid_news_candidate(c, last_message)]
        if query_horizon == "week":
            url_age_threshold = search_age_days or 14
            recent_diary_candidates = [
                c for c in diary_candidates
                if _candidate_url_is_recent(c.get("url", ""), url_age_threshold)
            ]
            strict_recent_candidates = [
                c for c in recent_diary_candidates
                if _candidate_url_has_date(c.get("url", "")) or not _is_invalid_news_candidate(c, last_message)
            ]
            article_recent_candidates = [
                c for c in strict_recent_candidates
                if _is_specific_article_hit(c) and not _is_invalid_news_candidate(c, last_message)
            ]
            _web_debug(
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
                diary_candidates = [c for c in strict_recent_candidates if not _is_invalid_news_candidate(c, last_message)]
        article_candidates = [c for c in diary_candidates if _is_specific_article_hit(c)]
        if article_candidates:
            raw_candidates.extend(article_candidates)
        else:
            raw_candidates.extend(diary_candidates)
            combined_domain_text = "\n".join(domain_search_texts)
            if not diary_candidates or combined_domain_text.startswith("Error en búsqueda:") or "No results found." in combined_domain_text:
                fallback_source = sources_by_domain.get(
                    domain,
                    {"title": press_name, "url": _default_press_homepage_url(domain)},
                )
                fallback_url = (fallback_source or {}).get("url", "").strip()
                if fallback_url:
                    homepage_prompt = _build_newspaper_homepage_fetch_prompt(last_message, fallback_source.get("title") or domain)
                    try:
                        fetched_home = await fetch_web_page(
                            url=fallback_url,
                            prompt=homepage_prompt,
                            use_dynamic=False,
                        )
                    except Exception:
                        fetched_home = ""
                    if not isinstance(fetched_home, str):
                        fetched_home = str(fetched_home)
                    homepage_lines = _filter_homepage_lines_for_query(
                        _extract_generic_content_lines(fetched_home, query_terms),
                        last_message,
                        query_terms,
                    )
                    homepage_lines = _dedupe_homepage_lines(homepage_lines)
                    if not homepage_lines and dynamic_fetch_available is not False:
                        try:
                            _web_debug(
                                "country_press.search.homepage_retry_dynamic",
                                domain=domain,
                                fallback_url=fallback_url,
                            )
                            fetched_home_dynamic = await fetch_web_page(
                                url=fallback_url,
                                prompt=homepage_prompt,
                                use_dynamic=True,
                            )
                        except Exception:
                            fetched_home_dynamic = ""
                        if not isinstance(fetched_home_dynamic, str):
                            fetched_home_dynamic = str(fetched_home_dynamic)
                        dynamic_issue = _classify_fetch_error(fetched_home_dynamic)
                        if dynamic_issue == "missing_playwright":
                            dynamic_fetch_available = False
                            _web_debug("country_press.search.dynamic_unavailable", reason="missing_playwright", domain=domain)
                        homepage_lines = _filter_homepage_lines_for_query(
                            _extract_generic_content_lines(fetched_home_dynamic, query_terms),
                            last_message,
                            query_terms,
                        )
                        homepage_lines = _dedupe_homepage_lines(homepage_lines)
                    section_candidates: list[dict[str, str]] = []
                    for section_url, section_label in _build_country_press_section_targets(domain, fallback_url, last_message):
                        section_prompt = _build_newspaper_section_fetch_prompt(
                            last_message,
                            fallback_source.get("title") or domain,
                            section_label,
                        )
                        _web_debug(
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
                            _web_debug(
                                "country_press.search.section_fetch_result",
                                domain=domain,
                                section_label=section_label,
                                section_url=section_url,
                                use_dynamic=use_dynamic,
                                content_length=len(fetched_section or ""),
                                preview=(fetched_section or "")[:500],
                            )
                            fetch_issue = _classify_fetch_error(fetched_section)
                            if fetch_issue == "missing_playwright":
                                dynamic_fetch_available = False
                                _web_debug("country_press.search.dynamic_unavailable", reason="missing_playwright", domain=domain)
                            elif fetch_issue in {"not_found", "dns", "fetch_error"}:
                                _web_debug(
                                    "country_press.search.section_unavailable",
                                    domain=domain,
                                    section_label=section_label,
                                    section_url=section_url,
                                    use_dynamic=use_dynamic,
                                    reason=fetch_issue,
                                )
                            elif fetch_issue == "blocked":
                                _web_debug(
                                    "country_press.search.section_blocked",
                                    domain=domain,
                                    section_label=section_label,
                                    section_url=section_url,
                                    use_dynamic=use_dynamic,
                                    reason=fetch_issue,
                                )
                            extracted_section_lines = _extract_section_content_lines(
                                fetched_section,
                                last_message,
                                section_label,
                            )
                            _web_debug(
                                "country_press.search.section_lines_extracted",
                                domain=domain,
                                section_label=section_label,
                                section_url=section_url,
                                use_dynamic=use_dynamic,
                                raw_line_count=len(extracted_section_lines),
                                raw_lines=extracted_section_lines[:5],
                            )
                            section_lines = _filter_section_lines_for_query(
                                extracted_section_lines,
                                last_message,
                                section_label,
                            )
                            section_lines = _dedupe_homepage_lines(section_lines)
                            _web_debug(
                                "country_press.search.section_lines_filtered",
                                domain=domain,
                                section_label=section_label,
                                section_url=section_url,
                                use_dynamic=use_dynamic,
                                filtered_count=len(section_lines),
                                filtered_lines=section_lines[:5],
                            )
                            if not fetched_section.strip():
                                _web_debug(
                                    "country_press.search.section_empty",
                                    domain=domain,
                                    section_label=section_label,
                                    section_url=section_url,
                                    use_dynamic=use_dynamic,
                                    reason="empty_fetch",
                                )
                            elif _is_no_info_response(fetched_section):
                                _web_debug(
                                    "country_press.search.section_empty",
                                    domain=domain,
                                    section_label=section_label,
                                    section_url=section_url,
                                    use_dynamic=use_dynamic,
                                    reason="no_info_response",
                                )
                            if any(_is_homepage_meta_line(line) for line in section_lines):
                                _web_debug(
                                    "country_press.search.section_rejected_meta",
                                    domain=domain,
                                    section_label=section_label,
                                    section_url=section_url,
                                    use_dynamic=use_dynamic,
                                    section_lines=section_lines[:5],
                                )
                                section_lines = []
                            else:
                                section_lines = [line for line in section_lines if _is_concrete_homepage_line(line)]
                                if extracted_section_lines and not section_lines:
                                    _web_debug(
                                        "country_press.search.section_rejected_non_concrete",
                                        domain=domain,
                                        section_label=section_label,
                                        section_url=section_url,
                                        use_dynamic=use_dynamic,
                                        raw_lines=extracted_section_lines[:5],
                                    )
                            if section_lines:
                                _web_debug(
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
                        if any(_is_homepage_meta_line(line) for line in homepage_lines):
                            _web_debug(
                                "country_press.search.homepage_rejected_meta",
                                domain=domain,
                                fallback_url=fallback_url,
                                homepage_lines=homepage_lines[:5],
                            )
                            homepage_lines = []
                        else:
                            homepage_lines = [line for line in homepage_lines if _is_concrete_homepage_line(line)]
                            if not homepage_lines:
                                _web_debug(
                                    "country_press.search.homepage_rejected_non_concrete",
                                    domain=domain,
                                    fallback_url=fallback_url,
                                )
                    if not section_candidates and homepage_lines:
                        _web_debug(
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

    ranked_candidates = _rank_candidates_by_source_policy(raw_candidates, query_terms, query_source_group)
    ranked_candidates = [
        c for c in ranked_candidates
        if (
            c.get("source_kind") in {"homepage_fallback", "section_fallback"}
            or not _is_invalid_news_candidate(c, last_message)
        )
    ]
    diverse_candidates = _dedup_candidates_by_event(ranked_candidates, query_terms)[:8]
    _web_debug(
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


def _build_generic_fetch_prompt(query: str) -> str:
    geography = _extract_query_geography(query)
    geography_line = f"Contexto geográfico: {geography}. " if geography else ""
    return (
        "Extraé únicamente la información relevante para responder la consulta del usuario. "
        f"{geography_line}"
        "Respondé con 4 párrafos breves sobre el mismo tema solicitado si la consulta es de noticias/actualidad, y con 3-5 viñetas solo si el contenido lo pide. "
        "Incluí el contexto inmediato de la noticia y por qué importa, sin inventar datos. "
        "Si la página mezcla varios temas, otros países o fuentes, devolvé solo la sección pertinente a la consulta. "
        "Si no hay datos claros, decilo.\n\n"
        f"Consulta: {query}"
    )


async def _run_week_search_candidates(
    last_message: str,
    search_age_days: Optional[int],
    query_terms: list[str],
    query_source_group: Optional[str],
    web_search_runtime_args: Optional[dict[str, Any]] = None,
) -> tuple[list[CandidateDict], str]:
    """Runs the generic OpenClaw-style web search path.

    The search provider decides the result set; this helper only normalizes, ranks,
    and deduplicates the returned hits.
    """
    source_terms = list(get_query_source_terms(last_message))
    country_press_candidates, country_press_search_text = await _run_country_press_search_candidates(
        last_message,
        search_age_days,
        query_terms,
        query_source_group,
        source_terms,
        web_search_runtime_args,
        query_horizon="week",
    )
    if country_press_candidates:
        url_age_threshold = search_age_days or 14
        filtered_candidates = [
            c for c in country_press_candidates
            if _candidate_url_is_recent(c.get("url", ""), url_age_threshold)
        ]
        _web_debug(
            "week_search.country_press",
            candidate_count=len(country_press_candidates),
            filtered_candidate_count=len(filtered_candidates),
            url_age_threshold=url_age_threshold,
            urls=[c.get("url", "") for c in filtered_candidates[:8]],
        )
        if filtered_candidates:
            return filtered_candidates[:8], country_press_search_text

    local_source_strategy = _country_press_strategy_cache_get(query_source_group, source_terms)
    if query_source_group and local_source_strategy in {"cache", "directory", "policy", "lookup"}:
        _web_debug(
            "week_search.global_skipped_no_local_sources",
            query=last_message,
            query_source_group=query_source_group,
            local_source_strategy=local_source_strategy,
        )
        return [], country_press_search_text

    from tools import search_web

    loop = asyncio.get_running_loop()
    search_invoke_args: dict = {"query": last_message, "use_cache": False, **(web_search_runtime_args or {})}
    if search_age_days is not None:
        search_invoke_args["max_age_days"] = search_age_days
    search_invoke_args["topic"] = "news"
    search_invoke_args["time_range"] = "week"

    search_text = await loop.run_in_executor(None, lambda: search_web.invoke(search_invoke_args))
    if not isinstance(search_text, str):
        search_text = str(search_text)

    url_age_threshold = search_age_days or 14
    candidates = [
        c for c in _extract_generic_search_candidates(search_text)
        if not _is_non_news_candidate(c)
        and _candidate_url_is_recent(c.get("url", ""), url_age_threshold)
        and not _is_invalid_news_candidate(c, last_message)
    ]
    ranked_candidates = _rank_candidates_by_source_policy(candidates, query_terms, query_source_group)
    diverse_candidates = _dedup_candidates_by_event(ranked_candidates, query_terms)[:8]
    _web_debug(
        "week_search.generic",
        invoke_args=search_invoke_args,
        url_age_threshold=url_age_threshold,
        extracted_candidate_count=len(candidates),
        ranked_candidate_count=len(ranked_candidates),
        diverse_candidate_count=len(diverse_candidates),
        search_preview=search_text[:500],
        diverse_urls=[c.get("url", "") for c in diverse_candidates],
    )

    return diverse_candidates, search_text


async def _fetch_web_page_follow_redirect(url: str, prompt: str, *, use_dynamic: bool = True) -> str:
    from tools.web_tools import fetch_web_page

    result = await fetch_web_page(url=url, prompt=prompt, use_dynamic=use_dynamic)
    if not isinstance(result, str):
        result = str(result)

    redirect_url = _extract_web_fetch_redirect_url(result)
    if redirect_url and redirect_url != url:
        redirected = await fetch_web_page(url=redirect_url, prompt=prompt, use_dynamic=use_dynamic)
        return redirected if isinstance(redirected, str) else str(redirected)
    return result


def _build_query_context(last_message: str) -> tuple[QueryContext, RecentPolicy]:
    query_terms = _extract_generic_query_terms(last_message)
    query_source_group = detect_query_source_group(last_message)
    source_terms = list(get_query_source_terms(last_message))
    if source_terms:
        merged: list[str] = []
        for term in query_terms + source_terms:
            if term not in merged:
                merged.append(term)
        query_terms = merged

    query_horizon = (
        detect_recent_query_horizon(last_message)
        if _is_recent_web_information_query(last_message)
        else None
    )
    reqs = get_recent_query_requirements(query_horizon)

    search_age_days: Optional[int] = None
    if query_horizon == "week":
        search_age_days = 14
    elif query_horizon == "month":
        search_age_days = 45
    elif _is_recent_web_information_query(last_message):
        search_age_days = 30

    ctx = QueryContext(
        query_terms=query_terms,
        query_source_group=query_source_group,
        source_terms=source_terms,
        query_horizon=query_horizon,
        search_age_days=search_age_days,
    )
    policy = RecentPolicy(
        min_score=reqs["min_score"],
        min_body_lines=reqs["min_body_lines"],
        min_sources=reqs["min_sources"],
        min_candidates=reqs["min_candidates"],
        candidate_min_body_lines=1 if query_horizon == "week" else reqs["min_body_lines"],
        candidate_min_sources=reqs["candidate_min_sources"] or 1,
    )
    return ctx, policy


async def _fetch_and_score_entries(
    ranked_candidates: list[CandidateDict],
    last_message: str,
    ctx: QueryContext,
    policy: RecentPolicy,
    search_text: str,
) -> list[dict[str, Any]]:
    fetch_prompt = _build_generic_fetch_prompt(last_message)
    query_terms = ctx.query_terms
    query_source_group = ctx.query_source_group

    async def _fetch_candidate(candidate: CandidateDict) -> tuple[CandidateDict, Any]:
        try:
            # use_dynamic=False: requests HTTP es suficiente para artículos de noticias
            # y evita lanzar N browsers Chromium en paralelo (que es lo que causaba el timeout)
            result = await _fetch_web_page_follow_redirect(candidate["url"], fetch_prompt, use_dynamic=False)
            return candidate, result
        except Exception as exc:  # pragma: no cover - defensive
            return candidate, exc

    fetched_results = await asyncio.gather(
        *(_fetch_candidate(candidate) for candidate in ranked_candidates),
        return_exceptions=False,
    )

    eligible_entries: list[dict[str, Any]] = []
    for candidate, result in fetched_results:
        if _is_invalid_news_candidate(candidate, last_message):
            _web_debug(
                "generic_fetch.entry_rejected_invalid_structure",
                url=candidate.get("url", ""),
                title=candidate.get("title", ""),
            )
            continue
        if isinstance(result, Exception):
            _web_debug("generic_fetch.fetch_exception", url=candidate.get("url", ""), error=repr(result))
            snippet_lines = _candidate_snippet_lines(candidate)
            if not snippet_lines:
                continue
            result = "\n".join(snippet_lines)
        if not isinstance(result, str):
            result = str(result)
        if result.startswith("Error") or result.startswith("URL rechazada") or _is_no_info_response(result):
            _web_debug(
                "generic_fetch.fetch_bad_result",
                url=candidate.get("url", ""),
                result_preview=result[:300],
            )
            snippet_lines = _candidate_snippet_lines(candidate)
            if not snippet_lines:
                continue
            result = "\n".join(snippet_lines)

        body_lines = _extract_generic_content_lines(result, query_terms)
        candidate_score = _score_generic_candidate(candidate, query_terms, query_source_group)
        content_score = len(body_lines) * 2
        if query_terms and not body_lines:
            result_blob = result.lower()
            if any(term in result_blob for term in query_terms):
                content_score = 1
        if not body_lines and content_score <= 1:
            snippet_lines = _candidate_snippet_lines(candidate)
            if snippet_lines:
                body_lines = snippet_lines
                content_score = 3
            else:
                continue

        score = candidate_score + content_score
        if score <= 0:
            continue
        if _is_recent_web_information_query(last_message):
            if score < policy.min_score or len(body_lines) < policy.candidate_min_body_lines:
                _web_debug(
                    "generic_fetch.entry_rejected_recent_threshold",
                    url=candidate.get("url", ""),
                    score=score,
                    min_score=policy.min_score,
                    body_lines_count=len(body_lines),
                    min_body_lines=policy.candidate_min_body_lines,
                )
                continue

        fallback_lines = [line.strip() for line in result.splitlines() if line.strip() and not line.strip().lower().startswith(("url:", "sources:", "http")) and "http" not in line.lower()]
        summary_lines = body_lines or _extract_generic_content_lines(search_text, query_terms) or fallback_lines[:5]
        if len(summary_lines) < 3 and fallback_lines:
            seen_lines = set(summary_lines)
            for line in fallback_lines:
                if line not in seen_lines:
                    summary_lines.append(line)
                    seen_lines.add(line)
                if len(summary_lines) >= 6:
                    break
        sources = _extract_sources_from_text(result)
        if not sources:
            sources = [{"title": candidate.get("title") or candidate["url"], "url": candidate["url"]}]
        if _is_recent_web_information_query(last_message) and len(sources) < policy.candidate_min_sources:
            _web_debug(
                "generic_fetch.entry_rejected_sources",
                url=candidate.get("url", ""),
                source_count=len(sources),
                min_sources=policy.candidate_min_sources,
            )
            continue

        eligible_entries.append({
            "summary_lines": summary_lines[:10],
            "sources": sources,
            "score": score,
            "candidate": candidate,
        })

    _web_debug(
        "generic_fetch.eligible_entries",
        eligible_count=len(eligible_entries),
        urls=[entry["candidate"].get("url", "") for entry in eligible_entries],
    )
    return eligible_entries


async def _run_week_search_pipeline(
    last_message: str,
    ctx: QueryContext,
    web_search_runtime_args: Optional[dict[str, Any]],
) -> tuple[list[CandidateDict], str, Optional[dict[str, Any]]]:
    """Returns (diverse_candidates, search_text, early_response). Caller returns early_response immediately if not None."""
    diverse_candidates, search_text = await _run_week_search_candidates(
        last_message, ctx.search_age_days, ctx.query_terms, ctx.query_source_group, web_search_runtime_args
    )
    local_source_strategy = _country_press_strategy_cache_get(ctx.query_source_group, ctx.source_terms)
    _web_debug(
        "generic_fetch.week_candidates",
        query=last_message,
        search_age_days=ctx.search_age_days,
        candidate_count=len(diverse_candidates),
        local_source_strategy=local_source_strategy,
        urls=[c.get("url", "") for c in diverse_candidates],
    )
    if ctx.query_source_group and not diverse_candidates and local_source_strategy in {"cache", "directory", "policy", "lookup"}:
        return [], search_text, _build_no_local_sources_response(last_message)
    return diverse_candidates, search_text, None


async def _run_general_search_pipeline(
    last_message: str,
    ctx: QueryContext,
    loop: asyncio.AbstractEventLoop,
    web_search_runtime_args: Optional[dict[str, Any]],
) -> tuple[list[CandidateDict], str]:
    from tools import search_web

    query_terms = ctx.query_terms
    query_source_group = ctx.query_source_group
    source_terms = ctx.source_terms
    query_horizon = ctx.query_horizon
    search_age_days = ctx.search_age_days

    country_press_domains, country_press_names = await _discover_country_press_sources(
        last_message,
        query_source_group,
        source_terms,
        web_search_runtime_args,
    )
    search_invoke_args: dict = {"query": last_message, "use_cache": False, **(web_search_runtime_args or {})}
    if search_age_days is not None:
        search_invoke_args["max_age_days"] = search_age_days
    if _is_recent_web_information_query(last_message):
        search_invoke_args["topic"] = "news"
        if query_horizon == "today":
            search_invoke_args["time_range"] = "day"
    if country_press_domains and not search_invoke_args.get("allowed_domains"):
        search_invoke_args["allowed_domains"] = country_press_domains
    if country_press_names:
        search_invoke_args["query"] = f"{last_message} {' '.join(country_press_names[:4])}".strip()

    search_text = await loop.run_in_executor(
        None,
        lambda: search_web.invoke(search_invoke_args),
    )
    if not isinstance(search_text, str):
        search_text = str(search_text)

    candidates = _extract_generic_search_candidates(search_text)
    ranked_candidates = _rank_candidates_by_source_policy(
        [
            c for c in candidates
            if not _is_non_news_candidate(c)
            and not _is_invalid_news_candidate(c, last_message)
        ],
        query_terms,
        query_source_group,
    )[:8]

    diverse_candidates = _dedup_candidates_by_event(ranked_candidates, query_terms)
    _web_debug(
        "generic_fetch.initial_search",
        query=last_message,
        invoke_args=search_invoke_args,
        candidate_count=len(candidates),
        ranked_candidate_count=len(ranked_candidates),
        diverse_candidate_count=len(diverse_candidates),
        search_preview=search_text[:500],
    )

    if len(diverse_candidates) < 3:
        alt_invoke_args: dict = {"query": last_message + " últimas noticias recientes", "use_cache": False, **(web_search_runtime_args or {})}
        if search_age_days is not None:
            alt_invoke_args["max_age_days"] = search_age_days
        if _is_recent_web_information_query(last_message):
            alt_invoke_args["topic"] = "news"
            if query_horizon == "today":
                alt_invoke_args["time_range"] = "day"
        if country_press_domains and not alt_invoke_args.get("allowed_domains"):
            alt_invoke_args["allowed_domains"] = country_press_domains
        if country_press_names:
            alt_invoke_args["query"] = f"{last_message} últimas noticias recientes {' '.join(country_press_names[:4])}".strip()
        alt_search_text = await loop.run_in_executor(
            None,
            lambda q=alt_invoke_args: search_web.invoke(q),
        )
        if not isinstance(alt_search_text, str):
            alt_search_text = str(alt_search_text)
        alt_candidates = [
            c for c in _extract_generic_search_candidates(alt_search_text)
            if not _is_non_news_candidate(c)
            and not _is_invalid_news_candidate(c, last_message)
        ]
        for c in _rank_candidates_by_source_policy(alt_candidates, query_terms, query_source_group):
            if len(diverse_candidates) >= 4:
                break
            if not any(_same_event(c, d, query_terms) for d in diverse_candidates):
                diverse_candidates.append(c)
        search_text = search_text + "\n" + alt_search_text
        _web_debug(
            "generic_fetch.alt_search",
            query=alt_invoke_args["query"],
            invoke_args=alt_invoke_args,
            alt_candidate_count=len(alt_candidates),
            diverse_candidate_count=len(diverse_candidates),
            alt_preview=alt_search_text[:500],
        )

    return diverse_candidates, search_text


async def _run_generic_web_search_strategy_impl(
    last_message: str,
    web_search_runtime_args: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    from tools.web_tools import _is_specific_article_hit
    ctx, policy = _build_query_context(last_message)
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
        diverse_candidates, search_text, early = await _run_week_search_pipeline(last_message, ctx, web_search_runtime_args)
        if early is not None:
            return early
    else:
        diverse_candidates, search_text = await _run_general_search_pipeline(last_message, ctx, loop, web_search_runtime_args)

    ranked_candidates = diverse_candidates[:4]
    _web_debug(
        "generic_fetch.ranked_candidates",
        query=last_message,
        query_horizon=query_horizon,
        ranked_candidate_count=len(ranked_candidates),
        urls=[c.get("url", "") for c in ranked_candidates],
    )

    # For week queries: hybrid approach —
    # 1. Fetch specific article URLs (non-hub) without dynamic JS (faster, avoids hallucination)
    # 2. Fall back to Tavily snippet when fetch fails or returns poor content
    # This is more reliable than full dynamic fetches for paywalled/dynamic pages.
    if query_horizon == "week":
        week_entry_lines: list[str] = []
        week_snippet_lines: list[str] = []
        week_entry_sources: list[dict[str, str]] = []
        week_snippet_sources: list[dict[str, str]] = []
        seen_week_urls: set[str] = set()
        fetch_prompt_week = _build_generic_fetch_prompt(last_message)

        async def _week_entry(c: dict[str, str]) -> tuple[str, str, bool]:
            url = c.get("url", "")
            snippet = c.get("snippet", "").strip()
            snippet = re.sub(r"^#+\s+", "", snippet)
            snippet = re.sub(r"\s+#+\s+", " ", snippet).strip()
            title = re.sub(r"^#+\s+", "", c.get("title", url)).strip()
            if _is_article_url(url):
                try:
                    fetched = await _fetch_web_page_follow_redirect(url, fetch_prompt_week, use_dynamic=False)
                    if (
                        isinstance(fetched, str)
                        and not fetched.startswith("Error")
                        and not fetched.startswith("URL rechazada")
                        and not _is_no_info_response(fetched)
                        and len(fetched.split()) >= 20
                    ):
                        lines = _extract_generic_content_lines(fetched, query_terms)
                        if lines:
                            return title, " ".join(lines[:3]), False
                except Exception:
                    pass
            return title, snippet, True

        week_results = await asyncio.gather(*[_week_entry(c) for c in diverse_candidates])

        for (title, content, from_snippet), c in zip(week_results, diverse_candidates):
            min_words = 4 if from_snippet else 8
            if not content or len(content.split()) < min_words:
                continue
            if _is_no_info_response(content):
                continue
            url = c.get("url", "")
            if url in seen_week_urls:
                continue
            seen_week_urls.add(url)
            week_entry_lines.append(f"[{title}] — {content}")
            if from_snippet:
                week_snippet_lines.append(f"[{title}] — {content}")
            if _is_article_url(url):
                week_entry_sources.append({"title": title, "url": url})
                if from_snippet:
                    week_snippet_sources.append({"title": title, "url": url})

        _web_debug(
            "generic_fetch.week_entries",
            entry_count=len(week_entry_lines),
            snippet_count=len(week_snippet_lines),
            entry_sources=week_entry_sources,
            snippet_sources=week_snippet_sources,
        )

        if len(week_entry_lines) >= 2:
            # Format bullets directly — each fetched entry is already LLM-processed.
            # Bypassing _synthesize_search_summary avoids the LLM merging distinct articles.
            import datetime as _dt_week
            _current_year = _dt_week.date.today().year
            _old_year_re = re.compile(r'\b(20\d{2})\b')
            paragraph_parts = []
            for (content_title, content, _from_snippet), c in zip(week_results, diverse_candidates):
                title = content_title or c.get("title") or c.get("url") or ""
                # Skip entries whose content ONLY references years before the current year.
                # "18 de noviembre de 2025" is 5 months old — not "this week".
                # Only kept if the content also mentions the current year as context.
                _years = [int(y) for y in _old_year_re.findall(content)]
                if _years and max(_years) <= _current_year - 1:
                    continue
                # Trim to 3 sentences max to avoid duplicated text in long fetches
                sentences = re.split(r"(?<=[.!?])\s+", content)
                trimmed = " ".join(sentences[:3]).strip()
                # Discard truncated snippets (Tavily cuts them with "…" or "...")
                is_truncated = trimmed.endswith(("…", "...")) or re.search(r"\w…$", trimmed)
                if trimmed and not _is_no_info_response(trimmed) and not is_truncated:
                    url = _clean_source_url(c.get("url", ""))
                    src_line = f"Fuente: [{title}]({url})" if url else (f"Fuente: {title}" if title else "")
                    entry = f"{title}: {trimmed}"
                    if src_line:
                        entry = f"{entry}\n\n{src_line}"
                    paragraph_parts.append(entry)
            summary = "\n\n".join(paragraph_parts)
            return {
                "summary": summary,
                "words": summary.split(),
                "source_type": "search",
                "sources": week_entry_sources,
                "pre_synthesized": True,
            }
        if week_snippet_lines:
            # Local snippet-backed fallback: keep country-local material visible even when
            # we do not reach the stronger multi-article threshold required for fetched content.
            snippet_sources = week_snippet_sources or week_entry_sources or [{"title": "search result", "url": ""}]
            snippet_summary = _build_source_backed_response(
                week_snippet_lines[:8],
                snippet_sources,
            )
            return {
                "summary": snippet_summary,
                "words": snippet_summary.split(),
                "source_type": "search",
                "sources": snippet_sources,
                "pre_synthesized": True,
            }
        # Not enough content — fall through to page fetch approach

    if not ranked_candidates:
        search_lines = _extract_generic_content_lines(search_text, query_terms)
        if not search_lines:
            _web_debug(
                "generic_fetch.no_ranked_candidates",
                query=last_message,
                search_preview=search_text[:500],
                search_lines_count=0,
            )
            return None
        sources = _extract_sources_from_text(search_text)
        if not sources:
            sources = [{"title": "search result", "url": ""}]
        summary = _build_source_backed_response(search_lines[:8], sources)
        _web_debug(
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

    eligible_entries = await _fetch_and_score_entries(ranked_candidates, last_message, ctx, policy, search_text)

    if query_horizon == "week" and eligible_entries:
        recent_article_entries = [
            entry for entry in eligible_entries
            if _candidate_url_is_recent(cast(dict[str, str], entry["candidate"]).get("url", ""), search_age_days or 14)
            and _is_specific_article_hit(cast(dict[str, str], entry["candidate"]))
            and not _is_invalid_news_candidate(cast(dict[str, str], entry["candidate"]), last_message)
        ]
        if recent_article_entries:
            best_recent_entry = sorted(
                recent_article_entries,
                key=lambda entry: (
                    _candidate_source_priority(cast(dict[str, str], entry["candidate"]), query_source_group),
                    -cast(int, entry["score"]),
                ),
            )[0]
            summary = _build_source_backed_response(
                cast(list[str], best_recent_entry["summary_lines"]),
                cast(list[dict[str, str]], best_recent_entry["sources"]),
            )
            _web_debug(
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
            if not _is_invalid_news_candidate(cast(dict[str, str], entry["candidate"]), last_message)
        ]
        if not eligible_entries:
            return None
        ordered_entries = sorted(
            eligible_entries,
            key=lambda entry: (
                _candidate_source_priority(cast(dict[str, str], entry["candidate"]), query_source_group),
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
                # Supplement with Tavily article URLs when web_fetch only got few sources
                final_sources = list(combined_sources)
                if len(final_sources) < 4:
                    seen_urls: set[str] = {str(s.get("url") or "") for s in final_sources if s.get("url")}
                    for extra in _extract_sources_from_text(search_text):
                        extra_url = str(extra.get("url") or "")
                        if extra_url and extra_url not in seen_urls and len(final_sources) < 5:
                            final_sources.append(extra)
                            seen_urls.add(extra_url)
                summary = _build_source_backed_response(combined_lines[:20], final_sources)
                _web_debug(
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

    # Week fallback: page fetches gave < min_candidates diverse entries —
    # use Tavily snippets from diverse_candidates directly (more reliable for recent paywalled articles)
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
            summary = _build_source_backed_response(snippet_lines, snippet_sources)
            _web_debug(
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
            if not _is_invalid_news_candidate(cast(dict[str, str], entry["candidate"]), last_message)
        ]
        if valid_entries:
            eligible_entries = valid_entries
        best_entry = sorted(
            eligible_entries,
            key=lambda entry: (
                _candidate_source_priority(cast(dict[str, str], entry["candidate"]), query_source_group),
                -cast(int, entry["score"]),
            ),
        )[0]
        summary = _build_source_backed_response(cast(list[str], best_entry["summary_lines"]), cast(list[dict[str, str]], best_entry["sources"]))
        _web_debug(
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

    search_lines = _extract_generic_content_lines(search_text, query_terms)
    if not search_lines:
        _web_debug("generic_fetch.search_lines_empty", query=last_message, search_preview=search_text[:500])
        return None
    if _is_recent_web_information_query(last_message):
        if len(search_lines) < recent_min_body_lines:
            strongest_candidate = ranked_candidates[0] if ranked_candidates else None
            strongest_score = _score_generic_candidate(strongest_candidate, query_terms, query_source_group) if strongest_candidate else 0
            if not (
                strongest_candidate is not None
                and strongest_score >= recent_min_score
                and _is_specific_article_hit({
                    "title": strongest_candidate.get("title") or "",
                    "link": strongest_candidate.get("url") or strongest_candidate.get("link") or "",
                    "snippet": strongest_candidate.get("snippet") or "",
                })
            ):
                _web_debug(
                    "generic_fetch.search_lines_rejected_recent",
                    search_lines_count=len(search_lines),
                    strongest_url=(strongest_candidate or {}).get("url", "") if strongest_candidate else "",
                    strongest_score=strongest_score,
                    min_score=recent_min_score,
                    min_body_lines=recent_min_body_lines,
                )
                return None
        sources = _extract_sources_from_text(search_text)
        if len(sources) < recent_min_sources:
            _web_debug(
                "generic_fetch.search_sources_rejected_recent",
                source_count=len(sources),
                min_sources=recent_min_sources,
            )
            return None

    sources = _extract_sources_from_text(search_text)
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

    summary = _build_source_backed_response(search_lines[:8], sources)
    _web_debug(
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


async def _run_generic_web_search_fetch(
    last_message: str,
    web_search_runtime_args: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    search_runtime = WebSearchRuntime()
    fetch_runtime = WebFetchRuntime()
    country_strategy = CountryRecentNewsStrategy(
        search_runtime=search_runtime,
        fetch_runtime=fetch_runtime,
    )
    local_result = await country_strategy.execute(last_message, web_search_runtime_args)
    if local_result is not None:
        _web_debug(
            "generic_fetch.strategy_selected",
            strategy="country_recent_news",
            query=last_message,
            source_count=len(cast(list[dict[str, str]], local_result.get("sources") or [])),
        )
        return local_result

    generic_strategy = GenericWebSearchStrategy(
        search_runtime=search_runtime,
        fetch_runtime=fetch_runtime,
    )
    result = await generic_strategy.execute(last_message, web_search_runtime_args)
    if result is not None:
        _web_debug(
            "generic_fetch.strategy_selected",
            strategy="generic_web_search",
            query=last_message,
            source_count=len(cast(list[dict[str, str]], result.get("sources") or [])),
        )
    return result



async def _synthesize_search_summary(
    raw_summary: str,
    query: str,
    get_llm_fn: Callable,
    sources: list[dict[str, str]],
    has_labeled_content: bool = False,
) -> str:
    """Passes raw search content through LLM to produce a clean, structured response."""
    try:
        llm = get_llm_fn()
        sources_block = _format_sources(sources)
        clean_lines = []
        for line in raw_summary.splitlines():
            stripped = line.strip()
            if not stripped or stripped in ("...", "[...]"):
                continue
            # Markdown headers from raw scraped pages
            if re.match(r"^#{1,3}\s", stripped):
                continue
            # Author bylines: "Name Name · date" or "Name Name - date"
            if re.search(r"\w+\s+\w+\s+[·\-]\s+\d{1,2}\s+de\s+\w+", stripped):
                continue
            # Image slugs and filenames with timestamps
            if re.match(r"^\d{8,14}[_\-]\w", stripped):
                continue
            # URL-path-like slugs without spaces
            if re.match(r"^[\w\-]+(?:[_\-][\w\-]+){3,}$", stripped) and " " not in stripped:
                continue
            # Lines with heavily hyphenated words (image alt text artifacts)
            if any(word.count("-") >= 3 for word in stripped.split()):
                continue
            clean_lines.append(stripped)
        import datetime
        today_str = datetime.date.today().strftime("%d de %B de %Y")
        clean_content = "\n\n".join(clean_lines[:40])
        query_terms_for_dedup = _extract_generic_query_terms(query)
        query_horizon_local = detect_recent_query_horizon(query) if _is_recent_web_information_query(query) else None
        # Cuando el contenido está en otro idioma, la instrucción de traducción debe
        # ir al INICIO del prompt — si va enterrada en las reglas, el LLM la ignora.
        translation_prefix = ""
        if has_labeled_content:
            translation_prefix = (
                "⚠️ TRADUCCIÓN OBLIGATORIA: El contenido de abajo puede estar en otro idioma "
                "(italiano, japonés, etc.). DEBES traducir TODO a español rioplatense. "
                "NUNCA copies texto en otro idioma — siempre traducí.\n\n"
            )
        prompt = (
            f"{translation_prefix}"
            f"Fecha actual: {today_str}\n"
            f"Consulta del usuario: {query}\n\n"
            f"Información recopilada de la web:\n{clean_content}\n\n"
            "Sintetizá una respuesta clara respondiendo ÚNICAMENTE con lo que está en el texto de arriba. "
            "PROHIBIDO usar conocimiento propio o información que no esté en el texto provisto.\n\n"
            "Reglas de contenido:\n"
            "- IDIOMA: respondé siempre en el mismo idioma que la consulta del usuario\n"
            "- IGNORÁ completamente: pie de fotos, descripciones de imágenes, nombres de personas sin contexto noticioso, títulos de anime/manga, fragmentos sin información útil\n"
            "- PRIORIZÁ: artículos con hechos concretos, cifras, eventos, decisiones o noticias verificables\n"
            "- Si el contenido disponible no responde bien la consulta, indicalo brevemente\n\n"
            "Reglas de formato:\n"
            "- Cada punto DEBE comenzar con '•' seguido de un espacio\n"
            "- Cada artículo/fuente del texto = UN punto separado, pero TODOS deben responder al mismo tema solicitado.\n"
            "  Ejemplo: si la consulta es seguridad japonesa, podés usar un artículo sobre misiles y otro sobre una embajada,\n"
            "  pero no mezcles clima, deportes o política general.\n"
            "- NUNCA combines dos artículos en un solo punto. Cada punto viene de UNA sola fuente y no debe repetir la misma noticia.\n"
            "- Si un artículo tiene información irrelevante para la consulta (noticias de otro país, entretenimiento, deportes sin relación), omitilo.\n"
            "- OBLIGATORIO: dejá UNA línea en blanco entre cada punto\n"
            "- Cada punto tiene 2-3 oraciones con el hecho concreto, quiénes están involucrados y por qué importa\n"
            "- NO uses títulos ni headers (##, ###) dentro de la respuesta\n"
        )
        if has_labeled_content:
            # Contenido etiquetado [titulo]: snippet — el LLM puede atribuir por punto.
            # No agregar sources_block al final: el LLM lo hace inline.
            prompt += (
                "- OBLIGATORIO: después del texto de cada punto, en una nueva línea escribí exactamente "
                "'Fuente: [titulo]' usando el título entre corchetes del texto de arriba "
                "(ej: '[La Repubblica]: ...' → 'Fuente: La Repubblica')\n"
                "- NO incluyas una sección Sources al final"
            )
        else:
            prompt += "- NO incluyas una sección Sources — se agrega automáticamente"
        if query_horizon_local == "week":
            cutoff = (datetime.date.today() - datetime.timedelta(days=30)).strftime('%d/%m/%Y')
            prompt += f"\n- Solo incluí eventos de los últimos 30 días (desde el {cutoff}). Descartá cualquier evento más antiguo aunque esté en el texto."
        elif query_horizon_local:
            prompt += f"\n- Solo incluí eventos ocurridos en los últimos 30 días. Descartá cualquier evento más antiguo."
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        synthesized = getattr(response, "content", str(response)).strip()
        # Strip any LLM-generated Sources section (always unreliable) and replace with real one
        synthesized = re.split(r"\n\s*sources\s*:", synthesized, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        synthesized = _enforce_synthesis_format(synthesized)
        synthesized = _dedup_synthesis_bullets(synthesized, query_terms_for_dedup)
        if not has_labeled_content and sources_block:
            synthesized = f"{synthesized}\n\n{sources_block}"
        return synthesized
    except Exception as _synth_exc:
        import logging
        logging.warning(f"_synthesize_search_summary FAILED: {type(_synth_exc).__name__}: {_synth_exc}")
        return _enforce_synthesis_format(raw_summary)


async def _guardrail_fast_result(
    summary: str,
    new_tracker: dict[str, Any],
    rid: str,
    t0: float,
    should_evaluate_guard_fn: Callable,
    evaluate_trajectory_safe_fn: Callable,
) -> dict[str, Any]:
    fast_result: dict[str, Any] = {
        "messages": [AIMessage(content=summary)],
        "scrape_tracker": new_tracker,
    }
    if should_evaluate_guard_fn("web_scraping_node"):
        _is_safe, _ = await evaluate_trajectory_safe_fn(fast_result, "web_scraping_node")
        if not _is_safe:
            _emit_node_outcome(
                rid, "web_scraping_node", "blocked", phase="post_guard",
                agent="web_scraping_agent",
                duration_ms=int((time.time() - t0) * 1000),
                reason="agentdog", followup_likely=True,
                **_extract_tokens({"messages": []}), **_extract_quality({"messages": []}),
                **_extract_followup({"messages": []}, "success"), **_node_meta(),
            )
            return {"messages": [AIMessage(content="Respuesta retenida por política de seguridad.")]}
    return fast_result


async def run_web_scraping_flow(
    state: AgentState,
    agent,
    get_llm_fn: Callable,
    *,
    hitl_enabled: bool,
    confirmation_handler: Optional[ConfirmationPort] = None,
    ask_confirmation_compat: Optional[Callable[[str], Awaitable[bool]]] = None,
    get_runtime_policy: Callable[[], dict],
    evaluate_trajectory_safe_fn=evaluate_trajectory_safe,
    should_evaluate_guard_fn=_should_evaluate_guard,
) -> dict[str, Any]:
    messages = state["messages"]
    last_message = get_last_message_text(messages)
    state_dict = cast(dict[str, Any], state)
    web_search_runtime_args = _web_search_runtime_args(state_dict)
    rid = get_or_create_request_id(state_dict, lambda: "")
    t0 = time.time()

    if not rid:
        rid = str(uuid.uuid4())

    explicit_urls = _extract_urls_from_text(last_message)
    if hitl_enabled:
        url_info = f" → URLs: {', '.join(explicit_urls)}" if explicit_urls else ""
        preview = last_message[:120] + ("..." if len(last_message) > 120 else "")

        confirmed = True
        if confirmation_handler is not None:
            confirmed = await confirmation_handler.confirm(
                f"\n[HITL] web_scraping_agent va a procesar: \"{preview}\"{url_info}\n¿Confirmar? [s/n]: "
            )
        elif ask_confirmation_compat is not None:
            confirmed = await ask_confirmation_compat(
                f"\n[HITL] web_scraping_agent va a procesar: \"{preview}\"{url_info}\n¿Confirmar? [s/n]: "
            )
        if not confirmed:
            _emit_node_outcome(
                rid, "web_scraping_node", "blocked", phase="pre_guard",
                agent="web_scraping_agent",
                duration_ms=int((time.time() - t0) * 1000),
                reason="hitl_rejected",
            )
            return {"messages": [AIMessage(content="Operación cancelada por el usuario.")]}

    ctx = _select_strategy_context(state, last_message, get_runtime_policy)
    tracker = ctx["tracker"]
    turn_count = ctx["turn_count"]
    category = ctx["category"]
    prior_score = ctx["prior_score"]
    prior_reliability = ctx["prior_reliability"]
    ml_recommended = ctx["ml_recommended"]
    prediction_match = ctx["prediction_match"]
    _web_debug(
        "run_web_scraping_flow.start",
        query=last_message,
        category=category,
        explicit_urls=explicit_urls,
        web_search_runtime_args=web_search_runtime_args,
    )

    try:
        if explicit_urls:
            fetch_prompt = last_message.strip() or "Extraé la información relevante de esta URL."
            fetch_result = await _fetch_web_page_follow_redirect(explicit_urls[0], fetch_prompt, use_dynamic=True)
            if isinstance(fetch_result, str) and not fetch_result.startswith("Error") and not fetch_result.startswith("URL rechazada"):
                summary = fetch_result.strip()
                words = summary.split()
                duration_ms = int((time.time() - t0) * 1000)
                reliability = _scrape_reliability(len(words))
                new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _update_scrape_tracker(
                    tracker, category, len(words), turn_count,
                    duration_ms=duration_ms, cost_usd=0.0,
                    source_type="webfetch", reliability_override=reliability,
                ))
                analytics = cast(dict[str, Any], analytics)
                new_score = _get_category_score(new_tracker, category, turn_count)

                _emit_node_outcome(
                    rid, "web_scraping_node", "success", phase="agent",
                    agent="web_scraping_agent", duration_ms=duration_ms,
                    category=category, exploring=False, strategy="web_fetch", exp_rate=0.0,
                    scrape_reliability=reliability, prior_reliability=prior_reliability,
                    prior_score=prior_score, scrape_score=new_score,
                    retry_done=False, source_type="webfetch",
                    **_extract_tokens({"messages": []}), **_extract_quality({"messages": []}),
                    **_extract_followup({"messages": []}, "success"), **analytics, **_node_meta(),
                )
                summary, _, _ = _finalize_web_user_summary(summary, last_message, None)
                return await _guardrail_fast_result(
                    summary, new_tracker, rid, t0,
                    should_evaluate_guard_fn, evaluate_trajectory_safe_fn,
                )

        if category in {"sports", "news"}:
            discovery = await _run_generic_web_search_fetch(last_message, web_search_runtime_args)
            if discovery is not None:
                _web_debug(
                    "run_web_scraping_flow.discovery_hit",
                    category=category,
                    source_type=discovery.get("source_type"),
                    source_count=len(cast(list[dict[str, str]], discovery.get("sources") or [])),
                    pre_synthesized=discovery.get("pre_synthesized"),
                    branch="news_sports",
                )
                _disc_raw = cast(str, discovery["summary"])
                _disc_sources = cast(list[dict[str, str]], discovery.get("sources") or [])
                if discovery.get("pre_synthesized"):
                    summary = _disc_raw
                else:
                    summary = await _synthesize_search_summary(
                        _disc_raw, last_message, get_llm_fn, _disc_sources,
                        has_labeled_content=bool(discovery.get("has_labeled_content")),
                    )
                summary, _disc_sources, words = _finalize_web_user_summary(summary, last_message, _disc_sources)
                duration_ms = int((time.time() - t0) * 1000)
                reliability = _scrape_reliability(len(words))
                source_type = cast(str, discovery.get("source_type") or "webfetch")
                new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _update_scrape_tracker(
                    tracker, category, len(words), turn_count,
                    duration_ms=duration_ms, cost_usd=0.0,
                    source_type=source_type, reliability_override=reliability,
                ))
                analytics = cast(dict[str, Any], analytics)
                new_score = _get_category_score(new_tracker, category, turn_count)

                _emit_node_outcome(
                    rid, "web_scraping_node", "success", phase="agent",
                    agent="web_scraping_agent", duration_ms=duration_ms,
                    category=category, exploring=False, strategy="search_web" if source_type == "search" else "web_search_fetch", exp_rate=0.0,
                    scrape_reliability=reliability, prior_reliability=prior_reliability,
                    prior_score=prior_score, scrape_score=new_score,
                    retry_done=False, source_type=source_type,
                    ml_recommended=ml_recommended, prediction_match=prediction_match,
                    ml_would_succeed=(bool(analytics.get("quality_target", 0)) if prediction_match is True else None),
                    **_extract_tokens({"messages": []}), **_extract_quality({"messages": []}), **_extract_followup({"messages": []}, "success"), **analytics, **_node_meta(),
                )
                return await _guardrail_fast_result(
                    summary, new_tracker, rid, t0,
                    should_evaluate_guard_fn, evaluate_trajectory_safe_fn,
                )
            _web_debug("run_web_scraping_flow.discovery_miss", category=category, branch="news_sports")

            from tools import search_web

            loop = asyncio.get_running_loop()
            _fb2_args: dict = {"query": last_message, "use_cache": False, **web_search_runtime_args}
            if _is_recent_web_information_query(last_message):
                _fb2_args["topic"] = "news"
            fallback_search = await loop.run_in_executor(
                None,
                lambda: search_web.invoke(_fb2_args),
            )
            if not isinstance(fallback_search, str):
                fallback_search = str(fallback_search)
            fallback_terms = _extract_generic_query_terms(last_message)
            fallback_query_source_group = detect_query_source_group(last_message)
            fallback_lines = _extract_generic_content_lines(fallback_search, fallback_terms)
            _web_debug(
                "run_web_scraping_flow.search_fallback",
                args=_fb2_args,
                fallback_lines_count=len(fallback_lines),
                search_preview=fallback_search[:500],
            )
            if fallback_lines:
                fallback_candidates = _extract_generic_search_candidates(fallback_search)
                fallback_sources = _extract_sources_from_text(fallback_search)
                if fallback_candidates:
                    top_candidate = max(
                        fallback_candidates,
                        key=lambda candidate: _score_generic_candidate(candidate, fallback_terms, fallback_query_source_group),
                    )
                    fallback_sources = [{
                        "title": top_candidate.get("title") or top_candidate.get("url") or "search result",
                        "url": top_candidate.get("url") or "",
                    }]
                elif not fallback_sources:
                    fallback_sources = [{"title": "search result", "url": ""}]
                _fallback_raw = _build_source_backed_response(fallback_lines[:10], fallback_sources)
                summary = await _synthesize_search_summary(_fallback_raw, last_message, get_llm_fn, fallback_sources)
                summary, fallback_sources, words = _finalize_web_user_summary(summary, last_message, fallback_sources)
                duration_ms = int((time.time() - t0) * 1000)
                reliability = _scrape_reliability(len(words))
                new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _update_scrape_tracker(
                    tracker, category, len(words), turn_count,
                    duration_ms=duration_ms, cost_usd=0.0,
                    source_type="search", reliability_override=reliability,
                ))
                analytics = cast(dict[str, Any], analytics)
                new_score = _get_category_score(new_tracker, category, turn_count)
                _emit_node_outcome(
                    rid, "web_scraping_node", "success", phase="agent",
                    agent="web_scraping_agent", duration_ms=duration_ms,
                    category=category, exploring=False, strategy="search_web", exp_rate=0.0,
                    scrape_reliability=reliability, prior_reliability=prior_reliability,
                    prior_score=prior_score, scrape_score=new_score,
                    retry_done=False, source_type="search",
                    ml_recommended=ml_recommended, prediction_match=prediction_match,
                    ml_would_succeed=(bool(analytics.get("quality_target", 0)) if prediction_match is True else None),
                    **_extract_tokens({"messages": []}), **_extract_quality({"messages": []}), **_extract_followup({"messages": []}, "success"), **analytics, **_node_meta(),
                )
                return await _guardrail_fast_result(
                    summary, new_tracker, rid, t0,
                    should_evaluate_guard_fn, evaluate_trajectory_safe_fn,
                )

        if is_web_information_query(last_message) or _is_recent_web_information_query(last_message):
            discovery = await _run_generic_web_search_fetch(last_message, web_search_runtime_args)
            if discovery is not None:
                _web_debug(
                    "run_web_scraping_flow.discovery_hit",
                    category=category,
                    source_type=discovery.get("source_type"),
                    source_count=len(cast(list[dict[str, str]], discovery.get("sources") or [])),
                    pre_synthesized=discovery.get("pre_synthesized"),
                    branch="generic_web_info",
                )
                _disc_raw = cast(str, discovery["summary"])
                _disc_sources = cast(list[dict[str, str]], discovery.get("sources") or [])
                if discovery.get("pre_synthesized"):
                    summary = _disc_raw
                else:
                    summary = await _synthesize_search_summary(
                        _disc_raw, last_message, get_llm_fn, _disc_sources,
                        has_labeled_content=bool(discovery.get("has_labeled_content")),
                    )
                summary, _disc_sources, words = _finalize_web_user_summary(summary, last_message, _disc_sources)
                duration_ms = int((time.time() - t0) * 1000)
                reliability = _scrape_reliability(len(words))
                source_type = cast(str, discovery.get("source_type") or "webfetch")
                new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _update_scrape_tracker(
                    tracker, category, len(words), turn_count,
                    duration_ms=duration_ms, cost_usd=0.0,
                    source_type=source_type, reliability_override=reliability,
                ))
                analytics = cast(dict[str, Any], analytics)
                new_score = _get_category_score(new_tracker, category, turn_count)

                _emit_node_outcome(
                    rid, "web_scraping_node", "success", phase="agent",
                    agent="web_scraping_agent", duration_ms=duration_ms,
                    category=category, exploring=False, strategy="search_web" if source_type == "search" else "web_search_fetch", exp_rate=0.0,
                    scrape_reliability=reliability, prior_reliability=prior_reliability,
                    prior_score=prior_score, scrape_score=new_score,
                    retry_done=False, source_type=source_type,
                    ml_recommended=ml_recommended, prediction_match=prediction_match,
                    ml_would_succeed=(bool(analytics.get("quality_target", 0)) if prediction_match is True else None),
                    **_extract_tokens({"messages": []}), **_extract_quality({"messages": []}), **_extract_followup({"messages": []}, "success"), **analytics, **_node_meta(),
                )
                return await _guardrail_fast_result(
                    summary, new_tracker, rid, t0,
                    should_evaluate_guard_fn, evaluate_trajectory_safe_fn,
                )
            _web_debug("run_web_scraping_flow.discovery_miss", category=category, branch="generic_web_info")

        agent_hint = (
            "[Sistema | web] Usa search_web para descubrir fuentes dinámicamente. "
            "Si la consulta es de información reciente, incluí el año actual en la búsqueda. "
            "Para noticias o información reciente, hacé varias búsquedas antes de responder; search_web puede usarse varias veces. "
            "Después usa web_fetch sobre varias URLs relevantes, no solo la primera. "
            "Si la consulta pide noticias o actualidad, reuní varias fuentes antes de responder. "
            "Si una fuente mezcla temas, países o resultados no relacionados, recházala y vuelve a buscar. "
            "Si web_fetch informa un redirect a otro host, repetí web_fetch con la URL de redirect. "
            "No respondas hasta tener fuentes que apoyen directamente la afirmación; si aparece una noticia vieja, un evento futuro o una página evergreen, descartala. "
            "Si la respuesta es de noticias o actualidad, desarrollala en 4 párrafos breves sobre el mismo tema solicitado, sin repetir noticias. "
            "Tu respuesta final debe incluir un bloque Sources con enlaces markdown.\n\n"
        )
        try:
            raw_result = await agent.ainvoke(
                {"messages": [HumanMessage(content=agent_hint + last_message)]},
                config=RunnableConfig(
                    tags=["web_scraping", "agent", "high_risk", "context_quarantine"],
                    metadata={
                        "node": "web_scraping_node",
                        "agent": "web_scraping_agent",
                        "request_id": rid,
                        "input_chars": len(last_message),
                        "prior_reliability": prior_reliability,
                    },
                    recursion_limit=16,
                ),
            )
        except Exception as exc:
            _web_debug("run_web_scraping_flow.agent_exception", error=repr(exc))
            fallback_discovery = await _run_generic_web_search_fetch(last_message, web_search_runtime_args)
            if fallback_discovery is not None:
                _web_debug(
                    "run_web_scraping_flow.agent_exception_recovered",
                    source_type=fallback_discovery.get("source_type"),
                    source_count=len(cast(list[dict[str, str]], fallback_discovery.get("sources") or [])),
                )
                summary = cast(str, fallback_discovery["summary"])
                words = cast(list[str], fallback_discovery.get("words") or summary.split())
                duration_ms = int((time.time() - t0) * 1000)
                reliability = _scrape_reliability(len(words))
                source_type = cast(str, fallback_discovery.get("source_type") or "search")
                new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _update_scrape_tracker(
                    tracker, category, len(words), turn_count,
                    duration_ms=duration_ms, cost_usd=0.0,
                    source_type=source_type, reliability_override=reliability,
                ))
                analytics = cast(dict[str, Any], analytics)
                new_score = _get_category_score(new_tracker, category, turn_count)
                _emit_node_outcome(
                    rid, "web_scraping_node", "success", phase="agent",
                    agent="web_scraping_agent", duration_ms=duration_ms,
                    category=category, exploring=False, strategy="search_web" if source_type == "search" else "web_search_fetch", exp_rate=0.0,
                    scrape_reliability=reliability, prior_reliability=prior_reliability,
                    prior_score=prior_score, scrape_score=new_score,
                    retry_done=False, source_type=source_type,
                    ml_recommended=ml_recommended, prediction_match=prediction_match,
                    ml_would_succeed=(bool(analytics.get("quality_target", 0)) if prediction_match is True else None),
                    **_extract_tokens({"messages": []}), **_extract_quality({"messages": []}), **_extract_followup({"messages": []}, "success"), **analytics, **_node_meta(),
                )
                return await _guardrail_fast_result(
                    summary, new_tracker, rid, t0,
                    should_evaluate_guard_fn, evaluate_trajectory_safe_fn,
                )
            _emit_node_outcome(
                rid, "web_scraping_node", "error", phase="agent",
                agent="web_scraping_agent",
                duration_ms=int((time.time() - t0) * 1000),
                reason=str(exc),
                followup_likely=True,
                **_node_meta(),
            )
            return {"messages": [AIMessage(content="No pude conectar con el motor de síntesis, pero no pude recuperar fuentes útiles tampoco. Probá de nuevo en unos minutos.")]}

        tokens = _extract_tokens(raw_result)
        quality = _extract_quality(raw_result)
        followup = _extract_followup(raw_result, "success")
        meta = _node_meta()

        if should_evaluate_guard_fn("web_scraping_node"):
            is_safe, _ = await evaluate_trajectory_safe_fn(
                {
                    "messages": raw_result.get("messages", []),
                    "next_agent": state.get("next_agent", ""),
                },
                "web_scraping_node",
            )
            if not is_safe:
                _emit_node_outcome(
                    rid, "web_scraping_node", "blocked", phase="post_guard",
                    agent="web_scraping_agent",
                    duration_ms=int((time.time() - t0) * 1000),
                    reason="agentdog",
                    followup_likely=True,
                    **tokens, **quality, **meta,
                )
                return {"messages": [AIMessage(content="Respuesta retenida por política de seguridad.")]}

        raw_messages = raw_result.get("messages", [])
        raw_text = extract_final_ai_text(raw_messages)
        _web_debug(
            "run_web_scraping_flow.agent_final",
            raw_text_preview=raw_text[:500],
            message_count=len(cast(list[Any], raw_messages)),
        )
        if not raw_text:
            _emit_node_outcome(
                rid, "web_scraping_node", "error", phase="agent",
                agent="web_scraping_agent",
                duration_ms=int((time.time() - t0) * 1000),
                reason="empty_response",
                followup_likely=True,
                **meta,
            )
            return {"messages": [AIMessage(content="No se pudo extraer información de la página.")]}

        sources_from_raw = _extract_sources_from_text(raw_text)
        if sources_from_raw or len(raw_text.split()) < 80:
            summary = await _synthesize_search_summary(raw_text, last_message, get_llm_fn, sources_from_raw)
        else:
            summary = await _summarize_if_long(raw_text, rid, get_llm_fn)
        summary, _, _ = _finalize_web_user_summary(summary, last_message, sources_from_raw or None)
        words = raw_text.split()
        summary_triggered = len(words) > 200
        duration_ms = int((time.time() - t0) * 1000)
        reliability = _scrape_reliability(len(words))
        source_type = "agent"
        retry_done = False

        if reliability in {"unreliable"}:
            _emit_node_outcome(
                rid, "web_scraping_node", "retry", phase="agent",
                agent="web_scraping_agent", duration_ms=duration_ms,
                reason=f"auto_retry:{reliability}",
                scrape_reliability=reliability, strategy="web_search_fetch",
                source_type=source_type, category=category, **tokens, **_node_meta(),
            )
            retry_summary, retry_words, retry_tokens, retry_quality = await _run_retry_agent(
                agent, last_message, rid, get_llm_fn,
            )
            if retry_summary is not None:
                summary = retry_summary
                words = cast(list[str], retry_words or [])
                tokens = cast(dict[str, Any], retry_tokens or {})
                quality = cast(dict[str, Any], retry_quality or {})
            retry_done = True
            duration_ms = int((time.time() - t0) * 1000)
            reliability = _scrape_reliability(len(words))

        cost_usd = tokens.get("estimated_cost_usd")
        new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _update_scrape_tracker(
            tracker, category, len(words), turn_count,
            duration_ms=duration_ms, cost_usd=cost_usd,
            source_type=source_type, reliability_override=reliability,
        ))
        analytics = cast(dict[str, Any], analytics)
        new_score = _get_category_score(new_tracker, category, turn_count)
        if reliability not in ("ok_weak", "ok_strong"):
            followup = {"followup_likely": True}

        _emit_node_outcome(
            rid, "web_scraping_node", "success", phase="agent",
            agent="web_scraping_agent", duration_ms=duration_ms,
            summary_triggered=summary_triggered, raw_words=len(words),
            category=category, exploring=False, strategy="web_search_fetch", exp_rate=0.0,
            scrape_reliability=reliability, prior_reliability=prior_reliability,
            prior_score=prior_score, scrape_score=new_score,
            retry_done=retry_done,
            source_type=source_type,
            **tokens, **quality, **followup, **analytics, **meta,
        )
        return {
            "messages": [AIMessage(content=summary)],
            "scrape_tracker": new_tracker,
        }

    except Exception as e:
        _emit_node_outcome(
            rid, "web_scraping_node", "error", phase="agent",
            agent="web_scraping_agent",
            duration_ms=int((time.time() - t0) * 1000),
            reason=str(e),
            followup_likely=True,
            **_node_meta(),
        )
        raise
