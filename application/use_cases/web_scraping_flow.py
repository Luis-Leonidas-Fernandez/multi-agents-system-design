"""Caso de uso para el flujo de web scraping.

Coordina HITL, estrategia, guardrails, retry y postcondiciones.
El nodo LangGraph queda como adaptador fino.
"""
import asyncio
import os
import re
import time
import uuid
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
from typing import Any, Optional, Callable, Awaitable, Mapping, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig

from application.policies.agentdog import evaluate_trajectory_safe, _should_evaluate_guard, _is_allowed_public_price_request
from application.helpers.audit_flow_helpers import (
    _emit_node_outcome,
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
    score_domain_boost,
)
from tools.web_tools import _is_specific_article_hit
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


def _extract_urls_from_text(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)\]]+", text or "")
    cleaned: list[str] = []
    seen: set[str] = set()
    for url in urls:
        normalized = url.rstrip(".,;:")
        if normalized and normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)
    return cleaned


def _clean_source_url(url: str) -> str:
    """Strip CITE_THIS artifacts (|domain=xxx>>>) from URLs."""
    return url.split("|")[0].rstrip(">").strip() if url else url


def _format_sources(sources: list[dict[str, str]]) -> str:
    if not sources:
        return ""
    lines = ["Sources:"]
    seen_domains: set[str] = set()
    for source in sources:
        raw_url = source.get("url") or ""
        url = _clean_source_url(raw_url)
        if not url:
            continue
        domain = source.get("domain") or (urlparse(url).hostname or "").replace("www.", "")
        if domain and domain in seen_domains:
            continue
        if domain:
            seen_domains.add(domain)
        title = source.get("title") or url
        # Clean CITE_THIS artifacts from title too
        if "|domain=" in title:
            title = url
        lines.append(f"- [{title}]({url})")
    return "\n".join(lines)


def _build_source_backed_response(summary_lines: list[str], sources: list[dict[str, str]]) -> str:
    body = []
    seen_lines: set[str] = set()
    for line in summary_lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"^[-•\u2022]\s+", "", cleaned)
        dedupe_key = re.sub(r"\s+", " ", cleaned).strip().lower()
        if dedupe_key in seen_lines:
            continue
        seen_lines.add(dedupe_key)
        body.append(cleaned)
    sources_block = _format_sources(sources)
    if sources_block:
        if body:
            body.append("")
        body.append(sources_block)
    return "\n".join(body).strip()


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


@dataclass(frozen=True)
class WebCandidateRecord:
    title: str
    url: str
    snippet: str
    source_kind: str
    evidence_kind: str
    recency: str
    specificity: str
    source_label: str = ""

    def as_candidate(self) -> dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source_kind": self.source_kind,
            "source_label": self.source_label,
        }


def _candidate_record_from_dict(candidate: dict[str, str], *, query: str, query_horizon: Optional[str]) -> WebCandidateRecord:
    source_kind = _classify_candidate_source_kind(candidate)
    evidence_kind = "section_lines" if source_kind == "section_hit" else "search_snippet"
    recency = _classify_candidate_recency(candidate, query_horizon)
    specificity = _classify_candidate_specificity(candidate, query)
    return WebCandidateRecord(
        title=str(candidate.get("title") or candidate.get("url") or "result"),
        url=str(candidate.get("url") or ""),
        snippet=str(candidate.get("snippet") or ""),
        source_kind=source_kind,
        evidence_kind=evidence_kind,
        recency=recency,
        specificity=specificity,
        source_label=str(candidate.get("source_label") or ""),
    )


def _classify_candidate_source_kind(candidate: dict[str, str]) -> str:
    if _is_hub_like_candidate(candidate):
        return "hub_hit"
    if candidate.get("source_kind") == "section_fallback":
        return "section_hit"
    if candidate.get("source_kind") == "homepage_fallback":
        return "homepage_hit"
    if _is_specific_article_hit(candidate):
        return "article_hit"
    return "topic_hit"


def _classify_candidate_recency(candidate: dict[str, str], query_horizon: Optional[str]) -> str:
    url = str(candidate.get("url") or "")
    if _candidate_url_has_date(url):
        threshold = 45 if query_horizon == "month" else 14 if query_horizon == "week" else 2 if query_horizon == "today" else 30
        return "dated_recent" if _candidate_url_is_recent(url, threshold) else "dated_old"
    if candidate.get("source_kind") == "section_fallback":
        return "dated_recent"
    return "undated"


def _classify_candidate_specificity(candidate: dict[str, str], query: str) -> str:
    if _is_invalid_news_candidate(candidate, query):
        return "structural"
    if candidate.get("source_kind") == "section_fallback" or _is_specific_article_hit(candidate):
        return "concrete"
    return "broad"


def _candidate_strategy_priority(candidate: dict[str, str], *, query: str, query_horizon: Optional[str]) -> tuple[int, int, int, int]:
    record = _candidate_record_from_dict(candidate, query=query, query_horizon=query_horizon)
    source_rank = {
        "section_hit": 0,
        "article_hit": 1,
        "homepage_hit": 2,
        "topic_hit": 3,
        "hub_hit": 4,
    }.get(record.source_kind, 5)
    specificity_rank = {"concrete": 0, "broad": 1, "structural": 2}.get(record.specificity, 3)
    recency_rank = {"dated_recent": 0, "undated": 1, "dated_old": 2}.get(record.recency, 3)
    return (source_rank, specificity_rank, recency_rank, -len(record.snippet.split()))


class CountryRecentNewsStrategy:
    """Estrategia principal para noticias locales recientes basadas en secciones."""

    def __init__(self, *, search_runtime: WebSearchRuntime, fetch_runtime: WebFetchRuntime) -> None:
        self._search_runtime = search_runtime
        self._fetch_runtime = fetch_runtime

    async def execute(
        self,
        last_message: str,
        web_search_runtime_args: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        query_source_group = detect_query_source_group(last_message)
        query_horizon = detect_recent_query_horizon(last_message) if _is_recent_web_information_query(last_message) else None
        if not _should_use_country_recent_news_strategy(last_message, query_source_group, query_horizon):
            return None

        source_terms = list(get_query_source_terms(last_message))
        query_terms = _extract_generic_query_terms(last_message)
        for term in source_terms:
            if term not in query_terms:
                query_terms.append(term)

        country_press_domains, country_press_names = await _discover_country_press_sources(
            last_message,
            query_source_group,
            source_terms,
            web_search_runtime_args,
        )
        if not country_press_domains:
            return None

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
            for section_url, section_label in _build_country_press_section_targets(domain, fallback_url, last_message):
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
                    structured_candidates.append(WebCandidateRecord(
                        title=f"{source_meta.get('title') or press_name} — {section_label}",
                        url=candidate_url,
                        snippet=item_text,
                        source_kind="section_hit",
                        evidence_kind="section_lines",
                        recency="dated_recent",
                        specificity="concrete",
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
                geography = _extract_query_geography(last_message) or ""
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
        # If the model backend is unavailable, preserve the raw content instead of
        # aborting the whole turn with a connection error.
        if sources_block:
            return f"{body_text.strip()}\n\n{sources_block.strip()}"
        return body_text.strip()


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


async def _legacy_run_web_scraping_flow(
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
    messages     = state["messages"]
    last_message = get_last_message_text(messages)
    state_dict   = cast(dict[str, Any], state)
    web_search_runtime_args = _web_search_runtime_args(state_dict)
    rid          = get_or_create_request_id(state_dict, lambda: "")
    import time
    import uuid
    t0           = time.time()

    if not rid:
        rid = str(uuid.uuid4())

    urls = []
    if hitl_enabled:
        urls     = re.findall(r'https?://\S+', last_message)
        url_info = f" → URLs: {', '.join(urls)}" if urls else ""
        preview  = last_message[:120] + ("..." if len(last_message) > 120 else "")
        needs_confirmation = bool(urls)
        if _is_allowed_public_price_request(messages, "web_scraping_node"):
            needs_confirmation = False

        confirmed = True
        if needs_confirmation:
            confirmed = False
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

    try:
        ctx = _select_strategy_context(state, last_message, get_runtime_policy)
        tracker          = ctx["tracker"]
        turn_count       = ctx["turn_count"]
        category         = ctx["category"]
        prior_score      = ctx["prior_score"]
        prior_reliability = ctx["prior_reliability"]
        ml_recommended   = ctx["ml_recommended"]
        strategy         = ctx["strategy"]
        exploring        = ctx["exploring"]
        exp_rate         = ctx["exp_rate"]
        prediction_match = ctx["prediction_match"]

        explicit_urls = _extract_urls_from_text(last_message)
        if explicit_urls:
            fetch_prompt = last_message.strip() or "Extraé la información relevante de esta URL."
            fetch_result = await _fetch_web_page_follow_redirect(explicit_urls[0], fetch_prompt, use_dynamic=True)
            if isinstance(fetch_result, str) and not fetch_result.startswith("Error") and not fetch_result.startswith("URL rechazada"):
                duration_ms = int((time.time() - t0) * 1000)
                words = fetch_result.split()
                source_type = "webfetch"
                reliability = _scrape_reliability(len(words))
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
                    category=category, exploring=False, strategy="web_fetch", exp_rate=0.0,
                    scrape_reliability=reliability, prior_reliability=prior_reliability,
                    prior_score=prior_score, scrape_score=new_score,
                    retry_done=False, source_type=source_type,
                    ml_recommended=ml_recommended, prediction_match=prediction_match,
                    ml_would_succeed=(bool(analytics.get("quality_target", 0)) if prediction_match is True else None),
                    **_extract_tokens({"messages": []}), **_extract_quality({"messages": []}), **_extract_followup({"messages": []}, "success"), **analytics, **_node_meta(),
                )
                return {
                    "messages": [AIMessage(content=fetch_result)],
                    "scrape_tracker": new_tracker,
                }

        if strategy == "api_price":
            from domain.tool_responses import PriceToolResponse
            coin     = _detect_coin_from_query(last_message)
            api_json: Optional[str] = None
            price_resp: Optional[Any] = None
            try:
                loop     = asyncio.get_running_loop()
                api_json = await loop.run_in_executor(
                    None, lambda: _get_crypto_price_fn(coin=coin, vs_currency="usd")
                )
            except Exception:
                pass

            if api_json:
                try:
                    price_resp = PriceToolResponse.model_validate_json(api_json)
                except Exception:
                    price_resp = None

            if price_resp and price_resp.is_valid_price():
                formatted   = _format_price_response(price_resp.model_dump())
                duration_ms = int((time.time() - t0) * 1000)
                tokens_fast = {
                    "model": _get_model_name(), "tokens_available": False,
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "total_tokens": 0, "estimated_cost_usd": 0.0,
                }
                quality_fast  = {"output_length": len(formatted), "tool_calls_count": 1}
                followup_fast = {"followup_likely": False}
                meta          = _node_meta()

                fast_path_result = {"messages": [AIMessage(content=formatted)], "next_agent": state.get("next_agent", "")}
                if should_evaluate_guard_fn("web_scraping_node"):
                    is_safe, _ = await evaluate_trajectory_safe_fn(fast_path_result, "web_scraping_node")
                    if not is_safe:
                        _emit_node_outcome(
                            rid, "web_scraping_node", "blocked", phase="post_guard",
                            agent="web_scraping_agent",
                            duration_ms=duration_ms,
                            reason="agentdog",
                            followup_likely=True,
                            **tokens_fast, **quality_fast, **meta,
                        )
                        return {"messages": [AIMessage(content="Respuesta retenida por política de seguridad.")]}

                new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _update_scrape_tracker(
                    tracker, category, 200, turn_count,
                    duration_ms=duration_ms, cost_usd=0.0,
                    source_type="structured", reliability_override="ok_strong",
                ))
                new_score          = _get_category_score(new_tracker, category, turn_count)
                quality_target_val = analytics.get("quality_target", 0)
                ml_would_succeed: Optional[bool] = (
                    bool(quality_target_val) if prediction_match is True else None
                )
                _emit_node_outcome(
                    rid, "web_scraping_node", "success", phase="agent",
                    agent="web_scraping_agent", duration_ms=duration_ms,
                    category=category, strategy=strategy, exploring=exploring,
                    exp_rate=exp_rate, source_type="structured",
                    price_extracted=price_resp.price, parse_success=True,
                    scrape_reliability="ok_strong",
                    prior_reliability=prior_reliability, prior_score=prior_score,
                    scrape_score=new_score, retry_done=False,
                    ml_recommended=ml_recommended, prediction_match=prediction_match,
                    ml_would_succeed=ml_would_succeed,
                    **tokens_fast, **quality_fast, **followup_fast, **analytics, **meta,
                )
                return {
                    "messages": [AIMessage(content=formatted)],
                    "scrape_tracker": new_tracker,
                }

        if category in {"sports", "news"}:
            discovery = await _run_generic_web_search_fetch(last_message, web_search_runtime_args)
            if discovery is not None:
                _disc_raw = cast(str, discovery["summary"])
                _disc_sources = cast(list[dict[str, str]], discovery.get("sources") or [])
                if discovery.get("pre_synthesized"):
                    summary = _disc_raw
                else:
                    summary = await _synthesize_search_summary(_disc_raw, last_message, get_llm_fn, _disc_sources)
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
                return {
                    "messages": [AIMessage(content=summary)],
                    "scrape_tracker": new_tracker,
                }

            from tools import search_web

            loop = asyncio.get_running_loop()
            _fb_args: dict = {"query": last_message, "use_cache": False, **web_search_runtime_args}
            if _is_recent_web_information_query(last_message):
                _fb_args["topic"] = "news"
            fallback_search = await loop.run_in_executor(
                None,
                lambda: search_web.invoke(_fb_args),
            )
            if not isinstance(fallback_search, str):
                fallback_search = str(fallback_search)
            fallback_terms = _extract_generic_query_terms(last_message)
            fallback_lines = _extract_generic_content_lines(fallback_search, fallback_terms)
            if fallback_lines:
                fallback_sources = _extract_sources_from_text(fallback_search)
                if not fallback_sources:
                    fallback_sources = [{"title": "search result", "url": ""}]
                summary = _build_source_backed_response(fallback_lines[:8], fallback_sources)
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
                return {
                    "messages": [AIMessage(content=summary)],
                    "scrape_tracker": new_tracker,
                }

        if is_web_information_query(last_message) or _is_recent_web_information_query(last_message):
            discovery = await _run_generic_web_search_fetch(last_message, web_search_runtime_args)
            if discovery is not None:
                _disc_raw = cast(str, discovery["summary"])
                _disc_sources = cast(list[dict[str, str]], discovery.get("sources") or [])
                if discovery.get("pre_synthesized"):
                    summary = _disc_raw
                else:
                    summary = await _synthesize_search_summary(_disc_raw, last_message, get_llm_fn, _disc_sources)
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
                return {
                    "messages": [AIMessage(content=summary)],
                    "scrape_tracker": new_tracker,
                }

        agent_hint = ""
        if strategy == "force_search":
            agent_hint = "[Sistema] Scraping falló repetidamente para este tipo de query. Usa search_web directamente.\n\n"
        elif strategy == "prefer_search":
            agent_hint = "[Sistema] Scraping devolvió contenido insuficiente o lento antes. Intenta search_web primero.\n\n"

        if agent_hint:
            agent_hint = (
                f"[Sistema | categoría={category} score={prior_score:+.2f} "
                f"estrategia={strategy} exploring={exploring} exp_rate={exp_rate:.0%}]\n{agent_hint}"
            )
        agent_message = agent_hint + last_message

        raw_result = await agent.ainvoke(
            {"messages": [HumanMessage(content=agent_message)]},
            config=RunnableConfig(
                tags=["web_scraping", "agent", "high_risk", "context_quarantine"],
                metadata={
                    "node":              "web_scraping_node",
                    "agent":             "web_scraping_agent",
                    "request_id":        rid,
                    "input_chars":       len(last_message),
                    "prior_reliability": prior_reliability,
                },
            ),
        )

        tokens   = _extract_tokens(raw_result)
        quality  = _extract_quality(raw_result)
        followup = _extract_followup(raw_result, "success")
        meta     = _node_meta()

        if should_evaluate_guard_fn("web_scraping_node"):
            is_safe, _ = await evaluate_trajectory_safe_fn(
                {
                    "messages":   raw_result.get("messages", []),
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

        summary           = await _summarize_if_long(raw_text, rid, get_llm_fn)
        words             = raw_text.split()
        summary_triggered = len(words) > 200

        duration_ms = int((time.time() - t0) * 1000)
        reliability = _scrape_reliability(len(words))
        retry_done  = False

        source_type   = "structured" if strategy in _STRUCTURED_SOURCE_STRATEGIES else "unstructured"
        parsed_price: Optional[float] = None
        parse_success: Optional[bool] = None
        price_data: Optional[dict]    = None

        if source_type == "structured":
            price_data = _extract_price_from_messages(raw_result)
            if price_data:
                parsed_price  = price_data["price"]
                parse_success = True
                reliability   = "ok_strong"
            elif raw_text:
                parsed_price  = _extract_structured_price(raw_text)
                parse_success = parsed_price is not None
                reliability   = "ok_strong" if parse_success else "unreliable"
            else:
                parse_success = False
                reliability   = "unreliable"

        if reliability in _RETRY_ON_RELIABILITY and strategy != "force_search":
            _emit_node_outcome(
                rid, "web_scraping_node", "retry", phase="agent",
                agent="web_scraping_agent", duration_ms=duration_ms,
                reason=f"auto_retry:{reliability}",
                scrape_reliability=reliability, strategy=strategy,
                source_type=source_type, category=category, **tokens, **_node_meta(),
            )
            retry_summary, retry_words, retry_tokens, retry_quality = await _run_retry_agent(
                agent, last_message, rid, get_llm_fn,
            )
            if retry_summary is not None:
                summary           = retry_summary
                words             = cast(list[str], retry_words or [])
                summary_triggered = len(words) > 200
                tokens            = cast(dict[str, Any], retry_tokens or {})
                quality           = cast(dict[str, Any], retry_quality or {})

            strategy    = "force_search"
            reliability = _scrape_reliability(len(words))
            retry_done  = True
            duration_ms = int((time.time() - t0) * 1000)

        if reliability == "unreliable":
            new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _update_scrape_tracker(
                tracker, category, len(words), turn_count,
                duration_ms=duration_ms, cost_usd=tokens.get("estimated_cost_usd"),
                source_type=source_type, reliability_override=reliability,
            ))
            analytics = cast(dict[str, Any], analytics)
            _emit_node_outcome(
                rid, "web_scraping_node", "low_confidence", phase="agent",
                agent="web_scraping_agent", duration_ms=duration_ms,
                scrape_reliability=reliability, strategy=strategy,
                retry_done=retry_done, category=category,
                source_type=source_type, price_extracted=parsed_price, parse_success=parse_success,
                ml_recommended=ml_recommended, prediction_match=prediction_match,
                ml_would_succeed=(False if prediction_match is True else None),
                **tokens, **quality, **_node_meta(), **analytics,
            )
            return {
                "messages": [AIMessage(content=(
                    "No pude obtener información confiable para esta consulta. "
                    "Intenta proporcionar una URL específica o reformular la pregunta."
                ))],
                "scrape_tracker": new_tracker,
            }

        cost_usd    = tokens.get("estimated_cost_usd")
        new_tracker, analytics = cast(tuple[dict[str, Any], dict[str, Any]], _update_scrape_tracker(
            tracker, category, len(words), turn_count,
            duration_ms=duration_ms, cost_usd=cost_usd,
            source_type=source_type, reliability_override=reliability,
        ))
        analytics = cast(dict[str, Any], analytics)
        new_score = _get_category_score(new_tracker, category, turn_count)

        if reliability not in ("ok_weak", "ok_strong"):
            followup = {"followup_likely": True}

        quality_target_val = analytics.get("quality_target", 0)
        ml_would_succeed = (
            bool(quality_target_val) if prediction_match is True else None
        )

        _emit_node_outcome(
            rid, "web_scraping_node", "success", phase="agent",
            agent="web_scraping_agent", duration_ms=duration_ms,
            summary_triggered=summary_triggered, raw_words=len(words),
            category=category, exploring=exploring, strategy=strategy, exp_rate=exp_rate,
            scrape_reliability=reliability, prior_reliability=prior_reliability,
            prior_score=prior_score, scrape_score=new_score,
            retry_done=retry_done,
            source_type=source_type, price_extracted=parsed_price, parse_success=parse_success,
            ml_recommended=ml_recommended, prediction_match=prediction_match,
            ml_would_succeed=ml_would_succeed,
            **tokens, **quality, **followup, **analytics, **meta,
        )
        summary, _, _ = _finalize_web_user_summary(summary, last_message, None)
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


# ============================================================================
# Generic Claude-style web flow
# ============================================================================

_GENERIC_WEB_STOPWORDS = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "cada", "como", "con",
    "contra", "cual", "cuál", "cuales", "cuáles", "cuando", "cuándo", "de", "del", "desde",
    "donde", "dónde", "durante", "e", "el", "ella", "ellas", "ellos", "en", "entre",
    "era", "eres", "es", "esa", "esas", "ese", "eso", "esta", "está", "están", "este",
    "esto", "estos", "fue", "ha", "han", "hay", "la", "las", "le", "les", "lo", "los",
    "mas", "más", "mi", "mis", "muy", "no", "nos", "nosotros", "o", "o", "para", "pero",
    "por", "que", "qué", "se", "sin", "sobre", "su", "sus", "te", "tu", "tus", "un",
    "una", "uno", "unos", "unas", "y", "ya", "hoy", "ayer", "mañana", "today", "latest",
    "current", "recent", "news", "noticias", "page", "web", "site",
}


def _extract_generic_query_terms(text: str) -> list[str]:
    terms: list[str] = []
    for raw in re.findall(r"[\wáéíóúñÁÉÍÓÚÑ]+", (text or "").lower()):
        if len(raw) < 3 or raw in _GENERIC_WEB_STOPWORDS:
            continue
        if raw not in terms:
            terms.append(raw)
    return terms


def _is_recent_web_information_query(text: str) -> bool:
    lowered = (text or "").lower()
    recent_terms = (
        "today", "hoy", "latest", "recent", "current", "actual", "actuales",
        "última", "últimas", "ultimo", "último", "ultimas", "ultimos",
        "this week", "esta semana", "semana", "week",
    )
    if not any(term in lowered for term in recent_terms):
        return False
    if any(term in lowered for term in ("price", "precio", "cotiza", "cotización", "cotizacion")):
        return False
    return True


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


_GEOGRAPHY_TERMS: tuple[tuple[str, str], ...] = (
    # Latin America
    ("ecuatoriano", "Ecuador"), ("ecuatoriana", "Ecuador"), ("ecuador", "Ecuador"),
    ("argentino", "Argentina"), ("argentina", "Argentina"),
    ("colombiano", "Colombia"), ("colombia", "Colombia"),
    ("venezolano", "Venezuela"), ("venezuela", "Venezuela"),
    ("chileno", "Chile"), ("chile", "Chile"),
    ("peruano", "Perú"), ("peru", "Perú"),
    ("boliviano", "Bolivia"), ("bolivia", "Bolivia"),
    ("paraguayo", "Paraguay"), ("paraguay", "Paraguay"),
    ("uruguayo", "Uruguay"), ("uruguay", "Uruguay"),
    ("guatemalteco", "Guatemala"), ("guatemala", "Guatemala"),
    ("hondureño", "Honduras"), ("honduras", "Honduras"),
    ("salvadoreño", "El Salvador"), ("el salvador", "El Salvador"),
    ("nicaragüense", "Nicaragua"), ("nicaragua", "Nicaragua"),
    ("costarricense", "Costa Rica"), ("costa rica", "Costa Rica"),
    ("panameño", "Panamá"), ("panama", "Panamá"),
    ("cubano", "Cuba"), ("cuba", "Cuba"),
    ("dominicano", "República Dominicana"), ("república dominicana", "República Dominicana"),
    ("haitiano", "Haití"), ("haiti", "Haití"),
    # North America
    ("mexicano", "México"), ("mexicana", "México"), ("mexico", "México"),
    ("estadounidense", "Estados Unidos"), ("estados unidos", "Estados Unidos"),
    ("usa", "Estados Unidos"), ("eeuu", "Estados Unidos"),
    ("canadiense", "Canadá"), ("canada", "Canadá"),
    # Europe
    ("español", "España"), ("espanol", "España"), ("españa", "España"),
    ("francés", "Francia"), ("frances", "Francia"), ("france", "Francia"), ("francia", "Francia"),
    ("alemán", "Alemania"), ("aleman", "Alemania"), ("alemania", "Alemania"),
    ("italiano", "Italia"), ("italiana", "Italia"), ("italia", "Italia"),
    ("británico", "Reino Unido"), ("britanico", "Reino Unido"), ("reino unido", "Reino Unido"),
    ("inglés", "Reino Unido"), ("ingles", "Reino Unido"),
    ("portugués", "Portugal"), ("portugues", "Portugal"), ("portugal", "Portugal"),
    ("holandés", "Países Bajos"), ("holanda", "Países Bajos"), ("países bajos", "Países Bajos"),
    ("belga", "Bélgica"), ("belgica", "Bélgica"), ("bélgica", "Bélgica"),
    ("suizo", "Suiza"), ("suiza", "Suiza"),
    ("sueco", "Suecia"), ("suecia", "Suecia"),
    ("noruego", "Noruega"), ("noruega", "Noruega"),
    ("danés", "Dinamarca"), ("danes", "Dinamarca"), ("dinamarca", "Dinamarca"),
    ("finlandés", "Finlandia"), ("finlandia", "Finlandia"),
    ("polaco", "Polonia"), ("polonia", "Polonia"),
    ("checo", "República Checa"), ("república checa", "República Checa"),
    ("húngaro", "Hungría"), ("hungria", "Hungría"),
    ("rumano", "Rumanía"), ("rumania", "Rumanía"),
    ("griego", "Grecia"), ("grecia", "Grecia"),
    ("turco", "Turquía"), ("turquia", "Turquía"), ("turquía", "Turquía"),
    ("ruso", "Rusia"), ("rusia", "Rusia"), ("russia", "Rusia"),
    ("ucraniano", "Ucrania"), ("ucrania", "Ucrania"),
    ("serbio", "Serbia"), ("serbia", "Serbia"),
    # Asia-Pacific
    ("japonés", "Japón"), ("japonesa", "Japón"), ("japones", "Japón"),
    ("japón", "Japón"), ("japon", "Japón"), ("japan", "Japón"),
    ("chino", "China"), ("china", "China"),
    ("surcoreano", "Corea del Sur"), ("corea del sur", "Corea del Sur"),
    ("norcoreano", "Corea del Norte"), ("corea del norte", "Corea del Norte"),
    ("coreano", "Corea"), ("corea", "Corea"),
    ("indio", "India"), ("india", "India"),
    ("paquistaní", "Pakistán"), ("pakistan", "Pakistán"),
    ("bangladesí", "Bangladesh"), ("bangladesh", "Bangladesh"),
    ("indonesio", "Indonesia"), ("indonesia", "Indonesia"),
    ("filipino", "Filipinas"), ("filipinas", "Filipinas"),
    ("vietnamita", "Vietnam"), ("vietnam", "Vietnam"),
    ("tailandés", "Tailandia"), ("tailandia", "Tailandia"),
    ("malayo", "Malasia"), ("malasia", "Malasia"),
    ("singapurense", "Singapur"), ("singapur", "Singapur"),
    ("australiano", "Australia"), ("australia", "Australia"),
    ("neozelandés", "Nueva Zelanda"), ("nueva zelanda", "Nueva Zelanda"),
    # Middle East & Africa
    ("israelí", "Israel"), ("israel", "Israel"),
    ("palestino", "Palestina"), ("palestina", "Palestina"),
    ("iraní", "Irán"), ("iran", "Irán"),
    ("iraquí", "Irak"), ("irak", "Irak"),
    ("sirio", "Siria"), ("siria", "Siria"),
    ("libanés", "Líbano"), ("libano", "Líbano"),
    ("saudí", "Arabia Saudita"), ("arabia saudita", "Arabia Saudita"), ("saudi", "Arabia Saudita"),
    ("emiratense", "Emiratos Árabes"), ("emiratos arabes", "Emiratos Árabes"),
    ("egipcio", "Egipto"), ("egipto", "Egipto"),
    ("nigeriano", "Nigeria"), ("nigeria", "Nigeria"),
    ("sudafricano", "Sudáfrica"), ("sudafrica", "Sudáfrica"),
    ("etíope", "Etiopía"), ("etiopia", "Etiopía"),
    ("keniata", "Kenia"), ("kenia", "Kenia"),
    ("marroquí", "Marruecos"), ("marruecos", "Marruecos"),
    # Brazil (no adjective form yet)
    ("brasileño", "Brasil"), ("brasileña", "Brasil"), ("brasil", "Brasil"),
)


def _extract_query_geography(text: str) -> Optional[str]:
    lowered = (text or "").lower()
    # 1. Check known country terms (longest match first to avoid "corea" matching before "corea del sur")
    for term, country in sorted(_GEOGRAPHY_TERMS, key=lambda x: -len(x[0])):
        if term in lowered:
            return country
    # 2. Pattern-based fallback: extract word after "de"/"en"/"sobre" before "de esta"/"semana"/"hoy"
    #    e.g. "noticias de turquía de esta semana" → "Turquía"
    #    e.g. "qué pasa en nigeria" → "Nigeria"
    fallback_patterns = [
        r"\b(?:de|en|sobre)\s+([a-záéíóúüñ]{4,})\s+(?:de\s+esta|esta\s+semana|hoy|del\b|esta\b)",
        r"\b(?:de|en|sobre)\s+([a-záéíóúüñ]{4,})\s*$",
        r"\b(?:de|en|sobre)\s+([a-záéíóúüñ]{4,})\s",
    ]
    _geo_stopwords = _GENERIC_WEB_STOPWORDS | {
        "noticia", "noticias", "semana", "semanas", "ultima", "ultimas",
        "ultimo", "ultimos", "última", "últimas", "último", "últimos",
        "reciente", "recientes", "informacion", "información", "tema",
        "seguridad", "economia", "politica", "deporte", "cultura",
    }
    for pattern in fallback_patterns:
        m = re.search(pattern, lowered)
        if m:
            word = m.group(1).strip()
            if word not in _geo_stopwords and len(word) >= 4:
                return word.capitalize()
    return None


_TOPIC_ANGLES: dict[str, list[str]] = {
    "security": [
        "{geo} crimen delincuencia seguridad interna {year}",
        "{geo} defensa militar despliegue fuerzas {year}",
        "{geo} diplomacia tensiones política exterior {year}",
        "{geo} desastre emergencia seguridad civil {year}",
    ],
    "economy": [
        "{geo} economía mercado inversión {year}",
        "{geo} empleo salario empresa {year}",
        "{geo} inflación precios comercio {year}",
        "{geo} tecnología industria energía {year}",
    ],
    "politics": [
        "{geo} gobierno elecciones política {year}",
        "{geo} congreso ley reforma legislación {year}",
        "{geo} oposición partido liderazgo {year}",
        "{geo} corrupción justicia tribunal {year}",
    ],
    "default": [
        "{geo} {topic} noticias recientes {year}",
        "{geo} {topic} novedades actualidad {year}",
        "{geo} {topic} últimas noticias semana {year}",
        "{geo} {topic} hoy noticia {year}",
    ],
}

# English equivalents used as supplementary search fallback when Spanish angles yield < 4 candidates.
_TOPIC_ANGLES_EN: dict[str, list[str]] = {
    "security": [
        "{geo_en} crime internal security {year}",
        "{geo_en} defense military deployment {year}",
        "{geo_en} diplomacy tensions foreign policy {year}",
        "{geo_en} disaster emergency civil security {year}",
    ],
    "economy": [
        "{geo_en} economy market investment {year}",
        "{geo_en} employment wages companies {year}",
        "{geo_en} inflation prices trade {year}",
        "{geo_en} technology industry energy {year}",
    ],
    "politics": [
        "{geo_en} government elections politics {year}",
        "{geo_en} congress law reform legislation {year}",
        "{geo_en} opposition party leadership {year}",
        "{geo_en} corruption justice tribunal {year}",
    ],
    "default": [
        "{geo_en} {topic} recent news {year}",
        "{geo_en} {topic} latest news this week {year}",
        "{geo_en} {topic} today news {year}",
        "{geo_en} {topic} updates {year}",
    ],
}

_GEO_ENGLISH: dict[str, str] = {
    # Latin America
    "Ecuador": "Ecuador", "Argentina": "Argentina", "Colombia": "Colombia",
    "Venezuela": "Venezuela", "Chile": "Chile", "Perú": "Peru",
    "Bolivia": "Bolivia", "Paraguay": "Paraguay", "Uruguay": "Uruguay",
    "Guatemala": "Guatemala", "Honduras": "Honduras", "El Salvador": "El Salvador",
    "Nicaragua": "Nicaragua", "Costa Rica": "Costa Rica", "Panamá": "Panama",
    "Cuba": "Cuba", "República Dominicana": "Dominican Republic", "Haití": "Haiti",
    # North America
    "México": "Mexico", "Estados Unidos": "United States", "Canadá": "Canada",
    # Europe
    "España": "Spain", "Francia": "France", "Alemania": "Germany",
    "Italia": "Italy", "Reino Unido": "United Kingdom", "Portugal": "Portugal",
    "Países Bajos": "Netherlands", "Bélgica": "Belgium", "Suiza": "Switzerland",
    "Suecia": "Sweden", "Noruega": "Norway", "Dinamarca": "Denmark",
    "Finlandia": "Finland", "Polonia": "Poland", "República Checa": "Czech Republic",
    "Hungría": "Hungary", "Rumanía": "Romania", "Grecia": "Greece",
    "Turquía": "Turkey", "Rusia": "Russia", "Ucrania": "Ukraine", "Serbia": "Serbia",
    # Asia-Pacific
    "Japón": "Japan", "China": "China", "Corea del Sur": "South Korea",
    "Corea del Norte": "North Korea", "Corea": "Korea",
    "India": "India", "Pakistán": "Pakistan", "Bangladesh": "Bangladesh",
    "Indonesia": "Indonesia", "Filipinas": "Philippines", "Vietnam": "Vietnam",
    "Tailandia": "Thailand", "Malasia": "Malaysia", "Singapur": "Singapore",
    "Australia": "Australia", "Nueva Zelanda": "New Zealand",
    # Middle East & Africa
    "Israel": "Israel", "Palestina": "Palestine", "Irán": "Iran",
    "Irak": "Iraq", "Siria": "Syria", "Líbano": "Lebanon",
    "Arabia Saudita": "Saudi Arabia", "Emiratos Árabes": "UAE",
    "Egipto": "Egypt", "Nigeria": "Nigeria", "Sudáfrica": "South Africa",
    "Etiopía": "Ethiopia", "Kenia": "Kenya", "Marruecos": "Morocco",
    # Brazil
    "Brasil": "Brazil",
}

_PERIODICOS_CONTINENT_SLUG_BY_COUNTRY: dict[str, str] = {
    # Latin America
    "Argentina": "sudamerica",
    "Bolivia": "sudamerica",
    "Brasil": "sudamerica",
    "Chile": "sudamerica",
    "Colombia": "sudamerica",
    "Ecuador": "sudamerica",
    "Paraguay": "sudamerica",
    "Perú": "sudamerica",
    "Uruguay": "sudamerica",
    "Venezuela": "sudamerica",
    # North/Central America + Caribbean
    "Canadá": "norteamerica",
    "Costa Rica": "centroamerica",
    "Cuba": "centroamerica",
    "El Salvador": "centroamerica",
    "Estados Unidos": "norteamerica",
    "Guatemala": "centroamerica",
    "Haití": "centroamerica",
    "Honduras": "centroamerica",
    "México": "norteamerica",
    "Nicaragua": "centroamerica",
    "Panamá": "centroamerica",
    "República Dominicana": "centroamerica",
    # Europe
    "Alemania": "europa",
    "Bélgica": "europa",
    "Dinamarca": "europa",
    "España": "europa",
    "Finlandia": "europa",
    "Francia": "europa",
    "Grecia": "europa",
    "Hungría": "europa",
    "Italia": "europa",
    "Noruega": "europa",
    "Países Bajos": "europa",
    "Polonia": "europa",
    "Portugal": "europa",
    "Reino Unido": "europa",
    "República Checa": "europa",
    "Rumanía": "europa",
    "Rusia": "europa",
    "Serbia": "europa",
    "Suecia": "europa",
    "Suiza": "europa",
    "Turquía": "europa",
    "Ucrania": "europa",
    # Asia-Pacific
    "Australia": "asia",
    "Bangladesh": "asia",
    "China": "asia",
    "Corea": "asia",
    "Corea del Norte": "asia",
    "Corea del Sur": "asia",
    "Filipinas": "asia",
    "India": "asia",
    "Indonesia": "asia",
    "Japón": "asia",
    "Malasia": "asia",
    "Nueva Zelanda": "asia",
    "Pakistán": "asia",
    "Singapur": "asia",
    "Tailandia": "asia",
    "Vietnam": "asia",
    # Middle East & Africa
    "Arabia Saudita": "medio-oriente",
    "Egipto": "africa",
    "Emiratos Árabes": "medio-oriente",
    "Etiopía": "africa",
    "Irak": "medio-oriente",
    "Irán": "medio-oriente",
    "Israel": "medio-oriente",
    "Kenia": "africa",
    "Líbano": "medio-oriente",
    "Marruecos": "africa",
    "Nigeria": "africa",
    "Palestina": "medio-oriente",
    "Siria": "medio-oriente",
    "Sudáfrica": "africa",
}


def _detect_news_topic(query: str) -> str:
    lowered = query.lower()
    if any(k in lowered for k in ["seguridad", "security", "crimen", "defensa", "militar", "policía", "policia", "terroris", "ataque", "atentado", "conflicto"]):
        return "security"
    if any(k in lowered for k in ["economía", "economia", "mercado", "bolsa", "precio", "inflacion", "inflación", "pib", "empleo", "comercio", "empresa"]):
        return "economy"
    if any(k in lowered for k in ["política", "politica", "gobierno", "elección", "eleccion", "presidente", "congreso", "partido", "ministro"]):
        return "politics"
    return "default"


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


_COUNTRY_PRESS_SECTION_PATHS: dict[str, dict[str, list[tuple[str, str]]]] = {
    # ── ITALIA ─────────────────────────────────────────────────────────────
    "ansa.it": {
        "security": [("/sito/notizie/cronaca/cronaca.shtml", "cronaca")],
        "politics": [("/sito/notizie/politica/politica.shtml", "politica")],
        "economy": [("/sito/notizie/economia/economia.shtml", "economia")],
        "default": [("/sito/notizie/cronaca/cronaca.shtml", "cronaca")],
    },
    "repubblica.it": {
        "security": [("/cronaca/", "cronaca")],
        "politics": [("/politica/", "politica")],
        "default": [("/cronaca/", "cronaca")],
    },
    "ilmessaggero.it": {
        "security": [("/italia/", "italia"), ("/roma/", "roma")],
        "politics": [("/politica/", "politica"), ("/italia/", "italia")],
        "default": [("/italia/", "italia")],
    },
    "ilfattoquotidiano.it": {
        "security": [("/cronaca/", "cronaca"), ("/", "homepage-cronaca")],
        "politics": [("/politica/", "politica"), ("/", "homepage-politica")],
        "default": [("/", "homepage")],
    },
    "ilfoglio.it": {
        "security": [("/cronaca/", "cronaca"), ("/", "homepage-cronaca")],
        "politics": [("/politica/", "politica"), ("/", "homepage-politica")],
        "default": [("/", "homepage")],
    },
    "ilmanifesto.it": {
        "security": [("/", "homepage-cronaca")],
        "politics": [("/sezioni/politica/", "politica"), ("/", "homepage-politica")],
        "default": [("/", "homepage")],
    },
    "huffingtonpost.it": {
        "security": [("/news/cronaca/", "cronaca"), ("/", "homepage-cronaca")],
        "politics": [("/politica/", "politica"), ("/", "homepage-politica")],
        "default": [("/", "homepage")],
    },
    # ── ESPAÑA ─────────────────────────────────────────────────────────────
    "elpais.com": {
        "security": [("/espana/", "españa"), ("/sociedad/", "sociedad")],
        "politics": [("/espana/", "españa"), ("/politica/", "política")],
        "economy": [("/economia/", "economía"), ("/negocios/", "negocios")],
        "default": [("/espana/", "españa"), ("/actualidad/", "actualidad")],
    },
    "elmundo.es": {
        "security": [("/espana/", "españa"), ("/cronica/", "crónica")],
        "politics": [("/espana/", "españa"), ("/politica/", "política")],
        "economy": [("/economia/", "economía"), ("/mercados/", "mercados")],
        "default": [("/espana/", "españa")],
    },
    "abc.es": {
        "security": [("/espana/", "españa"), ("/sociedad/", "sociedad")],
        "politics": [("/espana/", "españa"), ("/politica/", "política")],
        "economy": [("/economia/", "economía")],
        "default": [("/espana/", "españa")],
    },
    "lavanguardia.com": {
        "security": [("/sucesos/", "sucesos"), ("/vida/sucesos-y-tribunales/", "sucesos")],
        "politics": [("/politica/", "política"), ("/internacional/", "internacional")],
        "economy": [("/economia/", "economía"), ("/finanzas/", "finanzas")],
        "default": [("/politica/", "política"), ("/vida/", "vida")],
    },
    "elconfidencial.com": {
        "security": [("/espana/", "españa"), ("/sociedad/", "sociedad")],
        "politics": [("/espana/", "españa"), ("/politica/", "política")],
        "economy": [("/economia/", "economía"), ("/mercados/", "mercados")],
        "default": [("/espana/", "españa")],
    },
    "20minutos.es": {
        "security": [("/nacional/", "nacional"), ("/sociedad/", "sociedad")],
        "politics": [("/politica/", "política"), ("/nacional/", "nacional")],
        "economy": [("/economia/", "economía")],
        "default": [("/nacional/", "nacional")],
    },
    "eldiario.es": {
        "security": [("/sociedad/", "sociedad"), ("/espana/", "españa")],
        "politics": [("/politica/", "política"), ("/espana/", "españa")],
        "economy": [("/economia/", "economía")],
        "default": [("/espana/", "españa")],
    },
    "publico.es": {
        "security": [("/sociedad/", "sociedad"), ("/espana/", "españa")],
        "politics": [("/politica/", "política"), ("/espana/", "españa")],
        "economy": [("/economia/", "economía")],
        "default": [("/espana/", "españa")],
    },
    "larazon.es": {
        "security": [("/espana/", "españa"), ("/sociedad/", "sociedad")],
        "politics": [("/espana/", "españa"), ("/politica/", "política")],
        "default": [("/espana/", "españa")],
    },
    "cadenaser.com": {
        "security": [("/noticias/nacional/", "nacional"), ("/noticias/sociedad/", "sociedad")],
        "politics": [("/noticias/politica/", "política"), ("/noticias/nacional/", "nacional")],
        "default": [("/noticias/", "noticias")],
    },
    "rtve.es": {
        "security": [("/noticias/espana/", "españa"), ("/noticias/sociedad/", "sociedad")],
        "politics": [("/noticias/politica/", "política"), ("/noticias/espana/", "españa")],
        "economy": [("/noticias/economia/", "economía")],
        "default": [("/noticias/", "noticias")],
    },
    # ── ARGENTINA ──────────────────────────────────────────────────────────
    "lanacion.com.ar": {
        "security": [("/seguridad/", "seguridad"), ("/politica/", "política")],
        "politics": [("/politica/", "política"), ("/el-mundo/", "el-mundo")],
        "economy": [("/economia/", "economía"), ("/negocios/", "negocios")],
        "default": [("/ultimo-momento/", "último momento"), ("/", "homepage")],
    },
    "infobae.com": {
        "security": [("/sociedad/", "sociedad"), ("/politica/", "política")],
        "politics": [("/politica/", "política"), ("/america/america-latina/", "america")],
        "economy": [("/economia/", "economía"), ("/finanzas/", "finanzas")],
        "default": [("/sociedad/", "sociedad"), ("/", "homepage")],
    },
    "pagina12.com.ar": {
        "security": [("/secciones/el-pais/", "el-país"), ("/secciones/sociedad/", "sociedad")],
        "politics": [("/secciones/el-pais/", "el-país"), ("/secciones/", "secciones")],
        "economy": [("/secciones/economia/", "economía"), ("/secciones/", "secciones")],
        "default": [("/secciones/el-pais/", "el-país")],
    },
    "perfil.com": {
        "security": [("/noticias/policial/", "policial"), ("/noticias/policial.html", "policial")],
        "politics": [("/noticias/politica/", "política"), ("/noticias/politica.html", "política")],
        "economy": [("/noticias/economia/", "economía")],
        "default": [("/noticias/", "noticias")],
    },
    "cronica.com.ar": {
        "security": [("/categoria/policiales/", "policiales"), ("/categoria/", "noticias")],
        "politics": [("/categoria/politica/", "política")],
        "economy": [("/categoria/economia/", "economía")],
        "default": [("/categoria/policiales/", "policiales")],
    },
    "clarin.com": {
        "security": [("/policiales/", "policiales"), ("/sociedad/", "sociedad")],
        "politics": [("/politica/", "política"), ("/zona/", "zona")],
        "economy": [("/economia/", "economía"), ("/negocios/", "negocios")],
        "default": [("/ultimo-momento/", "último momento")],
    },
    "ambito.com": {
        "security": [("/politica/", "política"), ("/economia/", "economía")],
        "politics": [("/politica/", "política")],
        "economy": [("/economia/", "economía"), ("/finanzas/", "finanzas")],
        "default": [("/politica/", "política")],
    },
    "tiempoar.com.ar": {
        "security": [("/secciones/el-pais/", "el-país"), ("/", "homepage")],
        "politics": [("/secciones/el-pais/", "el-país")],
        "default": [("/", "homepage")],
    },
    # ── CHILE ──────────────────────────────────────────────────────────────
    "emol.com": {
        "security": [("/noticias/nacional/", "nacional"), ("/noticias/policial/", "policial")],
        "politics": [("/noticias/nacional/", "nacional"), ("/noticias/politica/", "política")],
        "economy": [("/noticias/economia/", "economía")],
        "default": [("/noticias/nacional/", "nacional")],
    },
    "latercera.com": {
        "security": [("/nacional/", "nacional"), ("/politica/", "política")],
        "politics": [("/politica/", "política"), ("/nacional/", "nacional")],
        "economy": [("/pulso/", "pulso"), ("/negocios/", "negocios")],
        "default": [("/nacional/", "nacional")],
    },
    # ── MÉXICO ─────────────────────────────────────────────────────────────
    "eluniversal.com.mx": {
        "security": [("/nacion/seguridad/", "seguridad"), ("/estados/", "estados")],
        "politics": [("/nacion/politica/", "política"), ("/nacion/", "nación")],
        "economy": [("/finanzas/", "finanzas"), ("/economia/", "economía")],
        "default": [("/nacion/", "nación")],
    },
    "milenio.com": {
        "security": [("/policia/", "policía"), ("/estados/", "estados")],
        "politics": [("/politica/", "política"), ("/mexico/", "méxico")],
        "economy": [("/negocios/", "negocios")],
        "default": [("/policia/", "policía")],
    },
    # ── COLOMBIA ───────────────────────────────────────────────────────────
    "eltiempo.com": {
        "security": [("/justicia/", "justicia"), ("/colombia/", "colombia")],
        "politics": [("/politica/", "política"), ("/colombia/", "colombia")],
        "economy": [("/economia/", "economía"), ("/negocios/", "negocios")],
        "default": [("/colombia/", "colombia")],
    },
}

_GENERIC_SECTION_PATHS: dict[str, list[tuple[str, str]]] = {
    "security": [
        ("/seguridad/", "seguridad"),
        ("/policiales/", "policiales"),
        ("/sociedad/", "sociedad"),
        ("/sucesos/", "sucesos"),
        ("/espana/", "españa"),
        ("/nacional/", "nacional"),
        ("/cronaca/", "cronaca"),
        ("/", "homepage"),
    ],
    "politics": [
        ("/politica/", "política"),
        ("/espana/", "españa"),
        ("/nacional/", "nacional"),
        ("/gobierno/", "gobierno"),
        ("/nacion/", "nación"),
        ("/secciones/el-pais/", "el-país"),
        ("/", "homepage"),
    ],
    "economy": [
        ("/economia/", "economía"),
        ("/finanzas/", "finanzas"),
        ("/negocios/", "negocios"),
        ("/mercados/", "mercados"),
        ("/", "homepage"),
    ],
    "default": [
        ("/noticias/", "noticias"),
        ("/actualidad/", "actualidad"),
        ("/ultimo-momento/", "último momento"),
        ("/nacional/", "nacional"),
        ("/espana/", "españa"),
        ("/", "homepage"),
    ],
}

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


def _build_country_press_section_targets(domain: str, fallback_url: str, last_message: str) -> list[tuple[str, str]]:
    topic = _detect_news_topic(last_message)
    base = (fallback_url or f"https://{domain}/").strip() or f"https://{domain}/"
    if not base.endswith("/"):
        base = base + "/"
    domain_map = _COUNTRY_PRESS_SECTION_PATHS.get(domain, {})
    candidates = list(domain_map.get(topic) or domain_map.get("default") or [])
    if not candidates:
        candidates = list(_GENERIC_SECTION_PATHS.get(topic, _GENERIC_SECTION_PATHS["default"]))
    built: list[tuple[str, str]] = []
    seen: set[str] = set()
    for path, label in candidates:
        full_url = base if path == "/" else urljoin(base, path.lstrip("/"))
        if full_url in seen:
            continue
        seen.add(full_url)
        built.append((full_url, label))
    return built[:4]


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


_MONTH_NAMES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}
_MONTH_NAMES_EN = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _candidate_url_is_recent(url: str, days_threshold: int) -> bool:
    """Returns True if the date embedded in the URL is within `days_threshold` days, or no date found.

    Handles separated dates (/2026/04/02/), compact dates (yjj20260402...), and
    month-name slugs in Spanish or English (e.g. julio-2025, march-2025).
    """
    import datetime
    today = datetime.date.today()
    cutoff = today - datetime.timedelta(days=days_threshold)
    lowered = (url or "").lower()
    # Separated numeric: /2026/04/02/ or /2026-04-02
    for match in re.finditer(r"[/\-](\d{4})[/\-](\d{2})[/\-](\d{2})", lowered):
        try:
            article_date = datetime.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            return article_date >= cutoff
        except ValueError:
            pass
    # Compact: YYYYMMDD anywhere in the URL slug (e.g. yjj20260212... or 20260402_news)
    for match in re.finditer(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])", lowered):
        try:
            article_date = datetime.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            return article_date >= cutoff
        except ValueError:
            pass
    # Month-name slug: "julio-2025", "march-2025", "2025-julio", "2025-march"
    all_months = {**_MONTH_NAMES_ES, **_MONTH_NAMES_EN}
    month_pattern = "|".join(re.escape(m) for m in all_months)
    for match in re.finditer(
        rf"(?:({month_pattern})[- ](\d{{4}})|(\d{{4}})[- ]({month_pattern}))", lowered
    ):
        month_name = match.group(1) or match.group(4)
        year_str = match.group(2) or match.group(3)
        try:
            article_date = datetime.date(int(year_str), all_months[month_name], 1)
            return article_date >= cutoff
        except (ValueError, KeyError):
            pass
    return True  # No date in URL → don't filter (assume recent)


_TITLE_STOPWORDS = {
    "de", "la", "el", "en", "a", "los", "las", "del", "que", "un", "una",
    "por", "con", "se", "ha", "al", "es", "su", "y", "e", "o", "the", "of",
    "in", "to", "a", "and", "for", "on", "at", "by", "with", "from",
}


def _text_keywords(text: str) -> set[str]:
    return {w.lower() for w in text.split() if len(w) > 4 and w.lower() not in _TITLE_STOPWORDS}


_NON_NEWS_DOMAINS = {
    "travel", "tourism", "tripadvisor", "lonelyplanet", "fodors", "frommers",
    "wikivoyage", "wikipedia", "wikitravel", "about.com", "tripsavvy",
    "smartertravel", "booking.com", "expedia", "airbnb",
    # statistics / data aggregators — evergreen content, never news
    "numbeo.com", "statista.com", "macrotrends.net", "worldometers.info",
    "tradingeconomics.com", "indexmundi.com", "globaleconomy.com",
    "countrymeters.info", "globalterrorismindex.org", "visionofhumanity.org",
    # think tanks / advocacy orgs — policy analysis, not news reporting
    "brennancenter.org", "aclu.org", "cato.org", "heritage.org",
    "brookings.edu", "cfr.org", "chathamhouse.org", "sipri.org",
    "amnesty.org", "hrw.org", "freedomhouse.org",
    "dialogopolitico.org", "csis.org", "rand.org", "wilsoncenter.org",
    # government travel advisory portals — evergreen safety ratings, not news
    "osac.gov", "travel.state.gov", "smartraveller.gov.au",
    "travel.gc.ca", "gov.uk/foreign-travel-advice",
    # travel/community forums — threads stay active for years, not current news
    "losviajeros.com", "foro.travel", "viajeros.com", "tripadvisor.com",
    "lonelyplanet.com/thorntree", "reddit.com/r/travel",
}

# URL path segments that indicate forums, threads, or community posts — not journalism.
_FORUM_PATH_SEGMENTS = {
    "/foros/", "/forum/", "/forums/", "/thread/", "/threads/",
    "/topic/", "/topics/", "/post/", "/posts/", "/discussion/",
    "/comunidad/", "/community/", "/board/", "/boards/",
}


def _is_non_news_candidate(candidate: dict[str, str]) -> bool:
    """Returns True if the candidate looks like evergreen/travel/wiki/government-PR content rather than news."""
    url = candidate.get("url", "").lower()
    title = candidate.get("title", "").lower()
    snippet = candidate.get("snippet", "").lower()

    if any(domain in url for domain in _NON_NEWS_DOMAINS):
        return True

    # Forum / community thread URLs — content is user-generated and not time-stamped journalism
    if any(seg in url for seg in _FORUM_PATH_SEGMENTS):
        return True

    # Government press-release sections (.gob. / .gov domains with /prensa/ or /comunicado/)
    # These publish project announcements, not journalistic news.
    _GOV_TLD = (".gob.", ".gov.", "/gob.", "/gov.")
    _PR_PATHS = ("/prensa/", "/comunicado", "/nota-de-prensa", "/press-release", "/sala-de-prensa")
    if any(tld in url for tld in _GOV_TLD) and any(path in url for path in _PR_PATHS):
        return True

    # Law firm / legal publisher URLs — client alerts, legal updates, publications
    _LEGAL_PATHS = (
        "/legal-update", "/client-alert", "/client-advisory", "/legal-alert",
        "/publications/", "/publication/", "/insights/", "/knowledge/",
        "/briefing/", "/memorandum/", "/legal-news/",
    )
    if any(seg in url for seg in _LEGAL_PATHS):
        return True

    # Snippets with generic travel-advice patterns (no concrete event, no date)
    evergreen_signals = [
        "se recomienda a los viajeros", "se aconseja a los viajeros",
        "para los turistas", "consejos de seguridad", "guía de viaje",
        "recomendaciones para viajeros", "baja tasa de criminalidad",
        "travel advisory", "safety tips for travelers",
        # travel advisory language
        "ejercer mayor precaución", "ejercer precaución",
        "se desaconseja viajar", "no se recomienda viajar",
        "nivel de alerta de viaje", "travel level", "do not travel",
        "reconsider travel", "exercise increased caution",
        "high threat location", "ubicación de alta amenaza",
    ]
    return any(signal in snippet or signal in title for signal in evergreen_signals)


def _same_event(
    candidate_a: dict[str, str],
    candidate_b: dict[str, str],
    query_terms: Optional[list[str]] = None,
) -> bool:
    """Returns True if two candidates appear to describe the same event.

    Title overlap ≥ 3 shared keywords (excluding query terms) → same event.
    Full-text (title+snippet) overlap ≥ 5 non-query keywords → same event.

    Thresholds intentionally conservative: for topic-rich queries (e.g. Japan security),
    many articles share 2 generic words (china, tensiones) without covering the same story.
    Requiring 3 title-keyword overlap prevents false deduplication across genuinely
    distinct events, which is what produces 4 diverse bullets.
    """
    excluded = set(t.lower() for t in (query_terms or []))
    excluded.update(_TITLE_STOPWORDS)

    def keywords(text: str) -> set[str]:
        return {w.lower() for w in text.split() if len(w) > 4 and w.lower() not in excluded}

    title_kw_a = keywords(candidate_a.get("title", ""))
    title_kw_b = keywords(candidate_b.get("title", ""))
    if len(title_kw_a & title_kw_b) >= 3:
        return True

    full_a = keywords(f"{candidate_a.get('title', '')} {candidate_a.get('snippet', '')}")
    full_b = keywords(f"{candidate_b.get('title', '')} {candidate_b.get('snippet', '')}")
    return len(full_a & full_b) >= 5


def _dedup_candidates_by_event(
    candidates: list[dict[str, str]],
    query_terms: Optional[list[str]] = None,
) -> list[dict[str, str]]:
    """Keep one candidate per event — drop articles that appear to cover the same story."""
    accepted: list[dict[str, str]] = []
    for candidate in candidates:
        if not any(_same_event(candidate, a, query_terms) for a in accepted):
            accepted.append(candidate)
    return accepted


def _extract_generic_search_candidates(search_text: str) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    current: Optional[dict[str, str]] = None

    for line in [line.rstrip() for line in (search_text or "").splitlines() if line.strip()]:
        # Handle both "1. [title](url)" and "1. [article] [title](url)" / "1. [hub] [title](url)"
        item_match = re.match(r"^\d+\. (?:\[(article|hub)\]\s*)?\[(.+?)\]\((https?://[^)]+)\)", line.strip())
        if item_match:
            if current:
                candidates.append(current)
            tag = item_match.group(1) or ""
            current = {
                "title": item_match.group(2).strip(),
                "url": item_match.group(3).strip(),
                "snippet": "",
                "hit_type": tag,
            }
            continue

        if current is not None and not line.startswith("Sources:") and not line.startswith("-") and not line.startswith("Call web_fetch") and not line.startswith("Next step"):
            snippet = line.strip()
            if snippet:
                current["snippet"] = (current.get("snippet", "") + " " + snippet).strip()

    if current:
        candidates.append(current)

    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for candidate in candidates:
        url = candidate.get("url", "")
        if url and url not in seen:
            seen.add(url)
            deduped.append(candidate)
    return deduped


def _score_generic_candidate(candidate: dict[str, str], query_terms: list[str], query_source_group: Optional[str] = None) -> int:
    blob = " ".join([candidate.get("title", ""), candidate.get("snippet", ""), candidate.get("url", "")]).lower()
    score = 0
    for term in query_terms:
        if term in blob:
            score += 3
    if re.search(r"\b\d+\s*-\s*\d+\b", blob):
        score += 2
    url = candidate.get("url", "")
    path = urlparse(url).path.lower()
    segments = [segment for segment in path.split("/") if segment]
    if re.search(r"(19|20)\d{2}[/\-]?\d{2}[/\-]?\d{2}", path) or re.search(r"\d{6,8}", path):
        score += 4
    if len(segments) >= 3:
        score += 2
    if len(segments) <= 2 and not re.search(r"(19|20)\d{2}[/\-]?\d{2}[/\-]?\d{2}", path):
        score -= 3
    if any(seg in {"topic", "topics", "tag", "tags", "category", "categories", "archive", "author"} for seg in segments):
        score -= 4
    if any(noise in blob for noise in ("login", "signin", "cookie", "privacy", "archive", "perfil")):
        score -= 2
    score += score_domain_boost(query_source_group, url)

    # Penalize candidates whose title matches zero query terms — likely off-topic.
    # e.g. a celebrity article mentions "seguridad" in the snippet but the title
    # ("Terrible momento en vivo: la cronista de TN") has no relevant term.
    # Stopword-only terms (len < 4) are excluded from this check to avoid false
    # positives on queries with no long meaningful terms.
    title_lower = candidate.get("title", "").lower()
    snippet_lower = candidate.get("snippet", "").lower()
    meaningful_terms = [t for t in query_terms if len(t) >= 4]
    if meaningful_terms and not any(t in title_lower for t in meaningful_terms):
        if candidate.get("source_kind") == "section_fallback" and any(t in snippet_lower for t in meaningful_terms):
            pass
        else:
            score -= 6
    if candidate.get("source_kind") == "homepage_fallback":
        score -= 8
    if candidate.get("source_kind") == "section_fallback":
        score -= 3
    if _is_hub_like_candidate(candidate):
        score -= 12

    return score


def _candidate_source_priority(candidate: dict[str, str], query_source_group: Optional[str]) -> int:
    url = candidate.get("url", "")
    return get_source_domain_priority(query_source_group, url)


def _rank_candidates_by_source_policy(
    candidates: list[dict[str, str]],
    query_terms: list[str],
    query_source_group: Optional[str],
) -> list[dict[str, str]]:
    if not candidates:
        return []
    if not query_source_group:
        return sorted(
            candidates,
            key=lambda c: _score_generic_candidate(c, query_terms, query_source_group),
            reverse=True,
        )
    return sorted(
        candidates,
        key=lambda c: (
            _candidate_source_priority(c, query_source_group),
            -_score_generic_candidate(c, query_terms, query_source_group),
        ),
    )


def _candidate_snippet_lines(candidate: dict[str, str]) -> list[str]:
    snippet = (candidate.get("snippet") or "").strip()
    if not snippet:
        return []
    snippet = re.sub(r"^#+\s+", "", snippet)
    snippet = re.sub(r"\s+#+\s+", " ", snippet).strip()
    if len(snippet.split()) < 4:
        return []
    return [snippet]


def _is_hub_like_candidate(candidate: dict[str, str]) -> bool:
    url = (candidate.get("url") or "").lower()
    hit_type = (candidate.get("hit_type") or "").lower()
    if hit_type == "hub":
        return True
    path = urlparse(url).path.lower().rstrip("/")
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return True
    structurally_invalid_segments = {
        "edizioni",
        "editioni",
        "dalle_sezioni_mobile.html",
        "gli-inserti-del-foglio",
        "conosci-i-foglianti",
        "ultima-ora",
    }
    if any(segment in structurally_invalid_segments for segment in segments):
        return True
    if any(token in path for token in ("/edizioni/", "/dalle_sezioni_mobile", "/gli-inserti-del-foglio", "/conosci-i-foglianti", "/t/")):
        return True
    if any(token in path for token in ("/tag/", "/tags/", "/autori/", "/authors/", "/argomenti/", "/rubriche/")):
        return True
    if path.endswith(("/news.shtml", "/index.shtml", "/cronaca.shtml", "/politica.shtml", "/economia.shtml")):
        return True
    if segments[-1] in {"politica", "cronaca", "economia", "sport", "archive", "archivio", "topnews", "ultimaora"}:
        return True
    if (
        len(segments) <= 2
        and any(seg in {"politica", "cronaca", "economia", "sport"} for seg in segments)
        and not any("-" in seg or "_" in seg or re.search(r"\d", seg) for seg in segments)
    ):
        return True
    return False


def _query_targets_public_safety(query: str) -> bool:
    normalized = _strip_accents((query or "").lower())
    signals = (
        "seguridad",
        "security",
        "sicurezza",
        "policia",
        "police",
        "polizia",
        "crime",
        "crimen",
        "cronaca",
        "public safety",
        "orden publico",
        "orden público",
    )
    return any(signal in normalized for signal in signals)


def _is_tangential_vertical_candidate(candidate: dict[str, str], query: str) -> bool:
    if not _query_targets_public_safety(query):
        return False
    path = urlparse(candidate.get("url", "")).path.lower()
    title = (candidate.get("title") or "").lower()
    blob = f"{path} {title}"
    return any(
        token in blob
        for token in (
            "/canale_motori/",
            "/motori/",
            "/auto/",
            "/sicurezza-informatica",
            "sicurezza informatica",
            "cybersecurity",
            "ciberseguridad",
            "motori",
            "automotive",
            "sicurezza stradale",
            "sicurezza vial",
            "road safety",
        )
    )


def _is_invalid_news_candidate(candidate: dict[str, str], query: str) -> bool:
    return _is_hub_like_candidate(candidate) or _is_tangential_vertical_candidate(candidate, query)


def _candidate_url_has_date(url: str) -> bool:
    lowered = (url or "").lower()
    if re.search(r"[/\-](\d{4})[/\-](\d{2})[/\-](\d{2})", lowered):
        return True
    if re.search(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])", lowered):
        return True
    all_months = {**_MONTH_NAMES_ES, **_MONTH_NAMES_EN}
    month_pattern = "|".join(re.escape(m) for m in all_months)
    return bool(
        re.search(rf"(?:({month_pattern})[- ](\d{{4}})|(\d{{4}})[- ]({month_pattern}))", lowered)
    )


def _strip_accents(text: str) -> str:
    """Remove diacritics so 'japon' matches 'japón', 'ultima' matches 'última', etc."""
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


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


_NO_INFO_RE = re.compile(
    # Pattern A: "no/sin + verb + noticias/información/news/contenido"
    # Catches: "no proporciona noticias", "no incluye noticias recientes", "no hay noticias"
    r"\b(?:no|sin)\b.{0,50}\b(?:noticias?|informacion|news|contenido relevante)\b"
    # Pattern B: "subject + no + verb" — "la información proporcionada no incluye"
    r"|\b(?:informacion|pagina|sitio|texto|contenido)\b.{0,40}\bno\b.{0,40}"
    r"\b(?:incluye|proporciona|contiene|ofrece|tiene|encontr)\b"
    # Pattern C: explicit "doesn't address this topic" meta-commentary
    r"|sin abordar (?:directamente|este tema|el tema)"
    r"|no aborda (?:directamente|este tema)"
    r"|no trata (?:directamente|este tema)"
    r"|informacion proporcionada se centra en"
    # Pattern D: English equivalents
    r"|does not (?:contain|provide|include) (?:information|news|relevant)"
    r"|no relevant information|no results found|without relevant information",
    re.DOTALL,
)


def _is_no_info_response(text: str) -> bool:
    lowered = _strip_accents((text or "").lower())
    return bool(_NO_INFO_RE.search(lowered))


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


def _slugify_periodicos_label(value: str) -> str:
    normalized = _strip_accents((value or "").lower())
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return normalized


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
    from tools.web_tools import fetch_web_page

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


def _extract_web_fetch_redirect_url(result_text: str) -> Optional[str]:
    match = re.search(r"^Redirect URL:\s*(https?://\S+)$", result_text or "", re.MULTILINE)
    if match:
        return match.group(1).strip().rstrip(".,;:")
    return None


async def _run_week_search_candidates(
    last_message: str,
    search_age_days: Optional[int],
    query_terms: list[str],
    query_source_group: Optional[str],
    web_search_runtime_args: Optional[dict[str, Any]] = None,
) -> tuple[list[dict[str, str]], str]:
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


async def _run_generic_web_search_strategy_impl(
    last_message: str,
    web_search_runtime_args: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    from tools import search_web
    from tools.web_tools import fetch_web_page

    query_terms = _extract_generic_query_terms(last_message)
    query_source_group = detect_query_source_group(last_message)
    source_terms = list(get_query_source_terms(last_message))
    if source_terms:
        merged_terms: list[str] = []
        for term in query_terms + source_terms:
            if term not in merged_terms:
                merged_terms.append(term)
        query_terms = merged_terms
    query_horizon = detect_recent_query_horizon(last_message) if _is_recent_web_information_query(last_message) else None
    recent_requirements = get_recent_query_requirements(query_horizon)
    recent_min_score = recent_requirements["min_score"]
    recent_min_body_lines = recent_requirements["min_body_lines"]
    recent_min_sources = recent_requirements["min_sources"]
    recent_min_candidates = recent_requirements["min_candidates"]
    recent_candidate_min_score = recent_requirements["candidate_min_score"] or recent_min_score
    recent_candidate_min_body_lines = 1 if query_horizon == "week" else recent_min_body_lines
    recent_candidate_min_sources = recent_requirements["candidate_min_sources"] or 1
    loop = asyncio.get_running_loop()

    # Date filter for search.
    # "esta semana/hoy" → 14 days (not 7 — Tavily returns only hub/portal pages with days=7,
    # no specific articles; 14 still excludes content older than 2 weeks like Dec-2025 events).
    # Any other recent news query → 30 days.
    search_age_days: Optional[int] = None
    if query_horizon == "week":
        search_age_days = 14
    elif query_horizon == "month":
        search_age_days = 45
    elif _is_recent_web_information_query(last_message):
        search_age_days = 30

    if query_horizon == "week":
        # OpenClaw-style search: one provider-backed result set, then rank/deduplicate
        # the returned candidates before fetching article pages.
        diverse_candidates, search_text = await _run_week_search_candidates(
            last_message, search_age_days, query_terms, query_source_group, web_search_runtime_args
        )
        local_source_strategy = _country_press_strategy_cache_get(query_source_group, source_terms)
        _web_debug(
            "generic_fetch.week_candidates",
            query=last_message,
            search_age_days=search_age_days,
            candidate_count=len(diverse_candidates),
            local_source_strategy=local_source_strategy,
            urls=[c.get("url", "") for c in diverse_candidates],
        )
        if query_source_group and not diverse_candidates and local_source_strategy in {"cache", "directory", "policy", "lookup"}:
            return _build_no_local_sources_response(last_message)
    else:
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

        # Run second search if fewer than 3 distinct events found
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
        # Match URLs that look like specific articles: date in path, or slug ≥15 chars
        # 15 chars covers nippon.com IDs like yjj2026040500456 (16 chars)
        _url_date_re = re.compile(
            r"/\d{4}/\d{2}/\d{2}/|/\d{8}[-_]|\d{4}-\d{2}-\d{2}"
            r"|/[a-z0-9-]{15,}/?$"  # article slug (was 30, lowered to 15)
        )
        week_entry_lines: list[str] = []
        week_snippet_lines: list[str] = []
        week_entry_sources: list[dict[str, str]] = []
        week_snippet_sources: list[dict[str, str]] = []
        seen_week_urls: set[str] = set()
        fetch_prompt_week = _build_generic_fetch_prompt(last_message)

        async def _week_entry(c: dict[str, str]) -> tuple[str, str, bool]:
            url = c.get("url", "")
            snippet = c.get("snippet", "").strip()
            # Strip markdown from snippet
            snippet = re.sub(r"^#+\s+", "", snippet)
            snippet = re.sub(r"\s+#+\s+", " ", snippet).strip()
            title = re.sub(r"^#+\s+", "", c.get("title", url)).strip()
            # Fetch if URL looks like a specific article: date in path, long slug,
            # or non-trivial path with ≥5-char last segment (catches nippon.com IDs like d01194)
            _path_fetch = urlparse(url).path.rstrip("/")
            _last_seg_fetch = _path_fetch.rsplit("/", 1)[-1] if _path_fetch else ""
            _is_article_url_fetch = bool(_url_date_re.search(url)) or (
                _path_fetch.count("/") >= 2 and len(_last_seg_fetch) >= 5
            )
            if _is_article_url_fetch:
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
            # Fallback to Tavily snippet
            return title, snippet, True

        week_results = await asyncio.gather(*[_week_entry(c) for c in diverse_candidates])

        for (title, content, from_snippet), c in zip(week_results, diverse_candidates):
            min_words = 4 if from_snippet else 8
            if not content or len(content.split()) < min_words:
                continue
            # Discard entries that are no-info placeholders
            if _is_no_info_response(content):
                continue
            url = c.get("url", "")
            if url in seen_week_urls:
                continue
            seen_week_urls.add(url)
            week_entry_lines.append(f"[{title}] — {content}")
            if from_snippet:
                week_snippet_lines.append(f"[{title}] — {content}")
            # Include URL in Sources if it looks like a specific article:
            # has date/long-slug pattern, OR has a non-trivial path (≥2 segments, last ≥5 chars)
            path = urlparse(url).path.rstrip("/")
            last_segment = path.rsplit("/", 1)[-1] if path else ""
            is_article_url = bool(_url_date_re.search(url)) or (
                path.count("/") >= 2 and len(last_segment) >= 5
            )
            if is_article_url:
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

    fetch_prompt = _build_generic_fetch_prompt(last_message)

    async def _fetch_candidate(candidate: dict[str, str]) -> tuple[dict[str, str], Any]:
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
            if score < recent_min_score or len(body_lines) < recent_candidate_min_body_lines:
                _web_debug(
                    "generic_fetch.entry_rejected_recent_threshold",
                    url=candidate.get("url", ""),
                    score=score,
                    min_score=recent_min_score,
                    body_lines_count=len(body_lines),
                    min_body_lines=recent_candidate_min_body_lines,
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
        if _is_recent_web_information_query(last_message) and len(sources) < recent_candidate_min_sources:
            _web_debug(
                "generic_fetch.entry_rejected_sources",
                url=candidate.get("url", ""),
                source_count=len(sources),
                min_sources=recent_candidate_min_sources,
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


def _enforce_synthesis_format(text: str) -> str:
    """Post-process LLM output to guarantee bullet spacing and strip header artifacts."""
    lines = text.splitlines()
    result: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Remove markdown headers the LLM may have emitted despite instructions
        if re.match(r"^#{1,4}\s", stripped):
            continue
        # Ensure blank line before every bullet (•, -, *) that starts a new point
        if re.match(r"^[•\-\*]\s", stripped) and result and result[-1].strip():
            result.append("")
        result.append(stripped)
        # Ensure blank line after every bullet line (before next non-empty line)
        if re.match(r"^[•\-\*]\s", stripped):
            # Peek ahead: if next non-empty line isn't a blank, we'll add one later
            pass
    # Second pass: ensure blank line after each bullet block
    final: list[str] = []
    for i, line in enumerate(result):
        final.append(line)
        if re.match(r"^[•\-\*]\s", line):
            # Add blank after bullet if next line is non-empty content
            if i + 1 < len(result) and result[i + 1].strip():
                final.append("")
    # Collapse 3+ consecutive blank lines to 2
    collapsed: list[str] = []
    blank_count = 0
    for line in final:
        if not line.strip():
            blank_count += 1
            if blank_count <= 2:
                collapsed.append(line)
        else:
            blank_count = 0
            collapsed.append(line)
    return "\n".join(collapsed).strip()


def _dedup_synthesis_bullets(text: str, query_terms: Optional[list[str]] = None) -> str:
    """Remove duplicate bullets from a synthesized response.

    Two bullets are considered duplicates when their non-query keyword overlap ≥ 3.
    Keeps the longer (more informative) bullet of each duplicate pair.
    """
    excluded = set(t.lower() for t in (query_terms or []))
    excluded.update(_TITLE_STOPWORDS)

    def kw(s: str) -> set[str]:
        words = set()
        for w in s.split():
            w = w.lower()
            if len(w) <= 4 or w in excluded:
                continue
            # Normalize plural: "terremotos" → "terremoto", "desastres" → "desastre"
            if w.endswith("s") and len(w) > 5:
                w = w[:-1]
            words.add(w)
        return words

    # Split into (bullet_block, non_bullet_prefix) sections
    # A bullet block = the • line plus any continuation lines before the next bullet
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if re.match(r"^[•\-\*]\s", line.strip()) and current:
            blocks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))

    # Separate bullet blocks from header/non-bullet prefix
    bullet_blocks: list[str] = []
    prefix_lines: list[str] = []
    for block in blocks:
        if re.match(r"^[•\-\*]\s", block.strip()):
            bullet_blocks.append(block)
        else:
            prefix_lines.append(block)

    _DISASTER_KW = {"terremoto", "sismo", "maremoto", "tsunami", "earthquake", "seismic"}

    # Dedup bullet blocks
    accepted: list[str] = []
    for block in bullet_blocks:
        block_kw = kw(block)
        duplicate = False
        for i, acc in enumerate(accepted):
            acc_kw = kw(acc)
            # Use a lower threshold when both bullets are about natural disasters
            # (earthquake variants share few unique words but are clearly the same topic)
            both_disaster = bool(block_kw & _DISASTER_KW) and bool(acc_kw & _DISASTER_KW)
            dedup_threshold = 2 if both_disaster else 4
            if len(block_kw & acc_kw) >= dedup_threshold:
                # Keep the longer one
                if len(block) > len(acc):
                    accepted[i] = block
                duplicate = True
                break
        if not duplicate:
            accepted.append(block)

    all_parts = prefix_lines + accepted
    return "\n\n".join(p.strip() for p in all_parts if p.strip())


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
        needs_confirmation = bool(explicit_urls)

        confirmed = True
        if needs_confirmation:
            confirmed = False
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
    strategy = ctx["strategy"]
    exploring = ctx["exploring"]
    exp_rate = ctx["exp_rate"]
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
                return {
                    "messages": [AIMessage(content=summary)],
                    "scrape_tracker": new_tracker,
                }

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
                return {
                    "messages": [AIMessage(content=summary)],
                    "scrape_tracker": new_tracker,
                }
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
                return {
                    "messages": [AIMessage(content=summary)],
                    "scrape_tracker": new_tracker,
                }

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
                return {
                    "messages": [AIMessage(content=summary)],
                    "scrape_tracker": new_tracker,
                }
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
                return {
                    "messages": [AIMessage(content=summary)],
                    "scrape_tracker": new_tracker,
                }
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
