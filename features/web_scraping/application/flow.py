"""Caso de uso para el flujo de web scraping.

Coordina HITL, estrategia, guardrails, retry y postcondiciones.
El nodo LangGraph queda como adaptador fino.
"""
import asyncio
import os
import re
import time
import uuid
from urllib.parse import urlparse
from typing import Any, Optional, Callable, Awaitable, Mapping, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig

from application.policies.agentdog import evaluate_trajectory_safe, _should_evaluate_guard, _is_allowed_public_price_request
from core.helpers.audit_flow_helpers import (
    _emit_node_outcome,
    _emit_country_news_metrics,
    _extract_tokens,
    _extract_quality,
    _extract_followup,
    _node_meta,
    _get_model_name,
)
from core.ports.confirmation_port import ConfirmationPort
from core.helpers.message_flow_helpers import extract_final_ai_text, get_last_message_text, is_web_information_query
from core.helpers.trace_flow_helpers import get_or_create_request_id
from application.services.prompt_loader import load_agent_prompt
from features.web_scraping.infrastructure.runtime import WebFetchRequest
from application.policies.security_flow import input_guard
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
from features.web_scraping.domain.country_profile import GEO_ENGLISH
from features.web_scraping.domain.country_resolver import GENERIC_WEB_STOPWORDS, GEOGRAPHY_TERMS, extract_query_geography
from features.web_scraping.domain.topic_detector import TOPIC_ANGLES, TOPIC_ANGLES_EN, detect_news_topic
from features.web_scraping.domain.section_path_resolver import (
    COUNTRY_PRESS_SECTION_PATHS,
    GENERIC_SECTION_PATHS,
    build_country_press_section_targets,
)
from features.web_scraping.infrastructure.country_profile_repo import PERIODICOS_CONTINENT_SLUG_BY_COUNTRY
from core.ports.country_news_ports import (
    ICountryResolver,
    ICountryProfileRepository,
    ISectionPathResolver,
    IPressSourceDiscovery,
    IDynamicPressSourceDiscovery,
)
from core.helpers.url_helpers import _is_article_url, _extract_web_fetch_redirect_url
from features.web_scraping.domain.models import (
    CandidateDict,
    EvidenceKind,
    Recency,
    SourceDict,
    SourceKind,
    Specificity,
    WebCandidate,
)
from features.web_scraping.domain.text_utils import (
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
from features.web_scraping.domain.classifier import (
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
from features.price.application.price_flow_helpers import (
    _detect_coin_from_query,
    _format_price_response,
    _extract_price_from_messages,
    _extract_structured_price,
    _get_crypto_price_fn,
)
from core.domain.models import AgentState


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
    from features.web_scraping.application.postprocess import _finalize_web_user_summary as _impl

    return _impl(summary, last_message, sources)


WebCandidateRecord = WebCandidate



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
    from features.web_scraping.application.postprocess import _build_no_local_sources_response as _impl

    return _impl(last_message)


def _debug_periodicos_fetch(url: str, stage: str) -> bytes:
    from features.web_scraping.infrastructure import scraping_infra

    _web_debug("country_press.directory.fetch_start", stage=stage, url=url)
    html = scraping_infra._fetch_html(url)
    _web_debug("country_press.directory.fetch_success", stage=stage, url=url, bytes=len(html))
    return html


def _web_search_runtime_args(state: Mapping[str, Any]) -> dict[str, Any]:
    from features.web_scraping.application.query_helpers import _web_search_runtime_args as _impl
    return _impl(state)


def _select_strategy_context(state: AgentState, last_message: str, get_runtime_policy: Callable[[], dict]) -> dict:
    from features.web_scraping.application.strategy_context import _select_strategy_context as _impl

    return _impl(state, last_message, get_runtime_policy)


async def _summarize_if_long(
    text: str, rid: str, get_llm_fn: Callable, *, is_retry: bool = False
) -> str:
    from features.web_scraping.application.retry_flow import _summarize_if_long as _impl

    return await _impl(text, rid, get_llm_fn, is_retry=is_retry)


async def _run_retry_agent(
    agent,
    last_message: str,
    rid: str,
    get_llm_fn: Callable,
) -> tuple[Optional[str], list[str], dict[str, Any], dict[str, Any]]:
    from features.web_scraping.application.retry_flow import _run_retry_agent as _impl

    return await _impl(agent, last_message, rid, get_llm_fn)


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
        if re.match(r"^\*\*.+\*\*$", line.strip()):
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

    from features.web_scraping.infrastructure.search_tools import search_web
    from features.web_scraping.infrastructure.scraping_tools import fetch_web_page

    geography = _extract_query_geography(last_message)
    if geography and os.getenv("WEB_PRESS_DIRECTORY_FIRST", "").strip().lower() == "true":
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
    from features.web_scraping.application.country_press_helpers import _run_country_press_search_candidates as _impl

    return await _impl(
        last_message,
        search_age_days,
        query_terms,
        query_source_group,
        source_terms,
        web_search_runtime_args,
        query_horizon=query_horizon,
    )


def _build_generic_fetch_prompt(query: str) -> str:
    from features.web_scraping.application.query_helpers import _build_generic_fetch_prompt as _impl
    return _impl(query)


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

    from features.web_scraping.infrastructure.search_tools import search_web

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
    from features.web_scraping.application.query_helpers import _fetch_web_page_follow_redirect as _impl
    return await _impl(url, prompt, use_dynamic=use_dynamic)


def _build_query_context(last_message: str) -> tuple[QueryContext, RecentPolicy]:
    from features.web_scraping.application.query_helpers import _build_query_context as _impl
    return _impl(last_message)


async def _fetch_and_score_entries(
    ranked_candidates: list[CandidateDict],
    last_message: str,
    ctx: QueryContext,
    policy: RecentPolicy,
    search_text: str,
) -> list[dict[str, Any]]:
    from features.web_scraping.application.search_pipeline import _fetch_and_score_entries as _impl

    return await _impl(ranked_candidates, last_message, ctx, policy, search_text)


async def _run_week_search_pipeline(
    last_message: str,
    ctx: QueryContext,
    web_search_runtime_args: Optional[dict[str, Any]],
) -> tuple[list[CandidateDict], str, Optional[dict[str, Any]]]:
    from features.web_scraping.application.search_pipeline import _run_week_search_pipeline as _impl

    return await _impl(last_message, ctx, web_search_runtime_args)


async def _run_general_search_pipeline(
    last_message: str,
    ctx: QueryContext,
    loop: asyncio.AbstractEventLoop,
    web_search_runtime_args: Optional[dict[str, Any]],
) -> tuple[list[CandidateDict], str]:
    from features.web_scraping.application.query_helpers import _run_general_search_pipeline as _impl

    return await _impl(last_message, ctx, loop, web_search_runtime_args)


async def _run_generic_web_search_strategy_impl(
    last_message: str,
    web_search_runtime_args: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    from features.web_scraping.application.generic_strategy import _run_generic_web_search_strategy_impl as _impl

    return await _impl(last_message, web_search_runtime_args)


async def _run_generic_web_search_fetch(
    last_message: str,
    web_search_runtime_args: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    from features.web_scraping.application.fetch_dispatch import _run_generic_web_search_fetch as _impl

    return await _impl(last_message, web_search_runtime_args)



async def _synthesize_search_summary(
    raw_summary: str,
    query: str,
    get_llm_fn: Callable,
    sources: list[dict[str, str]],
    has_labeled_content: bool = False,
) -> str:
    from features.web_scraping.application.synthesis import _synthesize_search_summary as _impl

    return await _impl(raw_summary, query, get_llm_fn, sources, has_labeled_content)


def _build_web_digest_contract(summary_lines: list[str], sources: list[dict[str, str]], *, intro: str | None = None, conclusion: str | None = None):
    from features.web_scraping.domain.text_utils import build_web_digest_contract as _impl

    return _impl(summary_lines, sources, intro=intro, conclusion=conclusion)


def _format_web_digest_contract(contract):
    from features.web_scraping.domain.text_utils import format_web_digest_contract as _impl

    return _impl(contract)


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

        if confirmation_handler is not None:
            confirmed = await confirmation_handler.confirm(
                f"\n[HITL] web_scraping_agent va a procesar: \"{preview}\"{url_info}\n¿Confirmar? [s/n]: "
            )
        elif ask_confirmation_compat is not None:
            confirmed = await ask_confirmation_compat(
                f"\n[HITL] web_scraping_agent va a procesar: \"{preview}\"{url_info}\n¿Confirmar? [s/n]: "
            )
        else:
            _emit_node_outcome(
                rid, "web_scraping_node", "blocked", phase="pre_guard",
                agent="web_scraping_agent",
                duration_ms=int((time.time() - t0) * 1000),
                reason="hitl_missing_confirmation_handler",
            )
            return {"messages": [AIMessage(content="Operación cancelada: falta un handler de confirmación.")]}
        if not confirmed:
            _emit_node_outcome(
                rid, "web_scraping_node", "blocked", phase="pre_guard",
                agent="web_scraping_agent",
                duration_ms=int((time.time() - t0) * 1000),
                reason="hitl_rejected",
            )
            return {"messages": [AIMessage(content="Operación cancelada por el usuario.")]}

    _MOODLE_KEYWORDS = (
        "moodle", "tarea", "tareas", "entrega", "entregas",
        "trabajo práctico", "trabajos prácticos", "actividad", "actividades",
        "pendiente", "pendientes", "vencida", "vencidas", "campus virtual",
    )
    if any(kw in last_message.lower() for kw in _MOODLE_KEYWORDS):
        from features.web_scraping.infrastructure.scraping_tools import scrape_moodle_assignments
        loop = asyncio.get_running_loop()
        moodle_result = await loop.run_in_executor(
            None, lambda: scrape_moodle_assignments.invoke({})
        )
        if not isinstance(moodle_result, str):
            moodle_result = str(moodle_result)
        _web_debug("run_web_scraping_flow.moodle_shortcut", result_preview=moodle_result[:200])
        return {"messages": [AIMessage(content=moodle_result)]}

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

    guard_result = input_guard({"messages": [HumanMessage(content=last_message)]})
    if isinstance(guard_result, dict) and guard_result.get("blocked"):
        return guard_result

    try:
        if explicit_urls:
            fetch_prompt = last_message.strip() or "Extraé la información relevante de esta URL."
            fetch_guard = input_guard({"messages": [HumanMessage(content=f"URL: {explicit_urls[0]}\n\nPrompt: {fetch_prompt}")]})
            if isinstance(fetch_guard, dict) and fetch_guard.get("blocked"):
                return fetch_guard
            fetch_result = await _fetch_web_page_follow_redirect(explicit_urls[0], fetch_prompt, use_dynamic=True)
            if isinstance(fetch_result, str) and not fetch_result.startswith("Error") and not fetch_result.startswith("URL rechazada"):
                summary = fetch_result.strip()
                summary, _, _ = _finalize_web_user_summary(summary, last_message, None)
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
                _disc_contract = discovery.get("digest_contract")
                if _disc_contract is not None:
                    summary = _format_web_digest_contract(cast(dict[str, Any], _disc_contract))
                elif discovery.get("pre_synthesized"):
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

            from features.web_scraping.infrastructure.search_tools import search_web

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
                _disc_contract = discovery.get("digest_contract")
                if _disc_contract is not None:
                    summary = _format_web_digest_contract(cast(dict[str, Any], _disc_contract))
                elif discovery.get("pre_synthesized"):
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

        from features.web_scraping.application.agent_strategy import _run_web_scraping_agent_strategy as _impl

        return await _impl(
            state=state,
            agent=agent,
            get_llm_fn=get_llm_fn,
            last_message=last_message,
            category=category,
            tracker=tracker,
            turn_count=turn_count,
            prior_score=prior_score,
            prior_reliability=prior_reliability,
            ml_recommended=ml_recommended,
            prediction_match=prediction_match,
            rid=rid,
            t0=t0,
            web_search_runtime_args=web_search_runtime_args,
            should_evaluate_guard_fn=should_evaluate_guard_fn,
            evaluate_trajectory_safe_fn=evaluate_trajectory_safe_fn,
        )

    except Exception as e:
        _emit_node_outcome(
            rid, "web_scraping_node", "error", phase="agent",
            agent="web_scraping_agent",
            duration_ms=int((time.time() - t0) * 1000),
            reason=str(e),
            followup_likely=True,
            **_node_meta(),
        )
        return {"messages": [AIMessage(content="No pude procesar la consulta de forma segura. Probá de nuevo en unos minutos.")]}
