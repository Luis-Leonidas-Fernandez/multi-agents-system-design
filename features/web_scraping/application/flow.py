"""Caso de uso para el flujo de web scraping.

Coordina estrategia, guardrails, retry y postcondiciones.
El nodo LangGraph queda como adaptador fino.
"""
import asyncio
import os
import re
import time
import unicodedata
import uuid
from urllib.parse import urljoin, urlparse
from typing import Any, Optional, Callable, Mapping, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig

from application.policies.agentdog import evaluate_trajectory_safe, _should_evaluate_guard, _is_allowed_public_price_request
from core.helpers.audit_flow_helpers import (
    _emit_guard_audit,
    _emit_node_outcome,
    _emit_country_news_metrics,
    _extract_tokens,
    _extract_quality,
    _extract_followup,
    _node_meta,
    _get_model_name,
)
from core.helpers.url_helpers import _normalize_http_url, _safe_hostname
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
from features.web_scraping.application.linkedin_intent import detect_linkedin_jobs_intent
from features.web_scraping.domain.section_path_resolver import (
    COUNTRY_PRESS_SECTION_PATHS,
    GENERIC_SECTION_PATHS,
    build_country_press_section_targets,
)
from features.web_scraping.domain.query_localization import (
    QuerySpec,
    LocalizedNewsQueryBuilder,
    QueryLocalizationContext,
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
    _is_dirty_section_label,
    _is_prompt_echo_line,
    _is_redirect_payload,
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


_MOODLE_ASSIGNMENT_KEYWORDS = (
    "moodle", "tarea", "tareas", "entrega", "entregas",
    "trabajo práctico", "trabajos prácticos", "actividad", "actividades",
    "pendiente", "pendientes", "vencida", "vencidas", "campus virtual",
)
_MOODLE_COURSE_LIST_KEYWORDS = (
    "mis materias",
    "mis cursos",
    "que materias tengo",
    "qué materias tengo",
    "que cursos tengo",
    "qué cursos tengo",
    "mostrame mis materias",
    "mostrame mis cursos",
    "mostrar mis materias",
    "mostrar mis cursos",
    "lista de materias",
    "lista de cursos",
)
_MOODLE_COURSE_AUDIT_PATTERNS = (
    re.compile(
        r"\b(?:audit[aá]|audita|revis[aá]|revisa|inspeccion[aá]|inspecciona|mostr[aá]|mostra|mostrame|muestrame|muéstrame|abr[ií]|abre|ver)\s+"
        r"(?:la\s+)?(?:materia|curso)\s+(.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:audit[aá]|audita|revis[aá]|revisa|inspeccion[aá]|inspecciona)\s+(.+)$",
        re.IGNORECASE,
    ),
)
_NOTION_SYNC_KEYWORDS = (
    "notion",
    "sincroniza",
    "sincronizá",
    "sincronizar",
    "registrá",
    "registra",
    "registrar",
    "guardá",
    "guarda",
    "guardar",
    "actualizá",
    "actualiza",
    "actualizar",
)


def _normalize_intent_text(message: str) -> str:
    lowered = (message or "").strip().lower()
    normalized = unicodedata.normalize("NFKD", lowered)
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", without_accents).strip()


def _extract_moodle_course_query(message: str) -> str:
    raw_message = (message or "").strip()
    normalized = _normalize_intent_text(raw_message)
    for pattern in _MOODLE_COURSE_AUDIT_PATTERNS:
        match = pattern.search(raw_message)
        if not match:
            continue
        candidate = (match.group(1) or "").strip(" .:;-")
        if not candidate:
            continue
        if pattern is _MOODLE_COURSE_AUDIT_PATTERNS[1] and not any(
            token in normalized for token in ("moodle", "campus virtual", "materia", "curso")
        ):
            continue
        return candidate
    return ""


def _detect_moodle_intent(message: str) -> tuple[Optional[str], str]:
    normalized = _normalize_intent_text(message)
    if any(keyword in normalized for keyword in _MOODLE_COURSE_LIST_KEYWORDS):
        return "course_list", ""

    course_query = _extract_moodle_course_query(message)
    if course_query:
        return "course_audit", course_query

    if any(keyword in normalized for keyword in _MOODLE_ASSIGNMENT_KEYWORDS):
        return "assignments", ""

    return None, ""


def _render_moodle_courses_chat(payload: Mapping[str, Any]) -> str:
    courses = payload.get("courses") if isinstance(payload.get("courses"), list) else []
    if not courses:
        return "No encontré materias visibles en Moodle."

    lines = ["Estas son tus materias visibles en Moodle:\n"]
    for course in courses:
        if not isinstance(course, Mapping):
            continue
        idx = int(course.get("index") or 0)
        name = str(course.get("course_name") or "Materia sin nombre").strip()
        if not name:
            name = "Materia sin nombre"
        lines.append(f"{idx}. {name}")

    lines.append("")
    lines.append("Si querés, decime `auditá la materia 1` o el nombre exacto y te traigo el contenido completo.")
    return "\n".join(lines).strip()


def _render_moodle_course_audit_chat(payload: Mapping[str, Any]) -> str:
    if not bool(payload.get("resolved", True)):
        candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
        lines = [str(payload.get("message") or "No pude resolver la materia de forma unívoca.").strip()]
        if candidates:
            lines.append("")
            lines.append("Candidatas encontradas:")
            for item in candidates[:10]:
                if not isinstance(item, Mapping):
                    continue
                idx = item.get("index")
                name = str(item.get("course_name") or "Materia sin nombre").strip()
                prefix = f"{idx}. " if idx is not None else "- "
                lines.append(f"{prefix}{name}")
        return "\n".join(lines).strip()

    course_name = str(payload.get("course_name") or payload.get("root_page_title") or "Materia Moodle").strip()
    page_count = int(payload.get("page_count") or 0)
    assignment_count = int(payload.get("assignment_count") or 0)
    warning_count = int(payload.get("warning_count") or 0)
    visited_count_raw = int(payload.get("visited_count_raw") or 0)
    retained_page_count = int(payload.get("retained_page_count") or page_count)
    resource_type_counts = payload.get("resource_type_counts") if isinstance(payload.get("resource_type_counts"), Mapping) else {}
    audit_json_path = str(payload.get("audit_json_path") or "").strip()
    audit_schema_path = str(payload.get("audit_schema_path") or "").strip()
    audit_summary_path = str(payload.get("audit_summary_path") or "").strip()

    lines = [
        f"Audité la materia **{course_name}**.",
        "",
        f"- Páginas auditadas: {page_count}",
        f"- Entregas detectadas: {assignment_count}",
        f"- Warnings: {warning_count}",
    ]
    if visited_count_raw or retained_page_count:
        lines.append(f"- Crawl: {visited_count_raw or retained_page_count} visitas / {retained_page_count} páginas retenidas")
    if assignment_count == 0 and resource_type_counts:
        details: list[str] = []
        for key, count_value in sorted(resource_type_counts.items()):
            count = int(count_value or 0)
            if count <= 0:
                continue
            label = str(key).replace("_", " ")
            details.append(f"{count} {label}")
        if details:
            lines.append(
                "- No se detectaron actividades de entrega tipo assignment/workshop. "
                f"Sí se detectaron: {', '.join(details)}."
            )
    if audit_json_path:
        lines.append(f"- JSON audit: {audit_json_path}")
    if audit_schema_path:
        lines.append(f"- Schema: {audit_schema_path}")
    if audit_summary_path:
        lines.append(f"- Resumen: {audit_summary_path}")
    return "\n".join(lines).strip()


def _is_notion_sync_request(message: str) -> bool:
    lowered = (message or "").lower()
    if "notion" not in lowered:
        return False
    return any(keyword in lowered for keyword in _NOTION_SYNC_KEYWORDS)



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
    query_source_group = detect_query_source_group(last_message)
    builder = LocalizedNewsQueryBuilder(debug_hook=_web_debug)
    return builder.build_terms(QueryLocalizationContext(
        geography=geography,
        geo_en=geo_en,
        topic=topic,
        horizon=horizon,
        query_source_group=query_source_group,
        public_safety_query=_query_targets_public_safety(last_message),
    ))


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
    query_source_group = detect_query_source_group(last_message)
    builder = LocalizedNewsQueryBuilder(debug_hook=_web_debug)
    return builder.build_queries(
        domain=domain,
        press_name=press_name,
        context=QueryLocalizationContext(
            geography=geography,
            geo_en=geo_en,
            topic=topic,
            horizon=horizon,
            query_source_group=query_source_group,
            public_safety_query=_query_targets_public_safety(last_message),
        ),
    )


def _build_country_press_search_query_specs(last_message: str, domain: str, press_name: str) -> list[QuerySpec]:
    geography = _extract_query_geography(last_message) or ""
    geo_en = _GEO_ENGLISH.get(geography, geography)
    topic = _detect_news_topic(last_message)
    horizon = detect_recent_query_horizon(last_message)
    query_source_group = detect_query_source_group(last_message)
    builder = LocalizedNewsQueryBuilder(debug_hook=_web_debug)
    return builder.build_query_specs(
        domain=domain,
        press_name=press_name,
        context=QueryLocalizationContext(
            geography=geography,
            geo_en=geo_en,
            topic=topic,
            horizon=horizon,
            query_source_group=query_source_group,
            public_safety_query=_query_targets_public_safety(last_message),
        ),
    )


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
    if _is_prompt_echo_line(line):
        return True
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

_SECTION_DISCOVERY_TIME_BUDGET_SECONDS = 20.0
_NAV_CONTAINER_HINTS = (
    "nav", "menu", "header", "section", "category", "topic", "subnav",
    "drawer", "navbar", "gnb", "lnb", "tab", "breadcrumb", "global", "local",
)
_SECTION_TOPIC_ALIAS_WEIGHTS: dict[str, dict[str, int]] = {
    "security": {
        "security": 6, "safety": 5, "crime": 6, "police": 6, "justice": 5, "court": 4,
        "law": 3, "public safety": 5, "incident": 4, "local": 2, "national": 3, "society": 4,
        "international": 2, "politics": 1, "world": 1,
        "seguridad": 6, "inseguridad": 5, "policial": 6, "policiales": 6, "policia": 6,
        "policía": 6, "sucesos": 5, "justicia": 5, "tribunales": 4, "sociedad": 4,
        "internacional": 2, "política": 1,
        "cronaca": 5, "sicurezza": 6, "polizia": 6, "giustizia": 5, "interni": 3,
        "사회": 5, "사건": 6, "범죄": 6, "경찰": 6, "치안": 6, "수사": 5, "법원": 4, "국내": 2,
        "국제": 2, "정치": 1,
        "事件": 6, "犯罪": 6, "警察": 6, "治安": 6, "司法": 5,
        "治安": 6, "警察": 6, "犯罪": 6, "事件": 6, "社会": 4,
    },
    "politics": {
        "politics": 6, "political": 5, "government": 6, "parliament": 5, "election": 5,
        "policy": 4, "national": 3, "politica": 6, "política": 6, "gobierno": 6,
        "parlamento": 5, "elecciones": 5, "nación": 3, "nacional": 3,
        "politica": 6, "governo": 6, "elezioni": 5,
        "정치": 6, "정부": 6, "국회": 5, "선거": 5,
        "政治": 6, "政府": 6, "議会": 5, "選挙": 5,
    },
    "economy": {
        "economy": 6, "economic": 5, "business": 5, "finance": 6, "market": 5, "markets": 5,
        "economia": 6, "economía": 6, "finanzas": 6, "mercado": 5, "mercados": 5,
        "negocios": 5, "empresas": 4, "finanza": 6, "mercato": 5, "mercati": 5,
        "economy": 6, "경제": 6, "금융": 6, "시장": 5, "기업": 4, "物価": 4,
        "経済": 6, "金融": 6, "市場": 5, "企業": 4,
    },
}


def _section_topic_aliases(last_message: str) -> dict[str, int]:
    return _SECTION_TOPIC_ALIAS_WEIGHTS.get(_detect_news_topic(last_message), {})


def _section_label_score(label_norm: str, path_norm: str, *, topic_aliases: dict[str, int]) -> int:
    score = 0
    for token, weight in topic_aliases.items():
        if token in label_norm:
            score += weight
        if token in path_norm:
            score += max(1, weight - 1)
    return score


def _section_area_score(area_blob: str, label: str) -> int:
    score = 0
    normalized_area = _strip_accents(area_blob.lower())
    if " role navigation " in f" {normalized_area} " or " nav " in f" {normalized_area} ":
        score += 4
    if any(hint in normalized_area for hint in _NAV_CONTAINER_HINTS):
        score += 3
    word_count = len(label.split())
    if 1 <= word_count <= 4:
        score += 1
    return score


def _extract_navigation_sources_from_html(html: str, *, base_url: str, domain: str) -> list[dict[str, Any]]:
    from bs4 import BeautifulSoup

    if not html or "<" not in html:
        return []

    normalized_domain = domain.lower().removeprefix("www.")
    soup = BeautifulSoup(html, "html.parser")
    collected: list[dict[str, Any]] = []
    seen: set[str] = set()
    banned_terms = {
        "facebook", "instagram", "twitter", "x.com", "youtube", "tiktok", "whatsapp",
        "author", "authors", "archive", "archives", "tag", "tags", "category", "categories",
        "newsletter", "subscribe", "login", "signin", "register", "advertise", "podcast",
    }

    for anchor in soup.find_all("a", href=True):
        raw_href = str(anchor.get("href") or "").strip()
        if not raw_href or raw_href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        raw_title = " ".join(anchor.stripped_strings)
        if not raw_title or len(raw_title) > 80 or _is_dirty_section_label(raw_title):
            continue
        absolute_url = raw_href if raw_href.startswith("http") else urljoin(base_url, raw_href)
        absolute_url = _normalize_http_url(absolute_url)
        if not absolute_url:
            continue
        parsed = urlparse(absolute_url)
        hostname = (_safe_hostname(absolute_url)).lower().removeprefix("www.")
        if hostname and hostname != normalized_domain:
            continue
        clean_url = f"{parsed.scheme or 'https'}://{parsed.netloc}{parsed.path}".rstrip("/")
        if not parsed.path or parsed.path == "/":
            continue
        if _is_article_url(clean_url) or _candidate_url_has_date(clean_url):
            continue
        title_norm = _strip_accents(raw_title.lower())
        path_norm = _strip_accents(parsed.path.lower())
        if any(term in f"{title_norm} {path_norm}" for term in banned_terms):
            continue

        area_parts: list[str] = []
        parent = anchor.parent
        depth = 0
        while parent is not None and getattr(parent, "name", None) and depth < 5:
            tag_name = str(parent.name or "")
            area_parts.append(tag_name)
            parent_id = parent.get("id")
            if parent_id:
                area_parts.append(str(parent_id))
            parent_classes = parent.get("class") or []
            area_parts.extend(str(value) for value in parent_classes if value)
            role = parent.get("role")
            if role:
                area_parts.append(f"role {role}")
            parent = parent.parent
            depth += 1
        area_blob = " ".join(area_parts)
        dedupe_key = f"{_slugify_periodicos_label(raw_title)}|{_slugify_periodicos_label(parsed.path)}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        collected.append(
            {
                "url": clean_url,
                "title": raw_title[:80],
                "area_score": _section_area_score(area_blob, raw_title),
            }
        )

    return collected


async def _fetch_homepage_document(url: str, *, use_dynamic: bool) -> str:
    from features.web_scraping.infrastructure import scraping_infra

    if not use_dynamic:
        try:
            html_bytes = await asyncio.to_thread(scraping_infra._fetch_html, url, 8)
        except Exception as exc:
            return f"Error al procesar la pagina web: {str(exc)}"
        return html_bytes.decode("utf-8", errors="replace")

    def _fetch_dynamic_html() -> str:
        try:
            browser = scraping_infra._get_browser()
            context = browser.new_context()
            page = context.new_page()
            scraping_infra._configure_page(page)
            page.goto(url, wait_until="domcontentloaded", timeout=8000)
            page.wait_for_timeout(500)
            html = page.content()
            context.close()
            return html
        except Exception as exc:
            return f"Error al procesar la pagina web: {str(exc)}"

    return await asyncio.to_thread(_fetch_dynamic_html)


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


def _build_newspaper_section_discovery_prompt(last_message: str, press_name: str) -> str:
    topic = _detect_news_topic(last_message)
    geography = _extract_query_geography(last_message) or ""
    geo_line = f"País objetivo: {geography}. " if geography else ""
    topic_line = {
        "security": "Tema objetivo: seguridad, crimen, policía, policiales, sucesos, justicia, ciberseguridad, defensa.",
        "politics": "Tema objetivo: política, gobierno, parlamento, elecciones, decretos, poder judicial.",
        "economy": "Tema objetivo: economía, finanzas, inflación, mercado, empresas, presupuesto.",
    }.get(topic, "Tema objetivo: noticias y actualidad.")
    return (
        f"Leé la homepage del diario {press_name}. "
        f"{geo_line}{topic_line} "
        "Identificá SOLO secciones o categorías del sitio que probablemente publiquen noticias sobre ese tema. "
        "No devuelvas artículos individuales, no devuelvas redes sociales, no devuelvas autores, no devuelvas tags, no devuelvas archives. "
        "Devolvé cada sección en una línea con formato markdown exacto: [Nombre de la sección](URL absoluta). "
        "Si hay secciones repetidas o equivalentes, devolvé solo la más específica. "
        "Si no encontrás secciones relevantes, devolvé exactamente: 'No hay secciones relevantes.'\n\n"
        f"Consulta original: {last_message}"
    )


def _topic_section_terms(last_message: str) -> set[str]:
    topic = _detect_news_topic(last_message)
    terms_map = {
        "security": {
            "security", "safety", "crime", "police", "policing", "incident", "incidents", "justice", "court", "law",
            "seguridad", "inseguridad", "policial", "policiales", "policia", "sucesos", "justicia", "tribunales",
            "cronaca", "sicurezza", "polizia", "giustizia",
            "사회", "사건", "범죄", "경찰", "치안", "수사", "법원",
            "事件", "犯罪", "警察", "治安", "司法",
        },
        "politics": {
            "politics", "political", "government", "election", "parliament", "policy", "congress", "senate",
            "politica", "gobierno", "elecciones", "parlamento", "congreso", "senado",
            "politica", "governo", "elezioni", "parlamento",
            "정치", "정부", "국회", "선거",
            "政治", "政府", "議会", "選挙",
        },
        "economy": {
            "economy", "economic", "business", "finance", "market", "markets", "companies", "inflation",
            "economia", "finanzas", "mercado", "mercados", "negocios", "empresas", "inflacion",
            "economia", "finanza", "mercato", "mercati", "imprese", "inflazione",
            "경제", "금융", "시장", "기업", "물가",
            "経済", "金融", "市場", "企業", "物価",
        },
    }
    return terms_map.get(topic, set())


def _extract_relevant_homepage_sections(
    text: str,
    *,
    domain: str,
    base_url: str,
    last_message: str,
) -> list[tuple[str, str]]:
    if not text or _classify_fetch_error(text) or _is_no_info_response(text):
        return []

    topical_terms = _topic_section_terms(last_message)
    topic_aliases = _section_topic_aliases(last_message)
    normalized_domain = domain.lower().removeprefix("www.")
    candidates: list[tuple[int, int, str, str]] = []
    seen_keys: set[str] = set()
    seen_paths: set[str] = set()
    banned_terms = {
        "facebook", "instagram", "twitter", "x.com", "youtube", "tiktok", "whatsapp",
        "author", "authors", "archive", "archives", "tag", "tags", "category", "categories",
        "newsletter", "subscribe", "login", "signin", "register",
    }

    structured_sources = [
        {**source, "area_score": 0}
        for source in _extract_sources_from_text(text)
    ]
    html_sources = _extract_navigation_sources_from_html(text, base_url=base_url, domain=domain)

    for source in [*html_sources, *structured_sources]:
        raw_url = (source.get("url") or "").strip()
        raw_title = " ".join((source.get("title") or "").split())
        if not raw_url or not raw_title or _is_dirty_section_label(raw_title):
            continue
        absolute_url = raw_url if raw_url.startswith("http") else urljoin(base_url, raw_url)
        absolute_url = _normalize_http_url(absolute_url)
        if not absolute_url:
            continue
        parsed = urlparse(absolute_url)
        hostname = (_safe_hostname(absolute_url)).lower().removeprefix("www.")
        if hostname and hostname != normalized_domain:
            continue
        clean_url = f"{parsed.scheme or 'https'}://{parsed.netloc}{parsed.path}".rstrip("/")
        if not parsed.path or parsed.path == "/":
            continue
        if _is_article_url(clean_url) or _candidate_url_has_date(clean_url):
            continue
        title_norm = _strip_accents(raw_title.lower())
        path_norm = _strip_accents(parsed.path.lower())
        blob = f"{title_norm} {path_norm}"
        if any(term in blob for term in banned_terms):
            continue
        topical_score = _section_label_score(title_norm, path_norm, topic_aliases=topic_aliases)
        if topical_terms:
            topical_score += sum(3 for term in topical_terms if term in title_norm)
            topical_score += sum(2 for term in topical_terms if term in path_norm)
        if topical_score <= 0:
            continue
        area_score = int(source.get("area_score") or 0)
        score = topical_score + area_score
        dedupe_key = f"{_slugify_periodicos_label(raw_title)}|{_slugify_periodicos_label(parsed.path)}"
        path_key = _slugify_periodicos_label(parsed.path)
        if dedupe_key in seen_keys or path_key in seen_paths:
            continue
        seen_keys.add(dedupe_key)
        seen_paths.add(path_key)
        label = raw_title[:80]
        candidates.append((score, area_score, clean_url, label))

    candidates.sort(key=lambda item: (-item[0], -item[1], len(urlparse(item[2]).path)))
    return [(url, label) for _, _, url, label in candidates[:4]]


async def _discover_homepage_section_targets(
    *,
    domain: str,
    fallback_url: str,
    last_message: str,
    press_name: str,
    dynamic_fetch_available: bool = True,
) -> tuple[list[tuple[str, str]], bool]:
    from time import perf_counter

    started_at = perf_counter()
    merged_targets: list[tuple[str, str]] = []
    seen_urls: set[str] = set()
    for use_dynamic in ((False, True) if dynamic_fetch_available else (False,)):
        fetch_started_at = perf_counter()
        homepage = await _fetch_homepage_document(fallback_url, use_dynamic=use_dynamic)
        issue = _classify_fetch_error(homepage)
        if issue == "missing_playwright":
            dynamic_fetch_available = False
        discovered_targets = _extract_relevant_homepage_sections(
            homepage,
            domain=domain,
            base_url=fallback_url,
            last_message=last_message,
        )
        _web_debug(
            "country_press.search.homepage_section_discovery",
            domain=domain,
            press_name=press_name,
            fallback_url=fallback_url,
            use_dynamic=use_dynamic,
            discovered_count=len(discovered_targets),
            discovered_targets=discovered_targets,
            discovered_labels=[label for _, label in discovered_targets],
            issue=issue,
            elapsed_ms=round((perf_counter() - fetch_started_at) * 1000, 1),
        )
        for section_url, section_label in discovered_targets:
            normalized_url = section_url.rstrip("/")
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            merged_targets.append((section_url, section_label))
        if discovered_targets:
            break

    for section_url, section_label in _build_country_press_section_targets(
        domain,
        fallback_url,
        last_message,
        allow_generic_fallback=False,
    ):
        normalized_url = section_url.rstrip("/")
        if normalized_url in seen_urls:
            continue
        seen_urls.add(normalized_url)
        merged_targets.append((section_url, section_label))

    _web_debug(
        "country_press.search.section_targets_merged",
        domain=domain,
        press_name=press_name,
        target_count=len(merged_targets),
        targets=merged_targets[:6],
        target_labels=[label for _, label in merged_targets[:10]],
        elapsed_ms=round((perf_counter() - started_at) * 1000, 1),
    )
    return merged_targets[:6], dynamic_fetch_available




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
    sports_results_query = any(
        term in {"futbol", "football", "soccer", "resultado", "resultados", "partido", "partidos", "marcador"}
        for term in (query_terms or [])
    )
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
                    if sports_results_query and re.search(r"\b\d+\s*-\s*\d+\b", next_line):
                        result.append(next_line)
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
    if _is_redirect_payload(text):
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
        if _is_prompt_echo_line(cleaned):
            continue
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
        if cleaned and not _is_homepage_meta_line(cleaned) and not _is_no_info_response(cleaned) and not _is_prompt_echo_line(cleaned):
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
        if _is_redirect_payload(line) or _is_prompt_echo_line(line):
            continue
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


def _is_same_site_redirect(original_url: str, redirect_url: str) -> bool:
    original_host = _safe_hostname(original_url).lower().removeprefix("www.")
    redirect_host = _safe_hostname(redirect_url).lower().removeprefix("www.")
    return bool(original_host and redirect_host and original_host == redirect_host)



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
        domain = _safe_hostname(normalized) or normalized
        sources.append({"title": title.strip() or normalized, "url": normalized, "domain": domain, "snippet": ""})

    if sources:
        return sources

    for url in _extract_urls_from_text(text):
        if url not in seen:
            seen.add(url)
            domain = _safe_hostname(url) or url
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
        hostname = _safe_hostname(url).lower()
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
    generic_titles = {"enlace", "link", "source", "fuente"}
    for source in _extract_sources_from_text(text):
        url = _normalize_http_url((source.get("url", "") or "").strip())
        if not url:
            continue
        if "periodicos.com.ar" in url:
            continue
        hostname = _safe_hostname(url).lower().removeprefix("www.")
        if not hostname:
            continue
        if url in seen:
            continue
        seen.add(url)
        title = (source.get("title", "") or "").strip()
        if not title or title.lower() in generic_titles:
            title = hostname
        sources.append({
            "title": title,
            "url": url,
            "domain": hostname,
            "snippet": source.get("snippet", "") or "",
        })
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
        hostname = _safe_hostname(url).lower()
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
        _is_safe, guard_meta = await evaluate_trajectory_safe_fn(fast_result, "web_scraping_node")
        guard_label = str(guard_meta.get("label") or "")
        verdict_source = str(guard_meta.get("verdict_source") or "")
        degraded_guard = verdict_source == "error" or guard_label == "error"
        _emit_guard_audit({
            "event_type": "node_guard_status",
            "request_id": rid,
            "node": "web_scraping_node",
            "guard_status": "degraded" if degraded_guard else "ok",
            "success_kind": "success_with_guard_degradation" if degraded_guard else "success_clean",
            "verdict_source": verdict_source,
            "guard_label": guard_label or "unknown",
            "ts_ms": int(time.time() * 1000),
        })
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

    if _is_notion_sync_request(last_message):
        print(
            f"[WEB_FLOW] branch=notion_sync_shortcut request_id={rid} query={last_message[:160]!r}",
            flush=True,
        )
        from integrations.notion_tasks_sync import sync_validated_moodle_artifact_to_notion_payload

        try:
            notion_payload = await asyncio.to_thread(
                sync_validated_moodle_artifact_to_notion_payload,
                "",
            )
        except Exception as exc:
            return {
                "messages": [
                    AIMessage(
                        content=f"No pude sincronizar a Notion el artifact Moodle aprobado: {exc}"
                    )
                ]
            }
        notion_result = (
            "Sincronización a Notion completada.\n\n"
            f"- Creadas: {int(notion_payload.get('created_count') or 0)}\n"
            f"- Actualizadas: {int(notion_payload.get('updated_count') or 0)}\n"
            f"- Sin cambios: {int(notion_payload.get('skipped_count') or 0)}\n"
            f"- Errores: {int(notion_payload.get('error_count') or 0)}"
        )
        _web_debug("run_web_scraping_flow.notion_sync_shortcut", result_preview=notion_result[:200])
        return {"messages": [AIMessage(content=notion_result)]}

    if detect_linkedin_jobs_intent(last_message):
        print(
            f"[WEB_FLOW] branch=linkedin_jobs_authenticated request_id={rid} "
            f"query={last_message[:160]!r}",
            flush=True,
        )
        linkedin_guard = input_guard({"messages": [HumanMessage(content=last_message)]})
        if isinstance(linkedin_guard, dict) and linkedin_guard.get("blocked"):
            return linkedin_guard

        from features.web_scraping.application.linkedin_service import run_linkedin_jobs_vertical

        linkedin_result = await asyncio.to_thread(
            run_linkedin_jobs_vertical,
            last_message,
        )
        summary = str(linkedin_result.user_summary or "").strip()
        source_type = "linkedin_authenticated"

        linkedin_ctx = _select_strategy_context(
            state,
            last_message,
            get_runtime_policy,
        )
        tracker = cast(dict[str, Any], linkedin_ctx["tracker"])
        duration_ms = int((time.time() - t0) * 1000)
        _emit_node_outcome(
            rid,
            "web_scraping_node",
            "success" if linkedin_result.status == "ok" else "degraded",
            phase="agent",
            agent="web_scraping_agent",
            duration_ms=duration_ms,
            category="jobs",
            exploring=False,
            strategy="linkedin_jobs_authenticated",
            source_type=source_type,
            linkedin_status=linkedin_result.status,
            result_count=len(linkedin_result.records),
            warning_count=len(linkedin_result.warnings),
            followup_likely=linkedin_result.status != "ok",
            **_extract_tokens({"messages": []}),
            **_extract_quality({"messages": []}),
            **_node_meta(),
        )
        return await _guardrail_fast_result(
            summary,
            tracker,
            rid,
            t0,
            should_evaluate_guard_fn,
            evaluate_trajectory_safe_fn,
        )

    moodle_intent, moodle_course_query = _detect_moodle_intent(last_message)
    if moodle_intent == "course_list":
        print(
            f"[WEB_FLOW] branch=moodle_course_list_shortcut request_id={rid} query={last_message[:160]!r}",
            flush=True,
        )
        from integrations.google_calendar_tools import prepare_moodle_courses_payload

        courses_payload = await asyncio.to_thread(
            prepare_moodle_courses_payload,
            "",
        )
        courses_result = _render_moodle_courses_chat(cast(Mapping[str, Any], courses_payload))
        _web_debug("run_web_scraping_flow.moodle_course_list_shortcut", result_preview=courses_result[:200])
        return {"messages": [AIMessage(content=courses_result)]}

    if moodle_intent == "course_audit":
        print(
            f"[WEB_FLOW] branch=moodle_course_audit_shortcut request_id={rid} query={last_message[:160]!r} course_query={moodle_course_query!r}",
            flush=True,
        )
        from integrations.google_calendar_tools import prepare_moodle_course_audit_by_name_payload

        course_payload = await asyncio.to_thread(
            prepare_moodle_course_audit_by_name_payload,
            moodle_course_query,
            "",
        )
        course_result = _render_moodle_course_audit_chat(cast(Mapping[str, Any], course_payload))
        _web_debug("run_web_scraping_flow.moodle_course_audit_shortcut", result_preview=course_result[:200])
        return {"messages": [AIMessage(content=course_result)]}

    if moodle_intent == "assignments":
        print(
            f"[WEB_FLOW] branch=moodle_review_shortcut request_id={rid} query={last_message[:160]!r}",
            flush=True,
        )
        from integrations.google_calendar_tools import prepare_moodle_assignments_payload

        moodle_payload = await asyncio.to_thread(
            prepare_moodle_assignments_payload,
            "",
        )
        moodle_result = str(moodle_payload.get("chat_response") or "").strip()
        if not moodle_result:
            moodle_result = (
                "Se generó el artifact de Moodle, pero no pude construir la vista humana. "
                f"JSON: {moodle_payload.get('json_path', '')}\n"
                f"Markdown: {moodle_payload.get('markdown_path', '')}"
            )
        _web_debug("run_web_scraping_flow.moodle_review_shortcut", result_preview=moodle_result[:200])
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
    query_source_group = detect_query_source_group(last_message)
    query_horizon = detect_recent_query_horizon(last_message) if _is_recent_web_information_query(last_message) else None
    recent_country_news_query = _should_use_country_recent_news_strategy(last_message, query_source_group, query_horizon)
    recent_country_news_agent_bypass = recent_country_news_query and query_source_group == "japan"

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
                sports_results_query = any(
                    term in {"futbol", "football", "soccer", "resultado", "resultados", "partido", "partidos", "marcador", "nba", "nfl", "mlb", "tenis"}
                    for term in fallback_terms
                )
                if sports_results_query and (fallback_candidates or fallback_sources):
                    fetch_prompt = _build_generic_fetch_prompt(last_message)
                    if fallback_candidates:
                        ordered_candidates = sorted(
                            fallback_candidates,
                            key=lambda candidate: _score_generic_candidate(candidate, fallback_terms, fallback_query_source_group),
                            reverse=True,
                        )
                    else:
                        ordered_candidates = [
                            {
                                "title": source.get("title") or source.get("url") or "search result",
                                "url": source.get("url") or "",
                            }
                            for source in fallback_sources
                        ]
                    for candidate in ordered_candidates[:3]:
                        candidate_url = candidate.get("url") or ""
                        if not candidate_url:
                            continue
                        try:
                            fetched_candidate = await _fetch_web_page_follow_redirect(candidate_url, fetch_prompt, use_dynamic=False)
                        except Exception:
                            continue
                        if not isinstance(fetched_candidate, str):
                            fetched_candidate = str(fetched_candidate)
                        if fetched_candidate.startswith("Error") or fetched_candidate.startswith("URL rechazada") or _is_no_info_response(fetched_candidate):
                            continue
                        fetched_lines = _extract_generic_content_lines(fetched_candidate, fallback_terms)
                        if not fetched_lines:
                            continue
                        fetched_sources = _extract_sources_from_text(fetched_candidate) or [{
                            "title": candidate.get("title") or candidate_url or "search result",
                            "url": candidate_url,
                        }]
                        _fallback_raw = _build_source_backed_response(fetched_lines[:10], fetched_sources)
                        summary = await _synthesize_search_summary(_fallback_raw, last_message, get_llm_fn, fetched_sources)
                        summary, fetched_sources, words = _finalize_web_user_summary(summary, last_message, fetched_sources)
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
                            category=category, exploring=False, strategy="web_search_fetch", exp_rate=0.0,
                            scrape_reliability=reliability, prior_reliability=prior_reliability,
                            prior_score=prior_score, scrape_score=new_score,
                            retry_done=False, source_type="webfetch",
                            ml_recommended=ml_recommended, prediction_match=prediction_match,
                            ml_would_succeed=(bool(analytics.get("quality_target", 0)) if prediction_match is True else None),
                            **_extract_tokens({"messages": []}), **_extract_quality({"messages": []}), **_extract_followup({"messages": []}, "success"), **analytics, **_node_meta(),
                        )
                        return await _guardrail_fast_result(
                            summary, new_tracker, rid, t0,
                            should_evaluate_guard_fn, evaluate_trajectory_safe_fn,
                        )
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

        if recent_country_news_agent_bypass:
            no_local = _build_no_local_sources_response(last_message)
            summary = cast(str, no_local["summary"])
            words = cast(list[str], no_local["words"])
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
                rid, "web_scraping_node", "low_confidence", phase="agent",
                agent="web_scraping_agent", duration_ms=duration_ms,
                category=category, exploring=False, strategy="country_recent_news", exp_rate=0.0,
                scrape_reliability=reliability, prior_reliability=prior_reliability,
                prior_score=prior_score, scrape_score=new_score,
                retry_done=False, source_type="search",
                evidence_status="insufficient_local_recent_evidence",
                ml_recommended=ml_recommended, prediction_match=prediction_match,
                followup_likely=True,
                **_extract_tokens({"messages": []}), **_extract_quality({"messages": []}), **analytics, **_node_meta(),
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
        print(
            f"[WEB_FLOW] branch=agent_strategy request_id={rid} category={category} query={last_message[:160]!r}",
            flush=True,
        )

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
