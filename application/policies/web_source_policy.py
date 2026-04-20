"""Política declarativa para ranking de fuentes web."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse


def _normalize_text(text: str) -> str:
    import unicodedata

    lowered = (text or "").lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", lowered)
        if unicodedata.category(c) != "Mn"
    )


def _unique_normalized_terms(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized_terms: list[str] = []
    for value in values:
        normalized = _normalize_text(str(value or "")).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            normalized_terms.append(normalized)
    return tuple(normalized_terms)


def _load_policy() -> dict:
    policy_path = Path(__file__).with_suffix(".json")
    with policy_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    country_suffix_hints = []
    for hint in raw.get("country_suffix_hints") or []:
        country_terms = _unique_normalized_terms(
            hint.get("country_terms") or hint.get("entity_terms") or hint.get("terms") or ()
        )
        suffixes = _unique_normalized_terms(hint.get("suffixes") or ())
        if country_terms and suffixes:
            country_suffix_hints.append({
                "country_terms": country_terms,
                "suffixes": suffixes,
            })

    source_groups = raw.get("source_groups") or []
    return {
        "country_suffix_hints": tuple(country_suffix_hints),
        "source_groups": source_groups,
        "global_trusted_domains": tuple(raw.get("global_trusted_domains") or ()),
        "global_trusted_boost": int(raw.get("global_trusted_boost") or 0),
        "recent_query_profiles": raw.get("recent_query_profiles") or {},
    }


WEB_SOURCE_POLICY = _load_policy()


def _link_hostname(link: str) -> str:
    parsed = urlparse(link or "")
    hostname = _normalize_text(parsed.hostname or "")
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname


def _matches_domain_hint(link: str, domain_hint: str) -> bool:
    lowered_link = _normalize_text(link)
    lowered_hint = _normalize_text(domain_hint).strip()
    if not lowered_hint:
        return False
    if "." in lowered_hint or lowered_hint.startswith("www."):
        return _matches_hostname_suffix(link, lowered_hint)
    return lowered_hint in lowered_link


def _matches_hostname_suffix(link: str, suffix: str) -> bool:
    hostname = _link_hostname(link)
    lowered_suffix = _normalize_text(suffix).strip()
    if not hostname or not lowered_suffix:
        return False
    if lowered_suffix.startswith("."):
        return hostname.endswith(lowered_suffix)
    return hostname == lowered_suffix or hostname.endswith("." + lowered_suffix)


def _get_group_terms(group: dict) -> tuple[str, ...]:
    terms: list[str] = []
    for key in ("entity_terms", "country_terms", "aliases"):
        terms.extend(group.get(key) or [])
    group_name = str(group.get("name") or "").strip()
    if group_name:
        terms.append(group_name.replace("_", " "))
    return _unique_normalized_terms(terms)


def _get_group_suffix_hints(group: dict) -> tuple[str, ...]:
    explicit_suffixes = _unique_normalized_terms(group.get("preferred_domain_suffixes") or ())
    if explicit_suffixes:
        return explicit_suffixes

    group_terms = set(_get_group_terms(group))
    resolved_suffixes: list[str] = []
    seen: set[str] = set()
    for hint in WEB_SOURCE_POLICY.get("country_suffix_hints") or ():
        hint_terms = set(hint.get("country_terms") or ())
        if not group_terms.intersection(hint_terms):
            continue
        for suffix in hint.get("suffixes") or ():
            normalized_suffix = _normalize_text(suffix).strip()
            if normalized_suffix and normalized_suffix not in seen:
                seen.add(normalized_suffix)
                resolved_suffixes.append(normalized_suffix)
    return tuple(resolved_suffixes)


def _get_source_group_policy(group_name: Optional[str]) -> dict:
    if not group_name:
        return {}
    for group in WEB_SOURCE_POLICY["source_groups"]:
        if str(group.get("name") or "") == group_name:
            return group
    return {}


def detect_query_source_group(query: str) -> Optional[str]:
    lowered = _normalize_text(query)
    for group in WEB_SOURCE_POLICY["source_groups"]:
        if any(term in lowered for term in _get_group_terms(group)):
            return str(group.get("name") or "")
    return None


def get_query_source_terms(query: str) -> tuple[str, ...]:
    group_name = detect_query_source_group(query)
    if not group_name:
        return ()
    for group in WEB_SOURCE_POLICY["source_groups"]:
        if str(group.get("name") or "") == group_name:
            return _get_group_terms(group)
    return ()


def get_preferred_domains_for_group(query_source_group: Optional[str]) -> tuple[str, ...]:
    group = _get_source_group_policy(query_source_group)
    if not group:
        return ()
    domains: list[str] = []
    seen: set[str] = set()
    for domain in group.get("preferred_domains", []):
        normalized = str(domain or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        domains.append(normalized)
    return tuple(domains)


def is_preferred_domain_for_group(query_source_group: Optional[str], link: str) -> bool:
    group = _get_source_group_policy(query_source_group)
    if not group:
        return False
    if any(_matches_domain_hint(link, domain) for domain in group.get("preferred_domains", [])):
        return True
    return any(_matches_hostname_suffix(link, suffix) for suffix in _get_group_suffix_hints(group))


def is_global_trusted_domain(link: str) -> bool:
    lowered_link = (link or "").lower()
    return any(domain and domain in lowered_link for domain in WEB_SOURCE_POLICY["global_trusted_domains"])


def get_source_domain_priority(query_source_group: Optional[str], link: str) -> int:
    group = _get_source_group_policy(query_source_group)
    if group:
        if any(_matches_domain_hint(link, domain) for domain in group.get("preferred_domains", [])):
            return 0
        if any(_matches_hostname_suffix(link, suffix) for suffix in _get_group_suffix_hints(group)):
            return 1
    if is_global_trusted_domain(link):
        return 2
    return 3


def score_domain_boost(query_source_group: Optional[str], link: str) -> int:
    if query_source_group:
        group = _get_source_group_policy(query_source_group)
        if group:
            preferred_boost = int(group.get("preferred_boost") or 0)
            if any(_matches_domain_hint(link, domain) for domain in group.get("preferred_domains", [])):
                return preferred_boost + 4
            if any(_matches_hostname_suffix(link, suffix) for suffix in _get_group_suffix_hints(group)):
                return preferred_boost + 2
    if is_global_trusted_domain(link):
        return int(WEB_SOURCE_POLICY["global_trusted_boost"])
    return 0


def get_recent_query_min_score() -> int:
    return int(((WEB_SOURCE_POLICY.get("recent_query_profiles") or {}).get("today") or {}).get("min_score") or 0)


def get_recent_query_min_body_lines() -> int:
    return int(((WEB_SOURCE_POLICY.get("recent_query_profiles") or {}).get("today") or {}).get("min_body_lines") or 0)


def get_recent_query_min_sources() -> int:
    return int(((WEB_SOURCE_POLICY.get("recent_query_profiles") or {}).get("today") or {}).get("min_sources") or 0)


def detect_recent_query_horizon(query: str) -> Optional[str]:
    lowered = (query or "").lower()
    if any(term in lowered for term in ("this week", "esta semana", "de esta semana", "semana")):
        return "week"
    if any(
        term in lowered
        for term in (
            "this month",
            "este mes",
            "de este mes",
            "último mes",
            "ultimo mes",
            "last month",
            "últimos 30 días",
            "ultimos 30 dias",
            "ultimos 30 días",
            "últimos 30 dias",
            "30 días",
            "30 dias",
        )
    ):
        return "month"
    if any(term in lowered for term in ("today", "hoy", "de hoy", "latest", "últimas noticias", "ultimas noticias", "últimas", "ultimas")):
        return "today"
    return None


def get_recent_query_requirements(horizon: Optional[str]) -> dict:
    profiles = WEB_SOURCE_POLICY.get("recent_query_profiles") or {}
    if horizon and horizon in profiles:
        profile = profiles[horizon] or {}
    else:
        profile = profiles.get("today") or {}
    return {
        "min_score": int(profile.get("min_score") or 0),
        "min_body_lines": int(profile.get("min_body_lines") or 0),
        "min_sources": int(profile.get("min_sources") or 0),
        "min_candidates": int(profile.get("min_candidates") or 0),
        "candidate_min_score": int(profile.get("candidate_min_score") or 0),
        "candidate_min_sources": int(profile.get("candidate_min_sources") or 0),
    }


def get_group_language(group_name: Optional[str]) -> Optional[str]:
    """Devuelve el código ISO 639-1 del idioma principal de la prensa local del grupo.

    Retorna None si el grupo no existe o no tiene campo ``language``.
    Los grupos con idioma "es" o "en" no necesitan traducción — el flujo de
    búsqueda ya genera variantes en esos idiomas.
    """
    group = _get_source_group_policy(group_name)
    if not group:
        return None
    lang = str(group.get("language") or "").strip().lower()
    return lang if lang else None


def _strip_accents(text: str) -> str:
    """Remove diacritics so 'japon' matches 'japón', 'ultima' matches 'última', etc."""
    return _normalize_text(text)


def is_recent_web_information_query(text: str) -> bool:
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
