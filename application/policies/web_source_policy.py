"""Política declarativa para ranking de fuentes web."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def _load_policy() -> dict:
    policy_path = Path(__file__).with_suffix(".json")
    with policy_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    source_groups = raw.get("source_groups") or []
    return {
        "source_groups": source_groups,
        "global_trusted_domains": tuple(raw.get("global_trusted_domains") or ()),
        "global_trusted_boost": int(raw.get("global_trusted_boost") or 0),
        "recent_query_profiles": raw.get("recent_query_profiles") or {},
    }


WEB_SOURCE_POLICY = _load_policy()


def detect_query_source_group(query: str) -> Optional[str]:
    lowered = (query or "").lower()
    for group in WEB_SOURCE_POLICY["source_groups"]:
        if any(term in lowered for term in group.get("entity_terms", [])):
            return str(group.get("name") or "")
    return None


def get_query_source_terms(query: str) -> tuple[str, ...]:
    group_name = detect_query_source_group(query)
    if not group_name:
        return ()
    for group in WEB_SOURCE_POLICY["source_groups"]:
        if str(group.get("name") or "") == group_name:
            return tuple(str(term).lower() for term in group.get("entity_terms", []) if term)
    return ()


def score_domain_boost(query_source_group: Optional[str], link: str) -> int:
    lowered_link = (link or "").lower()
    if query_source_group:
        for group in WEB_SOURCE_POLICY["source_groups"]:
            if str(group.get("name") or "") == query_source_group:
                if any(domain in lowered_link for domain in group.get("preferred_domains", [])):
                    return int(group.get("preferred_boost") or 0)
                break
    if any(domain in lowered_link for domain in WEB_SOURCE_POLICY["global_trusted_domains"]):
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
