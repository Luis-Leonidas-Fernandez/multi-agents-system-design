"""Helpers for the lightweight web runtime."""
from __future__ import annotations

from typing import Any


def _build_web_search_payload(request, selected_provider: str) -> dict[str, Any]:
    payload = {
        "query": request.query,
        "provider": selected_provider,
        "allowed_domains": request.allowed_domains,
        "blocked_domains": request.blocked_domains,
        "num_results": request.count,
        "max_age_days": request.max_age_days,
        "topic": request.topic,
        "time_range": request.time_range,
        "use_cache": request.use_cache,
    }
    return {k: v for k, v in payload.items() if v is not None}


def _classify_web_fetch_status(content: str) -> str:
    lowered = content.lower()
    if lowered.startswith("error"):
        return "error"
    if lowered.startswith("url rechazada"):
        return "rejected"
    if "redirect detected" in lowered:
        return "redirect"
    return "ok"
