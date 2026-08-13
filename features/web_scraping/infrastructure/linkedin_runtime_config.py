"""Runtime configuration helpers for LinkedIn jobs scraping."""
from __future__ import annotations

import os

from features.web_scraping.infrastructure.linkedin_query_navigation import (
    _HARD_MAX_QUERIES_PER_LOCATION,
)

_HARD_MAX_DETAIL_REQUESTS = 30
_DETAIL_CLICK_INTERVAL_MIN_MS = 750
_DETAIL_CLICK_INTERVAL_MAX_MS = 3000


def _configured_bounded_int(
    env_name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = (os.getenv(env_name) or str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{env_name} debe ser un entero entre {minimum} y {maximum}."
        ) from exc
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{env_name} debe ser un entero entre {minimum} y {maximum}."
        )
    return value


def configured_linkedin_detail_budget() -> int:
    return _configured_bounded_int(
        "LINKEDIN_DETAIL_BUDGET",
        default=25,
        minimum=0,
        maximum=_HARD_MAX_DETAIL_REQUESTS,
    )


def configured_linkedin_max_queries_per_location() -> int:
    return _configured_bounded_int(
        "LINKEDIN_MAX_QUERIES_PER_LOCATION",
        default=3,
        minimum=1,
        maximum=_HARD_MAX_QUERIES_PER_LOCATION,
    )


def configured_linkedin_query_interval_ms() -> int:
    return _configured_bounded_int(
        "LINKEDIN_QUERY_INTERVAL_MS",
        default=2750,
        minimum=2000,
        maximum=5000,
    )


def configured_linkedin_detail_click_interval_ms() -> int:
    return _configured_bounded_int(
        "LINKEDIN_DETAIL_CLICK_INTERVAL_MS",
        default=1200,
        minimum=_DETAIL_CLICK_INTERVAL_MIN_MS,
        maximum=_DETAIL_CLICK_INTERVAL_MAX_MS,
    )


def _configured_bool(env_name: str, *, default: bool = False) -> bool:
    raw_default = "true" if default else "false"
    raw = (os.getenv(env_name) or raw_default).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{env_name} debe ser true o false.")


def configured_linkedin_direct_detail_fallback() -> bool:
    return _configured_bool("LINKEDIN_DIRECT_DETAIL_FALLBACK")


def configured_linkedin_max_results() -> int:
    raw = (os.getenv("LINKEDIN_MAX_RESULTS") or "50").strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError("LINKEDIN_MAX_RESULTS debe ser un entero entre 1 y 50.") from exc
