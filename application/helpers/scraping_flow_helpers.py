"""Compatibilidad hacia helpers de scraping en domain."""

from domain.scraping_flow_helpers import (
    _CACHE_TTL_SECONDS,
    _SCRAPE_CACHE,
    _SCRAPE_CACHE_MAX,
    _build_result,
    _cache_key,
    _clean_text,
    _extract_links,
    _extract_text,
    _get_cache,
    _set_cache,
    _truncate_text,
    _validate_url,
)

__all__ = [
    "_CACHE_TTL_SECONDS",
    "_SCRAPE_CACHE_MAX",
    "_SCRAPE_CACHE",
    "_cache_key",
    "_get_cache",
    "_set_cache",
    "_validate_url",
    "_clean_text",
    "_truncate_text",
    "_extract_text",
    "_extract_links",
    "_build_result",
]
