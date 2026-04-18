"""Cachés en memoria para web search y web fetch."""

import time
from typing import Optional
from urllib.parse import urlparse
import os

_SEARCH_CACHE_TTL_SECONDS = 15 * 60
_SEARCH_CACHE_MAX = 128
_SEARCH_CACHE: dict[str, tuple[float, str]] = {}

_WEB_FETCH_CACHE_TTL_SECONDS = 15 * 60
_WEB_FETCH_CACHE_MAX = 128
_WEB_FETCH_CACHE: dict[str, tuple[float, str]] = {}


def _split_domain_list(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _search_cache_key(
    query: str,
    allowed_domains: Optional[list[str]],
    blocked_domains: Optional[list[str]],
    num_results: int,
) -> str:
    allowed = ",".join(sorted((allowed_domains or []) + _split_domain_list(os.getenv("WEB_ALLOWED_DOMAINS"))))
    blocked = ",".join(sorted((blocked_domains or []) + _split_domain_list(os.getenv("WEB_BLOCKED_DOMAINS"))))
    return f"{query}|allowed={allowed}|blocked={blocked}|n={num_results}"


def _get_search_cache(key: str) -> Optional[str]:
    entry = _SEARCH_CACHE.get(key)
    if not entry:
        return None
    ts, value = entry
    if time.time() - ts > _SEARCH_CACHE_TTL_SECONDS:
        _SEARCH_CACHE.pop(key, None)
        return None
    return value


def _set_search_cache(key: str, value: str) -> None:
    if len(_SEARCH_CACHE) >= _SEARCH_CACHE_MAX:
        oldest_key = next(iter(_SEARCH_CACHE))
        _SEARCH_CACHE.pop(oldest_key, None)
    _SEARCH_CACHE[key] = (time.time(), value)


def _web_fetch_cache_key(
    url: str,
    prompt: str,
    use_dynamic: bool,
    wait_for_selector: Optional[str],
    extract_selector: Optional[str],
    max_chars: int,
) -> str:
    return f"{url}|prompt={prompt}|dynamic={use_dynamic}|wait={wait_for_selector}|extract={extract_selector}|chars={max_chars}"


def _get_web_fetch_cache(key: str) -> Optional[str]:
    entry = _WEB_FETCH_CACHE.get(key)
    if not entry:
        return None
    ts, value = entry
    if time.time() - ts > _WEB_FETCH_CACHE_TTL_SECONDS:
        _WEB_FETCH_CACHE.pop(key, None)
        return None
    return value


def _set_web_fetch_cache(key: str, value: str) -> None:
    if len(_WEB_FETCH_CACHE) >= _WEB_FETCH_CACHE_MAX:
        oldest_key = next(iter(_WEB_FETCH_CACHE))
        _WEB_FETCH_CACHE.pop(oldest_key, None)
    _WEB_FETCH_CACHE[key] = (time.time(), value)
