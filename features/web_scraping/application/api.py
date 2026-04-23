"""Feature-level application API for web scraping."""
from application.use_cases.web_scraping_agent_strategy import _run_web_scraping_agent_strategy
from application.use_cases.web_scraping_country_press_helpers import _run_country_press_search_candidates
from application.use_cases.web_scraping_country_strategy import CountryRecentNewsStrategy
from application.use_cases.web_scraping_fetch_dispatch import _run_generic_web_search_fetch
from application.use_cases.web_scraping_flow import (
    run_web_scraping_flow,
    _select_strategy_context,
    _discover_country_press_sources,
    _discover_country_press_sources_via_directory,
    _extract_country_press_sources,
    _web_debug,
    _COUNTRY_PRESS_CACHE,
)
from application.use_cases.web_scraping_generic_strategy import GenericWebSearchStrategy, _run_generic_web_search_strategy_impl
from application.use_cases.web_scraping_query_helpers import _fetch_web_page_follow_redirect, _build_generic_fetch_prompt

__all__ = [
    "run_web_scraping_flow",
    "_select_strategy_context",
    "_run_generic_web_search_fetch",
    "_run_web_scraping_agent_strategy",
    "CountryRecentNewsStrategy",
    "_run_generic_web_search_strategy_impl",
    "GenericWebSearchStrategy",
    "_run_country_press_search_candidates",
    "_COUNTRY_PRESS_CACHE",
    "_discover_country_press_sources",
    "_discover_country_press_sources_via_directory",
    "_extract_country_press_sources",
    "_web_debug",
    "_fetch_web_page_follow_redirect",
    "_build_generic_fetch_prompt",
]
