# Web Scraping Ownership

- `web_scraping_flow.py` — thin orchestrator and compatibility entrypoint.
- `web_scraping_agent_strategy.py` — agent prompt loading, fallback discovery, guardrails, retry handling.
- `web_scraping_generic_strategy.py` — generic search strategy orchestration.
- `web_scraping_country_press_helpers.py` — country-press discovery/search helpers.
- `web_scraping_query_helpers.py` — query planning and redirect follow helpers.
- `web_scraping_fetch_dispatch.py` — fetch-path dispatch.
- `web_scraping_search_pipeline.py` — search result scoring and selection.
- `web_scraping_postprocess.py` — final response shaping.
- `web_scraping_api.py` — stable barrel import for web scraping consumers.

Rules of thumb:
- strategies own branching and decisions
- helpers own pure transformations or reusable plumbing
- flow keeps orchestration and compatibility only
- the barrel exposes the stable public API, but internal code should still prefer concrete modules when possible
