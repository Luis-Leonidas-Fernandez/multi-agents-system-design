"""Estrategia generalista de búsqueda web."""
from __future__ import annotations

from typing import Any, Optional

from application.services.web_runtime import WebFetchRuntime, WebSearchRuntime


class GenericWebSearchStrategy:
    """Estrategia generalista search+fetch con runtime encapsulado."""

    def __init__(self, *, search_runtime: WebSearchRuntime, fetch_runtime: WebFetchRuntime) -> None:
        self._search_runtime = search_runtime
        self._fetch_runtime = fetch_runtime

    async def execute(
        self,
        last_message: str,
        web_search_runtime_args: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        from application.use_cases.web_scraping_flow import _run_generic_web_search_strategy_impl

        return await _run_generic_web_search_strategy_impl(last_message, web_search_runtime_args)


__all__ = ["GenericWebSearchStrategy"]
