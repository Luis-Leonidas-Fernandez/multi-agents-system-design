"""Descubrimiento dinámico de fuentes de prensa para países no registrados.

Implementa IDynamicPressSourceDiscovery: intenta encontrar medios locales
para cualquier país, incluso si no está en el bootstrap estático.

Estrategia de descubrimiento (en orden de confianza):
  1. Caché de sesión — si ya se descubrió este país, retorna inmediatamente.
  2. Directorio periodicos.com.ar — navega hasta la página del país.
  3. Búsqueda web Tavily — `site:periodicos.com.ar {geography}` como fallback.

Trazabilidad: cada intento emite un evento _web_debug con el path tomado.
Caché: dict de sesión con TTL de 300 s por nombre canónico de país.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from ports.country_news_ports import IDynamicPressSourceDiscovery

# ---------------------------------------------------------------------------
# Caché de sesión — dominio → (domains, names, timestamp)
# Vive mientras viva el proceso. Fase 5 puede moverlo a infra/persistence.
# ---------------------------------------------------------------------------
_DYNAMIC_CACHE_TTL_SECONDS: int = 300
_dynamic_cache: dict[str, tuple[list[str], list[str], float]] = {}


def _dynamic_cache_get(geography: str) -> Optional[tuple[list[str], list[str]]]:
    entry = _dynamic_cache.get(geography)
    if entry is None:
        return None
    domains, names, ts = entry
    if time.time() - ts > _DYNAMIC_CACHE_TTL_SECONDS:
        del _dynamic_cache[geography]
        return None
    return domains, names


def _dynamic_cache_set(geography: str, domains: list[str], names: list[str]) -> None:
    _dynamic_cache[geography] = (domains, names, time.time())


class DefaultDynamicPressDiscovery(IDynamicPressSourceDiscovery):
    """Descubrimiento dinámico con caché de sesión y fallback a búsqueda web."""

    async def discover_for_unknown_country(
        self,
        query: str,
        geography: str,
        runtime_args: Optional[dict[str, Any]] = None,
    ) -> tuple[list[str], list[str]]:
        from application.use_cases.web_scraping_flow import (
            _web_debug,
            _discover_country_press_sources_via_directory,
            _slugify_periodicos_label,
        )

        # 1. Caché de sesión
        cached = _dynamic_cache_get(geography)
        if cached is not None:
            _web_debug(
                "country_press.dynamic.cache_hit",
                geography=geography,
                domains=cached[0],
            )
            return cached

        # 2. Directorio periodicos.com.ar
        _web_debug("country_press.dynamic.directory_attempt", geography=geography)
        try:
            domains, names, _ = await _discover_country_press_sources_via_directory(geography)
        except Exception as exc:
            _web_debug("country_press.dynamic.directory_error", geography=geography, error=str(exc))
            domains, names = [], []

        if domains:
            _web_debug(
                "country_press.dynamic.directory_hit",
                geography=geography,
                domains=domains,
                confidence=0.7,
            )
            _dynamic_cache_set(geography, domains, names)
            return domains, names

        # 3. Búsqueda web Tavily como fallback
        _web_debug("country_press.dynamic.search_fallback", geography=geography)
        domains, names = await self._discover_via_search(geography, runtime_args)
        if domains:
            _web_debug(
                "country_press.dynamic.search_hit",
                geography=geography,
                domains=domains,
                confidence=0.4,
            )
            _dynamic_cache_set(geography, domains, names)
        else:
            _web_debug("country_press.dynamic.not_found", geography=geography)

        return domains, names

    async def _discover_via_search(
        self,
        geography: str,
        runtime_args: Optional[dict[str, Any]] = None,
    ) -> tuple[list[str], list[str]]:
        """Busca medios del país vía Tavily usando site:periodicos.com.ar."""
        from application.use_cases.web_scraping_flow import (
            _web_debug,
            _extract_country_press_sources,
        )

        try:
            from tools import search_web
        except ImportError:
            return [], []

        geo_lower = geography.lower()
        lookup_query = f'site:periodicos.com.ar {geo_lower} periódicos diarios medios'
        lookup_args: dict[str, Any] = {
            "query": lookup_query,
            "use_cache": False,
            "allowed_domains": ["periodicos.com.ar"],
            "num_results": 5,
        }
        if runtime_args:
            lookup_args["blocked_domains"] = runtime_args.get("blocked_domains") or None

        try:
            lookup_text = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: search_web.invoke(lookup_args),
            )
        except Exception as exc:
            _web_debug("country_press.dynamic.search_error", geography=geography, error=str(exc))
            return [], []

        if not isinstance(lookup_text, str):
            lookup_text = str(lookup_text)

        from urllib.parse import urlparse as _urlparse
        sources = _extract_country_press_sources(lookup_text)
        seen_domains: set[str] = set()
        domains: list[str] = []
        names: list[str] = []
        for source in sources:
            url = source.get("url", "")
            hostname = (_urlparse(url).hostname or "").lower().removeprefix("www.")
            if hostname and hostname not in seen_domains:
                seen_domains.add(hostname)
                domains.append(hostname)
                names.append(source.get("title") or hostname)
        return domains, names


__all__ = ["DefaultDynamicPressDiscovery"]
