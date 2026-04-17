"""Adaptadores concretos para los puertos de noticias por país.

Cada clase envuelve la implementación de Fase 1 (domain/ e infra/).
Son los defaults que usa CountryRecentNewsStrategy cuando no se inyecta
ningún puerto explícito.
"""
from __future__ import annotations

from typing import Any, Optional

from domain.country_resolver import extract_query_geography
from domain.country_profile import GEO_ENGLISH
from domain.section_path_resolver import build_country_press_section_targets
from infra.country_profile_repo import PERIODICOS_CONTINENT_SLUG_BY_COUNTRY
from ports.country_news_ports import (
    ICountryProfileRepository,
    ICountryResolver,
    IPressSourceDiscovery,
    ISectionPathResolver,
)


class DefaultCountryResolver(ICountryResolver):
    """Delega en extract_query_geography del módulo de dominio."""

    def resolve(self, query: str) -> Optional[str]:
        return extract_query_geography(query)


class DefaultCountryProfileRepository(ICountryProfileRepository):
    """Lee GEO_ENGLISH y PERIODICOS_CONTINENT_SLUG_BY_COUNTRY del dominio."""

    def get_english_name(self, canonical: str) -> str:
        return GEO_ENGLISH.get(canonical, canonical)

    def get_continent_slug(self, canonical: str) -> Optional[str]:
        return PERIODICOS_CONTINENT_SLUG_BY_COUNTRY.get(canonical)


class DefaultSectionPathResolver(ISectionPathResolver):
    """Delega en build_country_press_section_targets del módulo de dominio."""

    def resolve_targets(
        self,
        domain: str,
        fallback_url: str,
        query: str,
    ) -> list[tuple[str, str]]:
        return build_country_press_section_targets(domain, fallback_url, query)


class DefaultPressSourceDiscovery(IPressSourceDiscovery):
    """Delega en _discover_country_press_sources del flujo de scraping.

    La importación es lazy para evitar ciclos de importación con web_scraping_flow.
    """

    async def discover(
        self,
        query: str,
        source_group: Optional[str],
        source_terms: list[str],
        runtime_args: Optional[dict[str, Any]] = None,
    ) -> tuple[list[str], list[str]]:
        from application.use_cases.web_scraping_flow import _discover_country_press_sources
        return await _discover_country_press_sources(
            query,
            source_group,
            source_terms,
            runtime_args,
        )


__all__ = [
    "DefaultCountryResolver",
    "DefaultCountryProfileRepository",
    "DefaultSectionPathResolver",
    "DefaultPressSourceDiscovery",
]
