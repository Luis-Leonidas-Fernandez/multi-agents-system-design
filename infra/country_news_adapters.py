"""Adaptadores concretos para los puertos de noticias por país.

Cada clase recibe un CountryBootstrap en construcción y lo usa para
resolver datos en lugar de importar constantes de módulo directamente.
Si no se pasa bootstrap, usan CountryBootstrap.default() (mismos datos
estáticos que antes — sin cambio de comportamiento).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from infra.country_bootstrap import CountryBootstrap

from domain.country_resolver import extract_query_geography
from domain.section_path_resolver import build_country_press_section_targets
from ports.country_news_ports import (
    ICountryProfileRepository,
    ICountryResolver,
    IPressSourceDiscovery,
    ISectionPathResolver,
)


class DefaultCountryResolver(ICountryResolver):
    """Resuelve geografía usando los demonyms del bootstrap."""

    def __init__(self, bootstrap: Optional["CountryBootstrap"] = None) -> None:
        from infra.country_bootstrap import CountryBootstrap
        self._bootstrap = bootstrap or CountryBootstrap.default()

    def resolve(self, query: str) -> Optional[str]:
        return extract_query_geography(query, self._bootstrap.geography_terms)


class DefaultCountryProfileRepository(ICountryProfileRepository):
    """Lee nombre en inglés y slug de continente del bootstrap."""

    def __init__(self, bootstrap: Optional["CountryBootstrap"] = None) -> None:
        from infra.country_bootstrap import CountryBootstrap
        self._bootstrap = bootstrap or CountryBootstrap.default()

    def get_english_name(self, canonical: str) -> str:
        return self._bootstrap.geo_english.get(canonical, canonical)

    def get_continent_slug(self, canonical: str) -> Optional[str]:
        return self._bootstrap.continent_slug_by_country.get(canonical)


class DefaultSectionPathResolver(ISectionPathResolver):
    """Resuelve paths de sección usando los mappings del bootstrap."""

    def __init__(self, bootstrap: Optional["CountryBootstrap"] = None) -> None:
        from infra.country_bootstrap import CountryBootstrap
        self._bootstrap = bootstrap or CountryBootstrap.default()

    def resolve_targets(
        self,
        domain: str,
        fallback_url: str,
        query: str,
    ) -> list[tuple[str, str]]:
        return build_country_press_section_targets(
            domain,
            fallback_url,
            query,
            section_paths=self._bootstrap.country_press_section_paths,
            generic_paths=self._bootstrap.generic_section_paths,
        )


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
        from application.services.press_discovery import discover_country_press_sources
        return await discover_country_press_sources(
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
