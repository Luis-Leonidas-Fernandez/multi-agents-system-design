"""Bootstrap estático del sistema de noticias por país.

CountryBootstrap agrupa todos los mappings hardcodeados en un único objeto
de configuración. Los adaptadores Default* lo reciben en construcción en
lugar de importar las constantes de módulo directamente.

Esto desacopla los datos del código que los usa y habilita:
- Reemplazar la configuración en tests sin tocar módulos globales.
- Extender con datos dinámicos en Fase 4 sin cambiar las interfaces.
- Agregar un país en un solo lugar en lugar de editar 5 diccionarios.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CountryBootstrap:
    """Configuración estática de países para el sistema de noticias.

    Todos los campos son opcionales en construcción; `default()` los rellena
    con los mappings actuales del dominio.
    """

    # Demonyms y nombres de país para resolver texto → nombre canónico.
    geography_terms: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    # Nombre canónico en español → nombre en inglés (para queries anglófonas).
    geo_english: dict[str, str] = field(default_factory=dict)

    # Nombre canónico → slug de continente para periodicos.com.ar.
    continent_slug_by_country: dict[str, str] = field(default_factory=dict)

    # Paths curados por dominio → tópico → [(path, label)].
    country_press_section_paths: dict[str, dict[str, list[tuple[str, str]]]] = field(
        default_factory=dict
    )

    # Paths genéricos de fallback por tópico → [(path, label)].
    generic_section_paths: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "CountryBootstrap":
        """Instancia con todos los mappings estáticos actuales."""
        from domain.country_resolver import GEOGRAPHY_TERMS
        from domain.country_profile import GEO_ENGLISH
        from infra.country_profile_repo import PERIODICOS_CONTINENT_SLUG_BY_COUNTRY
        from domain.section_path_resolver import (
            COUNTRY_PRESS_SECTION_PATHS,
            GENERIC_SECTION_PATHS,
        )
        return cls(
            geography_terms=GEOGRAPHY_TERMS,
            geo_english=GEO_ENGLISH,
            continent_slug_by_country=PERIODICOS_CONTINENT_SLUG_BY_COUNTRY,
            country_press_section_paths=COUNTRY_PRESS_SECTION_PATHS,
            generic_section_paths=GENERIC_SECTION_PATHS,
        )

    def with_country(
        self,
        *,
        canonical: str,
        english_name: str,
        continent_slug: str,
        demonyms: tuple[str, ...] = (),
    ) -> "CountryBootstrap":
        """Retorna una copia del bootstrap con un país adicional registrado.

        Útil para tests y para Fase 4 (fallback dinámico que descubre países
        nuevos y los persiste en memoria de sesión).
        """
        new_terms = self.geography_terms + tuple((d, canonical) for d in demonyms)
        new_geo_en = {**self.geo_english, canonical: english_name}
        new_slugs = {**self.continent_slug_by_country, canonical: continent_slug}
        return CountryBootstrap(
            geography_terms=new_terms,
            geo_english=new_geo_en,
            continent_slug_by_country=new_slugs,
            country_press_section_paths=self.country_press_section_paths,
            generic_section_paths=self.generic_section_paths,
        )


__all__ = ["CountryBootstrap"]
