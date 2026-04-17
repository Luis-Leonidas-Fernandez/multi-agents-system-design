"""Puertos (contratos) para el sistema de noticias por país."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class ICountryResolver(ABC):
    """Extrae el nombre canónico del país desde el texto del query."""

    @abstractmethod
    def resolve(self, query: str) -> Optional[str]:
        """Retorna el nombre canónico (ej. "Argentina") o None si no se detecta."""
        raise NotImplementedError


class ICountryProfileRepository(ABC):
    """Provee datos estáticos de un país: nombre en inglés y slug de continente."""

    @abstractmethod
    def get_english_name(self, canonical: str) -> str:
        """Mapea nombre canónico en español → inglés (ej. "Japón" → "Japan").
        Si no hay mapping, retorna el nombre tal cual."""
        raise NotImplementedError

    @abstractmethod
    def get_continent_slug(self, canonical: str) -> Optional[str]:
        """Retorna el slug de continente para periodicos.com.ar (ej. "sudamerica").
        None si el país no está registrado."""
        raise NotImplementedError


class ISectionPathResolver(ABC):
    """Resuelve los targets de sección (URL, label) a scrapear para un dominio."""

    @abstractmethod
    def resolve_targets(
        self,
        domain: str,
        fallback_url: str,
        query: str,
    ) -> list[tuple[str, str]]:
        """Retorna lista de (url_completa, label) a scrapear, máximo 4."""
        raise NotImplementedError


class IPressSourceDiscovery(ABC):
    """Descubre dominios de prensa local candidatos para un país."""

    @abstractmethod
    async def discover(
        self,
        query: str,
        source_group: Optional[str],
        source_terms: list[str],
        runtime_args: Optional[dict[str, Any]] = None,
    ) -> tuple[list[str], list[str]]:
        """Retorna (dominios, nombres_de_prensa) para el país del query."""
        raise NotImplementedError


__all__ = [
    "ICountryResolver",
    "ICountryProfileRepository",
    "ISectionPathResolver",
    "IPressSourceDiscovery",
]
