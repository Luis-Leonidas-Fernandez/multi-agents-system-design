"""Fachada pública para descubrimiento de fuentes de prensa.

Expone funciones de web_scraping_api como API pública para que infra/
pueda importarlas sin acceder a privadas de un use_case directamente.
"""
from application.use_cases.web_scraping_api import (
    _discover_country_press_sources as discover_country_press_sources,
    _discover_country_press_sources_via_directory as discover_country_press_sources_via_directory,
    _extract_country_press_sources as extract_country_press_sources,
    _web_debug as web_debug,
)
from domain.web_text_utils import _slugify_periodicos_label as slugify_periodicos_label

__all__ = [
    "discover_country_press_sources",
    "discover_country_press_sources_via_directory",
    "extract_country_press_sources",
    "web_debug",
    "slugify_periodicos_label",
]
