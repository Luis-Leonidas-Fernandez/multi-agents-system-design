"""Perfil de país: idioma, nombre canónico, traducción al inglés."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CountryProfile:
    """Datos estáticos de un país usados por las estrategias de scraping."""
    country_name: str          # nombre canónico en español
    country_name_en: str       # nombre en inglés para búsquedas
    language: Optional[str]    # ISO 639-1 del idioma local de la prensa
    continent_slug: Optional[str]  # slug usado en periodicos.com.ar


# Mapeo español → inglés para construir queries anglófonos.
# Si un país no está acá, se usa el nombre tal cual (fallback identidad).
GEO_ENGLISH: dict[str, str] = {
    # Latin America
    "Ecuador": "Ecuador", "Argentina": "Argentina", "Colombia": "Colombia",
    "Venezuela": "Venezuela", "Chile": "Chile", "Perú": "Peru",
    "Bolivia": "Bolivia", "Paraguay": "Paraguay", "Uruguay": "Uruguay",
    "Guatemala": "Guatemala", "Honduras": "Honduras", "El Salvador": "El Salvador",
    "Nicaragua": "Nicaragua", "Costa Rica": "Costa Rica", "Panamá": "Panama",
    "Cuba": "Cuba", "República Dominicana": "Dominican Republic", "Haití": "Haiti",
    # North America
    "México": "Mexico", "Estados Unidos": "United States", "Canadá": "Canada",
    # Europe
    "España": "Spain", "Francia": "France", "Alemania": "Germany",
    "Italia": "Italy", "Reino Unido": "United Kingdom", "Portugal": "Portugal",
    "Países Bajos": "Netherlands", "Bélgica": "Belgium", "Suiza": "Switzerland",
    "Suecia": "Sweden", "Noruega": "Norway", "Dinamarca": "Denmark",
    "Finlandia": "Finland", "Polonia": "Poland", "República Checa": "Czech Republic",
    "Hungría": "Hungary", "Rumanía": "Romania", "Grecia": "Greece",
    "Turquía": "Turkey", "Rusia": "Russia", "Ucrania": "Ukraine", "Serbia": "Serbia",
    # Asia-Pacific
    "Japón": "Japan", "China": "China", "Corea del Sur": "South Korea",
    "Corea del Norte": "North Korea", "Corea": "Korea",
    "India": "India", "Pakistán": "Pakistan", "Bangladesh": "Bangladesh",
    "Indonesia": "Indonesia", "Filipinas": "Philippines", "Vietnam": "Vietnam",
    "Tailandia": "Thailand", "Malasia": "Malaysia", "Singapur": "Singapore",
    "Australia": "Australia", "Nueva Zelanda": "New Zealand",
    # Middle East & Africa
    "Israel": "Israel", "Palestina": "Palestine", "Irán": "Iran",
    "Irak": "Iraq", "Siria": "Syria", "Líbano": "Lebanon",
    "Arabia Saudita": "Saudi Arabia", "Emiratos Árabes": "UAE",
    "Egipto": "Egypt", "Nigeria": "Nigeria", "Sudáfrica": "South Africa",
    "Etiopía": "Ethiopia", "Kenia": "Kenya", "Marruecos": "Morocco",
    # Brazil
    "Brasil": "Brazil",
}
