"""Repositorio de perfiles de país: slug de continente para periodicos.com.ar."""
from __future__ import annotations

from typing import Optional


# Mapeo de nombre canónico de país → slug de continente usado en
# periodicos.com.ar/periodicos/{continent_slug}/{country_slug}/
# Si un país no aparece acá, el directorio periodicos.com.ar se saltea.
PERIODICOS_CONTINENT_SLUG_BY_COUNTRY: dict[str, str] = {
    # Latin America
    "Argentina": "sudamerica",
    "Bolivia": "sudamerica",
    "Brasil": "sudamerica",
    "Chile": "sudamerica",
    "Colombia": "sudamerica",
    "Ecuador": "sudamerica",
    "Paraguay": "sudamerica",
    "Perú": "sudamerica",
    "Uruguay": "sudamerica",
    "Venezuela": "sudamerica",
    # North/Central America + Caribbean
    "Canadá": "norteamerica",
    "Costa Rica": "centroamerica",
    "Cuba": "centroamerica",
    "El Salvador": "centroamerica",
    "Estados Unidos": "norteamerica",
    "Guatemala": "centroamerica",
    "Haití": "centroamerica",
    "Honduras": "centroamerica",
    "México": "norteamerica",
    "Nicaragua": "centroamerica",
    "Panamá": "centroamerica",
    "República Dominicana": "centroamerica",
    # Europe
    "Alemania": "europa",
    "Bélgica": "europa",
    "Dinamarca": "europa",
    "España": "europa",
    "Finlandia": "europa",
    "Francia": "europa",
    "Grecia": "europa",
    "Hungría": "europa",
    "Italia": "europa",
    "Noruega": "europa",
    "Países Bajos": "europa",
    "Polonia": "europa",
    "Portugal": "europa",
    "Reino Unido": "europa",
    "República Checa": "europa",
    "Rumanía": "europa",
    "Rusia": "europa",
    "Serbia": "europa",
    "Suecia": "europa",
    "Suiza": "europa",
    "Turquía": "europa",
    "Ucrania": "europa",
    # Asia-Pacific
    "Australia": "asia",
    "Bangladesh": "asia",
    "China": "asia",
    "Corea": "asia",
    "Corea del Norte": "asia",
    "Corea del Sur": "asia",
    "Filipinas": "asia",
    "India": "asia",
    "Indonesia": "asia",
    "Japón": "asia",
    "Malasia": "asia",
    "Nueva Zelanda": "asia",
    "Pakistán": "asia",
    "Singapur": "asia",
    "Tailandia": "asia",
    "Vietnam": "asia",
    # Middle East & Africa
    "Arabia Saudita": "medio-oriente",
    "Egipto": "africa",
    "Emiratos Árabes": "medio-oriente",
    "Etiopía": "africa",
    "Irak": "medio-oriente",
    "Irán": "medio-oriente",
    "Israel": "medio-oriente",
    "Kenia": "africa",
    "Líbano": "medio-oriente",
    "Marruecos": "africa",
    "Nigeria": "africa",
    "Palestina": "medio-oriente",
    "Siria": "medio-oriente",
    "Sudáfrica": "africa",
}


def get_continent_slug(country_name: str) -> Optional[str]:
    """Retorna el slug de continente para periodicos.com.ar, o None si no está registrado."""
    return PERIODICOS_CONTINENT_SLUG_BY_COUNTRY.get(country_name)
