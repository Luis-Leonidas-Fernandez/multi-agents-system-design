"""Extracción del país desde el query del usuario."""
from __future__ import annotations

import re
from typing import Optional


# Stopwords genéricas para filtrar términos irrelevantes en queries.
GENERIC_WEB_STOPWORDS: frozenset[str] = frozenset({
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "cada", "como", "con",
    "contra", "cual", "cuál", "cuales", "cuáles", "cuando", "cuándo", "de", "del", "desde",
    "donde", "dónde", "durante", "e", "el", "ella", "ellas", "ellos", "en", "entre",
    "era", "eres", "es", "esa", "esas", "ese", "eso", "esta", "está", "están", "este",
    "esto", "estos", "fue", "ha", "han", "hay", "la", "las", "le", "les", "lo", "los",
    "mas", "más", "mi", "mis", "muy", "no", "nos", "nosotros", "o", "o", "para", "pero",
    "por", "que", "qué", "se", "sin", "sobre", "su", "sus", "te", "tu", "tus", "un",
    "una", "uno", "unos", "unas", "y", "ya", "hoy", "ayer", "mañana", "today", "latest",
    "current", "recent", "news", "noticias", "page", "web", "site",
})

# Demonyms y nombres de países → nombre canónico en español.
# Ordenados de más largo a más corto dentro de cada región para garantizar
# que "corea del sur" matchee antes que "corea".
GEOGRAPHY_TERMS: tuple[tuple[str, str], ...] = (
    # Latin America
    ("ecuatoriano", "Ecuador"), ("ecuatoriana", "Ecuador"), ("ecuador", "Ecuador"),
    ("argentino", "Argentina"), ("argentina", "Argentina"),
    ("colombiano", "Colombia"), ("colombia", "Colombia"),
    ("venezolano", "Venezuela"), ("venezuela", "Venezuela"),
    ("chileno", "Chile"), ("chile", "Chile"),
    ("peruano", "Perú"), ("peru", "Perú"),
    ("boliviano", "Bolivia"), ("bolivia", "Bolivia"),
    ("paraguayo", "Paraguay"), ("paraguay", "Paraguay"),
    ("uruguayo", "Uruguay"), ("uruguay", "Uruguay"),
    ("guatemalteco", "Guatemala"), ("guatemala", "Guatemala"),
    ("hondureño", "Honduras"), ("honduras", "Honduras"),
    ("salvadoreño", "El Salvador"), ("el salvador", "El Salvador"),
    ("nicaragüense", "Nicaragua"), ("nicaragua", "Nicaragua"),
    ("costarricense", "Costa Rica"), ("costa rica", "Costa Rica"),
    ("panameño", "Panamá"), ("panama", "Panamá"),
    ("cubano", "Cuba"), ("cuba", "Cuba"),
    ("dominicano", "República Dominicana"), ("república dominicana", "República Dominicana"),
    ("haitiano", "Haití"), ("haiti", "Haití"),
    # North America
    ("mexicano", "México"), ("mexicana", "México"), ("mexico", "México"),
    ("estadounidense", "Estados Unidos"), ("estados unidos", "Estados Unidos"),
    ("usa", "Estados Unidos"), ("eeuu", "Estados Unidos"),
    ("canadiense", "Canadá"), ("canada", "Canadá"),
    # Europe
    ("español", "España"), ("espanol", "España"), ("españa", "España"),
    ("francés", "Francia"), ("frances", "Francia"), ("france", "Francia"), ("francia", "Francia"),
    ("alemán", "Alemania"), ("aleman", "Alemania"), ("alemania", "Alemania"),
    ("italiano", "Italia"), ("italiana", "Italia"), ("italia", "Italia"),
    ("británico", "Reino Unido"), ("britanico", "Reino Unido"), ("reino unido", "Reino Unido"),
    ("inglés", "Reino Unido"), ("ingles", "Reino Unido"),
    ("portugués", "Portugal"), ("portugues", "Portugal"), ("portugal", "Portugal"),
    ("holandés", "Países Bajos"), ("holanda", "Países Bajos"), ("países bajos", "Países Bajos"),
    ("belga", "Bélgica"), ("belgica", "Bélgica"), ("bélgica", "Bélgica"),
    ("suizo", "Suiza"), ("suiza", "Suiza"),
    ("sueco", "Suecia"), ("suecia", "Suecia"),
    ("noruego", "Noruega"), ("noruega", "Noruega"),
    ("danés", "Dinamarca"), ("danes", "Dinamarca"), ("dinamarca", "Dinamarca"),
    ("finlandés", "Finlandia"), ("finlandia", "Finlandia"),
    ("polaco", "Polonia"), ("polonia", "Polonia"),
    ("checo", "República Checa"), ("república checa", "República Checa"),
    ("húngaro", "Hungría"), ("hungria", "Hungría"),
    ("rumano", "Rumanía"), ("rumania", "Rumanía"),
    ("griego", "Grecia"), ("grecia", "Grecia"),
    ("turco", "Turquía"), ("turquia", "Turquía"), ("turquía", "Turquía"),
    ("ruso", "Rusia"), ("rusia", "Rusia"), ("russia", "Rusia"),
    ("ucraniano", "Ucrania"), ("ucrania", "Ucrania"),
    ("serbio", "Serbia"), ("serbia", "Serbia"),
    # Asia-Pacific
    ("japonés", "Japón"), ("japonesa", "Japón"), ("japones", "Japón"),
    ("japón", "Japón"), ("japon", "Japón"), ("japan", "Japón"),
    ("chino", "China"), ("china", "China"),
    ("surcoreano", "Corea del Sur"), ("corea del sur", "Corea del Sur"),
    ("norcoreano", "Corea del Norte"), ("corea del norte", "Corea del Norte"),
    ("coreano", "Corea"), ("corea", "Corea"),
    ("indio", "India"), ("india", "India"),
    ("paquistaní", "Pakistán"), ("pakistan", "Pakistán"),
    ("bangladesí", "Bangladesh"), ("bangladesh", "Bangladesh"),
    ("indonesio", "Indonesia"), ("indonesia", "Indonesia"),
    ("filipino", "Filipinas"), ("filipinas", "Filipinas"),
    ("vietnamita", "Vietnam"), ("vietnam", "Vietnam"),
    ("tailandés", "Tailandia"), ("tailandia", "Tailandia"),
    ("malayo", "Malasia"), ("malasia", "Malasia"),
    ("singapurense", "Singapur"), ("singapur", "Singapur"),
    ("australiano", "Australia"), ("australia", "Australia"),
    ("neozelandés", "Nueva Zelanda"), ("nueva zelanda", "Nueva Zelanda"),
    # Middle East & Africa
    ("israelí", "Israel"), ("israel", "Israel"),
    ("palestino", "Palestina"), ("palestina", "Palestina"),
    ("iraní", "Irán"), ("iran", "Irán"),
    ("iraquí", "Irak"), ("irak", "Irak"),
    ("sirio", "Siria"), ("siria", "Siria"),
    ("libanés", "Líbano"), ("libano", "Líbano"),
    ("saudí", "Arabia Saudita"), ("arabia saudita", "Arabia Saudita"), ("saudi", "Arabia Saudita"),
    ("emiratense", "Emiratos Árabes"), ("emiratos arabes", "Emiratos Árabes"),
    ("egipcio", "Egipto"), ("egipto", "Egipto"),
    ("nigeriano", "Nigeria"), ("nigeria", "Nigeria"),
    ("sudafricano", "Sudáfrica"), ("sudafrica", "Sudáfrica"),
    ("etíope", "Etiopía"), ("etiopia", "Etiopía"),
    ("keniata", "Kenia"), ("kenia", "Kenia"),
    ("marroquí", "Marruecos"), ("marruecos", "Marruecos"),
    # Brazil
    ("brasileño", "Brasil"), ("brasileña", "Brasil"), ("brasil", "Brasil"),
)

_GEO_STOPWORDS: frozenset[str] = GENERIC_WEB_STOPWORDS | frozenset({
    "noticia", "noticias", "semana", "semanas", "ultima", "ultimas",
    "ultimo", "ultimos", "última", "últimas", "último", "últimos",
    "reciente", "recientes", "informacion", "información", "tema",
    "seguridad", "economia", "politica", "deporte", "cultura",
})


def extract_query_geography(
    text: str,
    terms: Optional[tuple[tuple[str, str], ...]] = None,
) -> Optional[str]:
    """Extrae el nombre canónico del país desde el texto del query.

    Primero busca por términos conocidos (longest match first).
    Si no hay match, aplica un fallback por regex buscando el sustantivo
    después de preposiciones geográficas comunes.

    Args:
        text: Query del usuario.
        terms: Tabla de demonyms a usar. Si es None usa GEOGRAPHY_TERMS del módulo.

    Retorna None si no puede determinar el país.
    """
    effective_terms = GEOGRAPHY_TERMS if terms is None else terms
    lowered = (text or "").lower()

    # 1. Términos conocidos — longest match first para evitar
    #    que "corea" gane ante "corea del sur"
    for term, country in sorted(effective_terms, key=lambda x: -len(x[0])):
        if term in lowered:
            return country

    # 2. Regex fallback: palabra después de "de"/"en"/"sobre" antes de horizonte temporal
    fallback_patterns = [
        r"\b(?:de|en|sobre)\s+([a-záéíóúüñ]{4,})\s+(?:de\s+esta|esta\s+semana|hoy|del\b|esta\b)",
        r"\b(?:de|en|sobre)\s+([a-záéíóúüñ]{4,})\s*$",
        r"\b(?:de|en|sobre)\s+([a-záéíóúüñ]{4,})\s",
    ]
    for pattern in fallback_patterns:
        m = re.search(pattern, lowered)
        if m:
            word = m.group(1).strip()
            if word not in _GEO_STOPWORDS and len(word) >= 4:
                return word.capitalize()

    return None
