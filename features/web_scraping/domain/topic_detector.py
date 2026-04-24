"""Detección del tópico de noticias desde el query del usuario."""
from __future__ import annotations


# Templates de búsqueda por ángulo en español, organizados por tópico.
TOPIC_ANGLES: dict[str, list[str]] = {
    "security": [
        "{geo} crimen delincuencia seguridad interna {year}",
        "{geo} defensa militar despliegue fuerzas {year}",
        "{geo} diplomacia tensiones política exterior {year}",
        "{geo} desastre emergencia seguridad civil {year}",
    ],
    "economy": [
        "{geo} economía mercado inversión {year}",
        "{geo} empleo salario empresa {year}",
        "{geo} inflación precios comercio {year}",
        "{geo} tecnología industria energía {year}",
    ],
    "politics": [
        "{geo} gobierno elecciones política {year}",
        "{geo} congreso ley reforma legislación {year}",
        "{geo} oposición partido liderazgo {year}",
        "{geo} corrupción justicia tribunal {year}",
    ],
    "default": [
        "{geo} {topic} noticias recientes {year}",
        "{geo} {topic} novedades actualidad {year}",
        "{geo} {topic} últimas noticias semana {year}",
        "{geo} {topic} hoy noticia {year}",
    ],
}

# Equivalentes en inglés — fallback cuando los ángulos en español
# retornan menos de 4 candidatos.
TOPIC_ANGLES_EN: dict[str, list[str]] = {
    "security": [
        "{geo_en} crime internal security {year}",
        "{geo_en} defense military deployment {year}",
        "{geo_en} diplomacy tensions foreign policy {year}",
        "{geo_en} disaster emergency civil security {year}",
    ],
    "economy": [
        "{geo_en} economy market investment {year}",
        "{geo_en} employment wages companies {year}",
        "{geo_en} inflation prices trade {year}",
        "{geo_en} technology industry energy {year}",
    ],
    "politics": [
        "{geo_en} government elections politics {year}",
        "{geo_en} congress law reform legislation {year}",
        "{geo_en} opposition party leadership {year}",
        "{geo_en} corruption justice tribunal {year}",
    ],
    "default": [
        "{geo_en} {topic} recent news {year}",
        "{geo_en} {topic} latest news this week {year}",
        "{geo_en} {topic} today news {year}",
        "{geo_en} {topic} updates {year}",
    ],
}


def detect_news_topic(query: str) -> str:
    """Clasifica el query en uno de los tópicos: security, economy, politics, default."""
    lowered = query.lower()
    if any(k in lowered for k in [
        "seguridad", "security", "crimen", "defensa", "militar",
        "policía", "policia", "terroris", "ataque", "atentado", "conflicto",
    ]):
        return "security"
    if any(k in lowered for k in [
        "economía", "economia", "mercado", "bolsa", "precio", "inflacion",
        "inflación", "pib", "empleo", "comercio", "empresa",
    ]):
        return "economy"
    if any(k in lowered for k in [
        "política", "politica", "gobierno", "elección", "eleccion",
        "presidente", "congreso", "partido", "ministro",
    ]):
        return "politics"
    return "default"
