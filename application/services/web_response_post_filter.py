"""Post-filtro liviano para respuestas web antes de mostrarlas al usuario.

El objetivo es mantener esta capa totalmente desacoplada del discovery/ranking:
- se aplica al final, antes de responder en UI
- se puede remover o ajustar sin tocar el pipeline principal
- debe respetar la temática detectada en la query del usuario
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text or "")
        if unicodedata.category(c) != "Mn"
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_accents(text).lower()).strip()


def _split_summary_and_sources(summary: str) -> tuple[list[str], list[str]]:
    lines = [line.rstrip() for line in (summary or "").splitlines()]
    body: list[str] = []
    sources: list[str] = []
    in_sources = False
    for line in lines:
        stripped = line.strip()
        if stripped == "Sources:":
            in_sources = True
            continue
        if in_sources:
            if stripped:
                sources.append(stripped)
            continue
        if stripped:
            body.append(stripped)
    return body, sources


_TOPIC_SIGNAL_PROFILES: dict[str, tuple[str, ...]] = {
    "security": (
        # italiano
        "allarme sicurezza", "sicurezza", "polizia", "police", "arresto", "omicidio",
        "coltell", "sabot", "ordigno", "41 bis", "blitz", "vandali", "vandalizz",
        "indagini", "inchiesta", "nas", "carabin", "guardia di finanza",
        "decreto sicurezza", "sicurezza pubblica",
        # español
        "seguridad", "policia", "policía", "detenid", "arrestad", "arrest",
        "operativo", "homicidio", "asesin", "robo", "narco", "policiales",
        "fiscal", "juzgado", "imputad", "sentencia", "condena", "carcel",
        "prision", "violencia", "sucesos", "crimen", "delito",
        "guardia civil", "ertzaintza", "mossos",
        # inglés
        "crime", "criminal", "terror",
    ),
    "economy": (
        "econom",
        "mercad",
        "mercato",
        "crecim",
        "crescit",
        "growth",
        "borsa",
        "inflaz",
        "inflacion",
        "inflazione",
        "pib",
        "inversion",
        "invest",
        "emploi",
        "empleo",
        "salari",
        "tasso",
        "banche",
        "banca",
        "imprese",
        "empresa",
        "industria",
        "deficit",
        "debito",
        "finanza",
        "finanzi",
        "commercio",
        "export",
    ),
    "politics": (
        "politic",
        "govern",
        "parlament",
        "congres",
        "senat",
        "decreto",
        "legge",
        "ley",
        "elezion",
        "oposic",
        "opposiz",
        "ministro",
        "president",
        "coalizion",
        "partito",
        "partid",
        "votacion",
        "votazione",
        "camera dei deputati",
    ),
    "sports": (
        "futbol",
        "football",
        "soccer",
        "gol",
        "partido",
        "partita",
        "liga",
        "serie a",
        "champions",
        "copa",
        "torneo",
        "nba",
        "nfl",
        "mlb",
        "tenis",
        "atp",
        "wta",
        "golazo",
        "marcador",
        "resultado",
        "resultados",
        "clasico",
    ),
}

_TOPIC_STRONG_SIGNALS: dict[str, tuple[str, ...]] = {
    "security": (
        "allarme sicurezza", "omicidio", "arresto", "ordigno", "terror",
        "decreto sicurezza", "sicurezza pubblica",
        # español
        "seguridad", "homicidio", "arrestad", "detenid", "operativo",
        "policiales", "guardia civil", "ertzaintza",
    ),
    "economy": (
        "crecim",
        "crescit",
        "growth",
        "inversion",
        "invest",
        "mercato",
        "mercad",
        "salari",
        "empleo",
        "pib",
        "inflaz",
        "borsa",
        "export",
    ),
    "politics": (
        "parlament",
        "govern",
        "elezion",
        "elezioni",
        "decreto",
        "ministro",
        "partito",
        "coalizion",
    ),
    "sports": (
        "partita",
        "partido",
        "gol",
        "serie a",
        "champions",
        "torneo",
        "resultado",
    ),
}

_TOPIC_QUERY_SIGNALS: dict[str, tuple[str, ...]] = {
    "security": (
        "seguridad",
        "security",
        "sicurezza",
        "policia",
        "polizia",
        "crimen",
        "crime",
        "orden publico",
        "public safety",
        "cronaca",
        "violencia",
        "defensa",
        "militar",
        "terror",
    ),
    "economy": (
        "economia",
        "economy",
        "mercado",
        "mercati",
        "inflacion",
        "inflazione",
        "finanza",
        "finanzas",
        "empresa",
        "empresas",
        "salario",
        "empleo",
        "comercio",
        "pib",
        "bolsa",
    ),
    "politics": (
        "politica",
        "politics",
        "gobierno",
        "governo",
        "parlamento",
        "parliament",
        "eleccion",
        "elecciones",
        "elezione",
        "elezioni",
        "presidente",
        "president",
        "congreso",
        "senado",
        "ministro",
        "partido",
    ),
    "sports": (
        "deporte",
        "deportes",
        "sport",
        "sports",
        "futbol",
        "football",
        "soccer",
        "tenis",
        "nba",
        "nfl",
        "mlb",
        "partido",
        "partidos",
        "resultado",
        "resultados",
        "liga",
    ),
}


def _detect_query_topic(query: str) -> str:
    normalized = _normalize(query)
    topic_scores: dict[str, int] = {}
    for topic, signals in _TOPIC_QUERY_SIGNALS.items():
        score = sum(1 for signal in signals if signal in normalized)
        if score:
            topic_scores[topic] = score
    if not topic_scores:
        return "default"
    return max(topic_scores.items(), key=lambda item: item[1])[0]


def _topic_signal_score(text: str, topic: str) -> int:
    normalized = _normalize(text)
    signals = _TOPIC_SIGNAL_PROFILES.get(topic, ())
    return sum(1 for signal in signals if signal in normalized)


def _line_matches_topic(line: str, topic: str) -> bool:
    if topic not in _TOPIC_SIGNAL_PROFILES:
        return True

    normalized = _normalize(line)
    target_score = _topic_signal_score(normalized, topic)
    if target_score == 0:
        return False

    strong_signals = _TOPIC_STRONG_SIGNALS.get(topic, ())
    has_strong_signal = any(signal in normalized for signal in strong_signals)
    competing_score = max(
        (_topic_signal_score(normalized, other_topic)
         for other_topic in _TOPIC_SIGNAL_PROFILES
         if other_topic != topic),
        default=0,
    )

    if has_strong_signal:
        return target_score >= competing_score

    return target_score >= 2 and target_score > competing_score


def _rebuild_summary(body_lines: list[str], source_lines: list[str]) -> str:
    parts: list[str] = []
    parts.extend(body_lines)
    if source_lines:
        if parts:
            parts.append("")
        parts.append("Sources:")
        parts.extend(source_lines)
    return "\n".join(parts).strip()


def apply_web_response_post_filter(
    summary: str,
    query: str,
    sources: Optional[list[dict[str, str]]] = None,
) -> tuple[str, Optional[list[dict[str, str]]]]:
    """Aplica un filtro final, conservador y temático.

    Reglas:
    - detecta la temática a partir de la query
    - solo filtra si encuentra al menos 2 líneas claramente alineadas con ese tema
    - si no tiene suficiente evidencia temática, deja la respuesta intacta
    """

    if not summary.strip():
        return summary, sources

    topic = _detect_query_topic(query)
    if topic == "default":
        return summary, sources

    body_lines, source_lines = _split_summary_and_sources(summary)
    if len(body_lines) <= 1:
        return summary, sources

    kept_lines = [line for line in body_lines if _line_matches_topic(line, topic)]
    if len(kept_lines) < 2 or len(kept_lines) == len(body_lines):
        return summary, sources

    filtered_sources = sources
    if sources and len(sources) > len(kept_lines):
        filtered_sources = list(sources[: len(kept_lines)])
        source_lines = [
            f"- [{source.get('title') or source.get('url') or 'source'}]({source.get('url') or ''})"
            for source in filtered_sources
            if (source.get("url") or "").strip()
        ]

    return _rebuild_summary(kept_lines, source_lines), filtered_sources
