"""Política de scoring y ranking de candidatos web.

Centraliza los criterios de puntuación para que no queden atrapados
dentro del use_case web_scraping_flow.
"""
import re
from typing import Optional
from urllib.parse import urlparse

from domain.web_classifier import _is_hub_like_candidate
from application.policies.web_source_policy import score_domain_boost, get_source_domain_priority

# --- Scores positivos ---
SCORE_TERM_MATCH = 3          # término de la query encontrado en blob
SCORE_PRICE_RANGE = 2         # patrón numérico tipo "100 - 200" en blob
SCORE_DATE_IN_URL = 4         # fecha YYYYMMDD o YYYY/MM/DD en path
SCORE_DEEP_URL = 2            # URL con 3 o más segmentos de path

# --- Penalizaciones ---
PENALTY_SHALLOW_URL = -3      # URL corta sin fecha → probable hub/listing
PENALTY_NAV_SEGMENT = -4      # segmento de navegación (tag, category, archive…)
PENALTY_NOISE_WORD = -2       # palabra de ruido (login, cookie, privacy…)
PENALTY_NO_TITLE_MATCH = -6   # título sin ningún término significativo de la query
PENALTY_HOMEPAGE_FALLBACK = -8
PENALTY_SECTION_FALLBACK = -3
PENALTY_HUB_LIKE = -12        # candidato que parece portada/hub de sitio

_NAV_SEGMENTS = {"topic", "topics", "tag", "tags", "category", "categories", "archive", "author"}
_NOISE_WORDS = {"login", "signin", "cookie", "privacy", "archive", "perfil"}


def _score_generic_candidate(
    candidate: dict[str, str],
    query_terms: list[str],
    query_source_group: Optional[str] = None,
) -> int:
    blob = " ".join([candidate.get("title", ""), candidate.get("snippet", ""), candidate.get("url", "")]).lower()
    score = 0

    for term in query_terms:
        if term in blob:
            score += SCORE_TERM_MATCH

    if re.search(r"\b\d+\s*-\s*\d+\b", blob):
        score += SCORE_PRICE_RANGE

    url = candidate.get("url", "")
    path = urlparse(url).path.lower()
    segments = [s for s in path.split("/") if s]

    if re.search(r"(19|20)\d{2}[/\-]?\d{2}[/\-]?\d{2}", path) or re.search(r"\d{6,8}", path):
        score += SCORE_DATE_IN_URL
    if len(segments) >= 3:
        score += SCORE_DEEP_URL
    if len(segments) <= 2 and not re.search(r"(19|20)\d{2}[/\-]?\d{2}[/\-]?\d{2}", path):
        score += PENALTY_SHALLOW_URL
    if any(seg in _NAV_SEGMENTS for seg in segments):
        score += PENALTY_NAV_SEGMENT
    if any(noise in blob for noise in _NOISE_WORDS):
        score += PENALTY_NOISE_WORD

    score += score_domain_boost(query_source_group, url)

    # Penalizar candidatos cuyo título no contiene ningún término significativo
    # (longitud ≥ 4 para excluir stopwords cortas). Excepción: section_fallback
    # con el término presente en el snippet.
    title_lower = candidate.get("title", "").lower()
    snippet_lower = candidate.get("snippet", "").lower()
    meaningful_terms = [t for t in query_terms if len(t) >= 4]
    if meaningful_terms and not any(t in title_lower for t in meaningful_terms):
        if candidate.get("source_kind") == "section_fallback" and any(t in snippet_lower for t in meaningful_terms):
            pass
        else:
            score += PENALTY_NO_TITLE_MATCH

    if candidate.get("source_kind") == "homepage_fallback":
        score += PENALTY_HOMEPAGE_FALLBACK
    if candidate.get("source_kind") == "section_fallback":
        score += PENALTY_SECTION_FALLBACK
    if _is_hub_like_candidate(candidate):
        score += PENALTY_HUB_LIKE

    return score


def _candidate_source_priority(candidate: dict[str, str], query_source_group: Optional[str]) -> int:
    return get_source_domain_priority(query_source_group, candidate.get("url", ""))


def _rank_candidates_by_source_policy(
    candidates: list[dict[str, str]],
    query_terms: list[str],
    query_source_group: Optional[str],
) -> list[dict[str, str]]:
    if not candidates:
        return []
    if not query_source_group:
        return sorted(
            candidates,
            key=lambda c: _score_generic_candidate(c, query_terms, query_source_group),
            reverse=True,
        )
    return sorted(
        candidates,
        key=lambda c: (
            _candidate_source_priority(c, query_source_group),
            -_score_generic_candidate(c, query_terms, query_source_group),
        ),
    )
