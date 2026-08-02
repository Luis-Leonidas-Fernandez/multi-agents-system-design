"""Detección acotada de intención para búsqueda de vacantes LinkedIn."""
from __future__ import annotations

import re
import unicodedata


_JOB_TERMS = (
    "vacante",
    "vacantes",
    "empleo",
    "empleos",
    "trabajo",
    "trabajos",
    "job",
    "jobs",
    "position",
    "positions",
    "oportunidad laboral",
)
_LINKEDIN_TERMS = ("linkedin", "linked in")
_LOCATION_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Japan", ("japon",)),
    ("South Korea", ("corea del sur", "south korea")),
)
_TARGET_TERMS = (
    "ai engineer",
    "ai engineering",
    "ingenieria de ia",
    "ingeniero de ia",
    "inteligencia artificial",
    "data science",
    "data scientist",
    "data analyst",
    "ciencia de datos",
    "machine learning",
    "deep learning",
    "deeplearning",
    "llm",
    "llm scientist",
    "generative ai",
    "genai",
    "producto con ia",
    "productos con ia",
    "ai product",
    "ai agent",
    "ai agents",
    "ai agent engineer",
    "ai agent developer",
    "ai architect",
    "ai specialist",
    "ai mentor",
    "ai solutions architect",
    "arquitecto de ia",
    "agentes de ia",
    "solution architect",
    "developer technology engineer",
    "developer technology engineer ai",
    "developer technology engineer -ai",
    "developer technology engineer - ai",
    "artificial intelligence engineer",
    "applied ai engineer",
    "automation",
    "ml engineer",
    "mlops",
    "mlops engineer",
    "speech llm engineer",
    "rag llm system",
    "rag & llm system",
    "rag and llm system",
    "dl",
    "dl engineer",
    "ai",
)


def normalize_linkedin_intent_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", (value or "").strip().lower())
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", without_accents).strip()


def detect_linkedin_jobs_intent(message: str) -> bool:
    normalized = normalize_linkedin_intent_text(message)
    has_job_intent = any(term in normalized for term in _JOB_TERMS)
    has_target = any(term in normalized for term in _TARGET_TERMS)
    has_linkedin = any(term in normalized for term in _LINKEDIN_TERMS)
    recent_hint = any(term in normalized for term in ("hoy", "ultimas 24", "últimas 24", "today", "24 hours"))
    return has_job_intent and (has_linkedin or has_target) and (has_target or recent_hint)


def extract_linkedin_locations(message: str) -> tuple[str, ...]:
    """Extrae ubicaciones conocidas preservando el orden en que aparecen."""
    normalized = normalize_linkedin_intent_text(message)
    matches: list[tuple[int, str]] = []
    for canonical_name, aliases in _LOCATION_ALIASES:
        positions = [
            match.start()
            for alias in aliases
            if (match := re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized))
        ]
        if positions:
            matches.append((min(positions), canonical_name))
    return tuple(name for _, name in sorted(matches))


__all__ = [
    "detect_linkedin_jobs_intent",
    "extract_linkedin_locations",
    "normalize_linkedin_intent_text",
]
