"""Estructuras de contexto y política para el pipeline de búsqueda web.

QueryContext  — intención del usuario (entrada).
RecentPolicy  — cómo evaluar resultados de queries recientes (política de negocio).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class QueryContext:
    query_terms: list[str]
    query_source_group: Optional[str]
    source_terms: list[str]
    query_horizon: Optional[str]
    search_age_days: Optional[int]


@dataclass
class RecentPolicy:
    min_score: int
    min_body_lines: int
    min_sources: int
    min_candidates: int
    candidate_min_body_lines: int
    candidate_min_sources: int
