"""Postprocesamiento de respuestas de web scraping."""
from __future__ import annotations

from typing import Any, Optional

from application.services.web_response_post_filter import apply_web_response_post_filter


def _finalize_web_user_summary(
    summary: str,
    last_message: str,
    sources: Optional[list[dict[str, str]]] = None,
) -> tuple[str, Optional[list[dict[str, str]]], list[str]]:
    filtered_summary, filtered_sources = apply_web_response_post_filter(
        summary=summary,
        query=last_message,
        sources=sources,
    )
    final_words = filtered_summary.split()
    return filtered_summary, filtered_sources, final_words


def _build_no_local_sources_response(last_message: str) -> dict[str, Any]:
    from application.use_cases import web_scraping_flow as _flow

    geography = _flow._extract_query_geography(last_message) or "ese país"
    summary = (
        f"No encontré fuentes locales confiables de {geography} para esta semana. "
        "Prefiero no mezclar resultados globales ruidosos o tangenciales."
    )
    return {
        "summary": summary,
        "words": summary.split(),
        "source_type": "search",
        "sources": [],
        "pre_synthesized": True,
    }


__all__ = ["_finalize_web_user_summary", "_build_no_local_sources_response"]
