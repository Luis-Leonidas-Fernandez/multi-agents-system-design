"""Enriquecimiento de estado real de entrega para tareas Moodle."""
from __future__ import annotations

from collections.abc import Mapping

from features.web_scraping.infrastructure.scraping_tools import fetch_moodle_submission_statuses


def enrich_moodle_submission_statuses(
    raw_assignments: list[Mapping[str, object]],
    *,
    base_url: str = "",
) -> list[dict[str, str]]:
    """Inyecta estado de entrega real desde el detalle de cada tarea sin alterar la extracción base."""
    normalized_items = [{str(key): str(value) for key, value in item.items()} for item in raw_assignments]
    enriched = [dict(item) for item in normalized_items]
    if not enriched:
        return enriched

    statuses = fetch_moodle_submission_statuses(enriched, base_url)
    for item in enriched:
        href = str(item.get("url") or "").strip()
        if not href:
            continue
        enriched_status = statuses.get(href, "")
        if enriched_status:
            item["status"] = enriched_status
    return enriched


__all__ = ["enrich_moodle_submission_statuses"]
