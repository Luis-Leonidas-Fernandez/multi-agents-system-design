"""Normalización de tareas Moodle a estructura estable."""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from features.web_scraping.domain.moodle_models import MoodleAssignment


def _normalize_status(raw_status: str) -> str:
    lowered = (raw_status or "").strip().lower()
    if any(token in lowered for token in ("vencida", "vencido", "overdue")):
        return "overdue"
    if any(token in lowered for token in ("entregado", "enviado", "submitted", "calificado", "graded")):
        return "submitted"
    if any(token in lowered for token in ("pendiente", "por entregar", "no entregado", "abierta", "abierto")):
        return "pending"
    return "unknown"


def _normalize_assignment_item(item: Mapping[str, object]) -> MoodleAssignment:
    title = str(item.get("name") or item.get("title") or "").strip()
    course = str(item.get("course") or "").strip()
    raw_date_text = str(item.get("date") or item.get("due_date") or "").strip()
    url = str(item.get("url") or "").strip()
    status = _normalize_status(str(item.get("status") or ""))
    return MoodleAssignment(
        title=title,
        course=course,
        due_date=raw_date_text,
        url=url,
        status=status,  # type: ignore[arg-type]
        source="moodle",
        raw_date_text=raw_date_text,
    )


def normalize_moodle_assignments(raw_assignments: Iterable[Mapping[str, object]]) -> list[MoodleAssignment]:
    return [_normalize_assignment_item(item) for item in raw_assignments]


def normalize_moodle_assignments_from_text(raw_text: str) -> list[MoodleAssignment]:
    assignments: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for raw_line in (raw_text or "").splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.startswith("─") or line.startswith("TAREAS EN MOODLE"):
            continue

        item_match = re.match(r"^\s*\d+\.\s+(?P<title>.+?)(?:\s+⚠\s+VENCIDA)?\s*$", line)
        if item_match:
            if current and current.get("name"):
                assignments.append(current)
            title = item_match.group("title").strip()
            current = {"name": title, "status": "VENCIDA" if "VENCIDA" in line else ""}
            continue

        if current is None:
            continue

        if line.strip().startswith("Curso"):
            current["course"] = line.split("Curso", 1)[1].strip(" :")
            continue
        if line.strip().startswith("Fecha"):
            current["date"] = line.split("Fecha", 1)[1].strip(" :")
            continue
        if line.strip().startswith("URL"):
            current["url"] = line.split("URL", 1)[1].strip(" :")
            continue

    if current and current.get("name"):
        assignments.append(current)

    return normalize_moodle_assignments(assignments)


__all__ = [
    "normalize_moodle_assignments",
    "normalize_moodle_assignments_from_text",
]
