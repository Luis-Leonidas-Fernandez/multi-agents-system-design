"""Política de clasificación de estado para tareas Moodle."""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from features.web_scraping.domain.moodle_models import MoodleAssignmentStatus


_STATUS_SUBMITTED_TOKENS = (
    "entregado",
    "entrega realizada",
    "enviado para calificar",
    "submitted",
    "submission submitted",
    "calificado",
    "graded",
)

_STATUS_PENDING_TOKENS = (
    "pendiente",
    "por entregar",
    "no entregado",
    "abierta",
    "abierto",
)

_STATUS_OVERDUE_TOKENS = (
    "vencida",
    "vencido",
    "atrasada",
    "overdue",
)


def _today_local() -> date:
    return datetime.now(ZoneInfo("America/Argentina/Buenos_Aires")).date()


def classify_moodle_assignment_status(
    raw_status: str,
    due_date_iso: str,
    *,
    today: date | None = None,
) -> MoodleAssignmentStatus:
    lowered = (raw_status or "").strip().lower()
    if any(token in lowered for token in _STATUS_SUBMITTED_TOKENS):
        return "submitted"
    if any(token in lowered for token in _STATUS_OVERDUE_TOKENS):
        return "overdue"
    if any(token in lowered for token in _STATUS_PENDING_TOKENS):
        return "pending"

    if not due_date_iso:
        return "unknown"

    try:
        due_date = date.fromisoformat(due_date_iso)
    except ValueError:
        return "unknown"

    current_day = today or _today_local()
    if due_date < current_day:
        return "overdue"
    if due_date == current_day:
        return "due_today"
    return "pending"


__all__ = ["classify_moodle_assignment_status"]
