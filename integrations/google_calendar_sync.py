"""Mapeo de tareas validadas Moodle a eventos de Google Calendar."""
from __future__ import annotations

from datetime import datetime, timedelta

from features.web_scraping.domain.moodle_models import CalendarDraftEvent, MoodleAssignment


def build_calendar_draft_events(assignments: list[MoodleAssignment]) -> list[CalendarDraftEvent]:
    drafts: list[CalendarDraftEvent] = []
    for assignment in assignments:
        if not assignment.due_date:
            continue
        if assignment.status not in {"pending", "due_today"}:
            continue
        start_date = datetime.fromisoformat(assignment.due_date).date()
        end_date = start_date + timedelta(days=1)
        description_lines = [
            f"Curso: {assignment.course or 'Sin curso informado'}",
            f"Estado: {assignment.status}",
            f"Fuente: {assignment.source}",
        ]
        if assignment.url:
            description_lines.append(f"URL: {assignment.url}")
        if assignment.raw_date_text and assignment.raw_date_text != assignment.due_date:
            description_lines.append(f"Fecha original: {assignment.raw_date_text}")
        drafts.append(
            CalendarDraftEvent(
                summary=f"[Entrega] {assignment.title}",
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                description="\n".join(description_lines),
                source_title=assignment.title,
            )
        )
    return drafts


__all__ = ["build_calendar_draft_events"]
