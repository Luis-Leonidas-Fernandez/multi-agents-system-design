"""Validación de tareas Moodle ya normalizadas."""
from __future__ import annotations

from datetime import date

from features.web_scraping.domain.moodle_models import (
    MoodleAssignment,
    MoodleAssignmentValidationIssue,
    ValidatedMoodleAssignments,
)
from features.web_scraping.application.moodle_date_parser import parse_moodle_due_date
from features.web_scraping.application.moodle_status_policy import classify_moodle_assignment_status


def validate_moodle_assignments(assignments: list[MoodleAssignment]) -> ValidatedMoodleAssignments:
    valid: list[MoodleAssignment] = []
    invalid: list[MoodleAssignment] = []
    issues: list[MoodleAssignmentValidationIssue] = []
    seen: set[tuple[str, str]] = set()

    for assignment in assignments:
        title = assignment.title.strip()
        if not title:
            invalid.append(assignment)
            issues.append(
                MoodleAssignmentValidationIssue(
                    assignment_title="(sin título)",
                    severity="error",
                    message="La tarea no tiene título.",
                )
            )
            continue

        due_date = parse_moodle_due_date(assignment.due_date)
        if due_date is None:
            invalid.append(assignment)
            issues.append(
                MoodleAssignmentValidationIssue(
                    assignment_title=title,
                    severity="error",
                    message="No se pudo inferir una fecha válida para la tarea.",
                )
            )
            continue

        dedupe_key = (title.lower(), due_date)
        if dedupe_key in seen:
            invalid.append(assignment)
            issues.append(
                MoodleAssignmentValidationIssue(
                    assignment_title=title,
                    severity="warning",
                    message=f"Tarea duplicada detectada para la fecha {due_date}.",
                )
            )
            continue
        seen.add(dedupe_key)

        if not assignment.course:
            issues.append(
                MoodleAssignmentValidationIssue(
                    assignment_title=title,
                    severity="warning",
                    message="La tarea no informa curso; se conservará vacía en el evento.",
                )
            )

        status = classify_moodle_assignment_status(assignment.status, due_date)
        valid.append(
            MoodleAssignment(
                title=title,
                course=assignment.course.strip(),
                due_date=due_date,
                url=assignment.url.strip(),
                status=status,
                source=assignment.source,
                raw_date_text=assignment.raw_date_text or assignment.due_date,
            )
        )

    def _sort_key(item: MoodleAssignment) -> tuple[date, str]:
        return (date.fromisoformat(item.due_date), item.title.lower())

    valid.sort(key=_sort_key)
    invalid.sort(key=lambda item: (item.due_date or "9999-99-99", item.title.lower()))

    return ValidatedMoodleAssignments(valid=valid, invalid=invalid, issues=issues)


__all__ = ["validate_moodle_assignments"]
