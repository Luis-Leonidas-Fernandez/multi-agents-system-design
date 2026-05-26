"""Render humano de tareas Moodle validadas."""
from __future__ import annotations

from features.web_scraping.domain.moodle_models import MoodleAssignment, MoodleAssignmentValidationIssue, ValidatedMoodleAssignments


def _status_label(status: str) -> str:
    labels = {
        "pending": "No entregado",
        "due_today": "Vence hoy",
        "overdue": "Vencido",
        "submitted": "Entregado",
        "unknown": "Desconocido",
    }
    return labels.get(status, status or "Desconocido")


def _status_summary(validated: ValidatedMoodleAssignments) -> dict[str, int]:
    summary = {
        "submitted": 0,
        "pending": 0,
        "due_today": 0,
        "overdue": 0,
        "unknown": 0,
    }
    for item in validated.valid:
        summary[item.status] = summary.get(item.status, 0) + 1
    for item in validated.invalid:
        summary[item.status] = summary.get(item.status, 0) + 1
    return summary


def _render_assignment(item: MoodleAssignment) -> list[str]:
    lines = [f"- **{item.title}**"]
    lines.append(f"  - Fecha: `{item.due_date}`")
    lines.append(f"  - Estado: `{_status_label(item.status)}`")
    if item.course:
        lines.append(f"  - Curso: {item.course}")
    if item.url:
        lines.append(f"  - URL: {item.url}")
    if item.raw_date_text and item.raw_date_text != item.due_date:
        lines.append(f"  - Fecha original: {item.raw_date_text}")
    return lines


def _render_issue(issue: MoodleAssignmentValidationIssue) -> str:
    return f"- `{issue.severity}` · **{issue.assignment_title}** · {issue.message}"


def render_moodle_assignments_chat(validated: ValidatedMoodleAssignments) -> str:
    summary = _status_summary(validated)
    lines: list[str] = [
        f"Tareas Moodle encontradas: {len(validated.valid)} válidas, {len(validated.invalid)} inválidas.",
        f"Estados → No entregado: {summary['pending']} · Vence hoy: {summary['due_today']} · Vencido: {summary['overdue']} · Entregado: {summary['submitted']}",
        "",
    ]

    if validated.valid:
        for index, item in enumerate(validated.valid, 1):
            lines.extend(
                [
                    f"{index}. {item.title}",
                    f"   Curso: {item.course or 'Sin curso'}",
                    f"   Fecha: {item.due_date}",
                    f"   Estado: {_status_label(item.status)}",
                    f"   URL: {item.url or 'Sin URL'}",
                ]
            )
            if item.raw_date_text and item.raw_date_text != item.due_date:
                lines.append(f"   Original: {item.raw_date_text}")
            lines.append("")
            lines.append("────────────────────────")
            lines.append("")
    else:
        lines.extend(
            [
                "No hay tareas válidas listas para revisar.",
                "",
            ]
        )

    if validated.issues:
        lines.append("Observaciones:")
        for issue in validated.issues:
            lines.append(f"- {issue.assignment_title}: {issue.message}")

    lines.append("")
    lines.append("Revisá el preview y el ícono JSON antes de crear eventos.")
    return "\n".join(lines).strip()


def render_moodle_assignments_review(validated: ValidatedMoodleAssignments) -> str:
    summary = _status_summary(validated)
    lines: list[str] = [
        "# Revisión de tareas Moodle",
        "",
        f"- Tareas válidas: **{len(validated.valid)}**",
        f"- Tareas inválidas: **{len(validated.invalid)}**",
        f"- Issues: **{len(validated.issues)}**",
        f"- No entregado: **{summary['pending']}**",
        f"- Vence hoy: **{summary['due_today']}**",
        f"- Vencido: **{summary['overdue']}**",
        f"- Entregado: **{summary['submitted']}**",
        "",
    ]

    lines.extend(["## Tareas válidas", ""])
    if not validated.valid:
        lines.append("- No hay tareas válidas listas para calendario.")
    else:
        for item in validated.valid:
            lines.extend(_render_assignment(item))
            lines.append("")

    lines.extend(["## Tareas inválidas", ""])
    if not validated.invalid:
        lines.append("- No hay tareas inválidas.")
    else:
        for item in validated.invalid:
            lines.extend(_render_assignment(item))
            lines.append("")

    lines.extend(["## Warnings y errores", ""])
    if not validated.issues:
        lines.append("- Sin warnings ni errores.")
    else:
        lines.extend(_render_issue(issue) for issue in validated.issues)

    return "\n".join(lines).strip() + "\n"


__all__ = ["render_moodle_assignments_chat", "render_moodle_assignments_review"]
