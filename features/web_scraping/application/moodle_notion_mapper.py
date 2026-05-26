"""Mapeo de tareas Moodle validadas al contrato interno de Notion."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1

from features.web_scraping.domain.moodle_models import MoodleAssignment
from integrations.notion_task_models import NotionTaskRecord, NotionTaskStatus


def build_moodle_external_id(assignment: MoodleAssignment) -> str:
    if assignment.url.strip():
        return f"moodle:{assignment.url.strip()}"
    fingerprint = f"{assignment.title.strip().lower()}|{assignment.due_date.strip()}|{assignment.course.strip().lower()}"
    return f"moodle:{sha1(fingerprint.encode('utf-8')).hexdigest()[:16]}"


def map_moodle_status_to_notion(status: str) -> NotionTaskStatus:
    mapping: dict[str, NotionTaskStatus] = {
        "pending": "No entregado",
        "due_today": "Vence hoy",
        "overdue": "Vencido",
        "submitted": "Entregado",
        "unknown": "Desconocido",
    }
    return mapping.get(str(status or "").strip(), "Desconocido")


def map_moodle_assignment_to_notion_record(
    assignment: MoodleAssignment,
    *,
    synced_at: str | None = None,
) -> NotionTaskRecord:
    return NotionTaskRecord(
        external_id=build_moodle_external_id(assignment),
        title=assignment.title.strip(),
        course=assignment.course.strip(),
        due_date=assignment.due_date.strip(),
        status=map_moodle_status_to_notion(assignment.status),
        source_url=assignment.url.strip(),
        source=assignment.source.strip() or "moodle",
        raw_date_text=assignment.raw_date_text.strip(),
        last_synced_at=synced_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )


def map_moodle_assignments_to_notion_records(
    assignments: list[MoodleAssignment],
    *,
    synced_at: str | None = None,
) -> list[NotionTaskRecord]:
    return [
        map_moodle_assignment_to_notion_record(item, synced_at=synced_at)
        for item in assignments
    ]


__all__ = [
    "build_moodle_external_id",
    "map_moodle_assignment_to_notion_record",
    "map_moodle_assignments_to_notion_records",
    "map_moodle_status_to_notion",
]
