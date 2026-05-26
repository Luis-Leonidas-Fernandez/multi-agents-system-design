"""Contratos de datos para la integración interna con Notion."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


NotionTaskStatus = Literal["No entregado", "Vence hoy", "Vencido", "Entregado", "Desconocido"]
NotionUpsertAction = Literal["created", "updated", "skipped", "error"]


@dataclass(frozen=True)
class NotionTaskRecord:
    external_id: str
    title: str
    course: str = ""
    due_date: str = ""
    status: NotionTaskStatus = "Desconocido"
    source_url: str = ""
    source: str = "moodle"
    raw_date_text: str = ""
    last_synced_at: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class NotionUpsertResult:
    action: NotionUpsertAction
    external_id: str
    title: str
    page_id: str = ""
    page_url: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class NotionSyncSummary:
    created: list[NotionUpsertResult] = field(default_factory=list)
    updated: list[NotionUpsertResult] = field(default_factory=list)
    skipped: list[NotionUpsertResult] = field(default_factory=list)
    errors: list[NotionUpsertResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "created_count": len(self.created),
            "updated_count": len(self.updated),
            "skipped_count": len(self.skipped),
            "error_count": len(self.errors),
            "created": [item.to_dict() for item in self.created],
            "updated": [item.to_dict() for item in self.updated],
            "skipped": [item.to_dict() for item in self.skipped],
            "errors": [item.to_dict() for item in self.errors],
        }


__all__ = [
    "NotionSyncSummary",
    "NotionTaskRecord",
    "NotionTaskStatus",
    "NotionUpsertAction",
    "NotionUpsertResult",
]
