"""Validación de duplicados antes de crear eventos en Google Calendar."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from features.web_scraping.domain.moodle_models import CalendarDraftEvent
from integrations.google_calendar_mcp import list_events


@dataclass(frozen=True)
class CalendarDuplicateCheckResult:
    to_create: list[CalendarDraftEvent]
    duplicates: list[dict[str, str]]


def _event_start_date(item: dict[str, Any]) -> str:
    start = item.get("start")
    if not isinstance(start, dict):
        return ""
    if isinstance(start.get("date"), str):
        return str(start["date"])
    if isinstance(start.get("dateTime"), str):
        return str(start["dateTime"])[:10]
    return ""


def _same_event(existing: dict[str, Any], draft: CalendarDraftEvent) -> bool:
    summary = str(existing.get("summary") or "").strip()
    return summary == draft.summary and _event_start_date(existing) == draft.start


def partition_calendar_drafts(
    drafts: list[CalendarDraftEvent],
    *,
    calendar_id: str | None = None,
) -> CalendarDuplicateCheckResult:
    to_create: list[CalendarDraftEvent] = []
    duplicates: list[dict[str, str]] = []

    for draft in drafts:
        payload = list_events(
            {
                "time_min": f"{draft.start}T00:00:00Z",
                "time_max": f"{draft.end}T00:00:00Z",
                "max_results": 20,
                "query": draft.summary,
                "calendar_id": calendar_id,
            }
        )
        items = payload.get("items") if isinstance(payload, dict) else []
        existing_items = items if isinstance(items, list) else []
        match = next((item for item in existing_items if isinstance(item, dict) and _same_event(item, draft)), None)
        if match is None:
            to_create.append(draft)
            continue
        duplicates.append(
            {
                "summary": draft.summary,
                "start": draft.start,
                "source_title": draft.source_title,
                "event_id": str(match.get("id") or ""),
                "html_link": str(match.get("htmlLink") or ""),
            }
        )

    return CalendarDuplicateCheckResult(to_create=to_create, duplicates=duplicates)


__all__ = ["CalendarDuplicateCheckResult", "partition_calendar_drafts"]
