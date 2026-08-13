"""Bounded, safe diagnostic context for LinkedIn job-detail hydration."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import re
from typing import Iterator

from features.web_scraping.domain.linkedin_models import LinkedInDetailDiagnostic


_ACTIVE_DETAIL_DIAGNOSTICS: ContextVar[LinkedInDetailDiagnosticsCollector | None] = (
    ContextVar("linkedin_detail_diagnostics", default=None)
)


class LinkedInDetailDiagnosticsCollector:
    """Collect only enum, boolean, count, and numeric job-ID diagnostic data."""

    _MAX_EVENTS_PER_JOB = 40

    def __init__(self) -> None:
        self._events_by_job: dict[str, list[LinkedInDetailDiagnostic]] = {}

    @staticmethod
    def _job_id(value: object) -> str:
        candidate = str(getattr(value, "linkedin_job_id", value) or "").strip()
        return candidate if re.fullmatch(r"\d{1,20}", candidate) else ""

    @property
    def events(self) -> list[LinkedInDetailDiagnostic]:
        return [
            event
            for events in self._events_by_job.values()
            for event in events
        ]

    def record(
        self,
        record_or_job_id: object,
        *,
        phase: str,
        mode: str = "none",
        outcome: str,
        include_description: bool = False,
        description_ready: bool = False,
        date_ready: bool = False,
        rejection: str = "none",
    ) -> None:
        job_id = self._job_id(record_or_job_id)
        events = self._events_by_job.setdefault(job_id, [])
        if len(events) >= self._MAX_EVENTS_PER_JOB:
            return
        events.append(
            LinkedInDetailDiagnostic(
                job_id=job_id,
                sequence=len(events) + 1,
                phase=phase,
                mode=mode,
                outcome=outcome,
                include_description=include_description,
                description_ready=description_ready,
                date_ready=date_ready,
                rejection=rejection,
            )
        )


def get_active_detail_diagnostics() -> LinkedInDetailDiagnosticsCollector | None:
    return _ACTIVE_DETAIL_DIAGNOSTICS.get()


@contextmanager
def detail_diagnostics_context() -> Iterator[LinkedInDetailDiagnosticsCollector]:
    collector = LinkedInDetailDiagnosticsCollector()
    token = _ACTIVE_DETAIL_DIAGNOSTICS.set(collector)
    try:
        yield collector
    finally:
        _ACTIVE_DETAIL_DIAGNOSTICS.reset(token)


__all__ = [
    "LinkedInDetailDiagnosticsCollector",
    "detail_diagnostics_context",
    "get_active_detail_diagnostics",
]
