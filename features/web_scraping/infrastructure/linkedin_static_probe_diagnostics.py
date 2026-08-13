"""Bounded, safe diagnostic context for LinkedIn static read-only probes."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import re
from typing import Iterator

from features.web_scraping.domain.linkedin_models import LinkedInStaticProbeDiagnostic


_ACTIVE_STATIC_PROBE_DIAGNOSTICS: ContextVar[
    LinkedInStaticProbeDiagnosticsCollector | None
] = ContextVar("linkedin_static_probe_diagnostics", default=None)


class LinkedInStaticProbeDiagnosticsCollector:
    """Collect only enum, count, status-code, query label, and numeric job IDs."""

    _MAX_EVENTS = 2000

    def __init__(self) -> None:
        self._events: list[LinkedInStaticProbeDiagnostic] = []

    @staticmethod
    def _job_id(value: object) -> str:
        candidate = str(getattr(value, "linkedin_job_id", value) or "").strip()
        return candidate if re.fullmatch(r"\d{1,20}", candidate) else ""

    @property
    def events(self) -> list[LinkedInStaticProbeDiagnostic]:
        return list(self._events)

    def record(
        self,
        *,
        kind: str,
        outcome: str,
        query: str = "",
        job_id: object = "",
        status_code: int = 0,
        candidate_count: int = 0,
        accepted_count: int = 0,
        body_source: str = "",
        description_length: int = 0,
        guest_status_code: int = 0,
        guest_retry_count: int = 0,
        identity_consistent: bool = True,
    ) -> None:
        if len(self._events) >= self._MAX_EVENTS:
            return
        self._events.append(
            LinkedInStaticProbeDiagnostic(
                kind=kind,
                query=query,
                job_id=self._job_id(job_id),
                sequence=len(self._events) + 1,
                status_code=max(0, min(999, int(status_code or 0))),
                candidate_count=max(0, min(10000, int(candidate_count or 0))),
                accepted_count=max(0, min(10000, int(accepted_count or 0))),
                outcome=outcome,
                body_source=re.sub(r"[^a-z0-9_:-]", "", body_source[:40]),
                description_length=max(
                    0,
                    min(200000, int(description_length or 0)),
                ),
                guest_status_code=max(0, min(999, int(guest_status_code or 0))),
                guest_retry_count=max(0, min(10, int(guest_retry_count or 0))),
                identity_consistent=bool(identity_consistent),
            )
        )


def get_active_static_probe_diagnostics() -> (
    LinkedInStaticProbeDiagnosticsCollector | None
):
    return _ACTIVE_STATIC_PROBE_DIAGNOSTICS.get()


@contextmanager
def static_probe_diagnostics_context() -> Iterator[
    LinkedInStaticProbeDiagnosticsCollector
]:
    collector = LinkedInStaticProbeDiagnosticsCollector()
    token = _ACTIVE_STATIC_PROBE_DIAGNOSTICS.set(collector)
    try:
        yield collector
    finally:
        _ACTIVE_STATIC_PROBE_DIAGNOSTICS.reset(token)


__all__ = [
    "LinkedInStaticProbeDiagnosticsCollector",
    "get_active_static_probe_diagnostics",
    "static_probe_diagnostics_context",
]
