"""Persistencia auditable del vertical LinkedIn Jobs."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from application.services.request_runtime import get_request_runtime_config
from features.web_scraping.domain.linkedin_models import (
    LinkedInAuditMeta,
    LinkedInAuditSnapshot,
    LinkedInQueryTiming,
    LinkedInRejectedRecord,
    LinkedInVacancyRecord,
)


@dataclass(frozen=True)
class LinkedInAuditPaths:
    job_uid: str
    audit_dir: Path
    json_path: Path
    schema_path: Path
    summary_path: Path


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-") or "local"


def _build_job_uid(session_id: str, request_id: str) -> str:
    return (
        f"linkedin-job-{_slug(session_id)}-{_slug(request_id)}-"
        f"{uuid4().hex[:12]}"
    )


def _warning_types(warnings: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for warning in warnings:
        key = str(warning or "").split(":", 1)[0].strip().lower() or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _audit_paths(job_uid: str, session_id: str, request_id: str) -> LinkedInAuditPaths:
    audit_dir = (
        Path("data")
        / "sessions"
        / _slug(session_id)
        / "linkedin"
        / _slug(request_id)
        / "audit"
    )
    audit_dir.mkdir(parents=True, exist_ok=True)
    return LinkedInAuditPaths(
        job_uid=job_uid,
        audit_dir=audit_dir,
        json_path=audit_dir / f"{job_uid}__linkedin_jobs_snapshot.json",
        schema_path=audit_dir / f"{job_uid}__linkedin_jobs_snapshot.schema.json",
        summary_path=audit_dir / f"{job_uid}__linkedin_jobs_summary.md",
    )


def _render_summary(snapshot: LinkedInAuditSnapshot) -> str:
    lines = [
        "# LinkedIn Jobs Audit Summary",
        "",
        f"- Job UID: `{snapshot.meta.job_uid}`",
        f"- Results: **{snapshot.meta.result_count}**",
        f"- Rejected: **{snapshot.meta.rejected_count}**",
        f"- Warnings: **{len(snapshot.warnings)}**",
        "",
        "## Extraction diagnostics",
        "",
    ]
    for timing in snapshot.timings:
        diagnostics = timing.diagnostics
        lines.extend(
            [
                f"- **{timing.query}**",
                f"  - Selector counts: `{diagnostics.selector_counts}`",
                f"  - Job hrefs: **{diagnostics.href_count}**",
                f"  - Candidates: **{diagnostics.candidate_count}**",
                f"  - Parseable: **{diagnostics.parseable_candidate_count}**",
                f"  - Discard reasons: `{diagnostics.discard_reasons}`",
            ]
        )
    lines.extend(
        [
        "",
        "## Vacancies",
        "",
        ]
    )
    for record in snapshot.vacancies:
        lines.extend(
            [
                f"- **{record.title}** — {record.company_name or 'Empresa no informada'}",
                f"  - {record.location or 'Ubicación no informada'}",
                f"  - Modalidad: {record.workplace_type or 'no informada'}",
                f"  - Publicada: {record.posted_at_text or 'sin fecha verificable'}",
                f"  - Idiomas: {', '.join(record.language_requirements) or 'no informado'}",
                f"  - Experiencia: {'; '.join(record.experience_requirements) or 'no informada'}",
                f"  - Hard skills: {', '.join(record.hard_skills) or 'no informadas'}",
                f"  - Soft skills: {', '.join(record.soft_skills) or 'no informadas'}",
                (
                    "  - Expectativas: "
                    f"{'; '.join(record.candidate_expectations) or 'no informadas'}"
                ),
                (
                    "  - Responsabilidades: "
                    f"{'; '.join(record.responsibilities) or 'no informadas'}"
                ),
                f"  - Extranjeros: {record.foreigner_acceptance}",
                f"  - Visa: {record.visa_status}",
                f"  - Relocation: {record.relocation_support}",
                f"  - URL: {record.canonical_url}",
            ]
        )
    if snapshot.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in snapshot.warnings)
    return "\n".join(lines).strip() + "\n"


def persist_linkedin_audit_snapshot(
    *,
    original_query: str,
    queries: list[str],
    timings: list[LinkedInQueryTiming],
    vacancies: list[LinkedInVacancyRecord],
    rejected: list[LinkedInRejectedRecord],
    warnings: list[str],
) -> LinkedInAuditPaths:
    runtime = get_request_runtime_config()
    session_id = str(runtime.session_id or "local")
    request_id = str(runtime.request_id or "manual")
    job_uid = _build_job_uid(session_id, request_id)
    paths = _audit_paths(job_uid, session_id, request_id)
    snapshot = LinkedInAuditSnapshot(
        meta=LinkedInAuditMeta(
            job_uid=job_uid,
            session_id=session_id,
            request_id=request_id,
            original_query=original_query,
            result_count=len(vacancies),
            rejected_count=len(rejected),
            warning_types=_warning_types(warnings),
        ),
        queries=list(queries),
        timings=list(timings),
        vacancies=list(vacancies),
        rejected=list(rejected),
        warnings=list(warnings),
    )
    payload: dict[str, Any] = snapshot.model_dump(mode="json")
    validated = LinkedInAuditSnapshot.model_validate(payload)
    paths.json_path.write_text(
        json.dumps(validated.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths.schema_path.write_text(
        json.dumps(LinkedInAuditSnapshot.model_json_schema(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths.summary_path.write_text(_render_summary(validated), encoding="utf-8")
    return paths


def load_linkedin_audit_snapshot(path: str | Path) -> LinkedInAuditSnapshot:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return LinkedInAuditSnapshot.model_validate(payload)


__all__ = [
    "LinkedInAuditPaths",
    "load_linkedin_audit_snapshot",
    "persist_linkedin_audit_snapshot",
]
