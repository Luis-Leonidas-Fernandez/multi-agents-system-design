"""Persistencia auditable del vertical LinkedIn Jobs."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from application.services.request_runtime import get_request_runtime_config
from features.web_scraping.application.session_retention import enforce_recent_session_retention
from features.web_scraping.domain.linkedin_models import (
    LinkedInAuditMeta,
    LinkedInAuditSnapshot,
    LinkedInDetailDiagnostic,
    LinkedInQueryTiming,
    LinkedInRejectedRecord,
    LinkedInSearchHydrationDiagnostic,
    LinkedInStaticProbeDiagnostic,
    LinkedInVacancyRecord,
    LinkedInVisualDiagnosticArtifact,
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


def new_linkedin_job_uid() -> str:
    """Allocate the stable UID shared by public audit and private diagnostics."""
    runtime = get_request_runtime_config()
    return _build_job_uid(
        str(runtime.session_id or "local"),
        str(runtime.request_id or "manual"),
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
    enforce_recent_session_retention()
    return LinkedInAuditPaths(
        job_uid=job_uid,
        audit_dir=audit_dir,
        json_path=audit_dir / f"{job_uid}__linkedin_jobs_snapshot.json",
        schema_path=audit_dir / f"{job_uid}__linkedin_jobs_snapshot.schema.json",
        summary_path=audit_dir / f"{job_uid}__linkedin_jobs_summary.md",
    )


def current_linkedin_audit_dir() -> Path:
    runtime = get_request_runtime_config()
    audit_dir = (
        Path("data")
        / "sessions"
        / _slug(str(runtime.session_id or "local"))
        / "linkedin"
        / _slug(str(runtime.request_id or "manual"))
        / "audit"
    )
    audit_dir.mkdir(parents=True, exist_ok=True)
    enforce_recent_session_retention()
    return audit_dir


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
                f"  - Signals: raw=**{diagnostics.raw_signal_count}**, "
                f"card=**{diagnostics.card_signal_count}**, "
                f"href=**{diagnostics.job_href_signal_count}**, "
                f"urn=**{diagnostics.urn_signal_count}**, "
                f"list_item=**{diagnostics.list_item_signal_count}**",
                f"  - Candidates: unique=**{diagnostics.unique_candidate_count}**, "
                f"new=**{diagnostics.new_candidate_count}**, "
                f"duplicate=**{diagnostics.duplicate_candidate_count}**, "
                f"queued=**{timing.retained_count}**",
                f"  - Discovery: mode=`{diagnostics.discovery_mode}`, "
                f"degraded=**{diagnostics.discovery_degraded}**",
                f"  - Row activation: attempts=**{diagnostics.row_activation_count}**, "
                f"success=**{diagnostics.row_activation_success_count}**, "
                f"no_change=**{diagnostics.row_activation_no_change_count}**, "
                f"no_job_id=**{diagnostics.row_activation_no_job_id_count}**, "
                f"duplicate=**{diagnostics.row_activation_duplicate_count}**, "
                f"scrolls=**{diagnostics.row_activation_scroll_count}**, "
                f"resolved=**{diagnostics.row_job_ids_resolved}**, "
                f"stop=`{diagnostics.row_activation_stop_reason}`",
                f"  - Row panel: score=**{diagnostics.selected_row_container_score}**, "
                f"candidates=**{diagnostics.row_candidate_count}**, "
                f"interactive=**{diagnostics.row_interactive_count}**",
                f"  - Parseable: **{diagnostics.parseable_candidate_count}**",
                f"  - Discard reasons: `{diagnostics.discard_reasons}`",
            ]
        )
    lines.extend(["", "## Search hydration diagnostics", ""])
    if snapshot.search_hydration_diagnostics:
        for diagnostic in snapshot.search_hydration_diagnostics:
            lines.append(
                "- "
                f"**{diagnostic.query}** "
                f"#{diagnostic.sequence}: {diagnostic.outcome} "
                f"at {diagnostic.elapsed_ms}ms "
                f"(cards={diagnostic.card_count}, "
                f"hrefs={diagnostic.href_count}, "
                f"raw_signals={diagnostic.raw_signal_count}, "
                f"unique_candidates={diagnostic.unique_candidate_count}, "
                "scroll_progress="
                f"{diagnostic.candidate_count_before_scroll}"
                f"→{diagnostic.candidate_count_after_scroll_1}"
                f"→{diagnostic.candidate_count_after_scroll_2}"
                f"→{diagnostic.candidate_count_after_scroll_3}, "
                f"scroll_container={diagnostic.selected_scroll_container}, "
                f"scroll_top={diagnostic.scroll_top_before}"
                f"→{diagnostic.scroll_top_after}, "
                f"scroll_size={diagnostic.client_height}"
                f"/{diagnostic.scroll_height}, "
                f"row_panel_score={diagnostic.selected_row_container_score}, "
                f"anchors={diagnostic.all_anchor_count}, "
                f"jobs_view={diagnostic.jobs_view_href_count}, "
                f"urn={diagnostic.job_urn_count}, "
                f"data_job_id={diagnostic.data_job_id_count}, "
                f"data_occludable={diagnostic.data_occludable_job_id_count}, "
                f"scrollables={diagnostic.scrollable_container_count}, "
                f"frames={diagnostic.frame_count}, "
                f"row_activation={diagnostic.row_activation_count}/"
                f"{diagnostic.row_activation_success_count}, "
                f"row_no_change={diagnostic.row_activation_no_change_count}, "
                f"row_no_job_id={diagnostic.row_activation_no_job_id_count}, "
                f"row_duplicate={diagnostic.row_activation_duplicate_count}, "
                f"row_candidates={diagnostic.row_candidate_count}, "
                f"row_interactive={diagnostic.row_interactive_count}, "
                f"row_resolved={diagnostic.row_job_ids_resolved}, "
                f"row_stop={diagnostic.row_activation_stop_reason}, "
                f"empty={diagnostic.empty_state_visible}, "
                f"auth_checkpoint={diagnostic.auth_checkpoint_visible})"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Static probe diagnostics", ""])
    if snapshot.static_probe_diagnostics:
        for diagnostic in snapshot.static_probe_diagnostics:
            label = diagnostic.query or diagnostic.job_id or "unknown"
            lines.append(
                "- "
                f"**{diagnostic.kind}** `{label}` "
                f"#{diagnostic.sequence}: {diagnostic.outcome} "
                f"(status={diagnostic.status_code}, "
                f"candidates={diagnostic.candidate_count}, "
                f"accepted={diagnostic.accepted_count}, "
                f"body_source={diagnostic.body_source or 'none'}, "
                f"description_len={diagnostic.description_length}, "
                f"guest_status={diagnostic.guest_status_code}, "
                f"guest_retries={diagnostic.guest_retry_count}, "
                f"identity={diagnostic.identity_consistent})"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Visual diagnostics", ""])
    if snapshot.visual_diagnostics:
        for diagnostic in snapshot.visual_diagnostics:
            lines.append(
                "- "
                f"`{diagnostic.manifest_path}` "
                f"for **{diagnostic.query}** "
                f"(sensitive_local_artifact={diagnostic.sensitive_local_artifact})"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Detail diagnostics", ""])
    if snapshot.detail_diagnostics:
        for diagnostic in snapshot.detail_diagnostics:
            lines.append(
                "- "
                f"Job `{diagnostic.job_id or 'unknown'}`: "
                f"{diagnostic.mode}/{diagnostic.phase}/{diagnostic.outcome} "
                f"(sequence={diagnostic.sequence}, "
                f"description_ready={diagnostic.description_ready}, "
                f"date_ready={diagnostic.date_ready}, "
                f"rejection={diagnostic.rejection})"
            )
    else:
        lines.append("- None")
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
    detail_diagnostics: list[LinkedInDetailDiagnostic] | None = None,
    search_hydration_diagnostics: (
        list[LinkedInSearchHydrationDiagnostic] | None
    ) = None,
    static_probe_diagnostics: list[LinkedInStaticProbeDiagnostic] | None = None,
    visual_diagnostics: list[LinkedInVisualDiagnosticArtifact] | None = None,
    job_uid: str | None = None,
) -> LinkedInAuditPaths:
    runtime = get_request_runtime_config()
    session_id = str(runtime.session_id or "local")
    request_id = str(runtime.request_id or "manual")
    job_uid = job_uid or _build_job_uid(session_id, request_id)
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
        detail_diagnostics=list(detail_diagnostics or []),
        search_hydration_diagnostics=list(search_hydration_diagnostics or []),
        static_probe_diagnostics=list(static_probe_diagnostics or []),
        visual_diagnostics=list(visual_diagnostics or []),
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
    "current_linkedin_audit_dir",
    "new_linkedin_job_uid",
    "load_linkedin_audit_snapshot",
    "persist_linkedin_audit_snapshot",
]
