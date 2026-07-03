"""Persistencia de snapshots auditables crudos del scraping Moodle."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from application.services.request_runtime import get_request_runtime_config
from features.web_scraping.application.moodle_artifacts import get_moodle_artifact_dir
from features.web_scraping.domain.moodle_audit_models import (
    MoodleAuditAssignmentRecord,
    MoodleAuditMeta,
    MoodleAuditPage,
    MoodleAuditSnapshot,
)


@dataclass(frozen=True)
class MoodleAuditPaths:
    job_uid: str
    audit_dir: Path
    json_path: Path
    schema_path: Path
    summary_path: Path


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-") or "page"


def _build_job_uid(session_id: str, request_id: str) -> str:
    session_slug = _slug(session_id or "local")
    request_slug = _slug(request_id or "manual")
    return f"moodle-job-{session_slug}-{request_slug}-{uuid4().hex[:12]}"


def _audit_paths(job_uid: str) -> MoodleAuditPaths:
    audit_dir = get_moodle_artifact_dir() / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    return MoodleAuditPaths(
        job_uid=job_uid,
        audit_dir=audit_dir,
        json_path=audit_dir / f"{job_uid}__moodle_audit_snapshot.json",
        schema_path=audit_dir / f"{job_uid}__moodle_audit_snapshot.schema.json",
        summary_path=audit_dir / f"{job_uid}__moodle_audit_summary.md",
    )


def _serialize_snapshot(snapshot: MoodleAuditSnapshot) -> dict[str, Any]:
    return snapshot.model_dump(mode="json")


def _persist_page_html(job_uid: str, audit_dir: Path, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    html_dir = audit_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    persisted_pages: list[dict[str, Any]] = []
    for idx, page in enumerate(pages, start=1):
        page_payload = dict(page)
        raw_html = str(page_payload.pop("raw_html", "") or "")
        if raw_html:
            name_seed = page_payload.get("page_kind") or page_payload.get("title") or f"page-{idx}"
            html_path = html_dir / f"{job_uid}__{idx:02d}_{_slug(str(name_seed))}.html"
            html_path.write_text(raw_html, encoding="utf-8")
            page_payload["html_snapshot_path"] = str(html_path)
        persisted_pages.append(page_payload)
    return persisted_pages


def _dedupe_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    index_by_key: dict[str, int] = {}
    for page in pages:
        payload = dict(page)
        dedupe_key = str(payload.get("final_url") or payload.get("url") or "").strip()
        if not dedupe_key:
            dedupe_key = f"{payload.get('page_kind','page')}::{payload.get('title','')}"
        payload["dedupe_key"] = dedupe_key
        if dedupe_key not in index_by_key:
            index_by_key[dedupe_key] = len(deduped)
            deduped.append(payload)
            continue

        existing = deduped[index_by_key[dedupe_key]]
        existing_notes = existing.setdefault("notes", [])
        if isinstance(existing_notes, list):
            existing_notes.append(f"deduped_duplicate:{payload.get('page_kind','page')}")

        for field in ("links", "attachments", "videos", "images", "breadcrumbs"):
            existing_values = existing.get(field)
            new_values = payload.get(field)
            if not isinstance(existing_values, list) or not isinstance(new_values, list):
                continue
            seen = {json.dumps(item, ensure_ascii=False, sort_keys=True) for item in existing_values if isinstance(item, (dict, list, str, int, float, bool, type(None)))}
            for item in new_values:
                try:
                    key = json.dumps(item, ensure_ascii=False, sort_keys=True)
                except TypeError:
                    key = str(item)
                if key in seen:
                    continue
                seen.add(key)
                existing_values.append(item)

        if not existing.get("description") and payload.get("description"):
            existing["description"] = payload["description"]
        if not existing.get("subtitle") and payload.get("subtitle"):
            existing["subtitle"] = payload["subtitle"]
        if not existing.get("external_resource") and payload.get("external_resource"):
            existing["external_resource"] = payload["external_resource"]
        if not existing.get("submission_state") and payload.get("submission_state"):
            existing["submission_state"] = payload["submission_state"]
        existing["extracted_items_count"] = max(
            int(existing.get("extracted_items_count") or 0),
            int(payload.get("extracted_items_count") or 0),
        )
    return deduped


def _score_page_confidence(page: dict[str, Any]) -> float:
    score = 0.0
    if page.get("title"):
        score += 0.2
    if page.get("description"):
        score += 0.15
    if page.get("final_url"):
        score += 0.1
    if page.get("breadcrumbs"):
        score += 0.1
    if page.get("attachments"):
        score += 0.15
    if page.get("videos"):
        score += 0.1
    if page.get("links"):
        score += 0.1
    if page.get("external_resource"):
        score += 0.1
    if page.get("submission_state"):
        score += 0.1
    return round(min(score, 1.0), 2)


def _warning_types(warnings: list[str]) -> dict[str, int]:
    buckets: dict[str, int] = {}
    for warning in warnings:
        key = str(warning or "").split(":", 1)[0].strip().lower() or "unknown"
        buckets[key] = buckets.get(key, 0) + 1
    return buckets


def _resource_type_counts(pages: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for page in pages:
        key = str(page.get("resource_type") or "unknown").strip() or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _host(url: str) -> str:
    try:
        return (urlparse((url or "").strip()).hostname or "").lower().removeprefix("www.")
    except Exception:
        return ""


def _render_summary(snapshot: MoodleAuditSnapshot) -> str:
    stats = snapshot.meta.stats
    lines = [
        "# Moodle Audit Summary",
        "",
        f"- Job UID: `{snapshot.meta.job_uid}`",
        f"- Session: `{snapshot.meta.session_id}`",
        f"- Request: `{snapshot.meta.request_id}`",
        f"- Base URL: `{snapshot.meta.base_url}`",
        f"- Assignments: **{len(snapshot.assignments)}**",
        f"- Pages: **{len(snapshot.pages)}**",
        f"- Warnings: **{len(snapshot.warnings)}**",
        "",
    ]
    if stats:
        lines.extend(
            [
                "## Stats",
                "",
                f"- Visitas crudas: **{stats.get('visited_count_raw', 0)}**",
                f"- Páginas retenidas: **{stats.get('retained_page_count', len(snapshot.pages))}**",
                f"- Redirects externos: **{stats.get('external_redirect_count', 0)}**",
                f"- Documentos download-backed: **{stats.get('download_document_count', 0)}**",
                f"- Assignment-like: **{stats.get('assignment_like_count', len(snapshot.assignments))}**",
                "",
            ]
        )
    if snapshot.meta.warning_types:
        lines.extend(["## Warning types", ""])
        for warning_type, count in sorted(snapshot.meta.warning_types.items()):
            lines.append(f"- {warning_type}: **{count}**")
        lines.append("")
    if snapshot.meta.resource_type_counts:
        lines.extend(["## Resource types", ""])
        for resource_type, count in sorted(snapshot.meta.resource_type_counts.items()):
            lines.append(f"- {resource_type}: **{count}**")
        lines.append("")
    lines.extend([
        "## Pages",
        "",
    ])
    for idx, page in enumerate(snapshot.pages, start=1):
        lines.append(
            f"{idx}. **{page.page_kind}** · `{page.resource_type or 'unknown'}` · score `{page.confidence_score:.2f}`"
        )
        lines.append(f"   - Title: {page.title or '(sin título)'}")
        lines.append(f"   - URL: {page.final_url or page.url}")
        lines.append(
            f"   - Artifacts: links={len(page.links)} attachments={len(page.attachments)} videos={len(page.videos)} images={len(page.images)}"
        )
        if page.external_resource:
            lines.append(
                "   - External: "
                f"{page.external_resource.provider} "
                f"id={page.external_resource.resource_id or '-'} "
                f"login={page.external_resource.requires_login}"
            )
            if page.external_resource.slide_count is not None:
                lines.append(f"   - Slides: {page.external_resource.slide_count}")
        if page.submission_state:
            lines.append(
                "   - Submission: "
                f"can_submit={page.submission_state.can_submit} "
                f"submitted={page.submission_state.is_submitted} "
                f"graded={page.submission_state.is_graded} "
                f"locked={page.submission_state.is_locked}"
            )
        lines.append("")
    if snapshot.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in snapshot.warnings)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def persist_moodle_audit_snapshot(
    raw_assignments: list[dict[str, str]],
    *,
    base_url: str = "",
    pages: list[dict[str, Any]] | None = None,
    warnings: list[str] | None = None,
    stats: dict[str, int] | None = None,
    resource_type_counts: dict[str, int] | None = None,
) -> MoodleAuditPaths:
    runtime = get_request_runtime_config()
    job_uid = _build_job_uid(str(runtime.session_id or ""), str(runtime.request_id or ""))
    paths = _audit_paths(job_uid)
    assignments = [MoodleAuditAssignmentRecord.model_validate(item) for item in raw_assignments]
    deduped_pages = _dedupe_pages(list(pages or []))
    raw_warning_list = list(warnings or [])
    for page in deduped_pages:
        page["confidence_score"] = _score_page_confidence(page)
    raw_pages = _persist_page_html(job_uid, paths.audit_dir, deduped_pages)
    validated_pages = [MoodleAuditPage.model_validate(page) for page in raw_pages]
    computed_resource_type_counts = _resource_type_counts(deduped_pages)
    computed_stats = {
        "visited_count_raw": max(len(deduped_pages), int(next((page.get("visited_count_raw") for page in deduped_pages if page.get("visited_count_raw")), 0) or 0)),
        "retained_page_count": len(deduped_pages),
        "external_redirect_count": sum(
            1
            for page in deduped_pages
            if str(page.get("page_kind") or "") == "linked_resource"
            and _host(str(page.get("final_url") or page.get("url") or ""))
            and _host(str(page.get("final_url") or page.get("url") or "")) != _host(str(page.get("url") or ""))
        ),
        "download_document_count": sum(
            1 for page in deduped_pages if str(page.get("resource_type") or "") == "document" and bool(page.get("attachments"))
        ),
        "assignment_like_count": len(assignments),
    }
    snapshot = MoodleAuditSnapshot(
        meta=MoodleAuditMeta(
            job_uid=job_uid,
            session_id=str(runtime.session_id or ""),
            request_id=str(runtime.request_id or ""),
            base_url=base_url,
            record_count=len(assignments),
            stats=dict(stats or computed_stats),
            resource_type_counts=dict(resource_type_counts or computed_resource_type_counts),
            warning_types=_warning_types(raw_warning_list),
        ),
        pages=validated_pages
        or [
            MoodleAuditPage(
                page_kind="dashboard",
                url=f"{base_url.rstrip('/')}/my/" if base_url else "/my/",
                extracted_items_count=len(assignments),
            ),
            MoodleAuditPage(
                page_kind="calendar_upcoming",
                url=(
                    f"{base_url.rstrip('/')}/calendar/view.php?view=upcoming&lookahead=365"
                    if base_url
                    else "/calendar/view.php?view=upcoming&lookahead=365"
                ),
                extracted_items_count=len(assignments),
            ),
        ],
        assignments=assignments,
        warnings=raw_warning_list,
    )
    payload = _serialize_snapshot(snapshot)
    validated_payload = MoodleAuditSnapshot.model_validate(payload)
    paths.json_path.write_text(
        json.dumps(validated_payload.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths.schema_path.write_text(
        json.dumps(MoodleAuditSnapshot.model_json_schema(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths.summary_path.write_text(_render_summary(validated_payload), encoding="utf-8")
    return paths


def load_moodle_audit_snapshot(json_path: str | Path) -> MoodleAuditSnapshot:
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return MoodleAuditSnapshot.model_validate(payload)


__all__ = [
    "MoodleAuditPaths",
    "load_moodle_audit_snapshot",
    "persist_moodle_audit_snapshot",
]
