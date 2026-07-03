"""Schemas auditables para snapshots crudos de scraping Moodle."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


_AUDIT_SCHEMA_VERSION = "1.1.0"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MoodleAuditPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_kind: str = Field(min_length=1)
    url: str = Field(min_length=1)
    final_url: str = ""
    title: str = ""
    subtitle: str = ""
    description: str = ""
    resource_type: str = ""
    parent_url: str = ""
    source_link_label: str = ""
    crawl_depth: int = Field(default=0, ge=0)
    visit_order: int = Field(default=0, ge=0)
    breadcrumbs: list[str] = Field(default_factory=list)
    html_snapshot_path: str = ""
    text_excerpt: str = ""
    extracted_items_count: int = Field(default=0, ge=0)
    dedupe_key: str = ""
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    links: list["MoodleAuditLink"] = Field(default_factory=list)
    attachments: list["MoodleAuditAttachment"] = Field(default_factory=list)
    videos: list["MoodleAuditVideo"] = Field(default_factory=list)
    images: list["MoodleAuditImage"] = Field(default_factory=list)
    external_resource: Optional["MoodleAuditExternalResource"] = None
    submission_state: Optional["MoodleAuditSubmissionState"] = None
    notes: list[str] = Field(default_factory=list)


class MoodleAuditAssignmentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = ""
    date: str = ""
    course: str = ""
    url: str = ""
    status: str = ""
    source_stage: str = "listing"


class MoodleAuditLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = ""
    url: str = ""
    final_url: str = ""
    redirect_target: str = ""
    redirect_chain: list[str] = Field(default_factory=list)
    resource_type: str = ""
    is_submission_target: bool = False


class MoodleAuditAttachment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = ""
    filename: str = ""
    url: str = ""
    final_url: str = ""
    redirect_target: str = ""
    redirect_chain: list[str] = Field(default_factory=list)
    mime_hint: str = ""
    kind: str = "file"
    content_type: str = ""
    content_length: Optional[int] = Field(default=None, ge=0)
    content_disposition: str = ""
    status_code: Optional[int] = Field(default=None, ge=100, le=599)
    is_download: bool = False


class MoodleAuditVideo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = ""
    embed_url: str = ""
    watch_url: str = ""
    provider: str = ""
    preview_url: str = ""


class MoodleAuditImage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = ""
    url: str = ""
    kind: str = "image"


class MoodleAuditExternalResource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = ""
    resource_id: str = ""
    resource_type: str = ""
    canonical_url: str = ""
    htmlpresent_url: str = ""
    preview_url: str = ""
    access_url: str = ""
    download_url: str = ""
    requires_login: bool = False
    slide_count: Optional[int] = Field(default=None, ge=0)
    content_blocks: list[str] = Field(default_factory=list)


class MoodleAuditSubmissionFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = ""
    filename: str = ""
    url: str = ""
    final_url: str = ""
    redirect_target: str = ""
    redirect_chain: list[str] = Field(default_factory=list)
    mime_hint: str = ""
    kind: str = "file"
    content_type: str = ""
    content_length: Optional[int] = Field(default=None, ge=0)
    content_disposition: str = ""
    status_code: Optional[int] = Field(default=None, ge=100, le=599)
    is_download: bool = False


class MoodleAuditSubmissionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission_status: str = ""
    grading_status: str = ""
    due_date_text: str = ""
    time_remaining_text: str = ""
    last_modified_text: str = ""
    attempt_text: str = ""
    instructions: list[str] = Field(default_factory=list)
    available_actions: list[str] = Field(default_factory=list)
    submitted_files: list[MoodleAuditSubmissionFile] = Field(default_factory=list)
    raw_fields: dict[str, str] = Field(default_factory=dict)
    field_confidence: dict[str, float] = Field(default_factory=dict)
    can_submit: bool = False
    is_submitted: bool = False
    is_graded: bool = False
    is_locked: bool = False


class MoodleAuditMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = _AUDIT_SCHEMA_VERSION
    job_uid: str = ""
    generated_at: datetime = Field(default_factory=_utc_now)
    session_id: str = ""
    request_id: str = ""
    base_url: str = ""
    extractor: str = "extract_moodle_assignments"
    record_count: int = Field(default=0, ge=0)
    stats: dict[str, int] = Field(default_factory=dict)
    resource_type_counts: dict[str, int] = Field(default_factory=dict)
    warning_types: dict[str, int] = Field(default_factory=dict)


class MoodleAuditSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_kind: str = "moodle_audit"
    schema_version: str = _AUDIT_SCHEMA_VERSION
    meta: MoodleAuditMeta
    pages: list[MoodleAuditPage] = Field(default_factory=list)
    assignments: list[MoodleAuditAssignmentRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


__all__ = [
    "MoodleAuditAttachment",
    "MoodleAuditAssignmentRecord",
    "MoodleAuditExternalResource",
    "MoodleAuditImage",
    "MoodleAuditLink",
    "MoodleAuditMeta",
    "MoodleAuditPage",
    "MoodleAuditSubmissionFile",
    "MoodleAuditSubmissionState",
    "MoodleAuditSnapshot",
    "MoodleAuditVideo",
]
