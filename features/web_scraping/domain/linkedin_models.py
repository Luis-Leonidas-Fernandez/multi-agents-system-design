"""Contratos tipados para el vertical read-only de vacantes LinkedIn."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


LINKEDIN_AUDIT_SCHEMA_VERSION = "1.3.0"

SkillItem = Annotated[str, StringConstraints(min_length=1, max_length=64)]
StructuredMetadataItem = Annotated[
    str,
    StringConstraints(min_length=1, max_length=180),
]

ForeignerAcceptance = Literal["yes", "no", "unknown", "ambiguous"]
VisaStatus = Literal[
    "sponsorship",
    "no_sponsorship",
    "unknown",
    "ambiguous",
]
RelocationSupport = Literal["yes", "no", "unknown", "ambiguous"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class LinkedInJobsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    location: str = ""
    locations: list[str] = Field(default_factory=list)
    max_results: int = Field(default=50, ge=1, le=50)
    include_description: bool = True


class LinkedInVacancyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    linkedin_job_id: str = ""
    title: str = Field(min_length=1)
    company_name: str = ""
    location: str = ""
    workplace_type: str = ""
    posted_at_text: str = ""
    published_at: datetime | None = None
    freshness_confidence: Literal["high", "medium", "low"] = "low"
    is_within_24_hours: bool = False
    canonical_url: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    description_excerpt: str = ""
    description_full_text: str = ""
    language_requirements: list[str] = Field(default_factory=list)
    experience_requirements: list[str] = Field(default_factory=list)
    hard_skills: list[SkillItem] = Field(default_factory=list, max_length=40)
    soft_skills: list[SkillItem] = Field(default_factory=list, max_length=20)
    candidate_expectations: list[StructuredMetadataItem] = Field(
        default_factory=list,
        max_length=6,
    )
    responsibilities: list[StructuredMetadataItem] = Field(
        default_factory=list,
        max_length=6,
    )
    foreigner_acceptance: ForeignerAcceptance = "unknown"
    visa_status: VisaStatus = "unknown"
    relocation_support: RelocationSupport = "unknown"
    matched_terms: list[str] = Field(default_factory=list)
    captured_at: datetime = Field(default_factory=utc_now)


class LinkedInRejectedRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_url: str = ""
    title: str = ""
    reason: str = Field(min_length=1)


class LinkedInParseDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selector_counts: dict[str, int] = Field(default_factory=dict)
    href_count: int = Field(default=0, ge=0)
    candidate_count: int = Field(default=0, ge=0)
    parseable_candidate_count: int = Field(default=0, ge=0)
    discard_reasons: dict[str, int] = Field(default_factory=dict)


class LinkedInQueryTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    started_at: datetime
    completed_at: datetime
    elapsed_ms: int = Field(ge=0)
    discovered_count: int = Field(default=0, ge=0)
    retained_count: int = Field(default=0, ge=0)
    error: str = ""
    diagnostics: LinkedInParseDiagnostics = Field(
        default_factory=LinkedInParseDiagnostics
    )


class LinkedInAuditMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = LINKEDIN_AUDIT_SCHEMA_VERSION
    job_uid: str = Field(min_length=1)
    session_id: str = ""
    request_id: str = ""
    original_query: str = Field(min_length=1)
    result_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    warning_types: dict[str, int] = Field(default_factory=dict)


class LinkedInAuditSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_kind: Literal["linkedin_jobs_audit"] = "linkedin_jobs_audit"
    meta: LinkedInAuditMeta
    queries: list[str] = Field(default_factory=list)
    timings: list[LinkedInQueryTiming] = Field(default_factory=list)
    vacancies: list[LinkedInVacancyRecord] = Field(default_factory=list)
    rejected: list[LinkedInRejectedRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LinkedInJobsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "ok",
        "extraction_incomplete",
        "validation_error",
        "auth_required",
        "blocked",
        "error",
    ]
    job_uid: str = ""
    records: list[LinkedInVacancyRecord] = Field(default_factory=list)
    rejected: list[LinkedInRejectedRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    timings: list[LinkedInQueryTiming] = Field(default_factory=list)
    audit_json_path: str = ""
    audit_schema_path: str = ""
    audit_summary_path: str = ""
    user_summary: str = ""


__all__ = [
    "LINKEDIN_AUDIT_SCHEMA_VERSION",
    "ForeignerAcceptance",
    "LinkedInAuditMeta",
    "LinkedInAuditSnapshot",
    "LinkedInJobsRequest",
    "LinkedInJobsResult",
    "LinkedInParseDiagnostics",
    "LinkedInQueryTiming",
    "LinkedInRejectedRecord",
    "LinkedInVacancyRecord",
    "RelocationSupport",
    "VisaStatus",
]
