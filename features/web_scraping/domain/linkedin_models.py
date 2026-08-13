"""Contratos tipados para el vertical read-only de vacantes LinkedIn."""
from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


LINKEDIN_AUDIT_SCHEMA_VERSION = "1.7.0"

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
    title: str = ""
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
    discovery_sources: list[str] = Field(default_factory=list)
    candidate_metadata_incomplete: bool = False
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
    raw_signal_count: int = Field(default=0, ge=0)
    card_signal_count: int = Field(default=0, ge=0)
    job_href_signal_count: int = Field(default=0, ge=0)
    urn_signal_count: int = Field(default=0, ge=0)
    list_item_signal_count: int = Field(default=0, ge=0)
    unique_candidate_count: int = Field(default=0, ge=0)
    new_candidate_count: int = Field(default=0, ge=0)
    duplicate_candidate_count: int = Field(default=0, ge=0)
    discovery_degraded: bool = False
    discovery_mode: str = Field(default="standard", max_length=40)


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


DetailDiagnosticPhase = Literal[
    "start",
    "wait_terminal",
    "fallback",
    "extraction",
    "validation",
]
DetailDiagnosticMode = Literal["panel", "direct", "none"]
DetailDiagnosticOutcome = Literal[
    "started",
    "ready",
    "timeout",
    "stale",
    "extracted",
    "rejected",
    "failed",
]
DetailDiagnosticRejection = Literal[
    "none",
    "missing_description",
    "incomplete_description",
    "missing_date",
    "outside_24_hours",
    "location_mismatch",
    "detail_failure",
]

SearchHydrationOutcome = Literal[
    "polling",
    "results",
    "empty",
    "timeout",
    "auth_checkpoint",
    "blocked",
    "failed",
]
SearchHydrationScrollContainer = Literal[
    "results_panel",
    "main",
    "unknown_scrollable",
    "none",
]

StaticProbeKind = Literal["search_static_probe", "detail_static_probe"]
StaticProbeOutcome = Literal[
    "attempted",
    "ok",
    "no_candidates",
    "missing_description",
    "incomplete_description",
    "missing_date",
    "outside_24_hours",
    "invalid_url",
    "auth_checkpoint",
    "blocked",
    "failed",
]


class LinkedInDetailDiagnostic(BaseModel):
    """Safe, bounded state-only trace for one LinkedIn job detail attempt."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(default="", pattern=r"^\d{0,20}$", max_length=20)
    sequence: int = Field(ge=1, le=40)
    phase: DetailDiagnosticPhase
    mode: DetailDiagnosticMode = "none"
    outcome: DetailDiagnosticOutcome
    include_description: bool = False
    description_ready: bool = False
    date_ready: bool = False
    rejection: DetailDiagnosticRejection = "none"


class LinkedInSearchHydrationDiagnostic(BaseModel):
    """Safe, bounded per-pass trace for LinkedIn search-list hydration."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(default="unknown", min_length=1, max_length=180)
    sequence: int = Field(ge=1, le=40)
    elapsed_ms: int = Field(default=0, ge=0)
    card_count: int = Field(default=0, ge=0, le=10000)
    href_count: int = Field(default=0, ge=0, le=10000)
    body_text_length: int = Field(default=0, ge=0, le=2_000_000)
    main_text_length: int = Field(default=0, ge=0, le=2_000_000)
    all_anchor_count: int = Field(default=0, ge=0, le=10000)
    jobs_href_count: int = Field(default=0, ge=0, le=10000)
    jobs_view_href_count: int = Field(default=0, ge=0, le=10000)
    li_count: int = Field(default=0, ge=0, le=10000)
    article_count: int = Field(default=0, ge=0, le=10000)
    job_urn_count: int = Field(default=0, ge=0, le=10000)
    data_job_id_count: int = Field(default=0, ge=0, le=10000)
    data_occludable_job_id_count: int = Field(default=0, ge=0, le=10000)
    scrollable_container_count: int = Field(default=0, ge=0, le=10000)
    frame_count: int = Field(default=0, ge=0, le=10000)
    raw_signal_count: int = Field(default=0, ge=0, le=10000)
    unique_candidate_count: int = Field(default=0, ge=0, le=10000)
    candidate_count_before_scroll: int = Field(default=0, ge=0, le=10000)
    candidate_count_after_scroll_1: int = Field(default=0, ge=0, le=10000)
    candidate_count_after_scroll_2: int = Field(default=0, ge=0, le=10000)
    candidate_count_after_scroll_3: int = Field(default=0, ge=0, le=10000)
    selected_scroll_container: SearchHydrationScrollContainer = "none"
    scroll_height: int = Field(default=0, ge=0, le=2_000_000)
    client_height: int = Field(default=0, ge=0, le=2_000_000)
    scroll_top_before: int = Field(default=0, ge=0, le=2_000_000)
    scroll_top_after: int = Field(default=0, ge=0, le=2_000_000)
    empty_state_visible: bool = False
    auth_checkpoint_visible: bool = False
    outcome: SearchHydrationOutcome

    @field_validator("query", mode="before")
    @classmethod
    def _sanitize_query(cls, value: object) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        folded = text.casefold()
        if (
            "://" in folded
            or "linkedin.com" in folded
            or ("<" in text and ">" in text)
        ):
            return "redacted"
        return text[:180] or "unknown"


class LinkedInStaticProbeDiagnostic(BaseModel):
    """Safe bounded trace for authenticated static read-only probes."""

    model_config = ConfigDict(extra="forbid")

    kind: StaticProbeKind
    query: str = Field(default="", max_length=180)
    job_id: str = Field(default="", pattern=r"^\d{0,20}$", max_length=20)
    sequence: int = Field(ge=1, le=80)
    status_code: int = Field(default=0, ge=0, le=999)
    candidate_count: int = Field(default=0, ge=0, le=10000)
    accepted_count: int = Field(default=0, ge=0, le=10000)
    outcome: StaticProbeOutcome
    body_source: str = Field(default="", max_length=40)
    description_length: int = Field(default=0, ge=0, le=200000)
    guest_status_code: int = Field(default=0, ge=0, le=999)
    guest_retry_count: int = Field(default=0, ge=0, le=10)
    identity_consistent: bool = True

    @field_validator("query", mode="before")
    @classmethod
    def _sanitize_query(cls, value: object) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        folded = text.casefold()
        if (
            "://" in folded
            or "linkedin.com" in folded
            or ("<" in text and ">" in text)
        ):
            return "redacted"
        return text[:180]


class LinkedInVisualDiagnosticArtifact(BaseModel):
    """Safe reference to local-only visual diagnostics artifacts."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(default="unknown", min_length=1, max_length=180)
    manifest_path: str = Field(min_length=1, max_length=240)
    sensitive_local_artifact: bool = True

    @field_validator("query", mode="before")
    @classmethod
    def _sanitize_query(cls, value: object) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        folded = text.casefold()
        if (
            "://" in folded
            or "linkedin.com" in folded
            or ("<" in text and ">" in text)
        ):
            return "redacted"
        return text[:180] or "unknown"

    @field_validator("manifest_path", mode="before")
    @classmethod
    def _reject_absolute_manifest_path(cls, value: object) -> str:
        text = str(value or "").replace("\\", "/").strip()
        if text.startswith("/") or "://" in text or ".." in text.split("/"):
            return "visual-diagnostics/manifest.json"
        return text[:240]


class LinkedInAuditSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_kind: Literal["linkedin_jobs_audit"] = "linkedin_jobs_audit"
    meta: LinkedInAuditMeta
    queries: list[str] = Field(default_factory=list)
    timings: list[LinkedInQueryTiming] = Field(default_factory=list)
    vacancies: list[LinkedInVacancyRecord] = Field(default_factory=list)
    rejected: list[LinkedInRejectedRecord] = Field(default_factory=list)
    detail_diagnostics: list[LinkedInDetailDiagnostic] = Field(
        default_factory=list,
        max_length=2000,
    )
    search_hydration_diagnostics: list[LinkedInSearchHydrationDiagnostic] = Field(
        default_factory=list,
        max_length=2000,
    )
    static_probe_diagnostics: list[LinkedInStaticProbeDiagnostic] = Field(
        default_factory=list,
        max_length=2000,
    )
    visual_diagnostics: list[LinkedInVisualDiagnosticArtifact] = Field(
        default_factory=list,
        max_length=20,
    )
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
    "LinkedInDetailDiagnostic",
    "LinkedInSearchHydrationDiagnostic",
    "LinkedInStaticProbeDiagnostic",
    "LinkedInVisualDiagnosticArtifact",
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
