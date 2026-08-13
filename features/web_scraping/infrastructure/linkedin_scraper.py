"""Extractor secuencial y read-only de vacantes LinkedIn autenticadas."""
from __future__ import annotations

from datetime import datetime
import time
from typing import Callable

from features.web_scraping.domain.linkedin_models import (
    LinkedInJobsRequest,
    LinkedInParseDiagnostics,
    LinkedInQueryTiming,
    LinkedInRejectedRecord,
    LinkedInVacancyRecord,
)
from features.web_scraping.infrastructure.linkedin_metadata import (
    _extract_structured_sections,
    _infer_experience_requirements,
    _infer_foreigner_acceptance,
    _infer_hard_skills,
    _infer_language_requirements,
    _infer_relocation_support,
    _infer_soft_skills,
    _infer_visa_status,
    _sanitize_description,
)
from features.web_scraping.infrastructure.linkedin_detail_diagnostics import (
    LinkedInDetailDiagnosticsCollector,
    get_active_detail_diagnostics,
)
from features.web_scraping.infrastructure.linkedin_detail_panel import (
    _AuthenticatedDetailProbeResult,
    _enrich_job_detail as _detail_panel_enrich_job_detail,
    _enrich_job_detail_via_panel as _detail_panel_enrich_job_detail_via_panel,
    _extract_detail_posted_date,
    _extract_job_detail_from_current_page as _detail_panel_extract_job_detail_from_current_page,
    _first_visible_raw_text,
    _first_visible_text,
    _is_incomplete_detail_body,
    _normalize_detail_location,
    _normalize_transient_description,
    _normalize_workplace_type,
    _probe_linkedin_detail_with_authenticated_request as _detail_panel_probe_detail_with_authenticated_request,
    _respect_detail_click_cadence,
    _safe_job_card_link,
    _wait_for_detail_hydration,
    _wait_for_detail_panel_hydration,
)
from features.web_scraping.infrastructure.linkedin_navigation import (
    LinkedInAuthRequiredError,
    LinkedInBlockedError,
    LinkedInDetailPanelError,
    _error_category,
    _is_http_response_code_failure,
    _is_page_recoverable_error,
    _safe_error_label,
    _safe_page_pause,
    _validate_authenticated_page,
)
from features.web_scraping.infrastructure.linkedin_parser import (
    MATCH_TERMS,
    _clean_posted_at_text,
    _normalize_repeated_title,
    _parse_linkedin_jobs_html_with_diagnostics,
    parse_linkedin_jobs_html,
    parse_linkedin_relative_time,
)
from features.web_scraping.infrastructure.authenticated_browser import (
    AuthenticatedBrowserLaunchConfig,
    configured_linkedin_headless,
    open_persistent_authenticated_context,
)
from features.web_scraping.infrastructure.linkedin_session_store import LinkedInSessionStore
from features.web_scraping.infrastructure.linkedin_url_policy import (
    canonicalize_linkedin_job_url,
    is_linkedin_auth_checkpoint,
    linkedin_job_id_from_url,
    validate_linkedin_jobs_url,
)


from features.web_scraping.infrastructure.linkedin_query_navigation import (
    CONSOLIDATED_QUERY_PLANS,
    LinkedInLocationResolution,
    _AuthenticatedSearchProbeResult,
    _DETAIL_NETWORK_CIRCUIT_THRESHOLD,
    _EMPTY_RESULTS_SELECTORS,
    _HARD_MAX_QUERIES_PER_LOCATION,
    _HARD_MAX_TOTAL_QUERY_ATTEMPTS,
    _QUERY_BACKOFF_BASE_MS,
    _QUERY_BACKOFF_MAX_MS,
    _QUERY_NETWORK_CIRCUIT_THRESHOLD,
    _QUERY_PAGE_RECOVERY_THRESHOLD,
    _QUERY_PROBE_TIMEOUT_MS,
    _SEARCH_HYDRATION_MAX_MS,
    _SEARCH_RESULT_SIGNAL_SELECTORS,
    _SearchNavigationState,
    _classify_probe_status,
    _query_location,
    build_linkedin_search_queries,
    resolve_linkedin_location,
)

from features.web_scraping.infrastructure.linkedin_auth_diagnostics import (
    _safe_auth_diagnostic,
)
from features.web_scraping.infrastructure.linkedin_dedupe import (
    _dedupe_linkedin_vacancies_semantically,
    _record_key,
    dedupe_linkedin_vacancies,
)
from features.web_scraping.infrastructure.linkedin_runtime_config import (
    configured_linkedin_detail_budget,
    configured_linkedin_detail_click_interval_ms,
    configured_linkedin_direct_detail_fallback,
    configured_linkedin_max_queries_per_location,
    configured_linkedin_max_results,
    configured_linkedin_query_interval_ms,
)


def _extract_job_detail_from_current_page(
    page,
    record: LinkedInVacancyRecord,
    *,
    include_description: bool,
    now: datetime | None = None,
    diagnostics: LinkedInDetailDiagnosticsCollector | None = None,
    diagnostic_mode: str = "none",
) -> LinkedInVacancyRecord:
    return _detail_panel_extract_job_detail_from_current_page(
        page,
        record,
        include_description=include_description,
        now=now,
        diagnostics=diagnostics,
        diagnostic_mode=diagnostic_mode,
    )


def _enrich_job_detail(
    page,
    record: LinkedInVacancyRecord,
    *,
    include_description: bool,
    now: datetime | None = None,
    diagnostics: LinkedInDetailDiagnosticsCollector | None = None,
) -> LinkedInVacancyRecord:
    diagnostics = diagnostics or get_active_detail_diagnostics()
    if diagnostics:
        diagnostics.record(
            record,
            phase="start",
            mode="direct",
            outcome="started",
            include_description=include_description,
            date_ready=record.published_at is not None,
        )
    page.goto(record.canonical_url, wait_until="domcontentloaded", timeout=30000)
    _validate_authenticated_page(page)
    hydration_state = _wait_for_detail_hydration(
        page,
        require_date=record.published_at is None,
        require_description=include_description,
    )
    if diagnostics:
        diagnostics.record(
            record,
            phase="wait_terminal",
            mode="direct",
            outcome=hydration_state,
            include_description=include_description,
            description_ready=include_description and hydration_state == "ready",
            date_ready=record.published_at is not None or hydration_state == "ready",
        )
    return _extract_job_detail_from_current_page(
        page,
        record,
        include_description=include_description,
        now=now,
        diagnostics=diagnostics,
        diagnostic_mode="direct",
    )


def _enrich_job_detail_via_panel(
    page,
    record: LinkedInVacancyRecord,
    *,
    card_link,
    include_description: bool,
    now: datetime | None = None,
    diagnostics: LinkedInDetailDiagnosticsCollector | None = None,
) -> LinkedInVacancyRecord:
    return _detail_panel_enrich_job_detail_via_panel(
        page,
        record,
        card_link=card_link,
        include_description=include_description,
        now=now,
        diagnostics=diagnostics or get_active_detail_diagnostics(),
    )































def _probe_linkedin_search_with_authenticated_request(
    session,
    *,
    source_url: str,
    now: datetime | None = None,
    allow_standalone_fallback: bool = False,
) -> _AuthenticatedSearchProbeResult:
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        _probe_linkedin_search_with_authenticated_request as _query_navigation_probe_search,
    )

    return _query_navigation_probe_search(
        session,
        source_url=source_url,
        now=now,
        allow_standalone_fallback=allow_standalone_fallback,
        parse_jobs_html_with_diagnostics=_parse_linkedin_jobs_html_with_diagnostics,
    )


def _probe_linkedin_detail_with_authenticated_request(
    session,
    record: LinkedInVacancyRecord,
    *,
    include_description: bool,
    now: datetime | None = None,
) -> _AuthenticatedDetailProbeResult:
    return _detail_panel_probe_detail_with_authenticated_request(
        session,
        record,
        include_description=include_description,
        now=now,
    )


def _respect_query_cadence(
    page,
    *,
    last_successful_query_at: float | None,
    interval_ms: int,
    now_fn=None,
) -> None:
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        _respect_query_cadence as _query_navigation_respect_query_cadence,
    )

    _query_navigation_respect_query_cadence(
        page,
        last_successful_query_at=last_successful_query_at,
        interval_ms=interval_ms,
        now_fn=now_fn or time.monotonic,
    )


def _ensure_search_source(
    page,
    *,
    source_url: str,
    navigation_state: _SearchNavigationState,
    interval_ms: int,
    now_fn=None,
) -> bool:
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        _ensure_search_source as _query_navigation_ensure_search_source,
    )

    return _query_navigation_ensure_search_source(
        page,
        source_url=source_url,
        navigation_state=navigation_state,
        interval_ms=interval_ms,
        now_fn=now_fn,
        wait_for_search_results_hydration=_wait_for_search_results_hydration,
        validate_authenticated_page=_validate_authenticated_page,
    )


def _ensure_search_source_with_single_retry(
    session,
    page,
    *,
    source_url: str,
    navigation_state: _SearchNavigationState,
    interval_ms: int,
    retry_allowed: bool,
    warning_scope: str,
    warnings: list[str],
    reserve_retry: Callable[[], None] | None = None,
    now_fn=None,
) -> tuple[object, bool, bool]:
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        _ensure_search_source_with_single_retry as _query_navigation_ensure_search_source_with_single_retry,
    )

    return _query_navigation_ensure_search_source_with_single_retry(
        session,
        page,
        source_url=source_url,
        navigation_state=navigation_state,
        interval_ms=interval_ms,
        retry_allowed=retry_allowed,
        warning_scope=warning_scope,
        warnings=warnings,
        reserve_retry=reserve_retry,
        now_fn=now_fn,
        ensure_search_source=_ensure_search_source,
    )


def _session_page_is_alive(session) -> bool:
    checker = getattr(session, "page_is_alive", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return False
    page = getattr(session, "page", None)
    if page is None:
        return False
    try:
        is_closed = getattr(page, "is_closed", None)
        return not (callable(is_closed) and is_closed() is True)
    except Exception:
        return False






def _wait_for_search_results_hydration(
    page,
    *,
    max_wait_ms: int = _SEARCH_HYDRATION_MAX_MS,
    query: str = "unknown",
    diagnostics=None,
) -> str:
    from features.web_scraping.infrastructure.linkedin_query_navigation import (
        _wait_for_search_results_hydration as _query_navigation_wait_for_search_results_hydration,
    )

    return _query_navigation_wait_for_search_results_hydration(
        page,
        max_wait_ms=max_wait_ms,
        query=query,
        diagnostics=diagnostics,
    )


def _record_key(record: LinkedInVacancyRecord) -> str:
    return record.linkedin_job_id or record.canonical_url



def _needs_detail_enrichment(record: LinkedInVacancyRecord) -> bool:
    """Return True when a listing record lacks useful structured metadata."""
    missing_core = not all(
        (
            record.company_name,
            record.location,
            record.workplace_type,
        )
    )
    missing_profile = not any(
        (
            record.hard_skills,
            record.soft_skills,
            record.language_requirements,
            record.experience_requirements,
            record.candidate_expectations,
            record.responsibilities,
        )
    )
    missing_mobility = all(
        value == "unknown"
        for value in (
            record.foreigner_acceptance,
            record.visa_status,
            record.relocation_support,
        )
    )
    return (
        record.published_at is None
        or missing_core
        or missing_profile
        or missing_mobility
    )


def _source_ordered_candidates_for_detail(
    candidates: list[LinkedInVacancyRecord],
    *,
    source_order: list[str],
) -> list[LinkedInVacancyRecord]:
    """Keep detail enrichment on one search source before switching context."""
    ordered_sources: list[str] = []
    for source_url in source_order:
        if source_url not in ordered_sources:
            ordered_sources.append(source_url)

    groups: dict[str, list[LinkedInVacancyRecord]] = {}
    for candidate in candidates:
        source_url = candidate.source_url or ""
        if source_url not in ordered_sources:
            ordered_sources.append(source_url)
        groups.setdefault(source_url, []).append(candidate)

    ordered: list[LinkedInVacancyRecord] = []
    for source_url in ordered_sources:
        ordered.extend(groups.get(source_url, []))
    return ordered


def _round_robin_candidates_by_location(
    candidates: list[LinkedInVacancyRecord],
    *,
    candidate_locations: dict[str, str],
    location_order: list[str],
) -> list[LinkedInVacancyRecord]:
    """Preserve per-country order while alternating countries fairly."""
    groups: dict[str, list[LinkedInVacancyRecord]] = {}
    ordered_locations: list[str] = []
    for location in location_order:
        if location not in ordered_locations:
            ordered_locations.append(location)
    for candidate in candidates:
        location = candidate_locations.get(_record_key(candidate), "")
        if location not in ordered_locations:
            ordered_locations.append(location)
        groups.setdefault(location, []).append(candidate)

    ordered: list[LinkedInVacancyRecord] = []
    index = 0
    while True:
        added = False
        for location in ordered_locations:
            group = groups.get(location, [])
            if index < len(group):
                ordered.append(group[index])
                added = True
        if not added:
            return ordered
        index += 1



def _scrape_linkedin_jobs_impl(
    request: LinkedInJobsRequest,
    *,
    session_store: LinkedInSessionStore | None = None,
) -> tuple[
    list[LinkedInVacancyRecord],
    list[LinkedInRejectedRecord],
    list[LinkedInQueryTiming],
    list[str],
    list[str],
]:
    from features.web_scraping.infrastructure.linkedin_jobs_pipeline import (
        LinkedInJobsPipelineDeps,
        scrape_linkedin_jobs_impl,
    )

    return scrape_linkedin_jobs_impl(
        request,
        session_store=session_store,
        deps=LinkedInJobsPipelineDeps.from_module(
            __import__(__name__, fromlist=["*"])
        ),
    )


def scrape_linkedin_jobs(
    request: LinkedInJobsRequest,
    *,
    session_store: LinkedInSessionStore | None = None,
) -> tuple[
    list[LinkedInVacancyRecord],
    list[LinkedInRejectedRecord],
    list[LinkedInQueryTiming],
    list[str],
    list[str],
]:
    """Public compatibility wrapper for the LinkedIn jobs pipeline."""

    from features.web_scraping.infrastructure.linkedin_jobs_pipeline import (
        scrape_linkedin_jobs as _pipeline_scrape_linkedin_jobs,
    )

    return _pipeline_scrape_linkedin_jobs(
        request,
        session_store=session_store,
    )



__all__ = [
    "LinkedInAuthRequiredError",
    "LinkedInBlockedError",
    "build_linkedin_search_queries",
    "configured_linkedin_detail_click_interval_ms",
    "configured_linkedin_detail_budget",
    "configured_linkedin_direct_detail_fallback",
    "configured_linkedin_max_results",
    "configured_linkedin_max_queries_per_location",
    "configured_linkedin_query_interval_ms",
    "dedupe_linkedin_vacancies",
    "parse_linkedin_jobs_html",
    "parse_linkedin_relative_time",
    "scrape_linkedin_jobs",
]
