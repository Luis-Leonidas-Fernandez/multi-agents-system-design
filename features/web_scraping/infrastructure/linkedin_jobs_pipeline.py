"""LinkedIn jobs scraping orchestration pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from features.web_scraping.domain.linkedin_models import (
        LinkedInJobsRequest,
        LinkedInQueryTiming,
        LinkedInRejectedRecord,
        LinkedInVacancyRecord,
    )
    from features.web_scraping.infrastructure.linkedin_session_store import (
        LinkedInSessionStore,
    )


@dataclass(frozen=True)
class LinkedInJobsPipelineDeps:
    session_store_cls: type[Any]
    launch_config_cls: type[Any]
    configured_headless: Callable[..., bool]
    open_authenticated_context: Callable[..., Any]
    validate_authenticated_page: Callable[..., None]
    configured_detail_budget: Callable[[], int]
    configured_detail_click_interval_ms: Callable[[], int]
    configured_direct_detail_fallback: Callable[[], bool]
    configured_max_queries_per_location: Callable[[], int]
    configured_query_interval_ms: Callable[[], int]
    build_search_queries: Callable[..., list[tuple[str, str]]]
    query_location: Callable[[str], str]
    hard_max_total_query_attempts: int
    search_navigation_state_cls: type[Any]
    parse_diagnostics_cls: type[Any]
    query_timing_cls: type[Any]
    rejected_record_cls: type[Any]
    auth_required_error_cls: type[Exception]
    blocked_error_cls: type[Exception]
    detail_panel_error_cls: type[Exception]
    respect_query_cadence: Callable[..., None]
    wait_for_search_results_hydration: Callable[..., str]
    parse_jobs_html_with_diagnostics: Callable[..., Any]
    record_key: Callable[[Any], str]
    safe_error_label: Callable[[Exception], str]
    is_http_response_code_failure: Callable[[str], bool]
    probe_search_with_authenticated_request: Callable[..., Any]
    error_category: Callable[[str], str]
    is_page_recoverable_error: Callable[[str], bool]
    query_page_recovery_threshold: int
    session_page_is_alive: Callable[[Any], bool]
    query_backoff_base_ms: int
    query_backoff_max_ms: int
    safe_page_pause: Callable[[Any, int], None]
    query_network_circuit_threshold: int
    source_ordered_candidates_for_detail: Callable[..., list[Any]]
    detail_network_circuit_threshold: int
    validate_jobs_url: Callable[[str], str]
    ensure_search_source_with_single_retry: Callable[..., tuple[Any, bool, bool]]
    safe_job_card_link: Callable[..., Any]
    respect_detail_click_cadence: Callable[..., None]
    enrich_job_detail_via_panel: Callable[..., Any]
    enrich_job_detail: Callable[..., Any]
    needs_detail_enrichment: Callable[[Any], bool]
    is_incomplete_detail_body: Callable[[str], bool]
    round_robin_candidates_by_location: Callable[..., list[Any]]
    dedupe_vacancies_semantically: Callable[..., tuple[list[Any], list[str]]]
    safe_auth_diagnostic: Callable[[Any], str]

    @classmethod
    def from_module(cls, module: Any) -> "LinkedInJobsPipelineDeps":
        return cls(
            session_store_cls=module.LinkedInSessionStore,
            launch_config_cls=module.AuthenticatedBrowserLaunchConfig,
            configured_headless=module.configured_linkedin_headless,
            open_authenticated_context=module.open_persistent_authenticated_context,
            validate_authenticated_page=module._validate_authenticated_page,
            configured_detail_budget=module.configured_linkedin_detail_budget,
            configured_detail_click_interval_ms=module.configured_linkedin_detail_click_interval_ms,
            configured_direct_detail_fallback=module.configured_linkedin_direct_detail_fallback,
            configured_max_queries_per_location=module.configured_linkedin_max_queries_per_location,
            configured_query_interval_ms=module.configured_linkedin_query_interval_ms,
            build_search_queries=module.build_linkedin_search_queries,
            query_location=module._query_location,
            hard_max_total_query_attempts=module._HARD_MAX_TOTAL_QUERY_ATTEMPTS,
            search_navigation_state_cls=module._SearchNavigationState,
            parse_diagnostics_cls=module.LinkedInParseDiagnostics,
            query_timing_cls=module.LinkedInQueryTiming,
            rejected_record_cls=module.LinkedInRejectedRecord,
            auth_required_error_cls=module.LinkedInAuthRequiredError,
            blocked_error_cls=module.LinkedInBlockedError,
            detail_panel_error_cls=module.LinkedInDetailPanelError,
            respect_query_cadence=module._respect_query_cadence,
            wait_for_search_results_hydration=module._wait_for_search_results_hydration,
            parse_jobs_html_with_diagnostics=module._parse_linkedin_jobs_html_with_diagnostics,
            record_key=module._record_key,
            safe_error_label=module._safe_error_label,
            is_http_response_code_failure=module._is_http_response_code_failure,
            probe_search_with_authenticated_request=module._probe_linkedin_search_with_authenticated_request,
            error_category=module._error_category,
            is_page_recoverable_error=module._is_page_recoverable_error,
            query_page_recovery_threshold=module._QUERY_PAGE_RECOVERY_THRESHOLD,
            session_page_is_alive=module._session_page_is_alive,
            query_backoff_base_ms=module._QUERY_BACKOFF_BASE_MS,
            query_backoff_max_ms=module._QUERY_BACKOFF_MAX_MS,
            safe_page_pause=module._safe_page_pause,
            query_network_circuit_threshold=module._QUERY_NETWORK_CIRCUIT_THRESHOLD,
            source_ordered_candidates_for_detail=module._source_ordered_candidates_for_detail,
            detail_network_circuit_threshold=module._DETAIL_NETWORK_CIRCUIT_THRESHOLD,
            validate_jobs_url=module.validate_linkedin_jobs_url,
            ensure_search_source_with_single_retry=module._ensure_search_source_with_single_retry,
            safe_job_card_link=module._safe_job_card_link,
            respect_detail_click_cadence=module._respect_detail_click_cadence,
            enrich_job_detail_via_panel=module._enrich_job_detail_via_panel,
            enrich_job_detail=module._enrich_job_detail,
            needs_detail_enrichment=module._needs_detail_enrichment,
            is_incomplete_detail_body=module._is_incomplete_detail_body,
            round_robin_candidates_by_location=module._round_robin_candidates_by_location,
            dedupe_vacancies_semantically=module._dedupe_linkedin_vacancies_semantically,
            safe_auth_diagnostic=module._safe_auth_diagnostic,
        )


def scrape_linkedin_jobs_impl(
    request: "LinkedInJobsRequest",
    *,
    session_store: "LinkedInSessionStore | None" = None,
    deps: LinkedInJobsPipelineDeps | None = None,
) -> tuple[
    list["LinkedInVacancyRecord"],
    list["LinkedInRejectedRecord"],
    list["LinkedInQueryTiming"],
    list[str],
    list[str],
]:
    if deps is None:
        from features.web_scraping.infrastructure import linkedin_scraper

        deps = LinkedInJobsPipelineDeps.from_module(linkedin_scraper)

    LinkedInSessionStore = deps.session_store_cls
    AuthenticatedBrowserLaunchConfig = deps.launch_config_cls
    configured_linkedin_headless = deps.configured_headless
    open_persistent_authenticated_context = deps.open_authenticated_context
    _validate_authenticated_page = deps.validate_authenticated_page
    configured_linkedin_detail_budget = deps.configured_detail_budget
    configured_linkedin_detail_click_interval_ms = deps.configured_detail_click_interval_ms
    configured_linkedin_direct_detail_fallback = deps.configured_direct_detail_fallback
    configured_linkedin_max_queries_per_location = deps.configured_max_queries_per_location
    configured_linkedin_query_interval_ms = deps.configured_query_interval_ms
    build_linkedin_search_queries = deps.build_search_queries
    _query_location = deps.query_location
    _HARD_MAX_TOTAL_QUERY_ATTEMPTS = deps.hard_max_total_query_attempts
    _SearchNavigationState = deps.search_navigation_state_cls
    LinkedInParseDiagnostics = deps.parse_diagnostics_cls
    LinkedInQueryTiming = deps.query_timing_cls
    LinkedInRejectedRecord = deps.rejected_record_cls
    LinkedInAuthRequiredError = deps.auth_required_error_cls
    LinkedInBlockedError = deps.blocked_error_cls
    LinkedInDetailPanelError = deps.detail_panel_error_cls
    _respect_query_cadence = deps.respect_query_cadence
    _wait_for_search_results_hydration = deps.wait_for_search_results_hydration
    _parse_linkedin_jobs_html_with_diagnostics = deps.parse_jobs_html_with_diagnostics
    _record_key = deps.record_key
    _safe_error_label = deps.safe_error_label
    _is_http_response_code_failure = deps.is_http_response_code_failure
    _probe_linkedin_search_with_authenticated_request = deps.probe_search_with_authenticated_request
    _error_category = deps.error_category
    _is_page_recoverable_error = deps.is_page_recoverable_error
    _QUERY_PAGE_RECOVERY_THRESHOLD = deps.query_page_recovery_threshold
    _session_page_is_alive = deps.session_page_is_alive
    _QUERY_BACKOFF_BASE_MS = deps.query_backoff_base_ms
    _QUERY_BACKOFF_MAX_MS = deps.query_backoff_max_ms
    _safe_page_pause = deps.safe_page_pause
    _QUERY_NETWORK_CIRCUIT_THRESHOLD = deps.query_network_circuit_threshold
    _source_ordered_candidates_for_detail = deps.source_ordered_candidates_for_detail
    _DETAIL_NETWORK_CIRCUIT_THRESHOLD = deps.detail_network_circuit_threshold
    validate_linkedin_jobs_url = deps.validate_jobs_url
    _ensure_search_source_with_single_retry = deps.ensure_search_source_with_single_retry
    _safe_job_card_link = deps.safe_job_card_link
    _respect_detail_click_cadence = deps.respect_detail_click_cadence
    _enrich_job_detail_via_panel = deps.enrich_job_detail_via_panel
    _enrich_job_detail = deps.enrich_job_detail
    _needs_detail_enrichment = deps.needs_detail_enrichment
    _is_incomplete_detail_body = deps.is_incomplete_detail_body
    _round_robin_candidates_by_location = deps.round_robin_candidates_by_location
    _dedupe_linkedin_vacancies_semantically = deps.dedupe_vacancies_semantically
    _safe_auth_diagnostic = deps.safe_auth_diagnostic

    store = session_store or LinkedInSessionStore()
    metadata = store.load_browser_metadata()
    if metadata is None:
        raise FileNotFoundError(
            "No hay un perfil persistente de LinkedIn inicializado. Ejecutá "
            "`python scripts/bootstrap_linkedin_session.py`."
        )
    launch_config = AuthenticatedBrowserLaunchConfig.from_env(
        persisted_browser=(metadata or {}).get("browser"),
        persisted_executable_path=(metadata or {}).get("executable_path"),
    )
    profile_path = store.resolve_profile_path(
        persisted_profile_path=metadata.get("profile_path"),
        create=False,
    )
    headless = configured_linkedin_headless(default=False)
    session = open_persistent_authenticated_context(
        profile_path=profile_path,
        headless=headless,
        launch_config=launch_config,
        reuse=True,
    )
    begin_job_pages = getattr(session, "begin_job_pages", None)
    job_pages = begin_job_pages() if callable(begin_job_pages) else None
    new_job_page = getattr(session, "new_job_page", None)
    if job_pages is not None and callable(new_job_page):
        new_job_page()
    records: list["LinkedInVacancyRecord"] = []
    candidates: list["LinkedInVacancyRecord"] = []
    seen_candidate_keys: set[str] = set()
    rejected: list["LinkedInRejectedRecord"] = []
    timings: list["LinkedInQueryTiming"] = []
    warnings: list[str] = []
    query_urls: list[str] = []
    try:
        page = session.page
        page.goto("https://www.linkedin.com/jobs/", wait_until="domcontentloaded", timeout=30000)
        _validate_authenticated_page(page)
        search_locations = request.locations or ([request.location] if request.location else [])
        normalized_locations = search_locations or [""]
        detail_budget = configured_linkedin_detail_budget()
        detail_click_interval_ms = configured_linkedin_detail_click_interval_ms()
        direct_detail_fallback = configured_linkedin_direct_detail_fallback()
        max_queries_per_location = configured_linkedin_max_queries_per_location()
        query_interval_ms = configured_linkedin_query_interval_ms()
        query_plan = build_linkedin_search_queries(search_locations)
        planned_locations = {
            _query_location(role)
            for role, _search_url in query_plan
        }
        location_count = max(1, len(planned_locations))
        detail_quota_per_location = (
            max(1, detail_budget // location_count)
            if detail_budget
            else 0
        )
        queries_by_location: dict[str, int] = {}
        stopped_locations: set[str] = set()
        consecutive_query_errors_by_location: dict[str, int] = {}
        recoverable_query_errors_by_location: dict[str, int] = {}
        query_circuit_reasons_by_location: dict[str, str] = {}
        # Los queries planificados siguen limitados por ubicación más abajo.
        # Este contador cubre navegaciones reales, incluyendo el único retry
        # permitido para una Page degradada.
        global_query_budget = _HARD_MAX_TOTAL_QUERY_ATTEMPTS
        global_query_attempts = 0
        search_navigation_state = _SearchNavigationState()
        candidate_locations: dict[str, str] = {}
        for role, search_url in query_plan:
            query_location = _query_location(role)
            if query_location in stopped_locations:
                continue
            if global_query_attempts >= global_query_budget:
                warnings.append(
                    f"query_global_budget_exhausted:{global_query_budget}"
                )
                break
            query_count = queries_by_location.get(query_location, 0)
            if query_count >= max_queries_per_location:
                stopped_locations.add(query_location)
                warnings.append(
                    f"query_budget_exhausted:{query_location or 'unspecified'}:"
                    f"{max_queries_per_location}"
                )
                continue
            queries_by_location[query_location] = query_count + 1
            started = datetime.now(timezone.utc)
            started_monotonic = time.monotonic()
            discovered_count = 0
            error = ""
            diagnostics = LinkedInParseDiagnostics()
            query_urls.append(search_url)
            navigation_attempt = 0
            while True:
                _respect_query_cadence(
                    page,
                    last_successful_query_at=search_navigation_state.completed_at,
                    interval_ms=query_interval_ms,
                )
                global_query_attempts += 1
                navigation_attempt += 1
                search_navigation_state.invalidate_source()
                try:
                    page.goto(
                        search_url,
                        wait_until="domcontentloaded",
                        timeout=30000,
                    )
                    _validate_authenticated_page(page)
                    hydration_state = _wait_for_search_results_hydration(page)
                    if hydration_state == "empty":
                        warnings.append(f"query_empty_results_explicit:{role}")
                    elif hydration_state == "timeout":
                        warnings.append(
                            f"query_hydration_timeout:no_terminal_signal:{role}"
                        )
                    discovered, diagnostics = (
                        _parse_linkedin_jobs_html_with_diagnostics(
                            page.content(),
                            source_url=search_url,
                            now=datetime.now(timezone.utc),
                        )
                    )
                    search_navigation_state.active_source_url = search_url
                    discovered_count = len(discovered)
                    for record in discovered:
                        dedupe_key = _record_key(record)
                        if dedupe_key in seen_candidate_keys:
                            rejected.append(
                                LinkedInRejectedRecord(
                                    source_url=record.canonical_url,
                                    title=record.title,
                                    reason="duplicate",
                                )
                            )
                            continue
                        seen_candidate_keys.add(dedupe_key)
                        candidate_locations[dedupe_key] = query_location
                        candidates.append(record)
                    consecutive_query_errors_by_location[query_location] = 0
                    recoverable_query_errors_by_location[query_location] = 0
                    error = ""
                    break
                except (LinkedInAuthRequiredError, LinkedInBlockedError):
                    raise
                except Exception as exc:
                    attempt_error = _safe_error_label(exc)
                    can_retry_degraded_page = (
                        navigation_attempt == 1
                        and _is_http_response_code_failure(attempt_error)
                        and global_query_attempts < global_query_budget
                    )
                    if can_retry_degraded_page:
                        warnings.append(
                            "query_navigation_retry:"
                            f"{query_location or 'unspecified'}:"
                            "http_response_code_failure"
                        )
                        try:
                            page = session.replace_page()
                            search_navigation_state.invalidate_source()
                            warnings.append(
                                "page_recovered:query:"
                                f"{query_location or 'unspecified'}:"
                                "http_response_code_failure"
                            )
                        except Exception as recovery_exc:
                            error = attempt_error
                            recovery_error = _safe_error_label(recovery_exc)
                            warnings.append(
                                "page_recovery_failed:query:"
                                f"{query_location or 'unspecified'}:"
                                f"{_error_category(recovery_error)}"
                            )
                            break
                        continue

                    can_probe_authenticated_request = (
                        navigation_attempt == 2
                        and _is_http_response_code_failure(attempt_error)
                        and global_query_attempts < global_query_budget
                    )
                    if can_probe_authenticated_request:
                        global_query_attempts += 1
                        warnings.append(
                            "query_probe_attempt:"
                            f"{query_location or 'unspecified'}"
                        )
                        probe = (
                            _probe_linkedin_search_with_authenticated_request(
                                session,
                                source_url=search_url,
                                now=datetime.now(timezone.utc),
                            )
                        )
                        diagnostics = probe.diagnostics
                        probe_warning = (
                            "query_probe_result:"
                            f"{query_location or 'unspecified'}:"
                            f"status_{probe.status_code}:"
                            f"{probe.category}"
                        )
                        if probe.detail:
                            probe_warning += f":{probe.detail}"
                        warnings.append(probe_warning)
                        if probe.records:
                            discovered_count = len(probe.records)
                            for record in probe.records:
                                dedupe_key = _record_key(record)
                                if dedupe_key in seen_candidate_keys:
                                    rejected.append(
                                        LinkedInRejectedRecord(
                                            source_url=record.canonical_url,
                                            title=record.title,
                                            reason="duplicate",
                                        )
                                    )
                                    continue
                                seen_candidate_keys.add(dedupe_key)
                                candidate_locations[dedupe_key] = (
                                    query_location
                                )
                                candidates.append(record)
                            consecutive_query_errors_by_location[
                                query_location
                            ] = 0
                            recoverable_query_errors_by_location[
                                query_location
                            ] = 0
                            error = ""
                            break
                        error = f"probe:{probe.category}"
                        warnings.append(f"query_failed:{role}:{error}")
                        consecutive_query_errors_by_location[
                            query_location
                        ] = (
                            consecutive_query_errors_by_location.get(
                                query_location,
                                0,
                            )
                            + 1
                        )
                        break

                    error = attempt_error
                    warnings.append(f"query_failed:{role}:{error}")
                    consecutive_query_errors_by_location[query_location] = (
                        consecutive_query_errors_by_location.get(
                            query_location,
                            0,
                        )
                        + 1
                    )
                    if _is_page_recoverable_error(error):
                        recoverable_query_errors_by_location[query_location] = (
                            recoverable_query_errors_by_location.get(
                                query_location,
                                0,
                            )
                            + 1
                        )
                    else:
                        recoverable_query_errors_by_location[query_location] = 0
                    break
                finally:
                    search_navigation_state.completed_at = time.monotonic()
            completed = datetime.now(timezone.utc)
            timings.append(
                LinkedInQueryTiming(
                    query=role,
                    started_at=started,
                    completed_at=completed,
                    elapsed_ms=int((time.monotonic() - started_monotonic) * 1000),
                    discovered_count=discovered_count,
                    retained_count=0,
                    error=error,
                    diagnostics=diagnostics,
                )
            )
            if error:
                error_category = _error_category(error)
                location_error_count = consecutive_query_errors_by_location.get(
                    query_location,
                    0,
                )
                recoverable_error_count = (
                    recoverable_query_errors_by_location.get(
                        query_location,
                        0,
                    )
                )
                query_circuit_reasons_by_location[query_location] = (
                    error_category
                )
                is_primary_attempt = (
                    queries_by_location.get(query_location, 0) == 1
                )
                retry_available = (
                    is_primary_attempt
                    and queries_by_location.get(query_location, 0)
                    < max_queries_per_location
                )
                if _is_http_response_code_failure(error):
                    warnings.append(
                        "query_location_circuit_open:"
                        f"{query_location or 'unspecified'}:"
                        "http_response_code_failure"
                    )
                    retry_available = False
                if error_category in {
                    "query_rate_limited",
                    "query_access_rejected",
                    "query_upstream_failure",
                    "query_navigation_failure",
                }:
                    warnings.append(
                        "query_location_circuit_open:"
                        f"{query_location or 'unspecified'}:"
                        f"{error_category}"
                    )
                    retry_available = False
                if (
                    not _is_http_response_code_failure(error)
                    and (
                        recoverable_error_count >= _QUERY_PAGE_RECOVERY_THRESHOLD
                        or not _session_page_is_alive(session)
                    )
                ):
                    try:
                        if not _session_page_is_alive(session):
                            page = session.replace_page()
                            search_navigation_state.invalidate_source()
                            warnings.append(
                                "page_recovered:query:"
                                f"{query_location or 'unspecified'}:"
                                f"{error_category}"
                            )
                        else:
                            warnings.append(
                                "page_recovery_not_required:query:"
                                f"{query_location or 'unspecified'}:"
                                f"{error_category}"
                            )
                    except Exception as recovery_exc:
                        warnings.append(
                            "page_recovery_failed:query:"
                            f"{query_location or 'unspecified'}:"
                            f"{_safe_error_label(recovery_exc)}"
                        )
                        warnings.append(
                            "query_location_circuit_open:"
                            f"{query_location or 'unspecified'}:"
                            "page_recovery_failed"
                        )
                        retry_available = False
                backoff_ms = min(
                    _QUERY_BACKOFF_BASE_MS * max(1, location_error_count),
                    _QUERY_BACKOFF_MAX_MS,
                )
                _safe_page_pause(page, backoff_ms)
                if recoverable_error_count >= _QUERY_NETWORK_CIRCUIT_THRESHOLD:
                    warnings.append(
                        "query_location_circuit_open:"
                        f"{query_location or 'unspecified'}:"
                        f"systemic_navigation_failure:{error_category}"
                    )
                    retry_available = False
                if not retry_available:
                    stopped_locations.add(query_location)
                    warnings.append(
                        "query_location_stopped:"
                        f"{query_location or 'unspecified'}:"
                        f"{error_category}"
                    )

        relevant_candidates: list["LinkedInVacancyRecord"] = []
        for candidate in candidates:
            if not candidate.matched_terms:
                rejected.append(
                    LinkedInRejectedRecord(
                        source_url=candidate.canonical_url,
                        title=candidate.title,
                        reason="low_topic_relevance",
                    )
                )
                continue
            if candidate.published_at is not None and not candidate.is_within_24_hours:
                rejected.append(
                    LinkedInRejectedRecord(
                        source_url=candidate.canonical_url,
                        title=candidate.title,
                        reason="outside_24_hours",
                    )
                )
                continue
            relevant_candidates.append(candidate)

        # Enrich details source-by-source. LinkedIn's side panel is stateful;
        # alternating countries before enrichment makes the panel often keep a
        # previous job hydrated and causes false `stale_detail_panel` rejects.
        shortlist = _source_ordered_candidates_for_detail(
            relevant_candidates,
            source_order=query_urls,
        )
        last_detail_click_at: float | None = None
        detail_attempts = 0
        detail_attempts_by_location: dict[str, int] = {}
        detail_network_circuit_locations: set[str] = set()
        direct_detail_fallback_disabled_locations: set[str] = set()
        consecutive_detail_network_failures_by_location: dict[str, int] = {}
        if not _session_page_is_alive(session):
            warnings.append(
                "detail_runtime_session_unavailable:"
                f"{next(iter(query_circuit_reasons_by_location.values()), 'page_unusable')}"
            )

        enriched_records: list["LinkedInVacancyRecord"] = []
        should_balance_locations = len([location for location in normalized_locations if location]) > 1

        for candidate in shortlist:
            candidate_key = _record_key(candidate)
            candidate_location = candidate_locations.get(candidate_key, "")
            verified = candidate
            needs_enrichment = _needs_detail_enrichment(candidate)
            detail_reason = ""

            if needs_enrichment:
                location_detail_attempts = detail_attempts_by_location.get(
                    candidate_location,
                    0,
                )
                if candidate_location in detail_network_circuit_locations:
                    detail_reason = "detail_network_failure"
                elif (
                    detail_attempts >= detail_budget
                    or location_detail_attempts >= detail_quota_per_location
                ):
                    detail_reason = "detail_budget_exhausted"
                else:
                    if not _session_page_is_alive(session):
                        try:
                            page = session.replace_page()
                            search_navigation_state.invalidate_source()
                            warnings.append(
                                "page_recovered:detail:"
                                f"{candidate_location or 'unspecified'}:"
                                "page_unusable"
                            )
                        except Exception as recovery_exc:
                            detail_network_circuit_locations.add(candidate_location)
                            detail_reason = "detail_network_failure"
                            warnings.append(
                                "page_recovery_failed:detail:"
                                f"{candidate_location or 'unspecified'}:"
                                f"{_safe_error_label(recovery_exc)}"
                            )

                    if not detail_reason:
                        detail_attempts += 1
                        detail_attempts_by_location[candidate_location] = (
                            location_detail_attempts + 1
                        )
                        try:
                            source_url = validate_linkedin_jobs_url(
                                candidate.source_url
                            )
                            def reserve_source_retry() -> None:
                                nonlocal detail_attempts
                                detail_attempts += 1
                                detail_attempts_by_location[
                                    candidate_location
                                ] = (
                                    detail_attempts_by_location.get(
                                        candidate_location,
                                        0,
                                    )
                                    + 1
                                )

                            source_retry_allowed = (
                                detail_attempts < detail_budget
                                and detail_attempts_by_location.get(
                                    candidate_location,
                                    0,
                                )
                                < detail_quota_per_location
                            )
                            page, _navigated, _source_retried = (
                                _ensure_search_source_with_single_retry(
                                    session,
                                    page,
                                    source_url=source_url,
                                    navigation_state=search_navigation_state,
                                    interval_ms=query_interval_ms,
                                    retry_allowed=source_retry_allowed,
                                    warning_scope=(
                                        candidate_location or "unspecified"
                                    ),
                                    warnings=warnings,
                                    reserve_retry=reserve_source_retry,
                                )
                            )
                            card_link = _safe_job_card_link(page, candidate)
                            if card_link is None:
                                raise LinkedInDetailPanelError(
                                    "card_click_failed"
                                )
                            _respect_detail_click_cadence(
                                page,
                                last_detail_click_at=last_detail_click_at,
                                interval_ms=detail_click_interval_ms,
                            )
                            verified = _enrich_job_detail_via_panel(
                                page,
                                candidate,
                                card_link=card_link,
                                include_description=request.include_description,
                                now=datetime.now(timezone.utc),
                            )
                            consecutive_detail_network_failures_by_location[
                                candidate_location
                            ] = 0
                        except (LinkedInAuthRequiredError, LinkedInBlockedError):
                            search_navigation_state.invalidate_source()
                            raise
                        except LinkedInDetailPanelError as panel_exc:
                            search_navigation_state.invalidate_source()
                            detail_reason = panel_exc.reason
                            warning = (
                                "detail_panel_failed:"
                                f"{candidate.linkedin_job_id or 'unknown'}:"
                                f"{panel_exc.reason}"
                            )
                            if panel_exc.safe_label:
                                warning += f":{panel_exc.safe_label}"
                            warnings.append(warning)
                            if (
                                panel_exc.reason == "stale_detail_panel"
                                and detail_attempts < detail_budget
                                and detail_attempts_by_location.get(
                                    candidate_location,
                                    0,
                                )
                                < detail_quota_per_location
                            ):
                                detail_attempts += 1
                                detail_attempts_by_location[
                                    candidate_location
                                ] = (
                                    detail_attempts_by_location.get(
                                        candidate_location,
                                        0,
                                    )
                                    + 1
                                )
                                warnings.append(
                                    "detail_panel_stale_retry:"
                                    f"{candidate.linkedin_job_id or 'unknown'}"
                                )
                                try:
                                    page, _navigated, _source_retried = (
                                        _ensure_search_source_with_single_retry(
                                            session,
                                            page,
                                            source_url=source_url,
                                            navigation_state=search_navigation_state,
                                            interval_ms=query_interval_ms,
                                            retry_allowed=False,
                                            warning_scope=(
                                                candidate_location or "unspecified"
                                            ),
                                            warnings=warnings,
                                        )
                                    )
                                    retry_card_link = _safe_job_card_link(
                                        page,
                                        candidate,
                                    )
                                    if retry_card_link is None:
                                        raise LinkedInDetailPanelError(
                                            "card_click_failed"
                                        )
                                    _respect_detail_click_cadence(
                                        page,
                                        last_detail_click_at=last_detail_click_at,
                                        interval_ms=detail_click_interval_ms,
                                    )
                                    verified = _enrich_job_detail_via_panel(
                                        page,
                                        candidate,
                                        card_link=retry_card_link,
                                        include_description=request.include_description,
                                        now=datetime.now(timezone.utc),
                                    )
                                    detail_reason = ""
                                    consecutive_detail_network_failures_by_location[
                                        candidate_location
                                    ] = 0
                                except LinkedInDetailPanelError as retry_exc:
                                    search_navigation_state.invalidate_source()
                                    detail_reason = retry_exc.reason
                                    retry_warning = (
                                        "detail_panel_stale_retry_failed:"
                                        f"{candidate.linkedin_job_id or 'unknown'}:"
                                        f"{retry_exc.reason}"
                                    )
                                    if retry_exc.safe_label:
                                        retry_warning += f":{retry_exc.safe_label}"
                                    warnings.append(retry_warning)
                                except Exception as retry_exc:
                                    search_navigation_state.invalidate_source()
                                    retry_error = _safe_error_label(retry_exc)
                                    detail_reason = (
                                        "detail_network_failure"
                                        if _is_page_recoverable_error(retry_error)
                                        else "detail_fetch_failed"
                                    )
                                    warnings.append(
                                        "detail_panel_stale_retry_failed:"
                                        f"{candidate.linkedin_job_id or 'unknown'}:"
                                        f"{retry_error}"
                                    )
                            if (
                                panel_exc.safe_label
                                and _is_http_response_code_failure(
                                    panel_exc.safe_label
                                )
                            ):
                                direct_detail_fallback_disabled_locations.add(
                                    candidate_location
                                )
                                detail_network_circuit_locations.add(
                                    candidate_location
                                )
                        except Exception as exc:
                            search_navigation_state.invalidate_source()
                            detail_error = _safe_error_label(exc)
                            detail_reason = (
                                "detail_network_failure"
                                if _is_page_recoverable_error(detail_error)
                                else "detail_fetch_failed"
                            )
                            warnings.append(
                                f"detail_failed:"
                                f"{candidate.linkedin_job_id or 'unknown'}:"
                                f"{detail_error}"
                            )
                            if detail_reason == "detail_network_failure":
                                failures = (
                                    consecutive_detail_network_failures_by_location.get(
                                        candidate_location,
                                        0,
                                    )
                                    + 1
                                )
                                consecutive_detail_network_failures_by_location[
                                    candidate_location
                                ] = failures
                                if (
                                    _is_http_response_code_failure(detail_error)
                                    or failures >= _DETAIL_NETWORK_CIRCUIT_THRESHOLD
                                ):
                                    direct_detail_fallback_disabled_locations.add(
                                        candidate_location
                                    )
                                    detail_network_circuit_locations.add(
                                        candidate_location
                                    )
                                    warnings.append(
                                        "detail_location_circuit_open:"
                                        f"{candidate_location or 'unspecified'}:"
                                        f"{_error_category(detail_error)}"
                                    )
                        finally:
                            last_detail_click_at = time.monotonic()

                        fallback_allowed = (
                            bool(detail_reason)
                            and direct_detail_fallback
                            and candidate_location
                            not in direct_detail_fallback_disabled_locations
                            and detail_reason
                            not in {
                                "detail_budget_exhausted",
                                "detail_network_failure",
                            }
                        )
                        if fallback_allowed:
                            try:
                                search_navigation_state.invalidate_source()
                                verified = _enrich_job_detail(
                                    page,
                                    candidate,
                                    include_description=request.include_description,
                                    now=datetime.now(timezone.utc),
                                )
                                detail_reason = ""
                            except (LinkedInAuthRequiredError, LinkedInBlockedError):
                                raise
                            except Exception as exc:
                                detail_reason = (
                                    "detail_network_failure"
                                    if _is_page_recoverable_error(
                                        _safe_error_label(exc)
                                    )
                                    else "detail_fetch_failed"
                                )
                                warnings.append(
                                    f"detail_fallback_failed:"
                                    f"{candidate.linkedin_job_id or 'unknown'}:"
                                    f"{_safe_error_label(exc)}"
                                )

            if request.include_description and not (
                verified.description_full_text or ""
            ).strip():
                rejected.append(
                    LinkedInRejectedRecord(
                        source_url=verified.canonical_url,
                        title=verified.title,
                        reason=detail_reason or "missing_description_full_text",
                    )
                )
                warnings.append(
                    "metadata_enrichment_incomplete:"
                    f"{verified.linkedin_job_id or 'unknown'}:"
                    f"{detail_reason or 'missing_description_full_text'}"
                )
                continue

            if (
                request.include_description
                and verified.description_full_text
                and _is_incomplete_detail_body(verified.description_full_text)
            ):
                rejected.append(
                    LinkedInRejectedRecord(
                        source_url=verified.canonical_url,
                        title=verified.title,
                        reason="detail_incomplete_body",
                    )
                )
                warnings.append(
                    "metadata_enrichment_incomplete:"
                    f"{verified.linkedin_job_id or 'unknown'}:detail_incomplete_body"
                )
                continue

            if verified.published_at is None:
                rejected.append(
                    LinkedInRejectedRecord(
                        source_url=verified.canonical_url,
                        title=verified.title,
                        reason=detail_reason or "unverified_posted_date",
                    )
                )
                continue
            if not verified.is_within_24_hours:
                rejected.append(
                    LinkedInRejectedRecord(
                        source_url=verified.canonical_url,
                        title=verified.title,
                        reason="outside_24_hours",
                    )
                )
                continue
            if detail_reason:
                warnings.append(
                    "metadata_enrichment_incomplete:"
                    f"{verified.linkedin_job_id or 'unknown'}:"
                    f"{detail_reason}"
                )
            enriched_records.append(verified)

            if request.max_results > 0:
                ordered_preview = (
                    _round_robin_candidates_by_location(
                        enriched_records,
                        candidate_locations=candidate_locations,
                        location_order=normalized_locations,
                    )
                    if should_balance_locations
                    else enriched_records
                )
                ordered_preview, _preview_duplicate_warnings = _dedupe_linkedin_vacancies_semantically(
                    ordered_preview
                )
                requested_location_count = len(
                    [location for location in normalized_locations if location]
                )
                required_location_coverage = min(
                    request.max_results,
                    requested_location_count,
                )
                covered_locations = {
                    candidate_locations.get(_record_key(record), "")
                    for record in enriched_records
                    if candidate_locations.get(_record_key(record), "")
                }
                if (
                    len(ordered_preview) >= request.max_results
                    and len(covered_locations) >= required_location_coverage
                ):
                    break

        ordered_records = (
            _round_robin_candidates_by_location(
                enriched_records,
                candidate_locations=candidate_locations,
                location_order=normalized_locations,
            )
            if should_balance_locations
            else enriched_records
        )
        ordered_records, semantic_duplicate_warnings = _dedupe_linkedin_vacancies_semantically(
            ordered_records
        )
        warnings.extend(semantic_duplicate_warnings)
        records = ordered_records[: request.max_results]

        for record in records:
            record_id = record.linkedin_job_id or "unknown"
            missing_fields = [
                field_name
                for field_name, value in (
                    ("company_name", record.company_name),
                    ("location", record.location),
                    ("workplace_type", record.workplace_type),
                    (
                        "description_full_text",
                        record.description_full_text if request.include_description else "skipped",
                    ),
                )
                if not value
            ]
            warnings.extend(
                f"metadata_missing:{record_id}:{field_name}"
                for field_name in missing_fields
            )
            if (
                request.include_description
                and record.description_full_text
                and not any(
                    (
                        record.hard_skills,
                        record.soft_skills,
                        record.candidate_expectations,
                        record.responsibilities,
                    )
                )
            ):
                warnings.append(
                    f"metadata_structured_missing:{record_id}"
                )

        retained_by_source: dict[str, int] = {}
        for record in records:
            retained_by_source[record.source_url] = (
                retained_by_source.get(record.source_url, 0) + 1
            )
        timings = [
            timing.model_copy(
                update={"retained_count": retained_by_source.get(query_url, 0)}
            )
            for timing, query_url in zip(timings, query_urls)
        ]
        return records, rejected, timings, warnings, query_urls
    except (LinkedInAuthRequiredError, LinkedInBlockedError) as exc:
        store.record_runtime_failure(
            type(exc).__name__,
            browser=launch_config.browser,
            headless=headless,
            profile_path=profile_path,
            diagnostic=_safe_auth_diagnostic(session.page),
        )
        raise
    finally:
        try:
            close_job_pages = getattr(session, "close_job_pages", None)
            if job_pages is not None and callable(close_job_pages):
                close_job_pages(job_pages)
        finally:
            # Los contextos reales son reutilizables y conservan el perfil
            # autenticado. Los doubles legacy sin ownership mantienen el
            # cierre anterior para no ocultar recursos en tests/integraciones.
            if job_pages is None:
                session.close()




def scrape_linkedin_jobs(
    request: "LinkedInJobsRequest",
    *,
    session_store: "LinkedInSessionStore | None" = None,
) -> tuple[
    list["LinkedInVacancyRecord"],
    list["LinkedInRejectedRecord"],
    list["LinkedInQueryTiming"],
    list[str],
    list[str],
]:
    """Run the LinkedIn jobs scraping pipeline."""

    return scrape_linkedin_jobs_impl(
        request,
        session_store=session_store,
    )
