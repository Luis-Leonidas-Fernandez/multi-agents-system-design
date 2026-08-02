"""Query planning and search-page navigation helpers for LinkedIn jobs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
import unicodedata
from typing import Callable
from urllib.parse import urlencode

from features.web_scraping.domain.linkedin_models import (
    LinkedInParseDiagnostics,
    LinkedInVacancyRecord,
)
from features.web_scraping.infrastructure.linkedin_navigation import (
    LinkedInAuthRequiredError,
    LinkedInBlockedError,
    _HYDRATION_POLL_INTERVALS_MS,
    _error_category,
    _is_http_response_code_failure,
    _locator_has_signal,
    _raise_for_terminal_page_signal,
    _safe_error_label,
    _safe_page_pause,
    _validate_authenticated_page,
)
from features.web_scraping.infrastructure.linkedin_parser import (
    _parse_linkedin_jobs_html_with_diagnostics,
)
from features.web_scraping.infrastructure.linkedin_url_policy import (
    is_linkedin_auth_checkpoint,
    validate_linkedin_jobs_url,
)


CONSOLIDATED_QUERY_PLANS = (
    (
        "AI/ML/Data/GenAI",
        (
            '"AI Engineer" OR "Artificial Intelligence Engineer" '
            'OR "Machine Learning Engineer" OR "ML Engineer" '
            'OR "Deep Learning Engineer" OR "DL Engineer" '
            'OR "Data Scientist" OR "Data Analyst" OR "MLOps Engineer" '
            'OR "Generative AI Engineer" OR "Generative AI" '
            'OR "LLM Engineer" OR "LLM Scientist" OR "Speech LLM Engineer"'
        ),
    ),
    (
        "AI Agents/Product/Architecture",
        (
            '"AI Agent Engineer" OR "AI Agent Developer" '
            'OR "AI Agent" OR "AI Product Engineer" '
            'OR "AI Product Manager" OR "AI Product" '
            'OR "Applied Scientist" OR "Applied AI Engineer" OR "AI Specialist" OR "AI Mentor" '
            'OR "AI Architect" OR "AI Solutions Architect" '
            'OR "Solution Architect AI" '
            'OR "Developer Technology Engineer AI" '
            'OR "Developer Technology Engineer - AI" '
            'OR "AI Automation Engineer" OR "RAG LLM System" OR "RAG & LLM System"'
        ),
    ),
)


@dataclass(frozen=True)
class LinkedInLocationResolution:
    canonical_label: str
    geo_id: str = ""


def _normalize_linkedin_location_alias(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", (value or "").strip().casefold())
    return "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )


_LINKEDIN_LOCATION_RESOLUTIONS = (
    (
        LinkedInLocationResolution("South Korea", "105149562"),
        ("South Korea", "Corea del Sur", "대한민국"),
    ),
    (
        LinkedInLocationResolution("Japan", "101355337"),
        ("Japan", "Japón", "日本"),
    ),
)
_LINKEDIN_LOCATION_BY_ALIAS = {
    _normalize_linkedin_location_alias(alias): resolution
    for resolution, aliases in _LINKEDIN_LOCATION_RESOLUTIONS
    for alias in aliases
}


def resolve_linkedin_location(value: str) -> LinkedInLocationResolution:
    """Resolve known aliases without guessing geoIds for unknown locations."""
    clean_value = (value or "").strip()
    if not clean_value:
        return LinkedInLocationResolution("")
    return _LINKEDIN_LOCATION_BY_ALIAS.get(
        _normalize_linkedin_location_alias(clean_value),
        LinkedInLocationResolution(clean_value),
    )


def build_linkedin_search_queries(
    locations: str | list[str] | tuple[str, ...] = "",
) -> list[tuple[str, str]]:
    if isinstance(locations, str):
        normalized_locations = [locations.strip()] if locations.strip() else [""]
    else:
        normalized_locations = [item.strip() for item in locations if item.strip()]
    if not normalized_locations:
        normalized_locations = [""]
    results: list[tuple[str, str]] = []
    seen_locations: set[str] = set()
    unique_locations: list[LinkedInLocationResolution] = []
    for location in normalized_locations:
        resolved_location = resolve_linkedin_location(location)
        location_key = resolved_location.canonical_label.casefold()
        if location_key in seen_locations:
            continue
        seen_locations.add(location_key)
        unique_locations.append(resolved_location)
    for query_name, keywords in CONSOLIDATED_QUERY_PLANS:
        for resolved_location in unique_locations:
            location = resolved_location.canonical_label
            params = {
                "keywords": keywords,
                "f_TPR": "r86400",
                "sortBy": "DD",
            }
            if location:
                params["location"] = location
            if resolved_location.geo_id:
                params["geoId"] = resolved_location.geo_id
            url = f"https://www.linkedin.com/jobs/search/?{urlencode(params)}"
            label = f"{query_name} @ {location}" if location else query_name
            results.append((label, validate_linkedin_jobs_url(url)))
    return results


_SEARCH_RESULT_SIGNAL_SELECTORS = (
    ".job-card-container",
    "li[data-occludable-job-id]",
    ".scaffold-layout__list-item",
    "[data-job-id]",
    "a[href*='/jobs/view/']",
)
_EMPTY_RESULTS_SELECTORS = (
    ".jobs-search-no-results-banner",
    ".jobs-search-no-results",
    ".jobs-search-two-pane__no-results-banner",
    "[data-test-no-results]",
    "text=No matching jobs found",
    "text=No jobs found",
    "text=No se encontraron empleos",
)
_SEARCH_HYDRATION_MAX_MS = 6500
_QUERY_BACKOFF_BASE_MS = 1000
_QUERY_BACKOFF_MAX_MS = 3000
_QUERY_PAGE_RECOVERY_THRESHOLD = 2
_QUERY_NETWORK_CIRCUIT_THRESHOLD = 3
_DETAIL_NETWORK_CIRCUIT_THRESHOLD = 2
_HARD_MAX_TOTAL_QUERY_ATTEMPTS = 8
_QUERY_PROBE_TIMEOUT_MS = 15000
_HARD_MAX_QUERIES_PER_LOCATION = len(CONSOLIDATED_QUERY_PLANS)


@dataclass
class _SearchNavigationState:
    active_source_url: str | None = None
    completed_at: float | None = None

    def invalidate_source(self) -> None:
        self.active_source_url = None


@dataclass(frozen=True)
class _AuthenticatedSearchProbeResult:
    records: list[LinkedInVacancyRecord]
    diagnostics: LinkedInParseDiagnostics
    status_code: int
    category: str
    detail: str = ""


def _classify_probe_status(status_code: int) -> str:
    if status_code == 429:
        return "query_rate_limited"
    if status_code in {401, 403, 999}:
        return "query_access_rejected"
    if 500 <= status_code <= 599:
        return "query_upstream_failure"
    return "query_navigation_failure"


def _probe_linkedin_search_with_authenticated_request(
    session,
    *,
    source_url: str,
    now: datetime | None = None,
    parse_jobs_html_with_diagnostics: Callable[..., tuple[list[LinkedInVacancyRecord], LinkedInParseDiagnostics]] = _parse_linkedin_jobs_html_with_diagnostics,
) -> _AuthenticatedSearchProbeResult:
    """Hace un único GET autenticado y descarta siempre la respuesta/HTML."""

    validated_source_url = validate_linkedin_jobs_url(source_url)
    response = None
    try:
        response = session.context.request.get(
            validated_source_url,
            fail_on_status_code=False,
            timeout=_QUERY_PROBE_TIMEOUT_MS,
        )
        status_code = int(getattr(response, "status", 0) or 0)
        final_url = str(getattr(response, "url", "") or "")
        if is_linkedin_auth_checkpoint(final_url):
            raise LinkedInAuthRequiredError(
                "La sesión LinkedIn requiere login, 2FA o checkpoint manual."
            )
        try:
            validate_linkedin_jobs_url(final_url)
        except ValueError:
            return _AuthenticatedSearchProbeResult(
                records=[],
                diagnostics=LinkedInParseDiagnostics(),
                status_code=status_code,
                category="query_navigation_failure",
                detail="final_url_rejected",
            )

        if not 200 <= status_code <= 299:
            return _AuthenticatedSearchProbeResult(
                records=[],
                diagnostics=LinkedInParseDiagnostics(),
                status_code=status_code,
                category=_classify_probe_status(status_code),
            )

        try:
            html = response.text()
            records, diagnostics = parse_jobs_html_with_diagnostics(
                html,
                source_url=validated_source_url,
                now=now or datetime.now(timezone.utc),
            )
        except Exception:
            return _AuthenticatedSearchProbeResult(
                records=[],
                diagnostics=LinkedInParseDiagnostics(),
                status_code=status_code,
                category="query_navigation_failure",
                detail="body_parse_failed",
            )
        return _AuthenticatedSearchProbeResult(
            records=records,
            diagnostics=diagnostics,
            status_code=status_code,
            category=("ok" if records else "query_navigation_failure"),
            detail="" if records else "no_cards",
        )
    except LinkedInAuthRequiredError:
        raise
    except Exception:
        return _AuthenticatedSearchProbeResult(
            records=[],
            diagnostics=LinkedInParseDiagnostics(),
            status_code=0,
            category="query_navigation_failure",
            detail="request_failed",
        )
    finally:
        if response is not None:
            try:
                response.dispose()
            except Exception:
                pass


def _respect_query_cadence(
    page,
    *,
    last_successful_query_at: float | None,
    interval_ms: int,
    now_fn=None,
) -> None:
    if last_successful_query_at is None:
        return
    now_fn = now_fn or time.monotonic
    elapsed_ms = max(0, round((now_fn() - last_successful_query_at) * 1000))
    remaining_ms = max(0, interval_ms - elapsed_ms)
    if remaining_ms:
        _safe_page_pause(page, remaining_ms)


def _wait_for_search_results_hydration(
    page,
    *,
    max_wait_ms: int = _SEARCH_HYDRATION_MAX_MS,
) -> str:
    elapsed_ms = 0
    poll_index = 0
    while True:
        _raise_for_terminal_page_signal(page)
        if any(
            _locator_has_signal(page, selector)
            for selector in _SEARCH_RESULT_SIGNAL_SELECTORS
        ):
            return "results"
        if any(
            _locator_has_signal(page, selector)
            for selector in _EMPTY_RESULTS_SELECTORS
        ):
            return "empty"
        if elapsed_ms >= max_wait_ms:
            return "timeout"
        interval_ms = _HYDRATION_POLL_INTERVALS_MS[
            min(poll_index, len(_HYDRATION_POLL_INTERVALS_MS) - 1)
        ]
        wait_ms = min(interval_ms, max_wait_ms - elapsed_ms)
        _safe_page_pause(page, wait_ms)
        elapsed_ms += wait_ms
        poll_index += 1


def _ensure_search_source(
    page,
    *,
    source_url: str,
    navigation_state: _SearchNavigationState,
    interval_ms: int,
    now_fn=None,
    wait_for_search_results_hydration: Callable[..., str] = _wait_for_search_results_hydration,
    validate_authenticated_page: Callable[[object], None] = _validate_authenticated_page,
) -> bool:
    now_fn = now_fn or time.monotonic
    validated_source_url = validate_linkedin_jobs_url(source_url)
    if navigation_state.active_source_url == validated_source_url:
        return False

    _respect_query_cadence(
        page,
        last_successful_query_at=navigation_state.completed_at,
        interval_ms=interval_ms,
        now_fn=now_fn,
    )
    navigation_state.invalidate_source()
    try:
        page.goto(
            validated_source_url,
            wait_until="domcontentloaded",
            timeout=30000,
        )
        validate_authenticated_page(page)
        wait_for_search_results_hydration(page)
        navigation_state.active_source_url = validated_source_url
        return True
    finally:
        navigation_state.completed_at = now_fn()


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
    ensure_search_source: Callable[..., bool] = _ensure_search_source,
) -> tuple[object, bool, bool]:
    """Carga un source y reemplaza una Page degradada como máximo una vez."""

    try:
        navigated = ensure_search_source(
            page,
            source_url=source_url,
            navigation_state=navigation_state,
            interval_ms=interval_ms,
            now_fn=now_fn,
        )
        return page, navigated, False
    except (LinkedInAuthRequiredError, LinkedInBlockedError):
        raise
    except Exception as exc:
        error = _safe_error_label(exc)
        if not (retry_allowed and _is_http_response_code_failure(error)):
            raise

        navigation_state.invalidate_source()
        if reserve_retry is not None:
            reserve_retry()
        warnings.append(
            f"detail_source_navigation_retry:{warning_scope}:"
            "http_response_code_failure"
        )
        try:
            replacement_page = session.replace_page()
        except Exception as recovery_exc:
            recovery_error = _safe_error_label(recovery_exc)
            warnings.append(
                f"page_recovery_failed:detail:{warning_scope}:"
                f"{_error_category(recovery_error)}"
            )
            raise exc from recovery_exc

        navigation_state.invalidate_source()
        warnings.append(
            f"page_recovered:detail:{warning_scope}:"
            "http_response_code_failure"
        )
        navigated = ensure_search_source(
            replacement_page,
            source_url=source_url,
            navigation_state=navigation_state,
            interval_ms=interval_ms,
            now_fn=now_fn,
        )
        return replacement_page, navigated, True


def _query_location(label: str) -> str:
    separator = " @ "
    return label.rsplit(separator, 1)[1] if separator in label else ""
