"""Query planning and search-page navigation helpers for LinkedIn jobs."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
import inspect
import re
import time
import unicodedata
from typing import Callable, Iterator
from urllib.parse import urlencode, urljoin

from features.web_scraping.domain.linkedin_models import (
    LinkedInParseDiagnostics,
    LinkedInSearchHydrationDiagnostic,
    LinkedInVacancyRecord,
)
from features.web_scraping.infrastructure.linkedin_navigation import (
    LinkedInAuthRequiredError,
    LinkedInBlockedError,
    _BLOCK_SIGNAL_SELECTORS,
    _HYDRATION_POLL_INTERVALS_MS,
    _LOGIN_SIGNAL_SELECTORS,
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
    canonicalize_linkedin_job_url,
    is_linkedin_auth_checkpoint,
    linkedin_job_id_from_url,
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

_COUNTRY_FOCUSED_LOCATION_TERMS = {
    "south korea": ('"South Korea"', 'Seoul', 'Korea'),
    "japan": ('"Japón"', 'Tokyo', '日本'),
}
_COUNTRY_FOCUSED_ROLE_TERMS = (
    '"AI Engineer"',
    '"Machine Learning Engineer"',
    '"Data Scientist"',
    '"Applied AI Engineer"',
    '"AI Architect"',
    '"LLM Engineer"',
)


def _country_focused_keywords(location: LinkedInLocationResolution) -> str:
    location_terms = _COUNTRY_FOCUSED_LOCATION_TERMS.get(
        location.canonical_label.casefold(),
        (),
    )
    if not location_terms:
        return ""
    return " OR ".join(
        f"{location_term} {role_term}"
        for location_term in location_terms
        for role_term in _COUNTRY_FOCUSED_ROLE_TERMS
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


def minutes_to_tpr(minutes: int) -> str:
    """Return LinkedIn's f_TPR value for a positive minute window."""
    if isinstance(minutes, bool) or not isinstance(minutes, int):
        raise ValueError("minutes must be a positive integer")
    if minutes <= 0:
        raise ValueError("minutes must be greater than zero")
    return f"r{minutes * 60}"


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
                "f_TPR": minutes_to_tpr(1440),
                "sortBy": "DD",
            }
            if location:
                params["location"] = location
            if resolved_location.geo_id:
                params["geoId"] = resolved_location.geo_id
            url = f"https://www.linkedin.com/jobs/search/?{urlencode(params)}"
            label = f"{query_name} @ {location}" if location else query_name
            results.append((label, validate_linkedin_jobs_url(url)))
    for resolved_location in unique_locations:
        keywords = _country_focused_keywords(resolved_location)
        if not keywords:
            continue
        location = resolved_location.canonical_label
        params = {
            "keywords": keywords,
            "f_TPR": minutes_to_tpr(1440),
            "sortBy": "DD",
            "location": location,
        }
        if resolved_location.geo_id:
            params["geoId"] = resolved_location.geo_id
        url = f"https://www.linkedin.com/jobs/search/?{urlencode(params)}"
        results.append(
            (
                f"Country-focused AI/Data @ {location}",
                validate_linkedin_jobs_url(url),
            )
        )
    return results


_SEARCH_LIST_AREA_SELECTORS = (
    ".jobs-search-results-list",
    ".scaffold-layout__list",
    "[role='listbox']",
)
_SEMANTIC_RESULT_WRAPPER_SELECTORS = (
    "[data-job-id]",
    "[data-occludable-job-id]",
    "[role='listitem']",
    "[role='option']",
    "li",
)
_SEARCH_RESULT_SIGNAL_SELECTORS = tuple(
    f"{area} {wrapper} a[href*='/jobs/view/']"
    for area in _SEARCH_LIST_AREA_SELECTORS
    for wrapper in _SEMANTIC_RESULT_WRAPPER_SELECTORS
)
_SEARCH_RESULT_CARD_COUNT_SELECTORS = tuple(
    f"{area} {wrapper}"
    for area in _SEARCH_LIST_AREA_SELECTORS
    for wrapper in _SEMANTIC_RESULT_WRAPPER_SELECTORS
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
MIN_EXPECTED_DISCOVERY_CANDIDATES = 2
MAX_DISCOVERY_SCROLLS = 3
DISCOVERY_SCROLL_WAIT_MS = 300
_SEARCH_HYDRATION_MAX_SCROLLS = MAX_DISCOVERY_SCROLLS
_QUERY_BACKOFF_BASE_MS = 1000
_QUERY_BACKOFF_MAX_MS = 3000
_QUERY_PAGE_RECOVERY_THRESHOLD = 2
_QUERY_NETWORK_CIRCUIT_THRESHOLD = 3
_DETAIL_NETWORK_CIRCUIT_THRESHOLD = 2
_HARD_MAX_TOTAL_QUERY_ATTEMPTS = 8
_QUERY_PROBE_TIMEOUT_MS = 15000
_HARD_MAX_QUERIES_PER_LOCATION = len(CONSOLIDATED_QUERY_PLANS) + 1
_MAX_SEARCH_HYDRATION_EVENTS_PER_QUERY = 40
_MAX_SEARCH_HYDRATION_COUNT = 10000
_MAX_SEARCH_TEXT_LENGTH = 2_000_000
_JOB_ID_PATTERN = re.compile(r"\b(\d{4,20})\b")
_JOB_POSTING_URN_PATTERN = re.compile(r"jobPosting:(\d{4,20})")
_BROAD_JOB_SIGNAL_SELECTORS = (
    "a[href*='/jobs/view/']",
    "[data-entity-urn*='jobPosting']",
    "[data-job-id]",
    "[data-occludable-job-id]",
)
_RESULTS_PANEL_SCROLL_SELECTORS = (
    ".jobs-search-results-list",
    ".scaffold-layout__list",
    "[role='listbox']",
)
_SCROLL_CONTAINER_PROBE_SCRIPT = """
node => {
  const jobSignalCount = node.querySelectorAll(
    "a[href*='/jobs/view/'], [data-entity-urn*='jobPosting'], [data-job-id], [data-occludable-job-id]"
  ).length;
  const scrollHeight = Math.max(0, Math.trunc(node.scrollHeight || 0));
  const clientHeight = Math.max(0, Math.trunc(node.clientHeight || 0));
  const scrollTop = Math.max(0, Math.trunc(node.scrollTop || 0));
  return {
    jobSignalCount,
    scrollHeight,
    clientHeight,
    scrollTop,
    scrollable: scrollHeight > clientHeight + 8
  };
}
"""
_SCROLL_CONTAINER_SCROLL_SCRIPT = """
node => {
  const before = Math.max(0, Math.trunc(node.scrollTop || 0));
  const scrollHeight = Math.max(0, Math.trunc(node.scrollHeight || 0));
  const clientHeight = Math.max(0, Math.trunc(node.clientHeight || 0));
  const delta = Math.min(clientHeight || 480, 480);
  if (typeof node.scrollBy === "function") {
    node.scrollBy({top: delta, behavior: "auto"});
  } else {
    node.scrollTop = before + delta;
  }
  const after = Math.max(0, Math.trunc(node.scrollTop || 0));
  return {
    scrollHeight,
    clientHeight,
    scrollTopBefore: before,
    scrollTopAfter: after
  };
}
"""
_UNKNOWN_SCROLLABLE_PROBE_SCRIPT = """
() => {
  const nodes = Array.from(document.querySelectorAll("body *"));
  const candidates = nodes
    .map((node) => {
      const scrollHeight = Math.max(0, Math.trunc(node.scrollHeight || 0));
      const clientHeight = Math.max(0, Math.trunc(node.clientHeight || 0));
      const scrollTop = Math.max(0, Math.trunc(node.scrollTop || 0));
      const jobSignalCount = node.querySelectorAll(
        "a[href*='/jobs/view/'], [data-entity-urn*='jobPosting'], [data-job-id], [data-occludable-job-id]"
      ).length;
      return {scrollHeight, clientHeight, scrollTop, jobSignalCount};
    })
    .filter((item) => item.scrollHeight > item.clientHeight + 8);
  candidates.sort((left, right) =>
    (right.jobSignalCount - left.jobSignalCount)
    || ((right.scrollHeight - right.clientHeight) - (left.scrollHeight - left.clientHeight))
  );
  const selected = candidates[0];
  if (!selected) {
    return {
      found: false,
      jobSignalCount: 0,
      scrollHeight: 0,
      clientHeight: 0,
      scrollTop: 0
    };
  }
  return {
    found: true,
    jobSignalCount: selected.jobSignalCount,
    scrollHeight: selected.scrollHeight,
    clientHeight: selected.clientHeight,
    scrollTop: selected.scrollTop
  };
}
"""
_UNKNOWN_SCROLLABLE_SCROLL_SCRIPT = """
() => {
  const nodes = Array.from(document.querySelectorAll("body *"));
  const candidates = nodes
    .map((node) => {
      const scrollHeight = Math.max(0, Math.trunc(node.scrollHeight || 0));
      const clientHeight = Math.max(0, Math.trunc(node.clientHeight || 0));
      const scrollTop = Math.max(0, Math.trunc(node.scrollTop || 0));
      const jobSignalCount = node.querySelectorAll(
        "a[href*='/jobs/view/'], [data-entity-urn*='jobPosting'], [data-job-id], [data-occludable-job-id]"
      ).length;
      return {node, scrollHeight, clientHeight, scrollTop, jobSignalCount};
    })
    .filter((item) => item.scrollHeight > item.clientHeight + 8);
  candidates.sort((left, right) =>
    (right.jobSignalCount - left.jobSignalCount)
    || ((right.scrollHeight - right.clientHeight) - (left.scrollHeight - left.clientHeight))
  );
  const selected = candidates[0];
  if (!selected) {
    return {
      found: false,
      jobSignalCount: 0,
      scrollHeight: 0,
      clientHeight: 0,
      scrollTopBefore: 0,
      scrollTopAfter: 0
    };
  }
  const before = Math.max(0, Math.trunc(selected.node.scrollTop || 0));
  const delta = Math.min(selected.clientHeight || 480, 480);
  if (typeof selected.node.scrollBy === "function") {
    selected.node.scrollBy({top: delta, behavior: "auto"});
  } else {
    selected.node.scrollTop = before + delta;
  }
  return {
    found: true,
    jobSignalCount: selected.jobSignalCount,
    scrollHeight: selected.scrollHeight,
    clientHeight: selected.clientHeight,
    scrollTopBefore: before,
    scrollTopAfter: Math.max(0, Math.trunc(selected.node.scrollTop || 0))
  };
}
"""


@dataclass
class _SearchNavigationState:
    active_source_url: str | None = None
    completed_at: float | None = None

    def invalidate_source(self) -> None:
        self.active_source_url = None


@dataclass(frozen=True)
class _SearchHydrationSample:
    card_count: int
    href_count: int
    empty_state_visible: bool
    auth_checkpoint_visible: bool
    body_text_length: int = 0
    main_text_length: int = 0
    all_anchor_count: int = 0
    jobs_href_count: int = 0
    jobs_view_href_count: int = 0
    li_count: int = 0
    article_count: int = 0
    job_urn_count: int = 0
    data_job_id_count: int = 0
    data_occludable_job_id_count: int = 0
    scrollable_container_count: int = 0
    frame_count: int = 0
    raw_signal_count: int = 0
    unique_candidate_count: int = 0
    candidate_count_before_scroll: int = 0
    candidate_count_after_scroll_1: int = 0
    candidate_count_after_scroll_2: int = 0
    candidate_count_after_scroll_3: int = 0
    selected_scroll_container: str = "none"
    scroll_height: int = 0
    client_height: int = 0
    scroll_top_before: int = 0
    scroll_top_after: int = 0


@dataclass(frozen=True)
class _SearchScrollMetrics:
    selected_scroll_container: str = "none"
    scroll_height: int = 0
    client_height: int = 0
    scroll_top_before: int = 0
    scroll_top_after: int = 0
    job_signal_count: int = 0


class LinkedInSearchHydrationDiagnosticsCollector:
    """Collect only safe count, boolean, elapsed, and enum search hydration data."""

    def __init__(self) -> None:
        self._events_by_query: dict[
            str,
            list[LinkedInSearchHydrationDiagnostic],
        ] = {}

    @property
    def events(self) -> list[LinkedInSearchHydrationDiagnostic]:
        return [
            event
            for events in self._events_by_query.values()
            for event in events
        ]

    def record(
        self,
        *,
        query: str,
        elapsed_ms: int,
        sample: _SearchHydrationSample | None = None,
        card_count: int = 0,
        href_count: int = 0,
        empty_state_visible: bool = False,
        auth_checkpoint_visible: bool = False,
        outcome: str,
    ) -> None:
        sample = sample or _SearchHydrationSample(
            card_count=card_count,
            href_count=href_count,
            empty_state_visible=empty_state_visible,
            auth_checkpoint_visible=auth_checkpoint_visible,
            unique_candidate_count=href_count,
            candidate_count_before_scroll=href_count,
        )
        event_query = LinkedInSearchHydrationDiagnostic(
            query=query,
            sequence=1,
            elapsed_ms=max(0, int(elapsed_ms)),
            card_count=_bounded_hydration_count(sample.card_count),
            href_count=_bounded_hydration_count(sample.href_count),
            empty_state_visible=bool(sample.empty_state_visible),
            auth_checkpoint_visible=bool(sample.auth_checkpoint_visible),
            outcome=outcome,
        ).query
        events = self._events_by_query.setdefault(event_query, [])
        event = LinkedInSearchHydrationDiagnostic(
            query=event_query,
            sequence=min(
                len(events) + 1,
                _MAX_SEARCH_HYDRATION_EVENTS_PER_QUERY,
            ),
            elapsed_ms=max(0, int(elapsed_ms)),
            card_count=_bounded_hydration_count(sample.card_count),
            href_count=_bounded_hydration_count(sample.href_count),
            empty_state_visible=bool(sample.empty_state_visible),
            auth_checkpoint_visible=bool(sample.auth_checkpoint_visible),
            body_text_length=_bounded_text_length(sample.body_text_length),
            main_text_length=_bounded_text_length(sample.main_text_length),
            all_anchor_count=_bounded_hydration_count(sample.all_anchor_count),
            jobs_href_count=_bounded_hydration_count(sample.jobs_href_count),
            jobs_view_href_count=_bounded_hydration_count(
                sample.jobs_view_href_count
            ),
            li_count=_bounded_hydration_count(sample.li_count),
            article_count=_bounded_hydration_count(sample.article_count),
            job_urn_count=_bounded_hydration_count(sample.job_urn_count),
            data_job_id_count=_bounded_hydration_count(
                sample.data_job_id_count
            ),
            data_occludable_job_id_count=_bounded_hydration_count(
                sample.data_occludable_job_id_count
            ),
            scrollable_container_count=_bounded_hydration_count(
                sample.scrollable_container_count
            ),
            frame_count=_bounded_hydration_count(sample.frame_count),
            raw_signal_count=_bounded_hydration_count(
                sample.raw_signal_count
            ),
            unique_candidate_count=_bounded_hydration_count(
                sample.unique_candidate_count
            ),
            candidate_count_before_scroll=_bounded_hydration_count(
                sample.candidate_count_before_scroll
            ),
            candidate_count_after_scroll_1=_bounded_hydration_count(
                sample.candidate_count_after_scroll_1
            ),
            candidate_count_after_scroll_2=_bounded_hydration_count(
                sample.candidate_count_after_scroll_2
            ),
            candidate_count_after_scroll_3=_bounded_hydration_count(
                sample.candidate_count_after_scroll_3
            ),
            selected_scroll_container=sample.selected_scroll_container,
            scroll_height=_bounded_text_length(sample.scroll_height),
            client_height=_bounded_text_length(sample.client_height),
            scroll_top_before=_bounded_text_length(sample.scroll_top_before),
            scroll_top_after=_bounded_text_length(sample.scroll_top_after),
            outcome=outcome,
        )
        if len(events) >= _MAX_SEARCH_HYDRATION_EVENTS_PER_QUERY:
            if outcome != "polling":
                events[-1] = event
            return
        events.append(event)


_ACTIVE_SEARCH_HYDRATION_DIAGNOSTICS: ContextVar[
    LinkedInSearchHydrationDiagnosticsCollector | None
] = ContextVar("linkedin_search_hydration_diagnostics", default=None)


def get_active_search_hydration_diagnostics() -> (
    LinkedInSearchHydrationDiagnosticsCollector | None
):
    return _ACTIVE_SEARCH_HYDRATION_DIAGNOSTICS.get()


@contextmanager
def search_hydration_diagnostics_context() -> Iterator[
    LinkedInSearchHydrationDiagnosticsCollector
]:
    collector = LinkedInSearchHydrationDiagnosticsCollector()
    token = _ACTIVE_SEARCH_HYDRATION_DIAGNOSTICS.set(collector)
    try:
        yield collector
    finally:
        _ACTIVE_SEARCH_HYDRATION_DIAGNOSTICS.reset(token)


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


def _callable_supports_keyword(callback: Callable[..., object], keyword: str) -> bool:
    try:
        parameters = inspect.signature(callback).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or (
            parameter.name == keyword
            and parameter.kind
            in {
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        )
        for parameter in parameters
    )


def _probe_linkedin_search_with_authenticated_request(
    session,
    *,
    source_url: str,
    now: datetime | None = None,
    allow_standalone_fallback: bool = False,
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
            parse_kwargs = {
                "source_url": validated_source_url,
                "now": now or datetime.now(timezone.utc),
            }
            if (
                allow_standalone_fallback
                and _callable_supports_keyword(
                    parse_jobs_html_with_diagnostics,
                    "allow_standalone_fallback",
                )
            ):
                parse_kwargs["allow_standalone_fallback"] = True
            records, diagnostics = parse_jobs_html_with_diagnostics(
                html,
                **parse_kwargs,
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
    query: str = "unknown",
    diagnostics: LinkedInSearchHydrationDiagnosticsCollector | None = None,
) -> str:
    if diagnostics is None and query != "unknown":
        diagnostics = get_active_search_hydration_diagnostics()
    elapsed_ms = 0
    poll_index = 0
    scroll_count = 0
    scroll_disabled = False
    reloaded = False
    candidate_count_before_scroll: int | None = None
    candidate_counts_after_scroll = [0, 0, 0]
    pending_scroll_unique_count: int | None = None
    last_scroll_metrics = _SearchScrollMetrics()
    while True:
        sample = _sample_search_hydration(page)
        if candidate_count_before_scroll is None:
            candidate_count_before_scroll = sample.unique_candidate_count
        if scroll_count > 0:
            candidate_counts_after_scroll[scroll_count - 1] = (
                sample.unique_candidate_count
            )
        if pending_scroll_unique_count is not None:
            unique_candidate_progressed = (
                sample.unique_candidate_count > pending_scroll_unique_count
            )
            scroll_position_changed = (
                last_scroll_metrics.scroll_top_after
                != last_scroll_metrics.scroll_top_before
            )
            if not unique_candidate_progressed and not scroll_position_changed:
                scroll_disabled = True
            pending_scroll_unique_count = None
        sample = _sample_with_scroll_progress(
            sample,
            candidate_count_before_scroll=candidate_count_before_scroll,
            candidate_counts_after_scroll=candidate_counts_after_scroll,
            scroll_metrics=last_scroll_metrics,
        )
        try:
            _raise_for_terminal_page_signal(page)
        except LinkedInAuthRequiredError:
            _record_search_hydration_diagnostic(
                diagnostics,
                query=query,
                elapsed_ms=elapsed_ms,
                sample=sample,
                outcome="auth_checkpoint",
            )
            raise
        except LinkedInBlockedError:
            _record_search_hydration_diagnostic(
                diagnostics,
                query=query,
                elapsed_ms=elapsed_ms,
                sample=sample,
                outcome="blocked",
            )
            raise
        except Exception:
            _record_search_hydration_diagnostic(
                diagnostics,
                query=query,
                elapsed_ms=elapsed_ms,
                sample=sample,
                outcome="failed",
            )
            raise
        if sample.unique_candidate_count >= MIN_EXPECTED_DISCOVERY_CANDIDATES:
            _record_search_hydration_diagnostic(
                diagnostics,
                query=query,
                elapsed_ms=elapsed_ms,
                sample=sample,
                outcome="results",
            )
            return "results"
        if sample.empty_state_visible:
            _record_search_hydration_diagnostic(
                diagnostics,
                query=query,
                elapsed_ms=elapsed_ms,
                sample=sample,
                outcome="empty",
            )
            return "empty"
        if elapsed_ms >= max_wait_ms:
            _record_search_hydration_diagnostic(
                diagnostics,
                query=query,
                elapsed_ms=elapsed_ms,
                sample=sample,
                outcome="timeout",
            )
            return "timeout"
        _record_search_hydration_diagnostic(
            diagnostics,
            query=query,
            elapsed_ms=elapsed_ms,
            sample=sample,
            outcome="polling",
        )
        if (
            not scroll_disabled
            and scroll_count < MAX_DISCOVERY_SCROLLS
            and sample.unique_candidate_count < MIN_EXPECTED_DISCOVERY_CANDIDATES
        ):
            last_scroll_metrics = _scroll_search_results_incrementally(page)
            if last_scroll_metrics.selected_scroll_container != "none":
                pending_scroll_unique_count = sample.unique_candidate_count
                scroll_count += 1
                scroll_wait_ms = min(
                    DISCOVERY_SCROLL_WAIT_MS,
                    max(0, max_wait_ms - elapsed_ms),
                )
                if scroll_wait_ms:
                    _safe_page_pause(page, scroll_wait_ms)
                    elapsed_ms += scroll_wait_ms
                continue
            scroll_disabled = True
        if (
            (scroll_disabled or scroll_count >= MAX_DISCOVERY_SCROLLS)
            and not reloaded
            and elapsed_ms < max_wait_ms
        ):
            reloaded = True
            try:
                page.reload(wait_until="domcontentloaded", timeout=30000)
            except Exception:
                pass
        interval_ms = _HYDRATION_POLL_INTERVALS_MS[
            min(poll_index, len(_HYDRATION_POLL_INTERVALS_MS) - 1)
        ]
        wait_ms = min(interval_ms, max_wait_ms - elapsed_ms)
        _safe_page_pause(page, wait_ms)
        elapsed_ms += wait_ms
        poll_index += 1


def _record_search_hydration_diagnostic(
    diagnostics: LinkedInSearchHydrationDiagnosticsCollector | None,
    *,
    query: str,
    elapsed_ms: int,
    sample: _SearchHydrationSample,
    outcome: str,
) -> None:
    if diagnostics is None:
        return
    diagnostics.record(
        query=query,
        elapsed_ms=elapsed_ms,
        sample=sample,
        outcome=outcome,
    )


def _bounded_hydration_count(value: object) -> int:
    try:
        return min(_MAX_SEARCH_HYDRATION_COUNT, max(0, int(value)))
    except (TypeError, ValueError):
        return 0


def _bounded_text_length(value: object) -> int:
    try:
        return min(_MAX_SEARCH_TEXT_LENGTH, max(0, int(value)))
    except (TypeError, ValueError):
        return 0


def _sample_with_scroll_progress(
    sample: _SearchHydrationSample,
    *,
    candidate_count_before_scroll: int,
    candidate_counts_after_scroll: list[int],
    scroll_metrics: _SearchScrollMetrics,
) -> _SearchHydrationSample:
    after_counts = (candidate_counts_after_scroll + [0, 0, 0])[:3]
    return _SearchHydrationSample(
        card_count=sample.card_count,
        href_count=sample.href_count,
        empty_state_visible=sample.empty_state_visible,
        auth_checkpoint_visible=sample.auth_checkpoint_visible,
        body_text_length=sample.body_text_length,
        main_text_length=sample.main_text_length,
        all_anchor_count=sample.all_anchor_count,
        jobs_href_count=sample.jobs_href_count,
        jobs_view_href_count=sample.jobs_view_href_count,
        li_count=sample.li_count,
        article_count=sample.article_count,
        job_urn_count=sample.job_urn_count,
        data_job_id_count=sample.data_job_id_count,
        data_occludable_job_id_count=sample.data_occludable_job_id_count,
        scrollable_container_count=sample.scrollable_container_count,
        frame_count=sample.frame_count,
        raw_signal_count=sample.raw_signal_count,
        unique_candidate_count=sample.unique_candidate_count,
        candidate_count_before_scroll=candidate_count_before_scroll,
        candidate_count_after_scroll_1=after_counts[0],
        candidate_count_after_scroll_2=after_counts[1],
        candidate_count_after_scroll_3=after_counts[2],
        selected_scroll_container=scroll_metrics.selected_scroll_container,
        scroll_height=scroll_metrics.scroll_height,
        client_height=scroll_metrics.client_height,
        scroll_top_before=scroll_metrics.scroll_top_before,
        scroll_top_after=scroll_metrics.scroll_top_after,
    )


def _safe_locator_count(page, selector: str) -> int:
    try:
        return _bounded_hydration_count(page.locator(selector).count())
    except Exception:
        return 0


def _safe_locator_text_length(page, selector: str) -> int:
    try:
        text = str(page.locator(selector).first.inner_text(timeout=250) or "")
        return _bounded_text_length(len(text))
    except Exception:
        return 0


def _safe_frame_count(page) -> int:
    try:
        return _bounded_hydration_count(len(getattr(page, "frames", []) or []))
    except Exception:
        return 0


def _safe_scrollable_container_count(page) -> int:
    try:
        count = page.evaluate(
            """
            () => Array.from(document.querySelectorAll("body *"))
              .filter((node) =>
                Math.max(0, Math.trunc(node.scrollHeight || 0))
                > Math.max(0, Math.trunc(node.clientHeight || 0)) + 8
              ).length
            """
        )
        return _bounded_hydration_count(count)
    except Exception:
        return 0


def _auth_checkpoint_visible(page) -> bool:
    current_url = str(getattr(page, "url", "") or "")
    if is_linkedin_auth_checkpoint(current_url):
        return True
    return any(
        _locator_has_signal(page, selector)
        for selector in _LOGIN_SIGNAL_SELECTORS
    )


def _job_id_from_attribute_value(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    urn_match = _JOB_POSTING_URN_PATTERN.search(text)
    if urn_match:
        return urn_match.group(1)
    generic_match = _JOB_ID_PATTERN.search(text)
    return generic_match.group(1) if generic_match else ""


def _add_job_ids_from_attribute(
    page,
    selector: str,
    attribute: str,
    job_ids: set[str],
) -> None:
    try:
        matches = page.locator(selector)
        for index in range(matches.count()):
            value = matches.nth(index).get_attribute(attribute)
            job_id = _job_id_from_attribute_value(value)
            if job_id:
                job_ids.add(job_id)
    except Exception:
        return


def _collect_broad_search_job_ids(page) -> set[str]:
    job_ids: set[str] = set()
    try:
        matches = page.locator("a[href*='/jobs/view/']")
        for index in range(matches.count()):
            href = str(matches.nth(index).get_attribute("href") or "").strip()
            if not href:
                continue
            try:
                canonical = canonicalize_linkedin_job_url(
                    urljoin("https://www.linkedin.com", href)
                )
            except ValueError:
                continue
            job_id = linkedin_job_id_from_url(canonical)
            if job_id:
                job_ids.add(job_id)
    except Exception:
        pass
    _add_job_ids_from_attribute(
        page,
        "[data-entity-urn*='jobPosting']",
        "data-entity-urn",
        job_ids,
    )
    _add_job_ids_from_attribute(page, "[data-job-id]", "data-job-id", job_ids)
    _add_job_ids_from_attribute(
        page,
        "[data-occludable-job-id]",
        "data-occludable-job-id",
        job_ids,
    )
    return job_ids


def _sample_search_hydration(page) -> _SearchHydrationSample:
    semantic_job_ids: set[str] = set()
    for selector in _SEARCH_RESULT_SIGNAL_SELECTORS:
        try:
            matches = page.locator(selector)
            for index in range(matches.count()):
                href = str(
                    matches.nth(index).get_attribute("href") or ""
                ).strip()
                if not href:
                    continue
                try:
                    canonical = canonicalize_linkedin_job_url(
                        urljoin("https://www.linkedin.com", href)
                    )
                except ValueError:
                    continue
                job_id = linkedin_job_id_from_url(canonical)
                if job_id:
                    semantic_job_ids.add(job_id)
        except Exception:
            continue
    broad_job_ids = _collect_broad_search_job_ids(page)
    jobs_view_href_count = _safe_locator_count(page, "a[href*='/jobs/view/']")
    job_urn_count = _safe_locator_count(
        page,
        "[data-entity-urn*='jobPosting']",
    )
    data_job_id_count = _safe_locator_count(page, "[data-job-id]")
    data_occludable_job_id_count = _safe_locator_count(
        page,
        "[data-occludable-job-id]",
    )
    raw_signal_count = (
        jobs_view_href_count
        + job_urn_count
        + data_job_id_count
        + data_occludable_job_id_count
    )
    card_count = max(
        (
            _safe_locator_count(page, selector)
            for selector in _SEARCH_RESULT_CARD_COUNT_SELECTORS
        ),
        default=0,
    )
    empty_state_visible = any(
        _locator_has_signal(page, selector)
        for selector in _EMPTY_RESULTS_SELECTORS
    )
    blocked_visible = any(
        _locator_has_signal(page, selector)
        for selector in _BLOCK_SIGNAL_SELECTORS
    )
    return _SearchHydrationSample(
        card_count=card_count,
        href_count=len(semantic_job_ids),
        empty_state_visible=empty_state_visible,
        auth_checkpoint_visible=_auth_checkpoint_visible(page) or blocked_visible,
        body_text_length=_safe_locator_text_length(page, "body"),
        main_text_length=_safe_locator_text_length(page, "main"),
        all_anchor_count=_safe_locator_count(page, "a"),
        jobs_href_count=_safe_locator_count(page, "a[href*='/jobs/']"),
        jobs_view_href_count=jobs_view_href_count,
        li_count=_safe_locator_count(page, "li"),
        article_count=_safe_locator_count(page, "article"),
        job_urn_count=job_urn_count,
        data_job_id_count=data_job_id_count,
        data_occludable_job_id_count=data_occludable_job_id_count,
        scrollable_container_count=_safe_scrollable_container_count(page),
        frame_count=_safe_frame_count(page),
        raw_signal_count=raw_signal_count,
        unique_candidate_count=len(broad_job_ids),
    )


def _has_semantic_search_result(page) -> bool:
    """Accept only parseable job links nested in stable list/card wrappers."""

    for selector in _SEARCH_RESULT_SIGNAL_SELECTORS:
        try:
            matches = page.locator(selector)
            for index in range(matches.count()):
                href = str(
                    matches.nth(index).get_attribute("href") or ""
                ).strip()
                if not href:
                    continue
                try:
                    canonical = canonicalize_linkedin_job_url(
                        urljoin("https://www.linkedin.com", href)
                    )
                except ValueError:
                    continue
                if linkedin_job_id_from_url(canonical):
                    return True
        except Exception:
            continue
    return False


def _safe_int_from_mapping(mapping: object, key: str) -> int:
    if not isinstance(mapping, dict):
        return 0
    return _bounded_text_length(mapping.get(key, 0))


def _scroll_container_priority(container: str) -> int:
    return {
        "results_panel": 3,
        "main": 2,
        "unknown_scrollable": 1,
        "none": 0,
    }.get(container, 0)


def _candidate_scroll_containers(page) -> list[tuple[object, _SearchScrollMetrics]]:
    candidates: list[tuple[object, _SearchScrollMetrics]] = []
    for label, selectors in (
        ("results_panel", _RESULTS_PANEL_SCROLL_SELECTORS),
        ("main", ("main",)),
    ):
        for selector in selectors:
            try:
                matches = page.locator(selector)
                for index in range(min(matches.count(), 8)):
                    locator = matches.nth(index)
                    metrics = locator.evaluate(_SCROLL_CONTAINER_PROBE_SCRIPT)
                    if not isinstance(metrics, dict) or not metrics.get(
                        "scrollable"
                    ):
                        continue
                    candidates.append(
                        (
                            locator,
                            _SearchScrollMetrics(
                                selected_scroll_container=label,
                                scroll_height=_safe_int_from_mapping(
                                    metrics,
                                    "scrollHeight",
                                ),
                                client_height=_safe_int_from_mapping(
                                    metrics,
                                    "clientHeight",
                                ),
                                scroll_top_before=_safe_int_from_mapping(
                                    metrics,
                                    "scrollTop",
                                ),
                                scroll_top_after=_safe_int_from_mapping(
                                    metrics,
                                    "scrollTop",
                                ),
                                job_signal_count=_bounded_hydration_count(
                                    metrics.get("jobSignalCount", 0)
                                ),
                            ),
                        )
                    )
            except Exception:
                continue
    return candidates


def _choose_scroll_container(
    candidates: list[tuple[object, _SearchScrollMetrics]],
) -> tuple[object, _SearchScrollMetrics] | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item[1].job_signal_count,
            _scroll_container_priority(item[1].selected_scroll_container),
            item[1].scroll_height - item[1].client_height,
        ),
    )


def _scroll_unknown_scrollable_container(page) -> _SearchScrollMetrics:
    try:
        metrics = page.evaluate(_UNKNOWN_SCROLLABLE_SCROLL_SCRIPT)
    except Exception:
        return _SearchScrollMetrics()
    if not isinstance(metrics, dict) or not metrics.get("found"):
        return _SearchScrollMetrics()
    return _SearchScrollMetrics(
        selected_scroll_container="unknown_scrollable",
        scroll_height=_safe_int_from_mapping(metrics, "scrollHeight"),
        client_height=_safe_int_from_mapping(metrics, "clientHeight"),
        scroll_top_before=_safe_int_from_mapping(metrics, "scrollTopBefore"),
        scroll_top_after=_safe_int_from_mapping(metrics, "scrollTopAfter"),
        job_signal_count=_bounded_hydration_count(
            metrics.get("jobSignalCount", 0)
        ),
    )


def _unknown_scrollable_container_metrics(page) -> _SearchScrollMetrics:
    try:
        metrics = page.evaluate(_UNKNOWN_SCROLLABLE_PROBE_SCRIPT)
    except Exception:
        return _SearchScrollMetrics()
    if not isinstance(metrics, dict) or not metrics.get("found"):
        return _SearchScrollMetrics()
    scroll_top = _safe_int_from_mapping(metrics, "scrollTop")
    return _SearchScrollMetrics(
        selected_scroll_container="unknown_scrollable",
        scroll_height=_safe_int_from_mapping(metrics, "scrollHeight"),
        client_height=_safe_int_from_mapping(metrics, "clientHeight"),
        scroll_top_before=scroll_top,
        scroll_top_after=scroll_top,
        job_signal_count=_bounded_hydration_count(
            metrics.get("jobSignalCount", 0)
        ),
    )


def _scroll_search_results_incrementally(page) -> _SearchScrollMetrics:
    """Nudge only the current search list; never navigate between routes."""

    selected = _choose_scroll_container(_candidate_scroll_containers(page))
    unknown_metrics = _unknown_scrollable_container_metrics(page)
    if selected is None:
        if unknown_metrics.selected_scroll_container == "none":
            return _SearchScrollMetrics()
        return _scroll_unknown_scrollable_container(page)
    locator, selected_metrics = selected
    if (
        unknown_metrics.selected_scroll_container != "none"
        and unknown_metrics.job_signal_count > selected_metrics.job_signal_count
    ):
        return _scroll_unknown_scrollable_container(page)
    try:
        metrics = locator.evaluate(_SCROLL_CONTAINER_SCROLL_SCRIPT)
    except Exception:
        return _SearchScrollMetrics()
    if not isinstance(metrics, dict):
        return _SearchScrollMetrics()
    return _SearchScrollMetrics(
        selected_scroll_container=selected_metrics.selected_scroll_container,
        scroll_height=_safe_int_from_mapping(metrics, "scrollHeight"),
        client_height=_safe_int_from_mapping(metrics, "clientHeight"),
        scroll_top_before=_safe_int_from_mapping(metrics, "scrollTopBefore"),
        scroll_top_after=_safe_int_from_mapping(metrics, "scrollTopAfter"),
        job_signal_count=selected_metrics.job_signal_count,
    )


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
