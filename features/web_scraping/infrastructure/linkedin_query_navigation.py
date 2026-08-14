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
from urllib.parse import urlencode, urljoin, urlsplit

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
    _clean_posted_at_text,
    _parse_linkedin_jobs_html_with_diagnostics,
    parse_linkedin_relative_time,
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
MAX_ROW_ACTIVATIONS_PER_QUERY = 20
DISCOVERY_SCROLL_WAIT_MS = 300
ROW_ACTIVATION_WAIT_MS = 1200
ROW_ACTIVATION_POLL_MS = 100
ROW_METADATA_WAIT_MS = 1600
ROW_METADATA_POLL_MS = 200
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
_ACTIVE_DETAIL_METADATA_DIAGNOSTIC: ContextVar[dict[str, object]] = ContextVar(
    "linkedin_active_detail_metadata_diagnostic",
    default={},
)


def latest_active_detail_metadata_diagnostic() -> dict[str, object]:
    return dict(_ACTIVE_DETAIL_METADATA_DIAGNOSTIC.get({}))


_JOB_ID_PATTERN = re.compile(r"\b(\d{1,20})\b")
_JOB_POSTING_URN_PATTERN = re.compile(r"jobPosting:(\d{4,20})")
_ROW_ALLOWLISTED_ATTRIBUTES = (
    "data-job-id",
    "data-occludable-job-id",
    "data-entity-urn",
)
_ROW_SELECTORS = tuple(
    f"{area} {wrapper}"
    for area in _SEARCH_LIST_AREA_SELECTORS
    for wrapper in _SEMANTIC_RESULT_WRAPPER_SELECTORS
)
_DETAIL_IDENTITY_SELECTORS = (
    "#job-details",
    ".jobs-search__job-details--container",
    ".jobs-details",
    ".job-details-jobs-unified-top-card",
    ".jobs-unified-top-card",
)
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
_STRUCTURAL_DISCOVERY_SCRIPT = r"""
() => {
  const viewportWidth = Math.max(0, Math.trunc(window.innerWidth || 0));
  const viewportHeight = Math.max(0, Math.trunc(window.innerHeight || 0));
  const nodes = Array.from(document.querySelectorAll("body *")).slice(0, 2500);
  const jobHref = "a[href*='/jobs/view/']";
  const jobSignal = "a[href*='/jobs/view/'], [data-entity-urn*='jobPosting'], [data-job-id], [data-occludable-job-id]";
  const interactive = "a, button, [role='button'], [tabindex], input, select, textarea";
  const boundedPath = (node) => {
    const parts = [];
    let current = node;
    while (current && current !== document.body && parts.length < 6) {
      const parent = current.parentElement;
      if (!parent) break;
      const sameTag = Array.from(parent.children).filter((child) => child.tagName === current.tagName);
      parts.unshift(`${String(current.tagName || '').toLowerCase()}:${Math.max(0, sameTag.indexOf(current))}`);
      current = parent;
    }
    return parts;
  };
  const safeAttributes = (node) => {
    const values = {};
    for (const name of ["data-job-id", "data-occludable-job-id", "data-entity-urn"]) {
      if (node.hasAttribute && node.hasAttribute(name)) values[name] = String(node.getAttribute(name) || '').slice(0, 120);
    }
    return values;
  };
  const safeHrefFragment = (node) => {
    const link = node.querySelector && node.querySelector(jobHref);
    const href = link ? String(link.getAttribute('href') || '') : String(node.getAttribute && node.getAttribute('href') || '');
    const match = href.match(/\/jobs\/view\/([^/?#]+)/);
    return match ? match[1].replace(/[^A-Za-z0-9_-]/g, '').slice(0, 80) : '';
  };
  const visible = (rect) => rect.width > 0 && rect.height > 0 && rect.bottom >= 0 && rect.right >= 0 && rect.top <= viewportHeight && rect.left <= viewportWidth;
  const structuralRow = (candidate) => {
    const candidateRect = candidate.getBoundingClientRect();
    if (!visible(candidateRect)) return false;
    const candidateStyle = window.getComputedStyle(candidate);
    const candidateRole = String(candidate.getAttribute && candidate.getAttribute('role') || '').toLowerCase();
    const candidateTabindex = candidate.hasAttribute && candidate.hasAttribute('tabindex') ? Number(candidate.getAttribute('tabindex')) : -1;
    const descendantAnchor = Boolean(candidate.querySelector && candidate.querySelector('a'));
    const pointer = candidateStyle && candidateStyle.cursor === 'pointer';
    const cardGeometry = candidateRect.width >= 180 && candidateRect.height >= 36 && candidateRect.height <= 360;
    return descendantAnchor || candidateRole === 'button' || (Number.isFinite(candidateTabindex) && candidateTabindex >= 0) || pointer || (cardGeometry && candidate.querySelectorAll && candidate.querySelectorAll(interactive).length > 0);
  };
  const descriptor = (node, index) => {
    const rect = node.getBoundingClientRect();
    const scrollHeight = Math.max(0, Math.trunc(node.scrollHeight || 0));
    const clientHeight = Math.max(0, Math.trunc(node.clientHeight || 0));
    const scrollTop = Math.max(0, Math.trunc(node.scrollTop || 0));
    const anchors = node.querySelectorAll ? node.querySelectorAll('a').length : 0;
    const jobLinks = node.querySelectorAll ? node.querySelectorAll(jobHref).length : 0;
    const interactiveCount = node.querySelectorAll ? Array.from(node.querySelectorAll(interactive)).filter((item) => visible(item.getBoundingClientRect())).length : 0;
    const structuralRows = node.querySelectorAll ? Array.from(node.querySelectorAll('*')).filter(structuralRow).length : 0;
    const rowCandidates = Math.max(node.querySelectorAll ? node.querySelectorAll(jobSignal).length : 0, structuralRows);
    const rows = Array.from(node.querySelectorAll ? node.querySelectorAll('*') : []).slice(0, 300)
      .filter((candidate) => candidate !== node)
      .map((candidate) => candidate.getBoundingClientRect())
      .filter((candidateRect) => candidateRect.width > 0 && candidateRect.height > 0 && candidateRect.height <= 360);
    const yBands = rows.map((item) => Math.round(item.top / 10)).sort((left, right) => left - right);
    let regularVerticalRepetition = 0;
    for (let position = 1; position < yBands.length; position += 1) {
      if (Math.abs(yBands[position] - yBands[position - 1]) >= 2) regularVerticalRepetition += 1;
    }
    const containsMain = Boolean(node.querySelector && node.querySelector('main, [role="main"]'));
    const isMain = Boolean(node.matches && node.matches('main, [role="main"]')) || containsMain;
    const fullPageContainer = rect.left <= 5 && rect.width >= viewportWidth * 0.9 && rect.height >= viewportHeight * 0.8;
    const isGlobal = Boolean(node.matches && node.matches('html, body, header, nav, footer, [role="banner"], [role="navigation"], [role="contentinfo"]')) || fullPageContainer;
    return {
      container_index: index,
      x: Math.max(-100000, Math.trunc(rect.left)),
      y: Math.max(-100000, Math.trunc(rect.top)),
      width: Math.max(0, Math.trunc(rect.width)),
      height: Math.max(0, Math.trunc(rect.height)),
      scrollHeight,
      clientHeight,
      scrollTop,
      row_candidate_count: Math.min(10000, Math.max(0, rowCandidates)),
      row_repetition: Math.min(10000, Math.max(0, regularVerticalRepetition)),
      anchor_count: Math.min(10000, Math.max(0, anchors)),
      interactive_count: Math.min(10000, Math.max(0, interactiveCount)),
      visible_row_count: Math.min(10000, Math.max(0, rows.length)),
      job_link_count: Math.min(10000, Math.max(0, jobLinks)),
      detail_or_main: isMain,
      header_nav_footer_global: isGlobal,
      scrollable: scrollHeight > clientHeight + 8,
    };
  };
  const rowDescriptor = (node, index) => {
    const rect = node.getBoundingClientRect();
    const style = window.getComputedStyle(node);
    const role = String(node.getAttribute && node.getAttribute('role') || '').toLowerCase();
    const isAnchor = String(node.tagName || '').toLowerCase() === 'a';
    const tabindex = node.hasAttribute && node.hasAttribute('tabindex') ? Number(node.getAttribute('tabindex')) : -1;
    const hasDescendantAnchor = Boolean(node.querySelector && node.querySelector('a'));
    const hasRoleButton = role === 'button';
    const hasTabindex = Number.isFinite(tabindex) && tabindex >= 0;
    const hasPointerCursor = style && style.cursor === 'pointer';
    const interactiveCount = node.querySelectorAll ? Array.from(node.querySelectorAll(interactive)).filter((item) => visible(item.getBoundingClientRect())).length : 0;
    const clickableCardGeometry = rect.width >= 180 && rect.height >= 36 && rect.height <= 360 && interactiveCount > 0;
    const isVisible = visible(rect);
    const isCandidate = isVisible && (isAnchor || hasDescendantAnchor || hasRoleButton || hasTabindex || hasPointerCursor || clickableCardGeometry);
    return {
      container_index: index,
      x: Math.max(-100000, Math.trunc(rect.left)),
      y: Math.max(-100000, Math.trunc(rect.top)),
      width: Math.max(0, Math.trunc(rect.width)),
      height: Math.max(0, Math.trunc(rect.height)),
      anchor_count: Math.min(10000, Math.max(0, node.querySelectorAll ? node.querySelectorAll('a').length : 0)),
      interactive_count: Math.min(10000, Math.max(0, interactiveCount)),
      has_descendant_anchor: hasDescendantAnchor,
      role_button: hasRoleButton,
      tabindex: hasTabindex,
      cursor_pointer: hasPointerCursor,
      clickable_card_geometry: clickableCardGeometry,
      visible: isVisible,
      row_candidate: isCandidate,
      href_fragment: safeHrefFragment(node),
      allowlisted_attributes: safeAttributes(node),
      structural_path: boundedPath(node),
    };
  };
  const containers = nodes
    .map((node, index) => descriptor(node, index))
    .filter((item) => item.scrollable || item.row_candidate_count > 0)
    .slice(0, 200);
  const rows = nodes
    .map((node, index) => rowDescriptor(node, index))
    .filter((item) => item.row_candidate)
    .slice(0, 500);
  return {viewport_width: viewportWidth, viewport_height: viewportHeight, containers, rows};
}
"""
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
    row_activation_count: int = 0
    row_activation_success_count: int = 0
    row_activation_no_change_count: int = 0
    row_activation_no_job_id_count: int = 0
    row_activation_duplicate_count: int = 0
    row_activation_scroll_count: int = 0
    selected_row_container_score: int = 0
    row_candidate_count: int = 0
    row_interactive_count: int = 0
    row_job_ids_resolved: int = 0
    row_activation_stop_reason: str = "none"


@dataclass(frozen=True)
class _SearchScrollMetrics:
    selected_scroll_container: str = "none"
    scroll_height: int = 0
    client_height: int = 0
    scroll_top_before: int = 0
    scroll_top_after: int = 0
    job_signal_count: int = 0
    selected_row_container_score: int = 0
    row_candidate_count: int = 0
    row_interactive_count: int = 0
    row_repetition: int = 0
    visible_row_count: int = 0


@dataclass(frozen=True)
class LinkedInDetailIdentity:
    """Bounded identity signals used to verify a row activated the panel."""

    job_id: str = ""
    canonical_detail_href: str = ""
    allowlisted_attributes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class LinkedInRowActivationOutcome:
    outcome: str
    signature: tuple[object, ...]
    job_id: str = ""
    canonical_detail_href: str = ""
    date_detected: bool = False
    date_verified: bool = False
    date_within_24_hours: bool = False


@dataclass(frozen=True)
class LinkedInRowDiscoveryResult:
    records: list[LinkedInVacancyRecord]
    outcomes: list[LinkedInRowActivationOutcome]
    scroll_count: int = 0
    structural_rows_found: int = 0
    stop_reason: str = "none"
    selected_row_container_score: int = 0
    row_interactive_count: int = 0

    @property
    def activation_count(self) -> int:
        return len(self.outcomes)

    @property
    def success_count(self) -> int:
        return sum(item.outcome == "row_activation_success" for item in self.outcomes)

    @property
    def no_change_count(self) -> int:
        return sum(item.outcome == "row_activation_no_change" for item in self.outcomes)

    @property
    def no_job_id_count(self) -> int:
        return sum(item.outcome == "row_activation_no_job_id" for item in self.outcomes)

    @property
    def duplicate_count(self) -> int:
        return sum(item.outcome == "row_activation_duplicate" for item in self.outcomes)

    @property
    def job_ids_resolved(self) -> int:
        return len({item.job_id for item in self.outcomes if item.job_id})


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
        row_activation_count: int = 0,
        row_activation_success_count: int = 0,
        row_activation_no_change_count: int = 0,
        row_activation_no_job_id_count: int = 0,
        row_activation_duplicate_count: int = 0,
        row_activation_scroll_count: int = 0,
        selected_row_container_score: int = 0,
        row_candidate_count: int = 0,
        row_interactive_count: int = 0,
        row_job_ids_resolved: int = 0,
        row_activation_stop_reason: str = "none",
        outcome: str,
    ) -> None:
        sample = sample or _SearchHydrationSample(
            card_count=card_count,
            href_count=href_count,
            empty_state_visible=empty_state_visible,
            auth_checkpoint_visible=auth_checkpoint_visible,
            unique_candidate_count=href_count,
            candidate_count_before_scroll=href_count,
            row_activation_count=row_activation_count,
            row_activation_success_count=row_activation_success_count,
            row_activation_no_change_count=row_activation_no_change_count,
            row_activation_no_job_id_count=row_activation_no_job_id_count,
            row_activation_duplicate_count=row_activation_duplicate_count,
            row_activation_scroll_count=row_activation_scroll_count,
            selected_row_container_score=selected_row_container_score,
            row_candidate_count=row_candidate_count,
            row_interactive_count=row_interactive_count,
            row_job_ids_resolved=row_job_ids_resolved,
            row_activation_stop_reason=row_activation_stop_reason,
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
            row_activation_count=_bounded_hydration_count(sample.row_activation_count),
            row_activation_success_count=_bounded_hydration_count(
                sample.row_activation_success_count
            ),
            row_activation_no_change_count=_bounded_hydration_count(
                sample.row_activation_no_change_count
            ),
            row_activation_no_job_id_count=_bounded_hydration_count(
                sample.row_activation_no_job_id_count
            ),
            row_activation_duplicate_count=_bounded_hydration_count(
                sample.row_activation_duplicate_count
            ),
            row_activation_scroll_count=min(3, _bounded_hydration_count(
                sample.row_activation_scroll_count
            )),
            selected_row_container_score=_bounded_score(
                sample.selected_row_container_score
            ),
            row_candidate_count=_bounded_hydration_count(sample.row_candidate_count),
            row_interactive_count=_bounded_hydration_count(sample.row_interactive_count),
            row_job_ids_resolved=_bounded_hydration_count(sample.row_job_ids_resolved),
            row_activation_stop_reason=_safe_enum_label(
                sample.row_activation_stop_reason,
                max_length=80,
            ),
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


def _safe_enum_label(value: object, *, max_length: int = 80) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value or "none")).strip("_")
    return label[:max_length] or "none"


def _bounded_score(value: object) -> int:
    try:
        return min(20, max(-20, int(value)))
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
        row_activation_count=sample.row_activation_count,
        row_activation_success_count=sample.row_activation_success_count,
        row_activation_no_change_count=sample.row_activation_no_change_count,
        row_activation_no_job_id_count=sample.row_activation_no_job_id_count,
        row_activation_duplicate_count=sample.row_activation_duplicate_count,
        row_activation_scroll_count=sample.row_activation_scroll_count,
        selected_row_container_score=scroll_metrics.selected_row_container_score or sample.selected_row_container_score,
        row_candidate_count=scroll_metrics.row_candidate_count or sample.row_candidate_count,
        row_interactive_count=scroll_metrics.row_interactive_count or sample.row_interactive_count,
        row_job_ids_resolved=sample.row_job_ids_resolved,
        row_activation_stop_reason=sample.row_activation_stop_reason,
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


def _safe_href_fragment(value: object) -> str:
    """Return only the bounded path fragment used in an ephemeral row key."""
    href = str(value or "").strip()
    if not href:
        return ""
    try:
        path = urlsplit(urljoin("https://www.linkedin.com", href)).path
    except Exception:
        return ""
    match = re.search(r"/jobs/view/([^/?#]*)", path)
    if not match:
        return ""
    return re.sub(r"[^A-Za-z0-9_-]", "", match.group(1))[:80]


def _safe_row_attributes(locator) -> tuple[tuple[str, str], ...]:
    values: list[tuple[str, str]] = []
    for name in _ROW_ALLOWLISTED_ATTRIBUTES:
        try:
            value = str(locator.get_attribute(name) or "").strip()
        except Exception:
            value = ""
        if value:
            values.append((name, value[:120]))
    return tuple(values)


def _row_container_index(locator) -> int:
    try:
        value = locator.evaluate(
            "node => Array.from(document.querySelectorAll('body *')).indexOf(node)"
        )
        return max(0, min(100000, int(value)))
    except Exception:
        return 0


def _row_signature(locator) -> tuple[object, ...]:
    href = ""
    try:
        href = str(locator.locator("a[href*='/jobs/view/']").first.get_attribute("href") or "")
    except Exception:
        try:
            href = str(locator.get_attribute("href") or "")
        except Exception:
            pass
    bbox = (0, 0, 0, 0)
    try:
        box = locator.bounding_box()
        if isinstance(box, dict):
            bbox = tuple(
                int(round(float(box.get(key, 0) or 0) / 10.0) * 10)
                for key in ("x", "y", "width", "height")
            )
    except Exception:
        pass
    path: tuple[str, ...] = ()
    try:
        raw_path = locator.evaluate(
            """node => {
              const parts = [];
              let current = node;
              while (current && current !== document.body && parts.length < 6) {
                const parent = current.parentElement;
                if (!parent) break;
                const same = Array.from(parent.children).filter(item => item.tagName === current.tagName);
                parts.unshift(`${String(current.tagName || '').toLowerCase()}:${Math.max(0, same.indexOf(current))}`);
                current = parent;
              }
              return parts;
            }"""
        )
        if isinstance(raw_path, list):
            path = tuple(str(item)[:40] for item in raw_path[:6])
    except Exception:
        pass
    # The DOM index is only a re-resolution hint. Identity comparisons use the
    # remaining ephemeral structural fields, so virtualization can move nodes.
    return (_row_container_index(locator), _safe_href_fragment(href), _safe_row_attributes(locator), bbox, path)


def _row_identity_key(signature: tuple[object, ...]) -> tuple[object, ...]:
    return tuple(signature[1:]) if len(signature) >= 5 else tuple(signature[1:])


def _descriptor_signature(descriptor: dict) -> tuple[object, ...]:
    bbox = tuple(
        int(round(_numeric_descriptor(descriptor.get(key)) / 10.0) * 10)
        for key in ("x", "y", "width", "height")
    )
    attributes = descriptor.get("allowlisted_attributes")
    if isinstance(attributes, dict):
        attrs = tuple(sorted((str(key), str(value)[:120]) for key, value in attributes.items() if key in _ROW_ALLOWLISTED_ATTRIBUTES and value))
    else:
        attrs = tuple()
    path = descriptor.get("structural_path")
    bounded_path = tuple(str(item)[:40] for item in path[:6]) if isinstance(path, list) else tuple()
    return (
        _numeric_descriptor(descriptor.get("container_index")),
        str(descriptor.get("href_fragment") or "")[:80],
        attrs,
        bbox,
        bounded_path,
    )


def _row_descriptor_is_compatible(candidate: dict, selected: dict) -> bool:
    candidate_y = _numeric_descriptor(candidate.get("y"))
    selected_y = _numeric_descriptor(selected.get("y"))
    candidate_h = max(1, _numeric_descriptor(candidate.get("height")))
    selected_h = max(1, _numeric_descriptor(selected.get("height")))
    candidate_x = _numeric_descriptor(candidate.get("x"))
    selected_x = _numeric_descriptor(selected.get("x"))
    vertical_overlap = min(candidate_y + candidate_h, selected_y + selected_h) - max(candidate_y, selected_y)
    same_band = vertical_overlap >= max(10, min(candidate_h, selected_h) // 2)
    width_compatible = (
        min(_numeric_descriptor(candidate.get("width")), _numeric_descriptor(selected.get("width")))
        >= max(120, int(max(_numeric_descriptor(candidate.get("width")), _numeric_descriptor(selected.get("width"))) * 0.55))
    )
    return same_band and width_compatible and abs(candidate_x - selected_x) <= max(40, min(candidate_h, selected_h))



def _descriptor_is_safe_activation_row(
    descriptor: dict,
    *,
    viewport_width: int = 0,
) -> bool:
    """Return True only for row-like targets inside the jobs results column.

    This is intentionally stricter than diagnostic row discovery: diagnostics may
    count broad structural candidates, but activation must not click global
    navigation, feed, profile, or the right-side detail panel.
    """
    x = _numeric_descriptor(descriptor.get("x"))
    y = _numeric_descriptor(descriptor.get("y"))
    width = _numeric_descriptor(descriptor.get("width"))
    height = _numeric_descriptor(descriptor.get("height"))
    if y < 70 or width < 160 or height < 36 or height > 260:
        return False
    if viewport_width > 0:
        if x >= int(viewport_width * 0.48):
            return False
        if width > int(viewport_width * 0.62):
            return False
    if descriptor.get("detail_or_main") or descriptor.get("header_nav_footer_global"):
        return False
    # A safe job href is always acceptable once geometry is in the left column.
    if str(descriptor.get("href_fragment") or ""):
        return True
    # Rows without href are provisional only when they look like left-panel list
    # cards, not giant layout wrappers or global navigation elements.
    rowish_geometry = 160 <= width <= 760 and 36 <= height <= 180
    interactive = any(
        bool(descriptor.get(key))
        for key in (
            "role_button",
            "tabindex",
            "cursor_pointer",
            "clickable_card_geometry",
            "has_descendant_anchor",
        )
    )
    return rowish_geometry and interactive

def _dedupe_structural_rows(descriptors: list[dict]) -> list[dict]:
    selected: list[dict] = []
    seen: set[tuple[object, ...]] = set()
    candidates = sorted(
        (item for item in descriptors if isinstance(item, dict) and item.get("row_candidate", True)),
        key=lambda item: (
            _numeric_descriptor(item.get("y")),
            _numeric_descriptor(item.get("x")),
            _numeric_descriptor(item.get("width")) * _numeric_descriptor(item.get("height")),
        ),
    )
    for candidate in candidates:
        signature = _descriptor_signature(candidate)
        identity = _row_identity_key(signature)
        if identity in seen:
            continue
        contained = False
        replace_indexes: list[int] = []
        for index, current in enumerate(selected):
            if _row_descriptor_is_compatible(candidate, current):
                candidate_area = _numeric_descriptor(candidate.get("width")) * _numeric_descriptor(candidate.get("height"))
                current_area = _numeric_descriptor(current.get("width")) * _numeric_descriptor(current.get("height"))
                if candidate_area >= current_area:
                    contained = True
                    break
                replace_indexes.append(index)
        if contained:
            continue
        if replace_indexes:
            selected = [item for index, item in enumerate(selected) if index not in replace_indexes]
        selected.append(candidate)
        seen.add(identity)
    return selected


def _is_visible_row(locator) -> bool:
    try:
        return bool(locator.is_visible(timeout=100))
    except Exception:
        try:
            return bool(locator.is_visible())
        except Exception:
            return True


def _enumerate_visible_job_rows(page) -> list[tuple[str, int, tuple[object, ...]]]:
    """Enumerate visible structural rows; indices are only locator handles."""
    try:
        result = page.evaluate(_STRUCTURAL_DISCOVERY_SCRIPT)
    except Exception:
        result = None
    if isinstance(result, dict) and isinstance(result.get("rows"), list):
        viewport_width = _numeric_descriptor(result.get("viewport_width"))
        structural_rows = _dedupe_structural_rows(
            [item for item in result["rows"] if isinstance(item, dict)]
        )
        return [
            ("body *", _numeric_descriptor(item.get("container_index")), _descriptor_signature(item))
            for item in structural_rows
            if item.get("visible", True)
            and _descriptor_is_safe_activation_row(
                item,
                viewport_width=viewport_width,
            )
        ]
    result: list[tuple[str, int, tuple[object, ...]]] = []
    seen_signatures: set[tuple[object, ...]] = set()
    for selector in _ROW_SELECTORS:
        try:
            matches = page.locator(selector)
            count = min(_bounded_hydration_count(matches.count()), 1000)
        except Exception:
            continue
        for index in range(count):
            try:
                locator = matches.nth(index)
                if not _is_visible_row(locator):
                    continue
                signature = _row_signature(locator)
            except Exception:
                continue
            if _row_identity_key(signature) in {_row_identity_key(item) for item in seen_signatures}:
                continue
            seen_signatures.add(signature)
            result.append((selector, index, signature))
    return result


def _resolve_row_locator(page, signature: tuple[object, ...]):
    """Re-enumerate and resolve a row immediately before its click."""
    for selector, index, current_signature in _enumerate_visible_job_rows(page):
        if _row_identity_key(current_signature) != _row_identity_key(signature):
            continue
        try:
            locator = page.locator(selector).nth(index)
            return locator if _is_visible_row(locator) else None
        except Exception:
            return None
    return None




def extract_card_local_posted_time_text(text: str) -> str:
    """Extract a posted/reposted relative time from one local job card text.

    This is intentionally independent and removable. It only receives text from
    the candidate card/row context; callers must not pass whole-page text.
    """
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized or len(normalized) > 1200:
        return ""
    patterns = (
        r"\bpublicado\s+de\s+nuevo\s+hace\s+(?:unos?\s+segundos?|un\s+momento|\d+\s+(?:minutos?|horas?|d[ií]as?))\b",
        r"\bpublicado\s+hace\s+(?:unos?\s+segundos?|un\s+momento|\d+\s+(?:minutos?|horas?|d[ií]as?))\b",
        r"\breposted\s+(?:just now|moments ago|a few seconds ago|\d+\s+(?:minutes?|hours?|days?)\s+ago)\b",
        r"\bposted\s+(?:just now|moments ago|a few seconds ago|\d+\s+(?:minutes?|hours?|days?)\s+ago)\b",
        r"\bhace\s+(?:unos?\s+segundos?|un\s+momento|\d+\s+(?:minutos?|horas?|d[ií]as?))\b",
        r"\b(?:just now|moments ago|a few seconds ago|\d+\s+(?:minutes?|hours?|days?)\s+ago)\b",
        r"\b(?:hoy|today)\b",
        r"\b\d+\s*(?:시간|분|일)\s*전\b",
        r"\b\d+\s*(?:時間|分|日)前\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match and match.group(0):
            return match.group(0).strip()[:80]
    return ""

def collect_visible_search_card_dates(page) -> dict[str, tuple[str, datetime, str, bool]]:
    """Collect safe relative dates from visible left-side search cards without clicking."""
    try:
        raw = page.evaluate(
            r"""
            () => {
              const patterns = [
                /\bpublicado\s+de\s+nuevo\s+hace\s+(?:unos?\s+segundos?|un\s+momento|\d+\s+(?:minutos?|horas?|d[ií]as?))\b/i,
                /\bpublicado\s+hace\s+(?:unos?\s+segundos?|un\s+momento|\d+\s+(?:minutos?|horas?|d[ií]as?))\b/i,
                /\breposted\s+(?:just now|moments ago|a few seconds ago|\d+\s+(?:minutes?|hours?|days?)\s+ago)\b/i,
                /\bposted\s+(?:just now|moments ago|a few seconds ago|\d+\s+(?:minutes?|hours?|days?)\s+ago)\b/i,
                /\bhace\s+(?:unos?\s+segundos?|un\s+momento|\d+\s+(?:minutos?|horas?|d[ií]as?))\b/i,
                /\b(?:just now|moments ago|a few seconds ago|\d+\s+(?:minutes?|hours?|days?)\s+ago)\b/i,
                /\b(?:hoy|today)\b/i,
                /\b\d+\s*(?:시간|분|일)\s*전\b/i,
                /\b\d+\s*(?:時間|分|日)前\b/i,
              ];
              const viewportWidth = Math.max(0, Number(window.innerWidth || 0));
              const textOf = (node) => String(
                node && (node.innerText || node.textContent) || ''
              ).replace(/\s+/g, ' ').trim();
              const matchDateText = (text) => {
                const normalized = String(text || '').replace(/\s+/g, ' ').trim();
                if (!normalized || normalized.length > 1200) return '';
                for (const pattern of patterns) {
                  const match = normalized.match(pattern);
                  if (match && match[0]) return match[0].slice(0, 80);
                }
                return '';
              };
              const matchDate = (node) => matchDateText(textOf(node));
              const findDateWithin = (node) => {
                const direct = matchDate(node);
                if (direct) return direct;
                const descendants = Array.from(
                  node && node.querySelectorAll ? node.querySelectorAll('*') : []
                ).slice(0, 240);
                for (const child of descendants) {
                  const rect = child.getBoundingClientRect ? child.getBoundingClientRect() : null;
                  if (!rect || rect.width <= 0 || rect.height <= 0) continue;
                  const text = textOf(child);
                  if (!text || text.length > 320) continue;
                  const matched = matchDateText(text);
                  if (matched) return matched;
                }
                return '';
              };
              const jobIdFromHref = (href) => {
                const match = String(href || '').match(/\/jobs\/view\/(?:[^/?#]*-)?(\d+)\/?(?:[?#]|$)/);
                return match && match[1] ? match[1].replace(/^0+/, '') || '0' : '';
              };
              const output = {};
              const rowLike = (node) => {
                const rect = node && node.getBoundingClientRect ? node.getBoundingClientRect() : null;
                if (!rect || rect.width <= 0 || rect.height <= 0 || rect.top < 70) return false;
                const width = Number(rect.width || 0);
                const height = Number(rect.height || 0);
                const left = Number(rect.left || 0);
                return (
                  left < viewportWidth * 0.58 &&
                  width >= 160 &&
                  width <= Math.max(760, viewportWidth * 0.62) &&
                  height >= 32 &&
                  height <= 360
                );
              };
              const jobIdFromNode = (node) => {
                for (const attr of ['data-job-id', 'data-occludable-job-id']) {
                  const value = node && node.getAttribute ? node.getAttribute(attr) : '';
                  const match = String(value || '').match(/(\d{3,})$/);
                  if (match && match[1]) return match[1].replace(/^0+/, '') || '0';
                }
                const link = node && node.querySelector ? node.querySelector('a[href*="/jobs/view/"]') : null;
                return link ? jobIdFromHref(link.getAttribute('href') || '') : '';
              };
              const collectFromRow = (node) => {
                if (!rowLike(node)) return;
                const jobId = jobIdFromNode(node);
                if (!jobId || output[jobId]) return;
                const date = findDateWithin(node);
                if (date) output[jobId] = date;
              };
              const rows = Array.from(document.querySelectorAll(
                'li[data-job-id], li[data-occludable-job-id], [role="listitem"][data-job-id], [role="option"][data-job-id], [role="listitem"][data-occludable-job-id], [role="option"][data-occludable-job-id], .job-card-container, .scaffold-layout__list-item'
              )).slice(0, 160);
              for (const row of rows) collectFromRow(row);
              const links = Array.from(document.querySelectorAll('a[href*="/jobs/view/"]')).slice(0, 120);
              for (const link of links) {
                const rect = link.getBoundingClientRect ? link.getBoundingClientRect() : null;
                if (!rect || rect.width <= 0 || rect.height <= 0 || rect.top < 70) continue;
                if (viewportWidth > 0 && rect.left > viewportWidth * 0.58) continue;
                const jobId = jobIdFromHref(link.getAttribute('href') || '');
                if (!jobId || output[jobId]) continue;
                let current = link;
                let depth = 0;
                while (current && current !== document.body && depth < 8) {
                  if (rowLike(current)) {
                    const date = findDateWithin(current);
                    if (date) {
                      output[jobId] = date;
                      break;
                    }
                  }
                  current = current.parentElement;
                  depth += 1;
                }
              }
              return output;
            }
            """
        )
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        return {}
    results: dict[str, tuple[str, datetime, str, bool]] = {}
    for job_id, raw_text in raw.items():
        normalized_job_id = _job_id_from_attribute_value(str(job_id or ""))
        if not normalized_job_id and re.fullmatch(r"\d{3,}", str(job_id or "").strip()):
            normalized_job_id = str(job_id or "").strip().lstrip("0") or "0"
        posted_text = extract_card_local_posted_time_text(str(raw_text or ""))
        if not posted_text:
            posted_text = _clean_posted_at_text(str(raw_text or ""))
        if not normalized_job_id or not posted_text or len(posted_text) > 80:
            continue
        published_at, confidence, within_24h = parse_linkedin_relative_time(posted_text)
        if published_at is None:
            continue
        results[normalized_job_id] = (posted_text, published_at, confidence, within_24h)
    return results


def _posted_at_from_row_locator(locator) -> tuple[str, datetime | None, str, bool]:
    """Extract only a safe relative date from the visible row text."""
    try:
        raw_text = str(
            locator.evaluate(
                r"""
                node => {
                  const patterns = [
                    /\bhace\s+(?:unos?\s+segundos?|un\s+momento|\d+\s+(?:minutos?|horas?|d[ií]as?))\b/i,
                    /\b(?:just now|moments ago|a few seconds ago|\d+\s+(?:minutes?|hours?|days?)\s+ago)\b/i,
                    /\b(?:hoy|today)\b/i,
                    /\b\d+\s*(?:시간|분|일)\s*전\b/i,
                    /\b\d+\s*(?:時間|分|日)前\b/i,
                  ];
                  const normalizedText = (candidate) => String(
                    candidate && (candidate.innerText || candidate.textContent) || ''
                  ).replace(/\s+/g, ' ').trim();
                  const findDate = (candidate) => {
                    const text = normalizedText(candidate);
                    for (const pattern of patterns) {
                      const match = text.match(pattern);
                      if (match && match[0]) return match[0].slice(0, 80);
                    }
                    return '';
                  };
                  let current = node;
                  let depth = 0;
                  while (current && current !== document.body && depth < 7) {
                    const rect = current.getBoundingClientRect ? current.getBoundingClientRect() : null;
                    const width = rect ? Number(rect.width || 0) : 0;
                    const height = rect ? Number(rect.height || 0) : 0;
                    const left = rect ? Number(rect.left || 0) : 0;
                    const rowLike = !rect || (
                      width >= 120 && width <= Math.max(760, window.innerWidth * 0.62) &&
                      height >= 24 && height <= 280 &&
                      left < window.innerWidth * 0.55
                    );
                    if (rowLike) {
                      const match = findDate(current);
                      if (match) return match;
                    }
                    current = current.parentElement;
                    depth += 1;
                  }
                  return findDate(node);
                }
                """
            )
            or ""
        )
    except Exception:
        raw_text = ""
    posted_text = _clean_posted_at_text(raw_text)
    published_at, confidence, within_24h = parse_linkedin_relative_time(
        posted_text
    )
    return posted_text, published_at, confidence, within_24h


def _canonical_detail_href(job_id: str) -> str:
    return f"https://www.linkedin.com/jobs/view/{job_id}"


def _active_detail_metadata_from_page(page) -> tuple[str, str, datetime | None, str, bool]:
    """Extract safe title/date evidence from the currently active right detail."""
    try:
        raw = page.evaluate(
            r"""
            () => {
              const viewportWidth = Math.max(0, Number(window.innerWidth || 0));
              const patterns = [
                /\bhace\s+(?:unos?\s+segundos?|un\s+momento|\d+\s+(?:minutos?|horas?|d[ií]as?))\b/i,
                /\b(?:just now|moments ago|a few seconds ago|\d+\s+(?:minutes?|hours?|days?)\s+ago)\b/i,
                /\b(?:hoy|today)\b/i,
                /\b\d+\s*(?:시간|분|일)\s*전\b/i,
                /\b\d+\s*(?:時間|分|日)前\b/i,
              ];
              const visibleRight = (node) => {
                const rect = node && node.getBoundingClientRect ? node.getBoundingClientRect() : null;
                return Boolean(rect && rect.width > 0 && rect.height > 0 && rect.left >= viewportWidth * 0.35);
              };
              const textOf = (node) => String(node && (node.innerText || node.textContent) || '').replace(/\s+/g, ' ').trim();
              const matchDate = (text) => {
                for (const pattern of patterns) {
                  const match = String(text || '').match(pattern);
                  if (match && match[0]) return match[0].slice(0, 80);
                }
                return '';
              };
              let title = '';
              let titleRect = null;
              let titleNode = null;
              for (const candidate of Array.from(document.querySelectorAll('main h1, [role="main"] h1, h1')).slice(0, 20)) {
                if (!visibleRight(candidate)) continue;
                title = textOf(candidate).slice(0, 180);
                if (title) {
                  titleNode = candidate;
                  titleRect = candidate.getBoundingClientRect();
                  break;
                }
              }
              let posted = '';
              const dateCandidates = [];
              const candidates = Array.from(document.querySelectorAll('main *, [role="main"] *, body *')).slice(0, 2500);
              for (const candidate of candidates) {
                if (!visibleRight(candidate)) continue;
                const rect = candidate.getBoundingClientRect();
                const text = textOf(candidate);
                if (!text || text.length > 900) continue;
                const matched = matchDate(text);
                if (!matched) continue;
                const titleDistance = titleRect ? Math.abs(rect.top - titleRect.bottom) : 10000;
                const leftDistance = titleRect ? Math.abs(rect.left - titleRect.left) : 10000;
                const abovePenalty = titleRect && rect.bottom < titleRect.top ? 1000 : 0;
                const broadContainerPenalty = rect.height > 220 || text.length > 500 ? 300 : 0;
                dateCandidates.push({
                  date: matched,
                  score: titleDistance + (leftDistance / 4) + abovePenalty + broadContainerPenalty,
                });
              }
              dateCandidates.sort((left, right) => left.score - right.score);
              posted = dateCandidates.length ? dateCandidates[0].date : '';
              const selectedScore = dateCandidates.length ? Math.max(0, Math.round(dateCandidates[0].score)) : 0;
              return {
                title,
                posted,
                date_candidate_count: Math.min(100, dateCandidates.length),
                selected_score: selectedScore,
              };
            }
            """
        )
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    title = re.sub(r"\s+", " ", str(raw.get("title") or "")).strip()[:180]
    posted_text = _clean_posted_at_text(str(raw.get("posted") or ""))
    published_at, confidence, within_24h = parse_linkedin_relative_time(posted_text)
    _ACTIVE_DETAIL_METADATA_DIAGNOSTIC.set(
        {
            "title_present": bool(title),
            "date_detected": bool(posted_text),
            "date_verified": published_at is not None,
            "date_within_24_hours": within_24h if published_at is not None else False,
            "date_candidate_count": max(0, min(100, _numeric_descriptor(raw.get("date_candidate_count")))),
            "selected_score": max(0, min(100000, _numeric_descriptor(raw.get("selected_score")))),
            "selected_date": posted_text[:80],
        }
    )
    return title, posted_text, published_at, confidence, within_24h


def _wait_for_active_detail_metadata(
    page,
    *,
    require_date: bool = True,
) -> tuple[str, str, datetime | None, str, bool]:
    elapsed_ms = 0
    latest: tuple[str, str, datetime | None, str, bool] = ("", "", None, "low", False)
    while True:
        latest = _active_detail_metadata_from_page(page)
        title, _posted_text, published_at, _confidence, _within_24h = latest
        if title and (not require_date or published_at is not None):
            return latest
        if elapsed_ms >= ROW_METADATA_WAIT_MS:
            return latest
        wait_ms = min(ROW_METADATA_POLL_MS, ROW_METADATA_WAIT_MS - elapsed_ms)
        _safe_page_pause(page, wait_ms)
        elapsed_ms += wait_ms


def _collect_visible_detail_job_links(page) -> tuple[set[str], set[str]]:
    """Collect safe job IDs from visible right-panel detail links.

    LinkedIn can obfuscate detail containers, but the active detail often still
    exposes a canonical /jobs/view/ link. Keep this fallback geometry-scoped so
    left-list cards and global navigation do not become the active identity.
    """
    job_ids: set[str] = set()
    hrefs: set[str] = set()
    try:
        viewport_width = int(page.evaluate("() => Math.max(0, Math.trunc(window.innerWidth || 0))") or 0)
    except Exception:
        viewport_width = 0
    try:
        links = page.locator("a[href*='/jobs/view/']")
        count = min(_bounded_hydration_count(links.count()), 50)
    except Exception:
        return job_ids, hrefs
    for index in range(count):
        try:
            locator = links.nth(index)
            href = str(locator.get_attribute("href") or "").strip()
        except Exception:
            continue
        if not href or "/apply" in href:
            continue
        try:
            box = locator.bounding_box()
        except Exception:
            box = None
        if not isinstance(box, dict):
            continue
        x = _numeric_descriptor(box.get("x"))
        y = _numeric_descriptor(box.get("y"))
        width = _numeric_descriptor(box.get("width"))
        height = _numeric_descriptor(box.get("height"))
        if width <= 0 or height <= 0 or y < 70:
            continue
        if viewport_width > 0 and x < int(viewport_width * 0.38):
            continue
        try:
            canonical = canonicalize_linkedin_job_url(
                urljoin("https://www.linkedin.com", href)
            )
        except ValueError:
            continue
        detail_id = linkedin_job_id_from_url(canonical)
        if detail_id:
            job_ids.add(detail_id)
            hrefs.add(canonical)
    return job_ids, hrefs

def _detail_identity_from_page(page) -> LinkedInDetailIdentity:
    job_ids: set[str] = set()
    hrefs: set[str] = set()
    attributes: set[tuple[str, str]] = set()

    def collect(locator) -> None:
        for name in _ROW_ALLOWLISTED_ATTRIBUTES:
            try:
                value = str(locator.get_attribute(name) or "").strip()
            except Exception:
                value = ""
            if value:
                attributes.add((name, value[:120]))
                job_id = _job_id_from_attribute_value(value)
                if job_id:
                    job_ids.add(job_id)
        try:
            links = locator.locator("a[href*='/jobs/view/']")
            for index in range(min(_bounded_hydration_count(links.count()), 20)):
                href = str(links.nth(index).get_attribute("href") or "").strip()
                if not href:
                    continue
                try:
                    canonical = canonicalize_linkedin_job_url(
                        urljoin("https://www.linkedin.com", href)
                    )
                except ValueError:
                    continue
                detail_id = linkedin_job_id_from_url(canonical)
                if detail_id:
                    job_ids.add(detail_id)
                    hrefs.add(canonical)
        except Exception:
            return

    for selector in _DETAIL_IDENTITY_SELECTORS:
        try:
            matches = page.locator(selector)
            for index in range(min(_bounded_hydration_count(matches.count()), 4)):
                collect(matches.nth(index))
        except Exception:
            continue

    visible_detail_job_ids, visible_detail_hrefs = _collect_visible_detail_job_links(page)
    job_ids.update(visible_detail_job_ids)
    hrefs.update(visible_detail_hrefs)

    try:
        current_url = str(getattr(page, "url", "") or "")
        canonical = canonicalize_linkedin_job_url(current_url)
        current_id = linkedin_job_id_from_url(canonical)
        if current_id:
            job_ids.add(current_id)
            hrefs.add(canonical)
    except ValueError:
        pass

    job_id = sorted(job_ids, key=lambda item: (len(item), item))[-1] if job_ids else ""
    canonical_href = _canonical_detail_href(job_id) if job_id else ""
    if hrefs:
        matching = sorted(hrefs)
        canonical_href = next(
            (href for href in matching if linkedin_job_id_from_url(href) == job_id),
            canonical_href,
        )
    return LinkedInDetailIdentity(
        job_id=job_id,
        canonical_detail_href=canonical_href,
        allowlisted_attributes=tuple(sorted(attributes)),
    )


def _detail_identity_changed(
    before: LinkedInDetailIdentity,
    after: LinkedInDetailIdentity,
) -> bool:
    return any(
        (
            before.job_id != after.job_id,
            before.canonical_detail_href != after.canonical_detail_href,
            before.allowlisted_attributes != after.allowlisted_attributes,
        )
    )


def _wait_for_changed_detail_identity(
    page,
    before: LinkedInDetailIdentity,
    *,
    max_wait_ms: int = ROW_ACTIVATION_WAIT_MS,
) -> tuple[LinkedInDetailIdentity, bool]:
    elapsed_ms = 0
    while True:
        after = _detail_identity_from_page(page)
        if _detail_identity_changed(before, after):
            return after, True
        if elapsed_ms >= max_wait_ms:
            return after, False
        wait_ms = min(ROW_ACTIVATION_POLL_MS, max_wait_ms - elapsed_ms)
        _safe_page_pause(page, wait_ms)
        elapsed_ms += wait_ms


def _scroll_results_panel_for_rows(page) -> bool:
    structural = _structural_panel_metrics(page)
    if structural is not None:
        locator, _metrics = structural
        try:
            metrics = locator.evaluate(_SCROLL_CONTAINER_SCROLL_SCRIPT)
        except Exception:
            return False
        return isinstance(metrics, dict) and _numeric_descriptor(metrics.get("scrollTopAfter")) != _numeric_descriptor(
            metrics.get("scrollTopBefore")
        )
    selected = None
    for selector in _RESULTS_PANEL_SCROLL_SELECTORS:
        try:
            matches = page.locator(selector)
            for index in range(min(_bounded_hydration_count(matches.count()), 8)):
                locator = matches.nth(index)
                metrics = locator.evaluate(_SCROLL_CONTAINER_PROBE_SCRIPT)
                if isinstance(metrics, dict) and metrics.get("scrollable"):
                    selected = locator
                    break
        except Exception:
            continue
        if selected is not None:
            break
    if selected is None:
        return False
    try:
        metrics = selected.evaluate(_SCROLL_CONTAINER_SCROLL_SCRIPT)
    except Exception:
        return False
    return isinstance(metrics, dict) and int(metrics.get("scrollTopAfter", 0) or 0) != int(
        metrics.get("scrollTopBefore", 0) or 0
    )


def discover_job_rows_via_activation(
    page,
    *,
    source_url: str,
    existing_job_ids: set[str] | None = None,
    max_activations: int = MAX_ROW_ACTIVATIONS_PER_QUERY,
    max_scrolls: int = MAX_DISCOVERY_SCROLLS,
    max_wait_ms: int = ROW_ACTIVATION_WAIT_MS,
    diagnostic_capture: Callable[[object, str], None] | None = None,
    diagnostic_detail_capture: Callable[[object, str], None] | None = None,
    diagnostic_scroll: Callable[[object, int], None] | None = None,
) -> LinkedInRowDiscoveryResult:
    """Discover IDs hidden behind virtualized rows using bounded read-only clicks."""
    validated_source_url = validate_linkedin_jobs_url(source_url)
    existing = set(existing_job_ids or ())
    outcomes: list[LinkedInRowActivationOutcome] = []
    records: list[LinkedInVacancyRecord] = []
    processed_signatures: set[tuple[object, ...]] = set()
    discovered_ids = set(existing)
    scroll_count = 0
    structural_rows_found = 0
    consecutive_scrolls_without_new_ids = 0
    stop_reason = "none"
    activation_limit = min(MAX_ROW_ACTIVATIONS_PER_QUERY, max(0, max_activations))
    selected_panel = _structural_panel_metrics(page)
    selected_panel_metrics = selected_panel[1] if selected_panel is not None else _SearchScrollMetrics()

    while len(outcomes) < activation_limit:
        rows = _enumerate_visible_job_rows(page)
        structural_rows_found = max(structural_rows_found, len(rows))
        if not rows:
            stop_reason = "no_structural_rows"
            break
        progressed = False
        ids_before_rows = len(discovered_ids)
        for _selector, _index, signature in rows:
            if len(outcomes) >= activation_limit:
                break
            if signature in processed_signatures:
                continue
            processed_signatures.add(signature)
            locator = _resolve_row_locator(page, signature)
            if locator is None:
                continue
            progressed = True
            before = _detail_identity_from_page(page)
            (
                row_posted_text,
                row_published_at,
                row_freshness_confidence,
                row_within_24h,
            ) = _posted_at_from_row_locator(locator)
            if diagnostic_detail_capture is not None:
                try:
                    diagnostic_detail_capture(page, "before_click")
                except Exception:
                    pass
            try:
                locator.click(timeout=5000)
            except Exception:
                outcome = "row_activation_no_change"
                outcomes.append(LinkedInRowActivationOutcome(outcome, signature))
                if diagnostic_capture is not None:
                    try:
                        diagnostic_capture(page, outcome)
                    except Exception:
                        pass
                continue
            after, changed = _wait_for_changed_detail_identity(
                page, before, max_wait_ms=max_wait_ms
            )
            detail_title = ""
            if changed and after.job_id:
                (
                    detail_title,
                    detail_posted_text,
                    detail_published_at,
                    detail_freshness_confidence,
                    detail_within_24h,
                ) = _wait_for_active_detail_metadata(
                    page,
                    require_date=row_published_at is None,
                )
                if row_published_at is None and detail_published_at is not None:
                    row_posted_text = detail_posted_text
                    row_published_at = detail_published_at
                    row_freshness_confidence = detail_freshness_confidence
                    row_within_24h = detail_within_24h
            if not changed:
                outcome = "row_activation_no_change"
            elif not after.job_id:
                outcome = "row_activation_no_job_id"
            elif after.job_id in discovered_ids:
                outcome = "row_activation_duplicate"
            else:
                outcome = "row_activation_success"
                discovered_ids.add(after.job_id)
                records.append(
                    LinkedInVacancyRecord(
                        linkedin_job_id=after.job_id,
                        title=detail_title,
                        posted_at_text=row_posted_text,
                        published_at=row_published_at,
                        freshness_confidence=(
                            row_freshness_confidence
                            if row_published_at is not None
                            else "low"
                        ),
                        is_within_24_hours=(
                            row_within_24h if row_published_at is not None else False
                        ),
                        canonical_url=after.canonical_detail_href
                        or _canonical_detail_href(after.job_id),
                        source_url=validated_source_url,
                        discovery_sources=["row_activation"],
                        candidate_metadata_incomplete=True,
                    )
                )
            outcomes.append(
                LinkedInRowActivationOutcome(
                    outcome=outcome,
                    signature=signature,
                    job_id=after.job_id,
                    canonical_detail_href=after.canonical_detail_href,
                    date_detected=bool(row_posted_text),
                    date_verified=row_published_at is not None,
                    date_within_24_hours=(
                        row_within_24h if row_published_at is not None else False
                    ),
                )
            )
            if diagnostic_detail_capture is not None:
                try:
                    diagnostic_detail_capture(page, "after_click")
                except Exception:
                    pass
            if diagnostic_capture is not None:
                try:
                    diagnostic_capture(page, outcome)
                except Exception:
                    # Diagnostics must never affect read-only discovery.
                    pass
        if len(outcomes) >= activation_limit:
            stop_reason = "activation_cap"
            break
        if scroll_count >= min(MAX_DISCOVERY_SCROLLS, max(0, max_scrolls)):
            stop_reason = "scroll_cap"
            break
        if not _scroll_results_panel_for_rows(page):
            stop_reason = "no_scroll_progress"
            break
        scroll_count += 1
        _safe_page_pause(page, min(DISCOVERY_SCROLL_WAIT_MS, max_wait_ms))
        if diagnostic_scroll is not None:
            try:
                diagnostic_scroll(page, scroll_count)
            except Exception:
                pass
        if len(discovered_ids) == ids_before_rows:
            consecutive_scrolls_without_new_ids += 1
        else:
            consecutive_scrolls_without_new_ids = 0
        if consecutive_scrolls_without_new_ids >= 2:
            stop_reason = "two_scrolls_without_new_job_ids"
            break
        if not progressed and not _enumerate_visible_job_rows(page):
            stop_reason = "no_new_structural_rows"
            break
    if stop_reason == "none":
        stop_reason = "structural_rows_found_but_unresolved" if structural_rows_found and not records else "exhausted"
    return LinkedInRowDiscoveryResult(
        records,
        outcomes,
        scroll_count,
        structural_rows_found,
        stop_reason,
        selected_panel_metrics.selected_row_container_score,
        selected_panel_metrics.row_interactive_count,
    )


def merge_row_activation_records(
    dom_records: list[LinkedInVacancyRecord],
    row_records: list[LinkedInVacancyRecord],
) -> tuple[list[LinkedInVacancyRecord], bool, bool]:
    """Merge sources by job ID before global same-run detail deduplication."""
    merged: list[LinkedInVacancyRecord] = []
    by_id: dict[str, int] = {}
    dom_contributed = False
    row_contributed = False
    for record, is_row in [*( (item, False) for item in dom_records), *((item, True) for item in row_records)]:
        job_id = _job_id_from_attribute_value(getattr(record, "linkedin_job_id", ""))
        if not job_id:
            continue
        if is_row:
            row_contributed = True
        else:
            dom_contributed = True
        index = by_id.get(job_id)
        if index is None:
            by_id[job_id] = len(merged)
            merged.append(record.model_copy(update={"linkedin_job_id": job_id}))
            continue
        current = merged[index]
        sources = sorted(set(current.discovery_sources) | set(record.discovery_sources))
        updates: dict[str, object] = {"discovery_sources": sources}
        for field_name in (
            "title",
            "company_name",
            "location",
            "workplace_type",
            "posted_at_text",
            "published_at",
            "freshness_confidence",
            "is_within_24_hours",
        ):
            current_value = getattr(current, field_name, None)
            incoming_value = getattr(record, field_name, None)
            if field_name in {"freshness_confidence", "is_within_24_hours"}:
                if current.published_at is None and record.published_at is not None:
                    updates[field_name] = incoming_value
                continue
            if not current_value and incoming_value:
                updates[field_name] = incoming_value
        merged[index] = current.model_copy(update=updates)
    return merged, dom_contributed, row_contributed


def discovery_mode_for_sources(
    *,
    dom_contributed: bool,
    row_contributed: bool,
    structural_rows_found: bool = False,
    row_job_ids_resolved: int = 0,
) -> str:
    if structural_rows_found and row_job_ids_resolved == 0:
        return "structural_rows_found_but_unresolved"
    if dom_contributed and row_contributed:
        return "multi_source_with_row_activation"
    if row_contributed:
        return "row_activation"
    return "standard"


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
    panel = _structural_panel_metrics(page)
    panel_metrics = panel[1] if panel is not None else _SearchScrollMetrics()
    structural_rows = _enumerate_visible_job_rows(page)
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
        selected_row_container_score=panel_metrics.selected_row_container_score,
        row_candidate_count=max(panel_metrics.row_candidate_count, len(structural_rows)),
        row_interactive_count=panel_metrics.row_interactive_count,
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


def _numeric_descriptor(value: object, *, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _normalised_score(value: int, maximum: int, weight: int) -> int:
    if maximum <= 0 or value <= 0:
        return 0
    return min(weight, max(0, round(weight * value / maximum)))


def _score_results_panel_descriptor(
    descriptor: dict,
    *,
    max_row_candidates: int,
    max_repetition: int,
    max_interactive: int,
    viewport_width: int,
) -> int:
    rows = _numeric_descriptor(descriptor.get("row_candidate_count", descriptor.get("row_count", 0)))
    repetition = _numeric_descriptor(
        descriptor.get("row_repetition", descriptor.get("vertical_repetition", descriptor.get("regular_vertical_repetition", 0)))
    )
    interactive = _numeric_descriptor(
        descriptor.get("interactive_count", descriptor.get("visible_interactive_count", 0))
    )
    scroll_height = _numeric_descriptor(descriptor.get("scrollHeight"))
    client_height = _numeric_descriptor(descriptor.get("clientHeight"))
    x = _numeric_descriptor(descriptor.get("x"))
    useful_scroll = scroll_height > client_height + max(80, client_height // 4)
    left_proximity = 2 if viewport_width <= 0 else max(0, min(2, round(2 * (1 - min(1, x / viewport_width)))))
    score = (
        _normalised_score(rows, max_row_candidates, 4)
        + _normalised_score(repetition, max_repetition, 3)
        + (2 if useful_scroll else 0)
        + _normalised_score(interactive, max_interactive, 2)
        + left_proximity
    )
    if descriptor.get("detail_or_main"):
        score -= 4
    if descriptor.get("header_nav_footer_global"):
        score -= 3
    return score


def _choose_structural_results_panel(
    descriptors: list[dict],
    *,
    viewport_width: int = 0,
) -> tuple[dict, int] | None:
    if not descriptors:
        return None
    max_rows = max(_numeric_descriptor(item.get("row_candidate_count", item.get("row_count", 0))) for item in descriptors)
    max_repetition = max(_numeric_descriptor(item.get("row_repetition", item.get("vertical_repetition", item.get("regular_vertical_repetition", 0)))) for item in descriptors)
    max_interactive = max(_numeric_descriptor(item.get("interactive_count", item.get("visible_interactive_count", 0))) for item in descriptors)
    scored = []
    for item in descriptors:
        score = _score_results_panel_descriptor(
            item,
            max_row_candidates=max_rows,
            max_repetition=max_repetition,
            max_interactive=max_interactive,
            viewport_width=viewport_width,
        )
        scored.append((
            score,
            _numeric_descriptor(item.get("row_repetition", item.get("vertical_repetition", item.get("regular_vertical_repetition", 0)))),
            -_numeric_descriptor(item.get("x")),
            -_numeric_descriptor(item.get("width")),
            _numeric_descriptor(item.get("visible_row_count", item.get("row_count", 0))),
            item,
        ))
    selected = max(scored, key=lambda item: item[:-1])
    return selected[-1], selected[0]


def _structural_discovery(page) -> tuple[dict, list[dict]]:
    try:
        result = page.evaluate(_STRUCTURAL_DISCOVERY_SCRIPT)
    except Exception:
        return {}, []
    if not isinstance(result, dict):
        return {}, []
    containers = [item for item in result.get("containers", []) if isinstance(item, dict)]
    rows = [item for item in result.get("rows", []) if isinstance(item, dict)]
    return result, rows


def _structural_panel_metrics(page) -> tuple[object, _SearchScrollMetrics] | None:
    try:
        result = page.evaluate(_STRUCTURAL_DISCOVERY_SCRIPT)
    except Exception:
        return None
    if not isinstance(result, dict):
        return None
    containers = [item for item in result.get("containers", []) if isinstance(item, dict)]
    selected = _choose_structural_results_panel(
        containers,
        viewport_width=_numeric_descriptor(result.get("viewport_width")),
    )
    if selected is None:
        return None
    descriptor, score = selected
    container_index = _numeric_descriptor(descriptor.get("container_index"))
    try:
        locator = page.locator("body *").nth(container_index)
    except Exception:
        return None
    return locator, _SearchScrollMetrics(
        selected_scroll_container="results_panel",
        scroll_height=_numeric_descriptor(descriptor.get("scrollHeight")),
        client_height=_numeric_descriptor(descriptor.get("clientHeight")),
        scroll_top_before=_numeric_descriptor(descriptor.get("scrollTop")),
        scroll_top_after=_numeric_descriptor(descriptor.get("scrollTop")),
        job_signal_count=_numeric_descriptor(descriptor.get("row_candidate_count", descriptor.get("row_count", 0))),
        selected_row_container_score=score,
        row_candidate_count=_numeric_descriptor(descriptor.get("row_candidate_count", descriptor.get("row_count", 0))),
        row_interactive_count=_numeric_descriptor(descriptor.get("interactive_count", descriptor.get("visible_interactive_count", 0))),
        row_repetition=_numeric_descriptor(descriptor.get("row_repetition", descriptor.get("vertical_repetition", descriptor.get("regular_vertical_repetition", 0)))),
        visible_row_count=_numeric_descriptor(descriptor.get("visible_row_count", descriptor.get("row_count", 0))),
    )


def _candidate_scroll_containers(page) -> list[tuple[object, _SearchScrollMetrics]]:
    candidates: list[tuple[object, _SearchScrollMetrics]] = []
    structural = _structural_panel_metrics(page)
    if structural is not None:
        candidates.append(structural)
        return candidates
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
            item[1].selected_row_container_score,
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
        selected_row_container_score=selected_metrics.selected_row_container_score,
        row_candidate_count=selected_metrics.row_candidate_count,
        row_interactive_count=selected_metrics.row_interactive_count,
        row_repetition=selected_metrics.row_repetition,
        visible_row_count=selected_metrics.visible_row_count,
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
