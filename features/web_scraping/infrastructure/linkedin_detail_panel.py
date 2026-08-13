"""Detail-panel extraction and enrichment helpers for LinkedIn jobs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
import time
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup

from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
from features.web_scraping.infrastructure.linkedin_detail_diagnostics import (
    LinkedInDetailDiagnosticsCollector,
    get_active_detail_diagnostics,
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
from features.web_scraping.infrastructure.linkedin_navigation import (
    LinkedInAuthRequiredError,
    LinkedInDetailPanelError,
    _DETAIL_HYDRATION_MAX_MS,
    _HYDRATION_POLL_INTERVALS_MS,
    _is_page_recoverable_error,
    _locator_has_signal,
    _raise_for_terminal_page_signal,
    _safe_error_label,
    _safe_page_pause,
    _validate_authenticated_page,
)
from features.web_scraping.infrastructure.linkedin_parser import (
    MATCH_TERMS,
    _clean_posted_at_text,
    _normalize_repeated_title,
    parse_linkedin_relative_time,
)
from features.web_scraping.infrastructure.linkedin_url_policy import (
    canonicalize_linkedin_job_url,
    is_linkedin_auth_checkpoint,
    linkedin_job_id_from_url,
    validate_linkedin_jobs_url,
)


_DETAIL_DATE_SELECTORS = (
    "time",
    ".jobs-unified-top-card__posted-date",
    ".job-details-jobs-unified-top-card__primary-description-container",
    ".jobs-unified-top-card__subtitle-primary-grouping",
    ".jobs-details-top-card__job-info",
    "[class*='posted-date']",
)


_DETAIL_TITLE_SELECTORS = (
    ".job-details-jobs-unified-top-card__job-title h1",
    ".job-details-jobs-unified-top-card__job-title",
    ".job-details-jobs-unified-top-card__job-title-link",
    ".jobs-unified-top-card__job-title",
    "h1.t-24",
    "h1",
)


_DETAIL_COMPANY_SELECTORS = (
    ".job-details-jobs-unified-top-card__company-name a",
    ".jobs-unified-top-card__company-name a",
    ".job-details-jobs-unified-top-card__company-name",
    ".jobs-unified-top-card__company-name",
    "a[href*='/company/']",
)


_DETAIL_LOCATION_SELECTORS = (
    ".job-details-jobs-unified-top-card__primary-description-container "
    ".tvm__text--low-emphasis",
    ".job-details-jobs-unified-top-card__tertiary-description-container "
    ".tvm__text--low-emphasis",
    ".jobs-unified-top-card__bullet",
    ".job-details-jobs-unified-top-card__bullet",
    ".job-details-jobs-unified-top-card__primary-description-container",
    ".jobs-unified-top-card__primary-description",
)


_DETAIL_WORKPLACE_SELECTORS = (
    ".job-details-jobs-unified-top-card__workplace-type",
    ".jobs-unified-top-card__workplace-type",
    ".jobs-unified-top-card__workplace-type-text",
    "[data-test-job-workplace-type]",
    "button[class*='workplace-type']",
    "[class*='workplace-type']",
)


_DETAIL_DESCRIPTION_SELECTORS = (
    ".jobs-description-content__text",
    ".jobs-description__content",
    ".show-more-less-html__markup",
    ".jobs-box__html-content",
    "#job-details",
    "[class*='jobs-description']",
)


_JOB_CARD_LINK_SELECTORS = (
    ".job-card-container a[href*='/jobs/view/']",
    "li[data-occludable-job-id] a[href*='/jobs/view/']",
    ".scaffold-layout__list-item a[href*='/jobs/view/']",
    "[data-job-id] a[href*='/jobs/view/']",
)


_DETAIL_PANEL_JOB_LINK_SELECTORS = (
    ".jobs-search__job-details--container a[href*='/jobs/view/']",
    ".jobs-details a[href*='/jobs/view/']",
    ".job-details-jobs-unified-top-card__job-title a[href*='/jobs/view/']",
)
_DETAIL_PROBE_TIMEOUT_MS = 10000
_GUEST_DETAIL_PROBE_TIMEOUT_MS = 10000
_GUEST_DETAIL_MAX_RETRIES = 2
_GUEST_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
_MIN_PLAUSIBLE_DESCRIPTION_CHARS = 16
_STATIC_DESCRIPTION_SELECTORS = (
    ("static_html_container", ".show-more-less-html__markup"),
    ("static_html_container", ".description__text"),
)
_STATIC_BLOCK_TAGS = (
    "br",
    "p",
    "li",
    "ul",
    "ol",
    "div",
    "section",
    "article",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
)
_STATIC_DESCRIPTION_PLACEHOLDERS = {
    "about the job",
    "about this job",
    "acerca del empleo",
    "job description",
    "loading",
    "loading...",
    "cargando",
    "show more",
    "show less",
    "ver más",
    "ver mas",
    "ver menos",
    "더 보기",
    "더보기",
}


@dataclass(frozen=True)
class _AuthenticatedDetailProbeResult:
    record: LinkedInVacancyRecord
    status_code: int
    category: str
    detail: str = ""
    body_source: str = ""
    description_length: int = 0
    guest_status_code: int = 0
    guest_retry_count: int = 0
    identity_consistent: bool = True


@dataclass(frozen=True)
class _StaticJobDetailParseResult:
    record: LinkedInVacancyRecord
    body_source: str = ""
    description_length: int = 0
    identity_title: str = ""
    identity_company: str = ""


def _extract_detail_posted_date(
    page,
    *,
    now: datetime | None = None,
) -> tuple[str, datetime | None, str, bool]:
    for selector in _DETAIL_DATE_SELECTORS:
        try:
            locator = page.locator(selector).first
            if not locator.count():
                continue
            text = re.sub(r"\s+", " ", locator.inner_text(timeout=3000) or "").strip()
            cleaned_text = _clean_posted_at_text(text)
            structured = ""
            try:
                structured = str(locator.get_attribute("datetime") or "")
            except Exception:
                pass
            relative_at, relative_confidence, relative_within_24h = (
                parse_linkedin_relative_time(
                    cleaned_text,
                    now=now,
                )
            )
            if relative_at is not None:
                return (
                    cleaned_text,
                    relative_at,
                    relative_confidence,
                    relative_within_24h,
                )
            published_at, confidence, within_24h = parse_linkedin_relative_time(
                cleaned_text,
                now=now,
                structured_datetime=structured,
            )
            if published_at is not None:
                return cleaned_text, published_at, confidence, within_24h
        except Exception:
            continue
    return "", None, "low", False


def _first_visible_text(page, selectors: tuple[str, ...]) -> str:
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if locator.count():
                value = re.sub(
                    r"\s+",
                    " ",
                    locator.inner_text(timeout=3000) or "",
                ).strip()
                if value:
                    return value
        except Exception:
            continue
    return ""


def _first_visible_raw_text(page, selectors: tuple[str, ...]) -> str:
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if locator.count():
                value = str(locator.inner_text(timeout=3000) or "").strip()
                if value:
                    return value
        except Exception:
            continue
    return ""


def _normalize_transient_description(value: str, limit: int | None = None) -> str:
    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in (value or "").replace("\r", "\n").splitlines()
    ]
    normalized = "\n".join(line for line in lines if line).strip()
    return normalized if limit is None else normalized[:limit]


def _is_incomplete_detail_body(value: str) -> bool:
    normalized = re.sub(r"\s+", " ", value or "").strip().casefold()
    return not normalized or normalized in {
        "acerca del empleo",
        "about the job",
        "about this job",
        "job description",
    }


def _normalize_identity_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _identity_texts_compatible(expected: object, observed: object) -> bool:
    expected_text = _normalize_identity_text(expected)
    observed_text = _normalize_identity_text(observed)
    if not expected_text or not observed_text:
        return True
    if expected_text == observed_text:
        return True
    return (
        len(expected_text) >= 6
        and len(observed_text) >= 6
        and (expected_text in observed_text or observed_text in expected_text)
    )


def _is_plausible_description(value: str) -> bool:
    normalized = re.sub(r"\s+", " ", value or "").strip()
    if not normalized:
        return False
    folded = normalized.strip(" .:;…").casefold()
    if folded in _STATIC_DESCRIPTION_PLACEHOLDERS:
        return False
    if _is_incomplete_detail_body(normalized):
        return False
    if len(normalized) < _MIN_PLAUSIBLE_DESCRIPTION_CHARS:
        return False
    if re.fullmatch(r"(show more|show less|ver más|ver mas|ver menos)", folded):
        return False
    return True


def _iter_jsonld_nodes(value: object):
    if isinstance(value, list):
        for item in value:
            yield from _iter_jsonld_nodes(item)
        return
    if not isinstance(value, dict):
        return
    yield value
    graph = value.get("@graph")
    if isinstance(graph, list):
        for item in graph:
            yield from _iter_jsonld_nodes(item)


def _jsonld_type_matches(value: object, expected: str) -> bool:
    if isinstance(value, list):
        return any(_jsonld_type_matches(item, expected) for item in value)
    return str(value or "").casefold() == expected.casefold()


def _extract_jsonld_job_posting(html: str) -> dict[str, object] | None:
    soup = BeautifulSoup(html or "", "html.parser")
    for script in soup.select("script[type='application/ld+json']"):
        raw = script.string or script.get_text("", strip=True)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            continue
        for node in _iter_jsonld_nodes(payload):
            if _jsonld_type_matches(node.get("@type"), "JobPosting"):
                return node
    return None


def _jsonld_name(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or "").strip()
    return str(value or "").strip()


def _jsonld_job_location(value: object) -> str:
    items = value if isinstance(value, list) else [value]
    parts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        address = item.get("address")
        if isinstance(address, dict):
            location_parts = [
                address.get("addressLocality"),
                address.get("addressRegion"),
                _jsonld_name(address.get("addressCountry")),
            ]
        else:
            location_parts = [item.get("name"), item.get("address")]
        text = ", ".join(
            str(part).strip()
            for part in location_parts
            if str(part or "").strip()
        )
        if text:
            parts.append(text)
    return "; ".join(dict.fromkeys(parts))


def _static_text_from_node(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    soup = BeautifulSoup(raw, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(_STATIC_BLOCK_TAGS):
        if tag.name != "br":
            tag.append("\n")
    text = soup.get_text("\n", strip=True)
    return _normalize_transient_description(text)


def _description_text_from_static_html(value: object) -> str:
    return _static_text_from_node(value)


def _static_title_from_soup(soup: BeautifulSoup) -> str:
    for selector in (
        ".top-card-layout__title",
        ".topcard__title",
        ".job-details-jobs-unified-top-card__job-title",
        "h1",
    ):
        node = soup.select_one(selector)
        text = _normalize_repeated_title(_static_text_from_node(node))
        if text:
            return text
    return ""


def _static_company_from_soup(soup: BeautifulSoup) -> str:
    for selector in (
        ".topcard__org-name-link",
        ".top-card-layout__entity-info a",
        ".job-details-jobs-unified-top-card__company-name",
        "a[href*='/company/']",
    ):
        node = soup.select_one(selector)
        text = _static_text_from_node(node)
        if text:
            return text
    return ""


def _static_location_from_soup(soup: BeautifulSoup) -> str:
    for selector in (
        ".topcard__flavor.topcard__flavor--bullet",
        ".topcard__flavor--bullet",
        ".job-details-jobs-unified-top-card__primary-description-container",
    ):
        node = soup.select_one(selector)
        text = _static_text_from_node(node)
        if text:
            return text
    return ""


def _select_static_description(
    soup: BeautifulSoup,
    job: dict[str, object] | None,
) -> tuple[str, str]:
    first_placeholder = ""
    if job is not None:
        jsonld_description = _description_text_from_static_html(
            job.get("description")
        )
        if _is_plausible_description(jsonld_description):
            return jsonld_description, "jsonld_description"
        if jsonld_description and not first_placeholder:
            first_placeholder = jsonld_description

    for source, selector in _STATIC_DESCRIPTION_SELECTORS:
        node = soup.select_one(selector)
        if node is None:
            continue
        text = _static_text_from_node(node)
        if _is_plausible_description(text):
            return text, source
        if text and not first_placeholder:
            first_placeholder = text

    return first_placeholder, ""


def _parse_static_job_detail_html_result(
    html: str,
    record: LinkedInVacancyRecord,
    *,
    include_description: bool,
    now: datetime | None = None,
) -> _StaticJobDetailParseResult:
    soup = BeautifulSoup(html or "", "html.parser")
    job = _extract_jsonld_job_posting(html)

    static_description, body_source = _select_static_description(soup, job)
    title = ""
    company = ""
    location = ""
    raw_workplace = ""
    posted_text = record.posted_at_text
    published_at = record.published_at
    confidence = record.freshness_confidence
    within_24h = record.is_within_24_hours

    if job is not None:
        title = _normalize_repeated_title(str(job.get("title") or "").strip())
        company = _jsonld_name(job.get("hiringOrganization"))
        location = _jsonld_job_location(job.get("jobLocation"))
        raw_workplace = " ".join(
            str(value or "")
            for value in (
                job.get("jobLocationType"),
                job.get("employmentType"),
                location,
            )
        )
        date_posted = str(job.get("datePosted") or "").strip()
        parsed_at, parsed_confidence, parsed_within_24h = (
            parse_linkedin_relative_time(
                date_posted,
                now=now,
                structured_datetime=date_posted,
            )
        )
        if parsed_at is not None:
            published_at = parsed_at
            confidence = parsed_confidence
            within_24h = parsed_within_24h
        posted_text = _clean_posted_at_text(date_posted) or posted_text

    title = title or _static_title_from_soup(soup)
    company = company or _static_company_from_soup(soup)
    location = location or _static_location_from_soup(soup)
    workplace_type = _normalize_workplace_type(raw_workplace or location)

    if job is None and not static_description and not title and not company:
        return _StaticJobDetailParseResult(record=record)

    inference_text = (
        f"{static_description}\n{title}\n{company}"
    ).strip()
    candidate_expectations, responsibilities = _extract_structured_sections(
        static_description
    )
    matched = sorted(
        {
            term
            for term in MATCH_TERMS
            if term in f"{record.title} {title} {static_description}".lower()
        }
    )
    enriched = record.model_copy(
        update={
            "title": title or _normalize_repeated_title(record.title),
            "company_name": company or record.company_name,
            "location": location or record.location,
            "workplace_type": workplace_type or record.workplace_type,
            "posted_at_text": posted_text or record.posted_at_text,
            "published_at": published_at or record.published_at,
            "freshness_confidence": (
                confidence
                if published_at is not None
                else record.freshness_confidence
            ),
            "is_within_24_hours": (
                within_24h
                if published_at is not None
                else record.is_within_24_hours
            ),
            "description_excerpt": (
                _sanitize_description(static_description)
                if include_description and static_description
                else record.description_excerpt
            ),
            "description_full_text": (
                static_description
                if include_description and static_description
                else record.description_full_text
            ),
            "language_requirements": _infer_language_requirements(inference_text),
            "experience_requirements": _infer_experience_requirements(inference_text),
            "hard_skills": _infer_hard_skills(static_description),
            "soft_skills": _infer_soft_skills(static_description),
            "candidate_expectations": candidate_expectations,
            "responsibilities": responsibilities,
            "foreigner_acceptance": _infer_foreigner_acceptance(inference_text),
            "visa_status": _infer_visa_status(inference_text),
            "relocation_support": _infer_relocation_support(inference_text),
            "matched_terms": matched or record.matched_terms,
        }
    )
    return _StaticJobDetailParseResult(
        record=enriched,
        body_source=body_source if include_description and body_source else "",
        description_length=len(static_description if include_description else ""),
        identity_title=title,
        identity_company=company,
    )


def _parse_static_job_detail_html(
    html: str,
    record: LinkedInVacancyRecord,
    *,
    include_description: bool,
    now: datetime | None = None,
) -> LinkedInVacancyRecord:
    return _parse_static_job_detail_html_result(
        html,
        record,
        include_description=include_description,
        now=now,
    ).record


def _static_detail_rejection(
    record: LinkedInVacancyRecord,
    *,
    include_description: bool,
) -> str:
    if include_description and not (record.description_full_text or "").strip():
        return "missing_description"
    if include_description and _is_incomplete_detail_body(record.description_full_text):
        return "incomplete_description"
    if record.published_at is None:
        return "missing_date"
    if not record.is_within_24_hours:
        return "outside_24_hours"
    return ""


def _classify_detail_probe_status(status_code: int) -> str:
    if status_code in {401, 403, 999}:
        return "detail_access_rejected"
    if status_code == 429:
        return "detail_rate_limited"
    if 500 <= status_code <= 599:
        return "detail_upstream_failure"
    return "detail_navigation_failure"


def _guest_detail_url(job_id: str) -> str:
    return f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"


def _extract_static_job_id(html: str) -> str:
    for pattern in (
        r"urn:li:jobPosting:(\d{6,})",
        r"/jobs/view/(\d{6,})",
        r"jobPosting/(\d{6,})",
    ):
        match = re.search(pattern, html or "")
        if match:
            return match.group(1)
    return ""


def _guest_identity_is_consistent(
    *,
    expected_record: LinkedInVacancyRecord,
    expected_job_id: str,
    html: str,
    parsed: _StaticJobDetailParseResult,
) -> bool:
    visible_job_id = _extract_static_job_id(html)
    if visible_job_id and expected_job_id and visible_job_id != expected_job_id:
        return False
    if not _identity_texts_compatible(
        expected_record.title,
        parsed.identity_title,
    ):
        return False
    if not _identity_texts_compatible(
        expected_record.company_name,
        parsed.identity_company,
    ):
        return False
    return True


def _pause_between_guest_retries(session, attempt: int) -> None:
    delay_ms = min(750, 250 * (attempt + 1))
    page = getattr(session, "page", None)
    if page is not None:
        try:
            _safe_page_pause(page, delay_ms)
            return
        except Exception:
            pass
    time.sleep(delay_ms / 1000)


def _try_guest_detail_body_recovery(
    session,
    record: LinkedInVacancyRecord,
    *,
    expected_record: LinkedInVacancyRecord,
    include_description: bool,
    now: datetime | None = None,
) -> _AuthenticatedDetailProbeResult:
    job_id = str(record.linkedin_job_id or "").strip()
    if not re.fullmatch(r"\d{1,20}", job_id):
        return _AuthenticatedDetailProbeResult(
            record=record,
            status_code=0,
            category="detail_navigation_failure",
            detail="job_id_mismatch",
            identity_consistent=False,
        )

    response = None
    last_status = 0
    retry_count = 0
    for attempt in range(_GUEST_DETAIL_MAX_RETRIES + 1):
        if attempt:
            retry_count += 1
        try:
            response = session.context.request.get(
                _guest_detail_url(job_id),
                fail_on_status_code=False,
                timeout=_GUEST_DETAIL_PROBE_TIMEOUT_MS,
            )
            status_code = int(getattr(response, "status", 0) or 0)
            last_status = status_code
            if status_code == 404:
                return _AuthenticatedDetailProbeResult(
                    record=record,
                    status_code=status_code,
                    category="detail_navigation_failure",
                    detail="guest_not_found",
                    guest_status_code=status_code,
                    guest_retry_count=retry_count,
                )
            if status_code in _GUEST_RETRY_STATUS_CODES:
                if attempt < _GUEST_DETAIL_MAX_RETRIES:
                    _pause_between_guest_retries(session, attempt)
                    continue
                return _AuthenticatedDetailProbeResult(
                    record=record,
                    status_code=status_code,
                    category=_classify_detail_probe_status(status_code),
                    detail="guest_retry_exhausted",
                    guest_status_code=status_code,
                    guest_retry_count=retry_count,
                )
            if not 200 <= status_code <= 299:
                return _AuthenticatedDetailProbeResult(
                    record=record,
                    status_code=status_code,
                    category=_classify_detail_probe_status(status_code),
                    guest_status_code=status_code,
                    guest_retry_count=retry_count,
                )

            html = response.text()
            parsed = _parse_static_job_detail_html_result(
                html,
                record,
                include_description=include_description,
                now=now,
            )
            identity_consistent = _guest_identity_is_consistent(
                expected_record=expected_record,
                expected_job_id=job_id,
                html=html,
                parsed=parsed,
            )
            if not identity_consistent:
                return _AuthenticatedDetailProbeResult(
                    record=record,
                    status_code=status_code,
                    category="detail_incomplete",
                    detail="missing_description",
                    guest_status_code=status_code,
                    guest_retry_count=retry_count,
                    identity_consistent=False,
                )
            if parsed.body_source:
                return _AuthenticatedDetailProbeResult(
                    record=parsed.record,
                    status_code=status_code,
                    category="ok",
                    detail="",
                    body_source="guest_html_container",
                    description_length=parsed.description_length,
                    guest_status_code=status_code,
                    guest_retry_count=retry_count,
                    identity_consistent=True,
                )
            return _AuthenticatedDetailProbeResult(
                record=record,
                status_code=status_code,
                category="detail_incomplete",
                detail="missing_description",
                guest_status_code=status_code,
                guest_retry_count=retry_count,
            )
        except Exception:
            if attempt < _GUEST_DETAIL_MAX_RETRIES:
                _pause_between_guest_retries(session, attempt)
                continue
            return _AuthenticatedDetailProbeResult(
                record=record,
                status_code=last_status,
                category="detail_navigation_failure",
                detail="request_failed",
                guest_status_code=last_status,
                guest_retry_count=retry_count,
            )
        finally:
            if response is not None:
                try:
                    response.dispose()
                except Exception:
                    pass
                response = None

    return _AuthenticatedDetailProbeResult(
        record=record,
        status_code=last_status,
        category="detail_navigation_failure",
        detail="request_failed",
        guest_status_code=last_status,
        guest_retry_count=retry_count,
    )


def _probe_linkedin_detail_with_authenticated_request(
    session,
    record: LinkedInVacancyRecord,
    *,
    include_description: bool,
    now: datetime | None = None,
) -> _AuthenticatedDetailProbeResult:
    """Fetch one exact job-detail URL with the authenticated context, then discard HTML."""

    job_id = str(record.linkedin_job_id or "").strip()
    source_url = validate_linkedin_jobs_url(record.canonical_url)
    source_job_id = linkedin_job_id_from_url(source_url)
    if job_id and source_job_id and job_id != source_job_id:
        return _AuthenticatedDetailProbeResult(
            record=record,
            status_code=0,
            category="detail_navigation_failure",
            detail="job_id_mismatch",
        )
    response = None
    try:
        response = session.context.request.get(
            source_url,
            fail_on_status_code=False,
            timeout=_DETAIL_PROBE_TIMEOUT_MS,
        )
        status_code = int(getattr(response, "status", 0) or 0)
        final_url = str(getattr(response, "url", "") or "")
        if is_linkedin_auth_checkpoint(final_url):
            raise LinkedInAuthRequiredError(
                "La sesión LinkedIn requiere login, 2FA o checkpoint manual."
            )
        try:
            final_url = validate_linkedin_jobs_url(final_url)
        except ValueError:
            return _AuthenticatedDetailProbeResult(
                record=record,
                status_code=status_code,
                category="detail_navigation_failure",
                detail="final_url_rejected",
            )
        final_job_id = linkedin_job_id_from_url(final_url)
        if job_id and final_job_id and job_id != final_job_id:
            return _AuthenticatedDetailProbeResult(
                record=record,
                status_code=status_code,
                category="detail_navigation_failure",
                detail="job_id_mismatch",
            )
        if not 200 <= status_code <= 299:
            return _AuthenticatedDetailProbeResult(
                record=record,
                status_code=status_code,
                category=_classify_detail_probe_status(status_code),
            )
        try:
            parsed = _parse_static_job_detail_html_result(
                response.text(),
                record,
                include_description=include_description,
                now=now or datetime.now(timezone.utc),
            )
        except Exception:
            return _AuthenticatedDetailProbeResult(
                record=record,
                status_code=status_code,
                category="detail_navigation_failure",
                detail="body_parse_failed",
            )
        enriched = parsed.record
        rejection = _static_detail_rejection(
            enriched,
            include_description=include_description,
        )
        if rejection in {"missing_description", "incomplete_description"}:
            guest_probe = _try_guest_detail_body_recovery(
                session,
                enriched,
                expected_record=record,
                include_description=include_description,
                now=now or datetime.now(timezone.utc),
            )
            if (
                getattr(guest_probe, "body_source", "")
                and guest_probe.identity_consistent
            ):
                enriched = guest_probe.record
                rejection = _static_detail_rejection(
                    enriched,
                    include_description=include_description,
                )
            return _AuthenticatedDetailProbeResult(
                record=enriched,
                status_code=status_code,
                category="ok" if not rejection else "detail_incomplete",
                detail=rejection,
                body_source=(
                    guest_probe.body_source
                    if getattr(guest_probe, "body_source", "")
                    else parsed.body_source
                ),
                description_length=max(
                    parsed.description_length,
                    guest_probe.description_length,
                ),
                guest_status_code=guest_probe.guest_status_code,
                guest_retry_count=guest_probe.guest_retry_count,
                identity_consistent=guest_probe.identity_consistent,
            )
        return _AuthenticatedDetailProbeResult(
            record=enriched,
            status_code=status_code,
            category="ok" if not rejection else "detail_incomplete",
            detail=rejection,
            body_source=parsed.body_source,
            description_length=parsed.description_length,
        )
    except LinkedInAuthRequiredError:
        raise
    except Exception:
        return _AuthenticatedDetailProbeResult(
            record=record,
            status_code=0,
            category="detail_navigation_failure",
            detail="request_failed",
        )
    finally:
        if response is not None:
            try:
                response.dispose()
            except Exception:
                pass


def _has_usable_detail_description(page) -> bool:
    description = _first_visible_raw_text(page, _DETAIL_DESCRIPTION_SELECTORS)
    return bool(description) and not _is_incomplete_detail_body(description)


def _normalize_workplace_type(value: str) -> str:
    normalized = (value or "").casefold()
    for label, tokens in (
        ("remote", ("remote", "remoto", "원격", "재택", "リモート")),
        ("hybrid", ("hybrid", "híbrido", "hibrido", "하이브리드", "ハイブリッド")),
        (
            "on-site",
            ("on-site", "onsite", "on site", "presencial", "출근", "オフィス勤務"),
        ),
    ):
        if any(token in normalized for token in tokens):
            return label
    return ""


def _normalize_detail_location(
    value: str,
    *,
    company_name: str,
    posted_at_text: str,
) -> str:
    segments = [
        segment.strip(" ·•|-")
        for segment in re.split(r"[·•\n]", value or "")
        if segment.strip(" ·•|-")
    ]
    for segment in segments:
        folded = segment.casefold()
        if company_name and folded == company_name.casefold():
            continue
        if posted_at_text and posted_at_text.casefold() in folded:
            continue
        if _clean_posted_at_text(segment) != segment:
            continue
        if parse_linkedin_relative_time(segment)[0] is not None:
            continue
        if _normalize_workplace_type(segment):
            continue
        if re.search(r"\b\d+\s+applicants?\b", folded):
            continue
        return segment
    return ""


def _extract_job_detail_from_current_page(
    page,
    record: LinkedInVacancyRecord,
    *,
    include_description: bool,
    now: datetime | None = None,
    diagnostics: LinkedInDetailDiagnosticsCollector | None = None,
    diagnostic_mode: str = "none",
) -> LinkedInVacancyRecord:
    detail_title = _normalize_repeated_title(
        _first_visible_text(page, _DETAIL_TITLE_SELECTORS)
    )
    company_name = _first_visible_text(page, _DETAIL_COMPANY_SELECTORS)
    raw_location = _first_visible_text(page, _DETAIL_LOCATION_SELECTORS)
    raw_workplace = _first_visible_text(page, _DETAIL_WORKPLACE_SELECTORS)
    workplace_type = _normalize_workplace_type(
        f"{raw_workplace} {raw_location} {detail_title}"
    )
    raw_visible_description = _first_visible_raw_text(
        page,
        _DETAIL_DESCRIPTION_SELECTORS,
    )
    transient_description = _normalize_transient_description(
        raw_visible_description
    )
    description_full_text = transient_description if include_description else ""
    description_excerpt = (
        _sanitize_description(transient_description)
        if include_description
        else ""
    )
    matched = sorted(
        {
            term
            for term in MATCH_TERMS
            if term in f"{record.title} {transient_description}".lower()
        }
    )
    posted_text = record.posted_at_text
    published_at = record.published_at
    confidence = record.freshness_confidence
    within_24h = record.is_within_24_hours
    if published_at is None:
        (
            detail_posted_text,
            published_at,
            confidence,
            within_24h,
        ) = _extract_detail_posted_date(page, now=now)
        if detail_posted_text:
            posted_text = detail_posted_text
    posted_text = _clean_posted_at_text(posted_text)
    location = _normalize_detail_location(
        raw_location,
        company_name=company_name,
        posted_at_text=posted_text,
    )
    inference_text = (
        f"{transient_description}\n{detail_title}\n{company_name}"
    ).strip()
    candidate_expectations, responsibilities = _extract_structured_sections(
        transient_description
    )
    diagnostics = diagnostics or get_active_detail_diagnostics()
    if diagnostics:
        diagnostics.record(
            record,
            phase="extraction",
            mode=diagnostic_mode,
            outcome="extracted",
            include_description=include_description,
            description_ready=bool(description_full_text),
            date_ready=published_at is not None,
        )
    return record.model_copy(
        update={
            "title": detail_title or _normalize_repeated_title(record.title),
            "company_name": company_name or record.company_name,
            "location": location or record.location,
            "workplace_type": workplace_type or record.workplace_type,
            "description_excerpt": description_excerpt,
            "description_full_text": description_full_text,
            "language_requirements": _infer_language_requirements(inference_text),
            "experience_requirements": _infer_experience_requirements(inference_text),
            "hard_skills": _infer_hard_skills(transient_description),
            "soft_skills": _infer_soft_skills(transient_description),
            "candidate_expectations": candidate_expectations,
            "responsibilities": responsibilities,
            "foreigner_acceptance": _infer_foreigner_acceptance(inference_text),
            "visa_status": _infer_visa_status(inference_text),
            "relocation_support": _infer_relocation_support(inference_text),
            "matched_terms": matched or record.matched_terms,
            "posted_at_text": posted_text,
            "published_at": published_at,
            "freshness_confidence": confidence,
            "is_within_24_hours": within_24h,
        }
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


def _respect_detail_click_cadence(
    page,
    *,
    last_detail_click_at: float | None,
    interval_ms: int,
    now_fn=None,
) -> None:
    if last_detail_click_at is None:
        return
    now_fn = now_fn or time.monotonic
    elapsed_ms = max(0, round((now_fn() - last_detail_click_at) * 1000))
    remaining_ms = max(0, interval_ms - elapsed_ms)
    if remaining_ms:
        _safe_page_pause(page, remaining_ms)


def _normalize_job_id(value: str) -> str:
    match = re.search(r"(\d+)\s*$", str(value or "").strip())
    if not match:
        return ""
    return match.group(1).lstrip("0") or "0"


def _safe_job_card_link(page, record: LinkedInVacancyRecord):
    expected_job_id = _normalize_job_id(
        record.linkedin_job_id or linkedin_job_id_from_url(record.canonical_url)
    )
    if not expected_job_id:
        return None
    canonicalize_linkedin_job_url(record.canonical_url)
    seen_hrefs: set[str] = set()
    for selector in _JOB_CARD_LINK_SELECTORS:
        try:
            matches = page.locator(selector)
            for index in range(matches.count()):
                locator = matches.nth(index)
                href = str(locator.get_attribute("href") or "").strip()
                if not href or href in seen_hrefs:
                    continue
                seen_hrefs.add(href)
                try:
                    canonical = canonicalize_linkedin_job_url(
                        urljoin("https://www.linkedin.com", href)
                    )
                except ValueError:
                    continue
                if _normalize_job_id(
                    linkedin_job_id_from_url(canonical)
                ) == expected_job_id:
                    return locator
        except Exception:
            continue
    return None


def _detail_panel_job_id_matches(page, expected_job_id: str) -> bool:
    expected_job_id = _normalize_job_id(expected_job_id)
    if not expected_job_id:
        return False
    try:
        current = urlparse(str(getattr(page, "url", "") or ""))
        if _normalize_job_id(
            linkedin_job_id_from_url(current.geturl())
        ) == expected_job_id:
            return True
        current_job_ids = parse_qs(current.query).get("currentJobId", [])
        if any(
            _normalize_job_id(current_job_id) == expected_job_id
            for current_job_id in current_job_ids
        ):
            return True
    except Exception:
        pass
    for selector in _DETAIL_PANEL_JOB_LINK_SELECTORS:
        try:
            matches = page.locator(selector)
            for index in range(matches.count()):
                href = str(matches.nth(index).get_attribute("href") or "").strip()
                try:
                    canonical = canonicalize_linkedin_job_url(
                        urljoin("https://www.linkedin.com", href)
                    )
                except ValueError:
                    continue
                if _normalize_job_id(
                    linkedin_job_id_from_url(canonical)
                ) == expected_job_id:
                    return True
        except Exception:
            continue
    return False


def _detail_panel_title_matches(page, expected_title: str) -> bool:
    actual = _normalize_repeated_title(
        _first_visible_text(page, _DETAIL_TITLE_SELECTORS)
    ).casefold()
    expected = _normalize_repeated_title(expected_title).casefold()
    if not actual or not expected:
        return False
    return actual == expected or actual in expected or expected in actual


def _wait_for_detail_panel_hydration(
    page,
    record: LinkedInVacancyRecord,
    *,
    include_description: bool,
    max_wait_ms: int = _DETAIL_HYDRATION_MAX_MS,
) -> str:
    expected_job_id = record.linkedin_job_id or linkedin_job_id_from_url(
        record.canonical_url
    )
    elapsed_ms = 0
    poll_index = 0
    saw_mismatched_panel = False
    while True:
        _raise_for_terminal_page_signal(page)
        id_matches = bool(expected_job_id) and _detail_panel_job_id_matches(
            page,
            expected_job_id,
        )
        title_matches = _detail_panel_title_matches(page, record.title)
        has_title = any(
            _locator_has_signal(page, selector, require_text=True)
            for selector in _DETAIL_TITLE_SELECTORS
        )
        has_description = (
            _has_usable_detail_description(page)
            if include_description
            else any(
                _locator_has_signal(page, selector, require_text=True)
                for selector in _DETAIL_DESCRIPTION_SELECTORS
            )
        )
        has_date = _extract_detail_posted_date(page)[1] is not None
        if id_matches and title_matches and has_date and (
            has_description or not include_description
        ):
            return "ready"
        if has_title and (not id_matches or not title_matches):
            saw_mismatched_panel = True
        if elapsed_ms >= max_wait_ms:
            return "stale" if saw_mismatched_panel else "timeout"
        interval_ms = _HYDRATION_POLL_INTERVALS_MS[
            min(poll_index, len(_HYDRATION_POLL_INTERVALS_MS) - 1)
        ]
        wait_ms = min(interval_ms, max_wait_ms - elapsed_ms)
        _safe_page_pause(page, wait_ms)
        elapsed_ms += wait_ms
        poll_index += 1


def _enrich_job_detail_via_panel(
    page,
    record: LinkedInVacancyRecord,
    *,
    card_link,
    include_description: bool,
    now: datetime | None = None,
    diagnostics: LinkedInDetailDiagnosticsCollector | None = None,
) -> LinkedInVacancyRecord:
    diagnostics = diagnostics or get_active_detail_diagnostics()
    if diagnostics:
        diagnostics.record(
            record,
            phase="start",
            mode="panel",
            outcome="started",
            include_description=include_description,
            date_ready=record.published_at is not None,
        )
    try:
        card_link.click(timeout=5000)
    except Exception as exc:
        if diagnostics:
            diagnostics.record(
                record,
                phase="wait_terminal",
                mode="panel",
                outcome="failed",
                include_description=include_description,
                date_ready=record.published_at is not None,
            )
        raise LinkedInDetailPanelError(
            "detail_network_failure"
            if _is_page_recoverable_error(_safe_error_label(exc))
            else "detail_click_failed",
            safe_label=_safe_error_label(exc),
        ) from exc
    state = _wait_for_detail_panel_hydration(
        page,
        record,
        include_description=include_description,
    )
    if diagnostics:
        diagnostics.record(
            record,
            phase="wait_terminal",
            mode="panel",
            outcome=state,
            include_description=include_description,
            description_ready=include_description and state == "ready",
            date_ready=record.published_at is not None or state == "ready",
        )
    if state == "stale":
        raise LinkedInDetailPanelError("stale_detail_panel")
    if state != "ready":
        raise LinkedInDetailPanelError("detail_panel_timeout")
    return _extract_job_detail_from_current_page(
        page,
        record,
        include_description=include_description,
        now=now,
        diagnostics=diagnostics,
        diagnostic_mode="panel",
    )


def _wait_for_detail_hydration(
    page,
    *,
    require_date: bool,
    require_description: bool = False,
    max_wait_ms: int = _DETAIL_HYDRATION_MAX_MS,
) -> str:
    elapsed_ms = 0
    poll_index = 0
    while True:
        _raise_for_terminal_page_signal(page)
        has_title = any(
            _locator_has_signal(page, selector, require_text=True)
            for selector in _DETAIL_TITLE_SELECTORS
        )
        has_description = (
            _has_usable_detail_description(page)
            if require_description
            else any(
                _locator_has_signal(page, selector, require_text=True)
                for selector in _DETAIL_DESCRIPTION_SELECTORS
            )
        )
        has_date = _extract_detail_posted_date(page)[1] is not None
        if require_date or require_description:
            if (not require_date or has_date) and (
                not require_description or has_description
            ):
                return "ready"
        elif has_title or has_description or has_date:
            return "ready"
        if elapsed_ms >= max_wait_ms:
            return "timeout"
        interval_ms = _HYDRATION_POLL_INTERVALS_MS[
            min(poll_index, len(_HYDRATION_POLL_INTERVALS_MS) - 1)
        ]
        wait_ms = min(interval_ms, max_wait_ms - elapsed_ms)
        _safe_page_pause(page, wait_ms)
        elapsed_ms += wait_ms
        poll_index += 1



__all__ = [
    "_AuthenticatedDetailProbeResult",
    "_enrich_job_detail",
    "_enrich_job_detail_via_panel",
    "_extract_detail_posted_date",
    "_extract_job_detail_from_current_page",
    "_first_visible_raw_text",
    "_first_visible_text",
    "_has_usable_detail_description",
    "_is_incomplete_detail_body",
    "_is_plausible_description",
    "_normalize_detail_location",
    "_normalize_transient_description",
    "_normalize_workplace_type",
    "_parse_static_job_detail_html",
    "_probe_linkedin_detail_with_authenticated_request",
    "_respect_detail_click_cadence",
    "_safe_job_card_link",
    "_wait_for_detail_hydration",
    "_wait_for_detail_panel_hydration",
]
