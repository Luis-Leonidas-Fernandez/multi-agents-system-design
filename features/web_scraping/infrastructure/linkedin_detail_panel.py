"""Detail-panel extraction and enrichment helpers for LinkedIn jobs."""
from __future__ import annotations

from datetime import datetime
import re
import time
from urllib.parse import parse_qs, urljoin, urlparse

from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord
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
    linkedin_job_id_from_url,
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
) -> LinkedInVacancyRecord:
    page.goto(record.canonical_url, wait_until="domcontentloaded", timeout=30000)
    _validate_authenticated_page(page)
    _wait_for_detail_hydration(
        page,
        require_date=record.published_at is None,
    )
    return _extract_job_detail_from_current_page(
        page,
        record,
        include_description=include_description,
        now=now,
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


def _safe_job_card_link(page, record: LinkedInVacancyRecord):
    expected_job_id = record.linkedin_job_id or linkedin_job_id_from_url(
        record.canonical_url
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
                if linkedin_job_id_from_url(canonical) == expected_job_id:
                    return locator
        except Exception:
            continue
    return None


def _detail_panel_job_id_matches(page, expected_job_id: str) -> bool:
    try:
        current = urlparse(str(getattr(page, "url", "") or ""))
        if linkedin_job_id_from_url(current.geturl()) == expected_job_id:
            return True
        current_job_ids = parse_qs(current.query).get("currentJobId", [])
        if expected_job_id in current_job_ids:
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
                if linkedin_job_id_from_url(canonical) == expected_job_id:
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
        has_description = any(
            _locator_has_signal(page, selector, require_text=True)
            for selector in _DETAIL_DESCRIPTION_SELECTORS
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
) -> LinkedInVacancyRecord:
    try:
        card_link.click(timeout=5000)
    except Exception as exc:
        raise LinkedInDetailPanelError(
            "detail_network_failure"
            if _is_page_recoverable_error(_safe_error_label(exc))
            else "card_click_failed",
            safe_label=_safe_error_label(exc),
        ) from exc
    state = _wait_for_detail_panel_hydration(
        page,
        record,
        include_description=include_description,
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
    )


def _wait_for_detail_hydration(
    page,
    *,
    require_date: bool,
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
        has_description = any(
            _locator_has_signal(page, selector, require_text=True)
            for selector in _DETAIL_DESCRIPTION_SELECTORS
        )
        has_date = _extract_detail_posted_date(page)[1] is not None
        if require_date:
            if has_date:
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
    "_enrich_job_detail",
    "_enrich_job_detail_via_panel",
    "_extract_detail_posted_date",
    "_extract_job_detail_from_current_page",
    "_first_visible_raw_text",
    "_first_visible_text",
    "_is_incomplete_detail_body",
    "_normalize_detail_location",
    "_normalize_transient_description",
    "_normalize_workplace_type",
    "_respect_detail_click_cadence",
    "_safe_job_card_link",
    "_wait_for_detail_hydration",
    "_wait_for_detail_panel_hydration",
]
