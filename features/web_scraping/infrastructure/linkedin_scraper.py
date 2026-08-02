"""Extractor secuencial y read-only de vacantes LinkedIn autenticadas."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
import re
import time
import unicodedata
from typing import Any, Callable
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

from features.web_scraping.domain.linkedin_models import (
    LinkedInJobsRequest,
    LinkedInParseDiagnostics,
    LinkedInQueryTiming,
    LinkedInRejectedRecord,
    LinkedInVacancyRecord,
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


CONSOLIDATED_QUERY_PLANS = (
    (
        "AI/ML/Data/GenAI",
        (
            '"AI Engineer" OR "Artificial Intelligence Engineer" '
            'OR "Machine Learning Engineer" OR "ML Engineer" '
            'OR "Deep Learning Engineer" OR "DL Engineer" '
            'OR "Data Scientist" OR "MLOps Engineer" '
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
            'OR "Applied Scientist" OR "Applied AI Engineer" OR "AI Specialist" '
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
MATCH_TERMS = (
    "ai",
    "artificial intelligence",
    "inteligencia artificial",
    "machine learning",
    "deep learning",
    "llm",
    "generative ai",
    "genai",
    "data science",
    "data scientist",
    "pytorch",
    "tensorflow",
    "rag",
    "mlops",
    "ai product",
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
_LOGIN_SIGNAL_SELECTORS = (
    "form[action*='login']",
    "input[name='session_key']",
    "input[name='session_password']",
)
_BLOCK_SIGNAL_SELECTORS = (
    "iframe[src*='captcha']",
    "[id*='captcha']",
    "[class*='captcha']",
    ".challenge-dialog",
    "text=Security verification",
    "text=Verificación de seguridad",
    "text=CAPTCHA",
)
_HYDRATION_POLL_INTERVALS_MS = (100, 150, 250, 400, 600, 800, 1000)
_SEARCH_HYDRATION_MAX_MS = 6500
_DETAIL_HYDRATION_MAX_MS = 6000
_QUERY_BACKOFF_BASE_MS = 1000
_QUERY_BACKOFF_MAX_MS = 3000
_QUERY_PAGE_RECOVERY_THRESHOLD = 2
_QUERY_NETWORK_CIRCUIT_THRESHOLD = 3
_DETAIL_NETWORK_CIRCUIT_THRESHOLD = 2
_HARD_MAX_DETAIL_REQUESTS = 30
_HARD_MAX_TOTAL_QUERY_ATTEMPTS = 8
_QUERY_PROBE_TIMEOUT_MS = 15000
_DETAIL_CLICK_INTERVAL_MIN_MS = 750
_DETAIL_CLICK_INTERVAL_MAX_MS = 3000
_HARD_MAX_QUERIES_PER_LOCATION = len(CONSOLIDATED_QUERY_PLANS)


class LinkedInAuthRequiredError(RuntimeError):
    pass


class LinkedInBlockedError(RuntimeError):
    pass


class LinkedInDetailPanelError(RuntimeError):
    def __init__(self, reason: str, *, safe_label: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.safe_label = safe_label


def _parse_iso_datetime(value: str) -> datetime | None:
    candidate = (value or "").strip()
    if not candidate:
        return None
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_linkedin_relative_time(
    text: str,
    *,
    now: datetime | None = None,
    structured_datetime: str = "",
) -> tuple[datetime | None, str, bool]:
    reference = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    structured = _parse_iso_datetime(structured_datetime)
    if structured is not None:
        age = reference - structured
        return structured, "high", timedelta(0) <= age <= timedelta(hours=24)

    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not normalized:
        return None, "low", False
    if any(
        token in normalized
        for token in (
            "just now",
            "moments ago",
            "a few seconds ago",
            "ahora mismo",
            "hace un momento",
            "hace unos segundos",
            "recién",
            "recien",
        )
    ):
        return reference, "medium", True
    if normalized in {"today", "hoy"} or "today" in normalized or "publicado hoy" in normalized:
        return reference.replace(hour=0, minute=0, second=0, microsecond=0), "medium", True

    patterns = (
        (r"(\d+)\s*(?:second|seconds|segundo|segundos)\b", "seconds"),
        (r"(\d+)\s*(?:minute|minutes|min|mins|minuto|minutos)\b", "minutes"),
        (r"(\d+)\s*(?:hour|hours|hr|hrs|hora|horas)\b", "hours"),
        (r"(\d+)\s*(?:day|days|dia|dias|día|días)\b", "days"),
    )
    for pattern, unit in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        amount = int(match.group(1))
        delta = timedelta(**{unit: amount})
        published = reference - delta
        return published, "medium", delta <= timedelta(hours=24)
    return None, "low", False


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


def _text(node: Any, selectors: tuple[str, ...]) -> str:
    for selector in selectors:
        found = node.select_one(selector)
        if found is not None:
            value = found.get_text(" ", strip=True)
            if value:
                return value
    return ""


def _workplace_type(card: Any) -> str:
    metadata = " ".join(
        node.get_text(" ", strip=True)
        for node in card.select(
            ".job-card-container__metadata-item, "
            ".job-search-card__location, "
            ".base-search-card__metadata"
        )
    ).lower()
    for label, tokens in (
        ("remote", ("remote", "remoto", "remota")),
        ("hybrid", ("hybrid", "híbrido", "hibrido", "híbrida", "hibrida")),
        ("on-site", ("on-site", "onsite", "presencial")),
    ):
        if any(token in metadata for token in tokens):
            return label
    return ""


_CARD_SELECTORS = (
    ".job-card-container",
    "li[data-occludable-job-id]",
    ".scaffold-layout__list-item",
    "[data-job-id]",
)
_JOB_LINK_SELECTOR = (
    "a.job-card-list__title--link, "
    "a.base-card__full-link, "
    "a[href*='/jobs/view/']"
)


def _increment_reason(reasons: dict[str, int], reason: str) -> None:
    reasons[reason] = reasons.get(reason, 0) + 1


def _normalize_repeated_title(value: str) -> str:
    title = re.sub(r"\s+", " ", value or "").strip()
    title = re.sub(r"\s+with verification\s*$", "", title, flags=re.IGNORECASE)
    words = title.split()
    if len(words) >= 2 and len(words) % 2 == 0:
        midpoint = len(words) // 2
        if [word.casefold() for word in words[:midpoint]] == [
            word.casefold() for word in words[midpoint:]
        ]:
            return " ".join(words[:midpoint])
    return title


_RELATIVE_DATE_PATTERNS = (
    r"\bhace\s+(?:unos?\s+segundos?|un\s+momento|\d+\s+(?:minutos?|horas?|d[ií]as?))\b",
    r"\b(?:just now|moments ago|a few seconds ago|\d+\s+(?:minutes?|hours?|days?)\s+ago)\b",
    r"\b(?:hoy|today)\b",
    r"\b\d+\s*(?:시간|분|일)\s*전\b",
    r"\b\d+\s*(?:時間|分|日)前\b",
)


def _clean_posted_at_text(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value or "").strip()
    for pattern in _RELATIVE_DATE_PATTERNS:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return normalized


def _parse_linkedin_jobs_html_with_diagnostics(
    html: str,
    *,
    source_url: str,
    now: datetime | None = None,
) -> tuple[list[LinkedInVacancyRecord], LinkedInParseDiagnostics]:
    from bs4 import BeautifulSoup

    validated_source_url = canonicalize_linkedin_job_url(source_url)
    soup = BeautifulSoup(html or "", "html.parser")
    selector_counts = {
        selector: len(soup.select(selector))
        for selector in _CARD_SELECTORS
    }
    href_count = len(soup.select("a[href*='/jobs/view/']"))
    candidates: list[Any] = []
    seen_nodes: set[int] = set()
    for selector in _CARD_SELECTORS:
        for node in soup.select(selector):
            node_key = id(node)
            if node_key in seen_nodes:
                continue
            seen_nodes.add(node_key)
            candidates.append(node)
    for link in soup.select("a[href*='/jobs/view/']"):
        node_key = id(link)
        if node_key in seen_nodes:
            continue
        seen_nodes.add(node_key)
        candidates.append(link)

    records: list[LinkedInVacancyRecord] = []
    discard_reasons: dict[str, int] = {}
    seen_jobs: set[str] = set()
    for card in candidates:
        link = card if getattr(card, "name", "") == "a" else card.select_one(
            _JOB_LINK_SELECTOR
        )
        href = str(link.get("href") or "").strip() if link is not None else ""
        if not href:
            _increment_reason(discard_reasons, "missing_href")
            continue
        try:
            absolute_href = urljoin("https://www.linkedin.com", href)
            canonical_url = canonicalize_linkedin_job_url(absolute_href)
        except ValueError:
            _increment_reason(discard_reasons, "invalid_url")
            continue
        job_key = linkedin_job_id_from_url(canonical_url) or canonical_url
        if job_key in seen_jobs:
            _increment_reason(discard_reasons, "duplicate_wrapper")
            continue
        title = _text(
            card,
            (
                ".job-card-list__title",
                ".base-search-card__title",
                ".job-card-container__link",
            ),
        ) or (link.get_text(" ", strip=True) if link is not None else "")
        title = _normalize_repeated_title(title)
        if not title:
            _increment_reason(discard_reasons, "missing_title")
            continue
        seen_jobs.add(job_key)
        company = _text(
            card,
            (
                ".job-card-container__primary-description",
                ".base-search-card__subtitle",
                ".job-card-container__company-name",
            ),
        )
        location = _text(
            card,
            (
                ".job-card-container__metadata-item",
                ".job-search-card__location",
                ".base-search-card__metadata",
            ),
        )
        time_node = card.select_one("time")
        posted_text = time_node.get_text(" ", strip=True) if time_node is not None else _text(
            card,
            (".job-card-container__listed-time", ".job-search-card__listdate"),
        )
        posted_text = _clean_posted_at_text(posted_text)
        structured = str(time_node.get("datetime") or "") if time_node is not None else ""
        published_at, confidence, within_24h = parse_linkedin_relative_time(
            posted_text,
            now=now,
            structured_datetime=structured,
        )
        blob = f"{title} {company}".lower()
        matched = sorted({term for term in MATCH_TERMS if term in blob})
        records.append(
            LinkedInVacancyRecord(
                linkedin_job_id=linkedin_job_id_from_url(canonical_url),
                title=title,
                company_name=company,
                location=location,
                workplace_type=_workplace_type(card),
                posted_at_text=posted_text,
                published_at=published_at,
                freshness_confidence=confidence,
                is_within_24_hours=within_24h,
                canonical_url=canonical_url,
                source_url=validated_source_url,
                matched_terms=matched,
            )
        )
    return records, LinkedInParseDiagnostics(
        selector_counts=selector_counts,
        href_count=href_count,
        candidate_count=len(candidates),
        parseable_candidate_count=len(records),
        discard_reasons=discard_reasons,
    )


def parse_linkedin_jobs_html(
    html: str,
    *,
    source_url: str,
    now: datetime | None = None,
) -> list[LinkedInVacancyRecord]:
    records, _ = _parse_linkedin_jobs_html_with_diagnostics(
        html,
        source_url=source_url,
        now=now,
    )
    return records


def dedupe_linkedin_vacancies(
    records: list[LinkedInVacancyRecord],
) -> list[LinkedInVacancyRecord]:
    deduped: list[LinkedInVacancyRecord] = []
    seen: set[str] = set()
    for record in records:
        key = record.linkedin_job_id or record.canonical_url
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _validate_authenticated_page(page) -> None:
    current_url = str(page.url or "")
    if is_linkedin_auth_checkpoint(current_url):
        raise LinkedInAuthRequiredError(
            "La sesión LinkedIn requiere login, 2FA o checkpoint manual."
        )
    try:
        validate_linkedin_jobs_url(current_url)
    except ValueError as exc:
        raise LinkedInBlockedError(
            "LinkedIn redirigió fuera del área de empleos permitida."
        ) from exc
    body_text = (page.locator("body").inner_text(timeout=5000) or "").lower()
    if any(token in body_text for token in ("security verification", "verificación de seguridad", "captcha")):
        raise LinkedInBlockedError(
            "LinkedIn solicitó una verificación manual. El scraper fue detenido."
        )
    cookie_names: set[str] = set()
    try:
        cookie_names = {
            str(cookie.get("name") or "")
            for cookie in page.context.cookies()
            if isinstance(cookie, dict)
        }
    except Exception:
        pass
    authenticated_marker = False
    for selector in (
        "#global-nav",
        ".global-nav",
        "a[href*='/mynetwork/']",
        "a[href*='/messaging/']",
    ):
        try:
            if page.locator(selector).count() > 0:
                authenticated_marker = True
                break
        except Exception:
            continue
    if "li_at" not in cookie_names and not authenticated_marker:
        raise LinkedInAuthRequiredError(
            "No se encontró una sesión autenticada válida de LinkedIn."
        )


def _safe_auth_diagnostic(page: object) -> dict[str, Any]:
    """Extrae señales booleanas y path; nunca cookies, query strings ni HTML."""

    raw_url = str(getattr(page, "url", "") or "")
    parsed = urlparse(raw_url)
    hostname = (parsed.hostname or "").lower()
    linkedin_host = hostname == "linkedin.com" or hostname.endswith(".linkedin.com")
    cookie_names: set[str] = set()
    try:
        cookie_names = {
            str(cookie.get("name") or "")
            for cookie in page.context.cookies()
            if isinstance(cookie, dict)
        }
    except Exception:
        pass

    def has_selector(selector: str) -> bool:
        try:
            return page.locator(selector).count() > 0
        except Exception:
            return False

    return {
        "final_path": (parsed.path or "/")[:160] if linkedin_host else "",
        "linkedin_host": linkedin_host,
        "is_auth_checkpoint": (
            is_linkedin_auth_checkpoint(raw_url) if linkedin_host else False
        ),
        "has_li_at": "li_at" in cookie_names,
        "has_global_nav": any(
            has_selector(selector)
            for selector in (
                "#global-nav",
                ".global-nav",
                "a[href*='/mynetwork/']",
                "a[href*='/messaging/']",
            )
        ),
        "has_login_form": any(
            has_selector(selector)
            for selector in (
                "form[action*='login']",
                "input[name='session_key']",
                "input[name='session_password']",
            )
        ),
    }


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


def _sanitize_description(value: str, limit: int = 1000) -> str:
    normalized = re.sub(r"\s+", " ", value or "").strip()
    normalized = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "[redacted-email]",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"https?://\S+", "[redacted-url]", normalized)
    normalized = re.sub(
        r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)",
        "[redacted-phone]",
        normalized,
    )
    return normalized[:limit].strip()


def _truncate_at_word(value: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", value or "").strip()
    if len(normalized) <= limit:
        return normalized
    cutoff = normalized[:limit].rsplit(" ", 1)[0].strip()
    return cutoff or normalized[:limit].strip()


def _sanitize_structured_item(value: str, limit: int = 180) -> str:
    sanitized = _sanitize_description(value, limit=max(limit * 3, limit))
    return _truncate_at_word(sanitized, limit)


def _is_incomplete_detail_body(value: str) -> bool:
    normalized = re.sub(r"\s+", " ", value or "").strip().casefold()
    return not normalized or normalized in {
        "acerca del empleo",
        "about the job",
        "about this job",
        "job description",
    }


_SECTION_PATTERNS = {
    "candidate_expectations": (
        r"requirements?",
        r"qualifications?",
        r"required qualifications?",
        r"minimum qualifications?",
        r"required skills?(?: and experience)?",
        r"what you(?:'|’)ll need(?: to succeed)?",
        r"what you need to succeed",
        r"what we need to see",
        r"who you are",
        r"who are we looking for\??",
        r"we are looking for",
        r"what you bring",
        r"応募資格",
        r"必須要件",
        r"歓迎要件",
        r"求める人物像",
        r"必要な経験",
        r"자격요건",
        r"지원자격",
        r"필수요건",
        r"우대사항",
    ),
    "responsibilities": (
        r"responsibilities",
        r"key responsibilities",
        r"role responsibilities",
        r"duties(?: and responsibilities)?",
        r"tasks",
        r"what you(?:'|’)ll do",
        r"what you will do",
        r"what you(?:'|’)ll be doing",
        r"in this role,? you(?:'|’)ll get to",
        r"in this role",
        r"業務内容",
        r"仕事内容",
        r"従事すべき業務の内容",
        r"担当業務",
        r"담당업무",
        r"주요업무",
        r"주요 업무",
    ),
}
_OTHER_SECTION_PATTERN = re.compile(
    r"^(?:about(?: us| the role)?|about .+|benefits?|preferred qualifications?|"
    r"nice to have|ways to stand out(?: from the crowd)?|company|company information|"
    r"compensation(?: & benefits)?|salary|location|working conditions?|work hours?|"
    r"holidays?(?: and leave)?|probation period|insurance|additional information|"
    r"discover more.*|equal opportunity employer|disclaimer|job type|"
    r"待遇|福利厚生|会社概要|勤務地|勤務時間|休日|給与|雇用形態|회사소개|혜택|복리후생)\s*:?\s*$",
    re.IGNORECASE,
)


def _clean_section_heading_candidate(line: str) -> str:
    cleaned = re.sub(r"^[#>*•●▪◦\-\s]+", "", line or "").strip()
    cleaned = cleaned.strip("[](){}【】「」『』〈〉《》")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _section_heading(line: str) -> tuple[str, str] | None:
    cleaned = _clean_section_heading_candidate(line)
    for kind, patterns in _SECTION_PATTERNS.items():
        for pattern in patterns:
            match = re.match(
                rf"^(?:{pattern})\s*:?\s*(.*)$",
                cleaned,
                re.IGNORECASE,
            )
            if match:
                return kind, match.group(1).strip()
    return None


def _bounded_unique_items(
    values: list[str],
    *,
    limit: int = 6,
    item_limit: int = 180,
) -> list[str]:
    bounded: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = re.sub(r"^[#>*•●▪◦\-\s]+", "", value or "").strip()
        item = re.sub(r"^\d+[.)]\s+", "", item).strip()
        item = _sanitize_structured_item(item, limit=item_limit)
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        bounded.append(item)
        if len(bounded) >= limit:
            break
    return bounded


def _extract_structured_sections(text: str) -> tuple[list[str], list[str]]:
    sections: dict[str, list[str]] = {
        "candidate_expectations": [],
        "responsibilities": [],
    }
    active = ""
    for line in (text or "").splitlines():
        heading = _section_heading(line)
        if heading:
            active, inline_value = heading
            if inline_value:
                sections[active].append(inline_value)
            continue
        if _OTHER_SECTION_PATTERN.match(line.strip()):
            active = ""
            continue
        if active and line.strip():
            sections[active].append(line)
    return (
        _bounded_unique_items(sections["candidate_expectations"]),
        _bounded_unique_items(sections["responsibilities"]),
    )


_HARD_SKILL_PATTERNS = (
    ("Python", r"\bpython\b"),
    ("SQL", r"\bsql\b"),
    ("R", r"(?<![\w.])R(?![\w.])"),
    ("Java", r"\bjava\b"),
    ("C++", r"(?<!\w)c\+\+(?!\w)"),
    ("PyTorch", r"\bpytorch\b"),
    ("TensorFlow", r"\btensorflow\b"),
    ("scikit-learn", r"\b(?:scikit-learn|sklearn)\b"),
    ("Spark", r"\b(?:apache\s+)?spark\b"),
    ("Databricks", r"\bdatabricks\b"),
    ("AWS", r"\b(?:aws|amazon web services)\b"),
    ("GCP", r"\b(?:gcp|google cloud(?: platform)?)\b"),
    ("Azure", r"\b(?:microsoft\s+)?azure\b"),
    ("Docker", r"\bdocker\b"),
    ("Kubernetes", r"\b(?:kubernetes|k8s)\b"),
    ("Git", r"\bgit\b"),
    ("MLOps", r"\bmlops\b"),
    ("LLM", r"\b(?:llms?|large language models?)\b"),
    ("RAG", r"\b(?:rag|retrieval[- ]augmented generation)\b"),
    ("NLP", r"\b(?:nlp|natural language processing)\b"),
    ("Computer Vision", r"\bcomputer vision\b|コンピュータビジョン|컴퓨터 비전"),
    ("Data pipelines", r"\bdata pipelines?\b|データパイプライン|데이터 파이프라인"),
    (
        "Vector DB",
        r"\b(?:vector (?:db|database)s?|pinecone|weaviate|milvus|pgvector)\b",
    ),
    ("Power BI", r"\bpower\s*bi\b"),
    ("Tableau", r"\btableau\b"),
    ("Looker", r"\blooker\b"),
    ("BI", r"\bbusiness intelligence\b"),
    ("pandas", r"\bpandas\b"),
    ("NumPy", r"\bnumpy\b"),
    ("Hugging Face", r"\bhugging\s*face\b"),
    ("LangChain", r"\blangchain\b"),
    ("MLflow", r"\bmlflow\b"),
    ("Airflow", r"\b(?:apache\s+)?airflow\b"),
    ("Kafka", r"\b(?:apache\s+)?kafka\b"),
    ("Snowflake", r"\bsnowflake\b"),
)


def _infer_hard_skills(text: str) -> list[str]:
    return [
        label
        for label, pattern in _HARD_SKILL_PATTERNS
        if re.search(pattern, text or "", re.IGNORECASE)
    ][:40]


_SOFT_SKILL_PATTERNS = (
    ("Communication", r"\bcommunication skills?\b|コミュニケーション(?:能力|スキル)|의사소통(?: 능력)?"),
    (
        "Collaboration",
        r"\b(?:collaboration|collaborative|teamwork)\b|"
        r"協調性|チームワーク|협업|팀워크",
    ),
    (
        "Problem solving",
        r"\bproblem[- ]solving\b|問題解決(?:能力)?|문제 해결(?: 능력)?",
    ),
    ("Ownership", r"\bownership\b|オーナーシップ|주인의식"),
    ("Leadership", r"\bleadership\b|リーダーシップ|리더십"),
    ("Adaptability", r"\badaptability\b|\badaptable\b|適応力|적응력"),
    (
        "Critical thinking",
        r"\bcritical thinking\b|批判的思考|クリティカルシンキング|비판적 사고",
    ),
    (
        "Stakeholder management",
        r"\bstakeholder(?: management)?\b|ステークホルダー|이해관계자",
    ),
    (
        "Cross-functional collaboration",
        r"\bcross[- ]functional\b|部門横断|クロスファンクショナル|부서 간 협업",
    ),
)


def _infer_soft_skills(text: str) -> list[str]:
    return [
        label
        for label, pattern in _SOFT_SKILL_PATTERNS
        if re.search(pattern, text or "", re.IGNORECASE)
    ][:20]


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


def _infer_explicit_status(
    text: str,
    *,
    positive_patterns: tuple[str, ...],
    negative_patterns: tuple[str, ...],
    mention_pattern: str,
    positive_value: str,
    negative_value: str,
) -> str:
    positive = any(re.search(pattern, text, re.IGNORECASE) for pattern in positive_patterns)
    negative = any(re.search(pattern, text, re.IGNORECASE) for pattern in negative_patterns)
    if positive and negative:
        return "ambiguous"
    if positive:
        return positive_value
    if negative:
        return negative_value
    if re.search(mention_pattern, text, re.IGNORECASE):
        return "ambiguous"
    return "unknown"


def _infer_visa_status(text: str) -> str:
    return _infer_explicit_status(
        text,
        positive_patterns=(
            r"\b(?:full\s+)?visa sponsorship\b.{0,80}\b(?:available|provided|offered)\b",
            r"\bvisa sponsorship (?:is )?(?:available|provided|offered)\b",
            r"\b(?:offer|provide) visa sponsorship\b",
            r"\b(?:we|company|employer) (?:can |will )?sponsor\b.{0,30}\bvisa\b",
            r"\bvisa (?:support|sponsorship) (?:available|provided|offered)\b",
            r"ビザ(?:サポート|支援)(?:あり|可能|提供)",
            r"(?:就労|勤務)ビザ.{0,20}(?:支援|サポート|取得可能)",
            r"ビザスポンサー(?:可能|あり|提供)",
            r"비자\s*(?:지원|스폰서십)(?:\s*(?:가능|제공))",
            r"(?:취업|근무)\s*비자.{0,20}(?:지원|발급 가능)",
        ),
        negative_patterns=(
            r"\bno visa sponsorship\b",
            r"\b(?:unable|cannot|can't|do not|does not|won't) sponsor\b",
            r"\bsponsorship (?:is )?not available\b",
            r"\bwithout (?:visa )?sponsorship\b",
            r"ビザ(?:サポート|支援)(?:不可|なし)",
            r"ビザスポンサー(?:不可|なし)",
            r"(?:就労|勤務)ビザ.{0,20}自己負担",
            r"비자\s*(?:지원|스폰서십)\s*(?:불가|없음)",
            r"(?:취업|근무)\s*비자.{0,20}(?:지원 불가|본인 부담)",
        ),
        mention_pattern=r"\bvisa\b|sponsorship|ビザ|비자",
        positive_value="sponsorship",
        negative_value="no_sponsorship",
    )


def _infer_relocation_support(text: str) -> str:
    return _infer_explicit_status(
        text,
        positive_patterns=(
            r"\brelocation (?:assistance|support|package)\b.{0,80}\b(?:available|provided|offered)\b",
            r"\brelocation (?:assistance|support|package) "
            r"(?:is )?(?:available|provided|offered)\b",
            r"\bwe (?:can |will )?(?:support|assist with) relocation\b",
            r"転居(?:支援|サポート)(?:あり|可能|提供)",
            r"引越し(?:支援|サポート|費用)(?:あり|可能|提供)",
            r"이주\s*(?:지원|패키지)(?:\s*(?:가능|제공))",
            r"이전\s*(?:지원|비용)(?:\s*(?:가능|제공))",
        ),
        negative_patterns=(
            r"\bno relocation (?:assistance|support|package)?\b",
            r"\brelocation (?:is )?not (?:available|provided|offered)\b",
            r"転居(?:支援|サポート)(?:不可|なし)",
            r"引越し(?:支援|サポート)(?:不可|なし)",
            r"이주\s*(?:지원|패키지)\s*(?:불가|없음)",
            r"이전\s*(?:지원|비용)\s*(?:불가|없음)",
        ),
        mention_pattern=r"\brelocation\b|転居|이주",
        positive_value="yes",
        negative_value="no",
    )


def _infer_foreigner_acceptance(text: str) -> str:
    return _infer_explicit_status(
        text,
        positive_patterns=(
            r"\b(?:foreign|overseas|international) "
            r"(?:applicants?|candidates?) (?:are )?(?:welcome|accepted|eligible)\b",
            r"\binternational applications?\b.{0,80}\b(?:welcome|accepted|eligible)\b",
            r"\bwe welcome both local and international applications?\b",
            r"\bapplications? from abroad (?:are )?(?:welcome|accepted)\b",
            r"外国人(?:応募者|候補者)?(?:歓迎|応募可)",
            r"海外(?:在住者|応募者)(?:歓迎|応募可)",
            r"国籍不問",
            r"외국인\s*(?:지원자)?\s*(?:환영|지원 가능)",
            r"해외\s*(?:거주자|지원자)\s*(?:환영|지원 가능)",
            r"국적\s*무관",
        ),
        negative_patterns=(
            r"\b(?:foreign|overseas|international) "
            r"(?:applicants?|candidates?) (?:are )?not (?:accepted|eligible)\b",
            r"\bdomestic applicants? only\b",
            r"\bmust already (?:reside|be based) in\b",
            r"外国人(?:応募者|候補者)?(?:不可|対象外)",
            r"海外(?:在住者|応募者)(?:不可|対象外)",
            r"日本国内在住者(?:のみ|限定)",
            r"외국인\s*(?:지원자)?\s*(?:불가|대상 아님)",
            r"해외\s*(?:거주자|지원자)\s*(?:불가|대상 아님)",
            r"국내\s*거주자만",
        ),
        mention_pattern=(
            r"\b(?:foreign|overseas|international) (?:applicants?|candidates?)\b"
            r"|外国人|海外(?:在住者|応募者)|외국인|해외\s*(?:거주자|지원자)"
        ),
        positive_value="yes",
        negative_value="no",
    )


def _infer_language_requirements(text: str) -> list[str]:
    requirements: list[str] = []

    def add(value: str) -> None:
        if value not in requirements:
            requirements.append(value)

    japanese_level = re.search(
        r"(?:JLPT\s*)?(N[1-5])(?:\s*(?:以上|レベル))?",
        text,
        re.IGNORECASE,
    )
    if japanese_level and re.search(r"JLPT|日本語|Japanese", text, re.IGNORECASE):
        add(f"Japanese (JLPT {japanese_level.group(1).upper()})")
    korean_level = re.search(
        r"\bTOPIK\s*(?:level|급)?\s*([1-6])(?:급)?\b",
        text,
        re.IGNORECASE,
    )
    if korean_level:
        add(f"Korean (TOPIK {korean_level.group(1)})")

    language_specs = (
        ("Japanese", r"Japanese|日本語|일본어"),
        ("Korean", r"Korean|韓国語|한국어"),
        ("English", r"English|英語|영어"),
    )
    qualifier_specs = (
        ("business", r"business(?:[- ]level)?|ビジネスレベル|비즈니스\s*수준"),
        ("native", r"native(?:[- ]level)?|ネイティブ|원어민"),
        ("fluent", r"fluent|流暢|유창"),
        (
            "conversational",
            r"conversational(?:[- ]level)?|日常会話|회화\s*(?:수준|가능)",
        ),
        (
            "professional",
            r"professional(?:[- ]level)?|業務レベル|업무\s*수준",
        ),
    )
    requirement_cue = (
        r"required|must|preferred|proficiency|必須|歓迎|必要|"
        r"필수|우대|능통|가능"
    )
    for language, alias in language_specs:
        occurrences = list(re.finditer(alias, text, re.IGNORECASE))
        if not occurrences:
            continue
        qualifier = ""
        explicit = False
        for occurrence in occurrences:
            line_start = text.rfind("\n", 0, occurrence.start()) + 1
            line_end = text.find("\n", occurrence.end())
            if line_end < 0:
                line_end = len(text)
            start = max(line_start, occurrence.start() - 80)
            end = min(line_end, occurrence.end() + 80)
            context = text[start:end]
            explicit = explicit or bool(
                re.search(requirement_cue, context, re.IGNORECASE)
            )
            for label, pattern in qualifier_specs:
                if re.search(pattern, context, re.IGNORECASE):
                    qualifier = label
                    explicit = True
                    break
            if qualifier:
                break
        if qualifier:
            add(f"{language} ({qualifier})")
        elif explicit:
            add(language)
    return requirements


def _infer_experience_requirements(text: str) -> list[str]:
    patterns = (
        r"[^.;\n]{0,70}\b(?:at least|minimum(?: of)?|min\.?)?\s*"
        r"\d+\+?(?:\s*[-–]\s*\d+)?\s+years?"
        r"(?:\s+of)?\s+experience[^.;\n]{0,80}",
        r"[^.;\n]{0,70}\b\d+\+?\s+años?(?:\s+de)?\s+experiencia[^.;\n]{0,80}",
        r"[^。；;\n]{0,70}(?:経験)?\s*\d+\s*年以上[^。；;\n]{0,80}",
        r"[^。；;\n]{0,70}\d+\s*年(?:程度|以上)の経験[^。；;\n]{0,80}",
        r"[^.;\n]{0,70}(?:경력\s*)?\d+\s*년\s*이상[^.;\n]{0,80}",
        r"[^.;\n]{0,70}\d+\s*년(?:의)?\s*경력[^.;\n]{0,80}",
    )
    line_patterns = (
        r"\b(?:at least|minimum(?: of)?|min\.?)?\s*\d+\+?(?:\s*[-–]\s*\d+)?\s+years?\b",
        r"\b\d+\+?\s+años?\b",
        r"\d+\s*年以上|\d+\s*年(?:程度|以上)の経験",
        r"\d+\s*년\s*이상|\d+\s*년(?:의)?\s*경력",
    )
    experience_cue = re.compile(
        r"experience|hands[- ]on|industry|academic|relevant work|"
        r"professional background|proven expertise|実務経験|経験|경력|경험|experiencia",
        re.IGNORECASE,
    )
    exclusion_cue = re.compile(
        r"salary|compensation|payout|annual salary|fixed overtime|"
        r"勤務時間|休日|給与|年収|月収|보험|급여",
        re.IGNORECASE,
    )
    requirements: list[str] = []

    def add(fragment: str) -> None:
        cleaned = _truncate_at_word(
            re.sub(r"\s+", " ", fragment or "").strip(" ,:-•-"),
            160,
        )
        if cleaned and cleaned not in requirements:
            requirements.append(cleaned)

    for line in (text or "").splitlines():
        normalized_line = re.sub(r"\s+", " ", line).strip(" ,:-•-")
        if not normalized_line or exclusion_cue.search(normalized_line):
            continue
        if experience_cue.search(normalized_line) and any(
            re.search(pattern, normalized_line, re.IGNORECASE)
            for pattern in line_patterns
        ):
            add(normalized_line)
            if len(requirements) >= 3:
                return requirements

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            fragment = match.group(0)
            if exclusion_cue.search(fragment):
                continue
            add(fragment)
            if len(requirements) >= 3:
                return requirements
    return requirements


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


def _safe_error_label(exc: Exception) -> str:
    raw_message = str(exc or "")
    message = raw_message.casefold()
    exception_type = type(exc).__name__
    exception_type_folded = exception_type.casefold()
    if (
        "targetclosed" in exception_type_folded
        or "target closed" in message
        or "has been closed" in message
        or "persistent context closed" in message
        or "replacement page closed" in message
    ):
        category = "target_closed"
    elif "err_name_not_resolved" in message:
        category = "dns"
    elif "err_internet_disconnected" in message:
        category = "offline"
    elif any(
        token in message
        for token in (
            "err_connection_reset",
            "err_connection_refused",
            "err_connection_closed",
            "err_connection_timed_out",
        )
    ):
        category = "connection"
    elif "err_aborted" in message:
        category = "network_aborted"
    elif chromium_error := re.search(r"\bERR_[A-Z_]+\b", raw_message.upper()):
        category = f"chromium_{chromium_error.group(0).lower()}"
    elif "timeout" in message:
        category = "timeout"
    elif any(token in message for token in ("net::", "network", "connection")):
        category = "network"
    elif any(token in message for token in ("429", "rate limit", "too many requests")):
        category = "rate_limited"
    elif any(token in message for token in ("navigation", "goto")):
        category = "navigation"
    else:
        category = "runtime"
    return f"{exception_type}:{category}"


def _error_category(label: str) -> str:
    return label.rsplit(":", 1)[-1] if ":" in label else "runtime"


def _is_page_recoverable_error(label: str) -> bool:
    category = _error_category(label)
    return category.startswith("chromium_err_") or category in {
        "dns",
        "offline",
        "connection",
        "network_aborted",
        "network",
        "target_closed",
    }


def _is_http_response_code_failure(label: str) -> bool:
    return _error_category(label) == "chromium_err_http_response_code_failure"


def _safe_page_pause(page, milliseconds: int) -> None:
    try:
        page.wait_for_timeout(milliseconds)
    except Exception:
        time.sleep(milliseconds / 1000)


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
            records, diagnostics = _parse_linkedin_jobs_html_with_diagnostics(
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
            category=(
                "ok" if records else "query_navigation_failure"
            ),
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


def _ensure_search_source(
    page,
    *,
    source_url: str,
    navigation_state: _SearchNavigationState,
    interval_ms: int,
    now_fn=None,
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
        _validate_authenticated_page(page)
        _wait_for_search_results_hydration(page)
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
) -> tuple[object, bool, bool]:
    """Carga un source y reemplaza una Page degradada como máximo una vez."""

    try:
        navigated = _ensure_search_source(
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
        navigated = _ensure_search_source(
            replacement_page,
            source_url=source_url,
            navigation_state=navigation_state,
            interval_ms=interval_ms,
            now_fn=now_fn,
        )
        return replacement_page, navigated, True


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


def _locator_has_signal(page, selector: str, *, require_text: bool = False) -> bool:
    try:
        locator = page.locator(selector).first
        if not locator.count():
            return False
        try:
            if not locator.is_visible(timeout=0):
                return False
        except Exception:
            pass
        if not require_text:
            return True
        return bool(
            re.sub(
                r"\s+",
                " ",
                locator.inner_text(timeout=500) or "",
            ).strip()
        )
    except Exception:
        return False


def _raise_for_terminal_page_signal(page) -> None:
    current_url = str(getattr(page, "url", "") or "")
    if is_linkedin_auth_checkpoint(current_url):
        raise LinkedInAuthRequiredError(
            "La sesión LinkedIn requiere login, 2FA o checkpoint manual."
        )
    try:
        validate_linkedin_jobs_url(current_url)
    except ValueError as exc:
        raise LinkedInBlockedError(
            "LinkedIn redirigió fuera del área de empleos permitida."
        ) from exc
    if any(_locator_has_signal(page, selector) for selector in _LOGIN_SIGNAL_SELECTORS):
        raise LinkedInAuthRequiredError(
            "La sesión LinkedIn requiere login manual."
        )
    if any(_locator_has_signal(page, selector) for selector in _BLOCK_SIGNAL_SELECTORS):
        raise LinkedInBlockedError(
            "LinkedIn solicitó una verificación manual."
        )


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


def _record_key(record: LinkedInVacancyRecord) -> str:
    return record.linkedin_job_id or record.canonical_url


def _query_location(label: str) -> str:
    separator = " @ "
    return label.rsplit(separator, 1)[1] if separator in label else ""


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


def _configured_bounded_int(
    env_name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = (os.getenv(env_name) or str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{env_name} debe ser un entero entre {minimum} y {maximum}."
        ) from exc
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{env_name} debe ser un entero entre {minimum} y {maximum}."
        )
    return value


def configured_linkedin_detail_budget() -> int:
    return _configured_bounded_int(
        "LINKEDIN_DETAIL_BUDGET",
        default=25,
        minimum=0,
        maximum=_HARD_MAX_DETAIL_REQUESTS,
    )


def configured_linkedin_max_queries_per_location() -> int:
    return _configured_bounded_int(
        "LINKEDIN_MAX_QUERIES_PER_LOCATION",
        default=2,
        minimum=1,
        maximum=_HARD_MAX_QUERIES_PER_LOCATION,
    )


def configured_linkedin_query_interval_ms() -> int:
    return _configured_bounded_int(
        "LINKEDIN_QUERY_INTERVAL_MS",
        default=2750,
        minimum=2000,
        maximum=5000,
    )


def configured_linkedin_detail_click_interval_ms() -> int:
    return _configured_bounded_int(
        "LINKEDIN_DETAIL_CLICK_INTERVAL_MS",
        default=1200,
        minimum=_DETAIL_CLICK_INTERVAL_MIN_MS,
        maximum=_DETAIL_CLICK_INTERVAL_MAX_MS,
    )


def configured_linkedin_direct_detail_fallback() -> bool:
    raw = (os.getenv("LINKEDIN_DIRECT_DETAIL_FALLBACK") or "false").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "LINKEDIN_DIRECT_DETAIL_FALLBACK debe ser true o false."
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
    records: list[LinkedInVacancyRecord] = []
    candidates: list[LinkedInVacancyRecord] = []
    seen_candidate_keys: set[str] = set()
    rejected: list[LinkedInRejectedRecord] = []
    timings: list[LinkedInQueryTiming] = []
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

        relevant_candidates: list[LinkedInVacancyRecord] = []
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

        enriched_records: list[LinkedInVacancyRecord] = []
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

        records = (
            _round_robin_candidates_by_location(
                enriched_records,
                candidate_locations=candidate_locations,
                location_order=normalized_locations,
            )
            if should_balance_locations
            else enriched_records
        )[: request.max_results]

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


def configured_linkedin_max_results() -> int:
    raw = (os.getenv("LINKEDIN_MAX_RESULTS") or "50").strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError("LINKEDIN_MAX_RESULTS debe ser un entero entre 1 y 50.") from exc


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
