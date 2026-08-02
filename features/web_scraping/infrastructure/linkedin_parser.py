"""HTML parsing helpers for LinkedIn job search result pages."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from features.web_scraping.domain.linkedin_models import (
    LinkedInParseDiagnostics,
    LinkedInVacancyRecord,
)
from features.web_scraping.infrastructure.linkedin_url_policy import (
    canonicalize_linkedin_job_url,
    linkedin_job_id_from_url,
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
    "data analyst",
    "ai mentor",
    "pytorch",
    "tensorflow",
    "rag",
    "mlops",
    "ai product",
)

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
_RELATIVE_DATE_PATTERNS = (
    r"\bhace\s+(?:unos?\s+segundos?|un\s+momento|\d+\s+(?:minutos?|horas?|d[ií]as?))\b",
    r"\b(?:just now|moments ago|a few seconds ago|\d+\s+(?:minutes?|hours?|days?)\s+ago)\b",
    r"\b(?:hoy|today)\b",
    r"\b\d+\s*(?:시간|분|일)\s*전\b",
    r"\b\d+\s*(?:時間|分|日)前\b",
)


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


__all__ = [
    "MATCH_TERMS",
    "_clean_posted_at_text",
    "_normalize_repeated_title",
    "_parse_linkedin_jobs_html_with_diagnostics",
    "parse_linkedin_jobs_html",
    "parse_linkedin_relative_time",
]
