"""HTML parsing helpers for LinkedIn job search result pages."""
from __future__ import annotations

from dataclasses import dataclass, field
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
_SEARCH_LIST_AREA_SELECTORS = (
    ".jobs-search-results-list",
    ".scaffold-layout__list",
    "[role='listbox']",
    "main .jobs-search-results-list",
    "main [data-view-name*='job-search']",
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


def _normalize_job_id(value: str) -> str:
    match = re.search(r"(\d+)\s*$", str(value or "").strip())
    if not match:
        return ""
    return match.group(1).lstrip("0") or "0"


def _is_excluded_job_link(link: Any) -> bool:
    for parent in link.parents:
        name = str(getattr(parent, "name", "") or "").casefold()
        role = str(parent.get("role") or "").casefold()
        classes = " ".join(parent.get("class") or []).casefold()
        node_id = str(parent.get("id") or "").casefold()
        if name in {"nav", "header"} or role == "navigation":
            return True
        marker = f"{classes} {node_id}"
        if any(
            token in marker
            for token in (
                "jobs-search__job-details",
                "job-details",
                "jobs-details",
                "global-nav",
            )
        ):
            return True
    return False


def _nearest_semantic_wrapper(link: Any) -> Any | None:
    for parent in link.parents:
        if getattr(parent, "name", "") in {"html", "body"}:
            return None
        role = str(parent.get("role") or "").casefold()
        if (
            parent.has_attr("data-job-id")
            or parent.has_attr("data-occludable-job-id")
            or role in {"listitem", "option"}
            or getattr(parent, "name", "") == "li"
        ):
            return parent
    return None


def _wrapper_matches_job_id(wrapper: Any, job_id: str) -> bool:
    wrapper_ids = [
        _normalize_job_id(wrapper.get(attribute))
        for attribute in ("data-job-id", "data-occludable-job-id")
        if wrapper.has_attr(attribute)
    ]
    return not wrapper_ids or job_id in wrapper_ids


@dataclass
class _CandidateSignal:
    job_id: str
    source: str
    node: Any
    link: Any | None = None
    canonical_url: str = ""


@dataclass
class _CandidateBucket:
    job_id: str
    node: Any
    link: Any | None = None
    canonical_url: str = ""
    discovery_sources: set[str] = field(default_factory=set)


def _job_id_from_urn(value: str) -> str:
    match = re.search(r"urn:li:jobPosting:(\d+)", str(value or ""))
    return _normalize_job_id(match.group(1) if match else "")


def _safe_canonical_url_from_href(href: str) -> tuple[str, str]:
    if not str(href or "").strip():
        return "", ""
    try:
        canonical_url = canonicalize_linkedin_job_url(
            urljoin("https://www.linkedin.com", str(href or "").strip())
        )
    except ValueError:
        return "", ""
    job_id = _normalize_job_id(linkedin_job_id_from_url(canonical_url))
    return canonical_url, job_id


def _extract_candidate_signals(
    soup: BeautifulSoup,
    *,
    allow_standalone_fallback: bool = False,
) -> list[_CandidateSignal]:
    signals: list[_CandidateSignal] = []

    def add_signal(
        source: str,
        job_id: str,
        node: Any,
        *,
        link: Any | None = None,
        canonical_url: str = "",
    ) -> None:
        normalized_job_id = _normalize_job_id(job_id)
        if not normalized_job_id:
            return
        signals.append(
            _CandidateSignal(
                job_id=normalized_job_id,
                source=source,
                node=node,
                link=link,
                canonical_url=canonical_url,
            )
        )

    seen_card_nodes: set[int] = set()
    for selector in _CARD_SELECTORS:
        for card in soup.select(selector):
            if id(card) in seen_card_nodes:
                continue
            seen_card_nodes.add(id(card))
            card_link = card.select_one(_JOB_LINK_SELECTOR)
            href_job_id = ""
            href_canonical_url = ""
            if card_link is not None and not _is_excluded_job_link(card_link):
                href_canonical_url, href_job_id = _safe_canonical_url_from_href(
                    str(card_link.get("href") or "").strip()
                )
            attr_mismatched_href = False
            attr_seen = False
            for attribute in ("data-job-id", "data-occludable-job-id"):
                if card.has_attr(attribute):
                    attr_seen = True
                    attr_job_id = _normalize_job_id(str(card.get(attribute) or ""))
                    attr_mismatched_href = bool(
                        attr_job_id and href_job_id and href_job_id != attr_job_id
                    )
                    if attr_job_id and (not href_job_id or href_job_id == attr_job_id):
                        add_signal(
                            "card",
                            attr_job_id,
                            card,
                            link=card_link,
                            canonical_url=href_canonical_url,
                        )
                    break
            if href_job_id and not (attr_seen and attr_mismatched_href):
                add_signal(
                    "card",
                    href_job_id,
                    card,
                    link=card_link,
                    canonical_url=href_canonical_url,
                )
            urn_job_id = _job_id_from_urn(str(card.get("data-entity-urn") or ""))
            if urn_job_id:
                add_signal("card", urn_job_id, card)

    for node in soup.select('[data-entity-urn*="urn:li:jobPosting:"]'):
        add_signal("urn", _job_id_from_urn(str(node.get("data-entity-urn") or "")), node)

    seen_list_nodes: set[int] = set()
    for node in soup.select('li[data-job-id], li[data-occludable-job-id], [role="listitem"][data-job-id], [role="option"][data-job-id], [role="listitem"][data-occludable-job-id], [role="option"][data-occludable-job-id]'):
        if id(node) in seen_list_nodes:
            continue
        seen_list_nodes.add(id(node))
        node_link = node.select_one(_JOB_LINK_SELECTOR)
        href_job_id = ""
        if node_link is not None and not _is_excluded_job_link(node_link):
            _href_canonical, href_job_id = _safe_canonical_url_from_href(
                str(node_link.get("href") or "").strip()
            )
        for attribute in ("data-job-id", "data-occludable-job-id"):
            if node.has_attr(attribute):
                attr_job_id = _normalize_job_id(str(node.get(attribute) or ""))
                if attr_job_id and (not href_job_id or href_job_id == attr_job_id):
                    add_signal("list_item", attr_job_id, node)
                break

    seen_link_nodes: set[int] = set()
    for link in soup.select("a[href*='/jobs/view/']"):
        if id(link) in seen_link_nodes:
            continue
        seen_link_nodes.add(id(link))
        if _is_excluded_job_link(link):
            continue
        href = str(link.get("href") or "").strip()
        canonical_url, job_id = _safe_canonical_url_from_href(href)
        if not job_id:
            continue
        wrapper = _nearest_semantic_wrapper(link)
        if wrapper is not None and _wrapper_matches_job_id(wrapper, job_id):
            source = "job_href"
            node = wrapper
        elif allow_standalone_fallback:
            title = link.get_text(" ", strip=True) or str(
                link.get("aria-label") or ""
            ).strip()
            if not title:
                continue
            source = "standalone_fallback"
            node = link
        else:
            continue
        add_signal(source, job_id, node, link=link, canonical_url=canonical_url)

    return signals


def _bucket_candidate_signals(
    signals: list[_CandidateSignal],
) -> list[_CandidateBucket]:
    buckets: dict[str, _CandidateBucket] = {}
    for signal in signals:
        bucket = buckets.get(signal.job_id)
        if bucket is None:
            bucket = _CandidateBucket(
                job_id=signal.job_id,
                node=signal.node,
                link=signal.link,
                canonical_url=signal.canonical_url,
                discovery_sources={signal.source},
            )
            buckets[signal.job_id] = bucket
            continue
        bucket.discovery_sources.add(signal.source)
        if bucket.link is None and signal.link is not None:
            bucket.link = signal.link
            bucket.node = signal.node
        if not bucket.canonical_url and signal.canonical_url:
            bucket.canonical_url = signal.canonical_url
    return list(buckets.values())


def _semantic_candidates(soup: BeautifulSoup) -> list[tuple[Any, Any]]:
    return []


_DEFAULT_SEMANTIC_CANDIDATES = _semantic_candidates


def _standalone_fallback_candidates(soup: BeautifulSoup) -> list[tuple[Any, Any]]:
    # Compatibility shim for older tests/callers; the parser now uses the
    # signal/bucket path above so discovery source names stay unique per job.
    return [
        (bucket.node, bucket.link or bucket.node)
        for bucket in _bucket_candidate_signals(
            _extract_candidate_signals(soup, allow_standalone_fallback=True)
        )
        if "standalone_fallback" in bucket.discovery_sources
    ]


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
    allow_standalone_fallback: bool = False,
) -> tuple[list[LinkedInVacancyRecord], LinkedInParseDiagnostics]:
    validated_source_url = canonicalize_linkedin_job_url(source_url)
    soup = BeautifulSoup(html or "", "html.parser")
    selector_counts = {
        selector: len(soup.select(selector))
        for selector in _CARD_SELECTORS
    }
    href_count = len(soup.select("a[href*='/jobs/view/']"))
    signals = _extract_candidate_signals(
        soup,
        allow_standalone_fallback=allow_standalone_fallback,
    )
    semantic_override = globals().get("_semantic_candidates")
    if semantic_override is not _DEFAULT_SEMANTIC_CANDIDATES:
        for node, link in semantic_override(soup):
            canonical_url, job_id = _safe_canonical_url_from_href(
                str(link.get("href") or "").strip() if link is not None else ""
            )
            if job_id:
                signals.append(
                    _CandidateSignal(
                        job_id=job_id,
                        source="job_href",
                        node=node,
                        link=link,
                        canonical_url=canonical_url,
                    )
                )
    buckets = _bucket_candidate_signals(signals)
    source_counts: dict[str, int] = {
        "card": 0,
        "job_href": 0,
        "urn": 0,
        "list_item": 0,
        "standalone_fallback": 0,
    }
    for signal in signals:
        source_counts[signal.source] = source_counts.get(signal.source, 0) + 1

    records: list[LinkedInVacancyRecord] = []
    discard_reasons: dict[str, int] = {}
    if allow_standalone_fallback and source_counts.get("standalone_fallback", 0):
        discard_reasons["standalone_link_fallback"] = sum(
            1 for bucket in buckets if "standalone_fallback" in bucket.discovery_sources
        )
    duplicate_signal_count = max(0, len(signals) - len(buckets))
    if duplicate_signal_count and not (
        allow_standalone_fallback and source_counts.get("standalone_fallback", 0)
    ):
        discard_reasons["duplicate_wrapper"] = duplicate_signal_count
    if not buckets and any(selector_counts.values()):
        _increment_reason(discard_reasons, "missing_job_identity")

    for bucket in buckets:
        try:
            link = bucket.link or bucket.node.select_one(_JOB_LINK_SELECTOR)
            href = str(link.get("href") or "").strip() if link is not None else ""
            canonical_url = bucket.canonical_url
            if not canonical_url and href:
                canonical_url, href_job_id = _safe_canonical_url_from_href(href)
                if href_job_id and href_job_id != bucket.job_id:
                    _increment_reason(discard_reasons, "job_id_mismatch")
                    canonical_url = ""
            if not canonical_url:
                canonical_url = f"https://www.linkedin.com/jobs/view/{bucket.job_id}"
            if link is not None and _is_excluded_job_link(link):
                _increment_reason(discard_reasons, "excluded_link_area")
                continue
            link_title = ""
            if link is not None:
                link_title = link.get_text(" ", strip=True) or str(
                    link.get("aria-label") or ""
                ).strip()
            title = _text(
                bucket.node,
                (
                    ".job-card-list__title",
                    ".base-search-card__title",
                    ".job-card-container__link",
                ),
            ) or link_title
            title = _normalize_repeated_title(title)
            company = _text(
                bucket.node,
                (
                    ".job-card-container__primary-description",
                    ".base-search-card__subtitle",
                    ".job-card-container__company-name",
                ),
            )
            location = _text(
                bucket.node,
                (
                    ".job-card-container__metadata-item",
                    ".job-search-card__location",
                    ".base-search-card__metadata",
                ),
            )
            time_node = bucket.node.select_one("time")
            posted_text = time_node.get_text(" ", strip=True) if time_node is not None else _text(
                bucket.node,
                (".job-card-container__listed-time", ".job-search-card__listdate"),
            )
            posted_text = _clean_posted_at_text(posted_text)
            if not posted_text and time_node is None:
                posted_text = _clean_posted_at_text(
                    bucket.node.get_text(" ", strip=True)
                )
            structured = str(time_node.get("datetime") or "") if time_node is not None else ""
            published_at, confidence, within_24h = parse_linkedin_relative_time(
                posted_text,
                now=now,
                structured_datetime=structured,
            )
            blob = f"{title} {company}".lower()
            matched = sorted({term for term in MATCH_TERMS if term in blob})
            missing_metadata = not (title and company and location)
            records.append(
                LinkedInVacancyRecord(
                    linkedin_job_id=bucket.job_id,
                    title=title,
                    company_name=company,
                    location=location,
                    workplace_type=_workplace_type(bucket.node),
                    posted_at_text=posted_text,
                    published_at=published_at,
                    freshness_confidence=confidence,
                    is_within_24_hours=within_24h,
                    canonical_url=canonical_url,
                    source_url=validated_source_url,
                    matched_terms=matched,
                    discovery_sources=sorted(bucket.discovery_sources),
                    candidate_metadata_incomplete=missing_metadata,
                )
            )
        except Exception as exc:
            exception_type = re.sub(r"[^A-Za-z0-9_.-]", "_", type(exc).__name__)
            _increment_reason(
                discard_reasons,
                f"card_parse_exception:{exception_type}",
            )
    unique_candidate_count = len(buckets)
    return records, LinkedInParseDiagnostics(
        selector_counts=selector_counts,
        href_count=href_count,
        candidate_count=unique_candidate_count,
        parseable_candidate_count=len(records),
        discard_reasons=discard_reasons,
        raw_signal_count=len(signals),
        card_signal_count=source_counts.get("card", 0),
        job_href_signal_count=source_counts.get("job_href", 0),
        urn_signal_count=source_counts.get("urn", 0),
        list_item_signal_count=source_counts.get("list_item", 0),
        unique_candidate_count=unique_candidate_count,
        new_candidate_count=unique_candidate_count,
        duplicate_candidate_count=0,
        discovery_degraded=allow_standalone_fallback
        and source_counts.get("standalone_fallback", 0) > 0,
        discovery_mode=(
            "standalone_fallback"
            if allow_standalone_fallback and source_counts.get("standalone_fallback", 0) > 0
            else "standard"
        ),
    )

def parse_linkedin_jobs_html(
    html: str,
    *,
    source_url: str,
    now: datetime | None = None,
    allow_standalone_fallback: bool = False,
) -> list[LinkedInVacancyRecord]:
    records, _ = _parse_linkedin_jobs_html_with_diagnostics(
        html,
        source_url=source_url,
        now=now,
        allow_standalone_fallback=allow_standalone_fallback,
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
