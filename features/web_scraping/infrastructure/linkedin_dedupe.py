"""Deduplication helpers for LinkedIn vacancy records."""
from __future__ import annotations

import re
import unicodedata

from features.web_scraping.domain.linkedin_models import LinkedInVacancyRecord


def _record_key(record: LinkedInVacancyRecord) -> str:
    return record.linkedin_job_id or record.canonical_url


def dedupe_linkedin_vacancies(
    records: list[LinkedInVacancyRecord],
) -> list[LinkedInVacancyRecord]:
    deduped: list[LinkedInVacancyRecord] = []
    seen: set[str] = set()
    for record in records:
        key = _record_key(record)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _semantic_text_fingerprint(value: str, *, limit: int = 1400) -> str:
    normalized = unicodedata.normalize("NFKD", value or "").casefold()
    normalized = "".join(
        ch for ch in normalized if not unicodedata.combining(ch)
    )
    normalized = re.sub(r"https?://\S+", " ", normalized)
    normalized = re.sub(r"\b\d{4,}\b", " ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()[:limit]


def _location_specificity_score(value: str) -> int:
    normalized = _semantic_text_fingerprint(value, limit=220)
    if not normalized:
        return 0
    generic_locations = {"japan", "japon", "south korea", "corea del sur"}
    if normalized in generic_locations:
        return 1
    location_parts = [
        part for part in re.split(r"[,/|-]", value or "") if part.strip()
    ]
    return min(20, 2 + len(location_parts))


def _semantic_duplicate_key(record: LinkedInVacancyRecord) -> str:
    description_signature = _semantic_text_fingerprint(
        record.description_full_text or record.description_excerpt,
    )
    if len(description_signature) < 160:
        return ""
    title = _semantic_text_fingerprint(record.title, limit=160)
    company = _semantic_text_fingerprint(record.company_name, limit=160)
    if not title or not company:
        return ""
    return f"{title}|{company}|{description_signature}"


_RECRUITER_WRAPPER_SIGNALS = (
    "the client is hiring for this role",
    "career agent",
    "skip the application form",
    "puts strong candidates straight in front of the client",
)
_RECRUITER_CAMPAIGN_STOP_WORDS = {
    "about",
    "after",
    "agent",
    "backed",
    "best",
    "building",
    "candidate",
    "career",
    "client",
    "closed",
    "company",
    "description",
    "frontier",
    "hiring",
    "important",
    "location",
    "models",
    "other",
    "roles",
    "salary",
    "seoul",
    "teams",
    "their",
    "there",
    "world",
    "worlds",
    "youll",
    "your",
}


def _is_recruiter_wrapped(record: LinkedInVacancyRecord) -> bool:
    body = _semantic_text_fingerprint(
        record.description_full_text or record.description_excerpt,
        limit=1600,
    )
    return any(signal in body for signal in _RECRUITER_WRAPPER_SIGNALS)


def _company_description_segment(record: LinkedInVacancyRecord) -> str:
    body = _semantic_text_fingerprint(
        record.description_full_text or record.description_excerpt,
        limit=2400,
    )
    marker = "company description"
    start = body.find(marker)
    if start < 0:
        return body[:900]
    start += len(marker)
    end_candidates = [
        position
        for marker in (
            "what you ll do",
            "what youll do",
            "responsibilities",
            "requirements",
        )
        if (position := body.find(marker, start)) > start
    ]
    end = min(end_candidates) if end_candidates else start + 900
    return body[start:end].strip()


def _campaign_token_set(record: LinkedInVacancyRecord) -> set[str]:
    return {
        token
        for token in _company_description_segment(record).split()
        if len(token) >= 4 and token not in _RECRUITER_CAMPAIGN_STOP_WORDS
    }


def _same_recruiter_campaign(
    left: LinkedInVacancyRecord,
    right: LinkedInVacancyRecord,
) -> bool:
    if not (_is_recruiter_wrapped(left) and _is_recruiter_wrapped(right)):
        return False
    if _semantic_text_fingerprint(left.company_name, limit=120) != _semantic_text_fingerprint(
        right.company_name,
        limit=120,
    ):
        return False
    if _semantic_text_fingerprint(left.location, limit=120) != _semantic_text_fingerprint(
        right.location,
        limit=120,
    ):
        return False

    left_tokens = _campaign_token_set(left)
    right_tokens = _campaign_token_set(right)
    if min(len(left_tokens), len(right_tokens)) < 8:
        return False
    overlap = len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))
    return overlap >= 0.62


def _semantic_quality_score(record: LinkedInVacancyRecord) -> tuple[int, int, float, int]:
    published_timestamp = (
        record.published_at.timestamp()
        if record.published_at is not None
        else 0.0
    )
    return (
        1 if record.description_full_text else 0,
        _location_specificity_score(record.location),
        published_timestamp,
        len(record.description_full_text or record.description_excerpt),
    )


def _dedupe_linkedin_vacancies_semantically(
    records: list[LinkedInVacancyRecord],
) -> tuple[list[LinkedInVacancyRecord], list[str]]:
    deduped: list[LinkedInVacancyRecord] = []
    duplicate_warnings: list[str] = []
    semantic_index: dict[str, int] = {}
    seen_identity: set[str] = set()

    for record in records:
        identity = _record_key(record)
        if identity and identity in seen_identity:
            continue
        if identity:
            seen_identity.add(identity)

        recruiter_duplicate_index = next(
            (
                index
                for index, existing_record in enumerate(deduped)
                if _same_recruiter_campaign(existing_record, record)
            ),
            None,
        )
        if recruiter_duplicate_index is not None:
            existing = deduped[recruiter_duplicate_index]
            keep_new = _semantic_quality_score(record) > _semantic_quality_score(existing)
            kept = record if keep_new else existing
            dropped = existing if keep_new else record
            if keep_new:
                deduped[recruiter_duplicate_index] = record
            duplicate_warnings.append(
                "recruiter_campaign_duplicate_dropped:"
                f"{dropped.linkedin_job_id or 'unknown'}:"
                f"kept:{kept.linkedin_job_id or 'unknown'}"
            )
            continue

        semantic_key = _semantic_duplicate_key(record)
        if semantic_key and semantic_key in semantic_index:
            existing_index = semantic_index[semantic_key]
            existing = deduped[existing_index]
            keep_new = _semantic_quality_score(record) > _semantic_quality_score(existing)
            kept = record if keep_new else existing
            dropped = existing if keep_new else record
            if keep_new:
                deduped[existing_index] = record
            duplicate_warnings.append(
                "semantic_duplicate_dropped:"
                f"{dropped.linkedin_job_id or 'unknown'}:"
                f"kept:{kept.linkedin_job_id or 'unknown'}"
            )
            continue

        if semantic_key:
            semantic_index[semantic_key] = len(deduped)
        deduped.append(record)

    return deduped, duplicate_warnings
