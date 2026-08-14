"""LinkedIn jobs scraping orchestration pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import inspect
import re
import time
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import urljoin, urlparse, urlunparse, unquote_plus

from features.web_scraping.infrastructure.linkedin_detail_diagnostics import (
    LinkedInDetailDiagnosticsCollector,
    get_active_detail_diagnostics,
)
from features.web_scraping.infrastructure.linkedin_query_navigation import (
    LinkedInSearchHydrationDiagnosticsCollector,
    MIN_EXPECTED_DISCOVERY_CANDIDATES,
    _wait_for_active_detail_metadata,
    collect_visible_search_card_dates,
    discovery_mode_for_sources,
    latest_active_detail_metadata_diagnostic,
    discover_job_rows_via_activation,
    merge_row_activation_records,
    get_active_search_hydration_diagnostics,
)
from features.web_scraping.infrastructure.linkedin_static_probe_diagnostics import (
    LinkedInStaticProbeDiagnosticsCollector,
    get_active_static_probe_diagnostics,
)
from features.web_scraping.infrastructure.linkedin_visual_diagnostics import (
    get_active_visual_diagnostics,
)

from features.web_scraping.infrastructure.linkedin_url_policy import (
    canonicalize_linkedin_url,
)

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


_KOREA_LOCATION_MARKERS = (
    "south korea",
    "corea del sur",
    "korea, republic of",
    "republic of korea",
    "seoul",
    "대한민국",
    "서울",
)
_JAPAN_LOCATION_MARKERS = (
    "japan",
    "japón",
    "tokyo",
    "日本",
    "東京",
)
_REGIONAL_REMOTE_LOCATION_MARKERS = (
    "asia-pacific",
    "asia pacific",
    "asia-pacífico",
    "asia pacífico",
    "apac",
    "remote",
    "remoto",
)



_LINKEDIN_MATCH_TERMS = (
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


def _safe_candidate_identity(record: Any) -> str:
    job_id = _normalized_numeric_job_id(getattr(record, "linkedin_job_id", ""))
    if job_id:
        return job_id
    source_url = str(getattr(record, "canonical_url", "") or "")
    match = re.search(r"/jobs/view/(?:[^/?#]*-)?(\d+)/?(?:[?#]|$)", source_url)
    if not match:
        return ""
    return match.group(1).lstrip("0") or "0"


def _candidate_metadata_complete(record: Any) -> bool:
    return bool(
        str(getattr(record, "title", "") or "").strip()
        and str(getattr(record, "company_name", "") or "").strip()
        and str(getattr(record, "location", "") or "").strip()
    )


def _safe_active_detail_date_label(value: str) -> str:
    label = re.sub(r"\s+", "_", (value or "").strip())[:40]
    label = re.sub(r"[^0-9A-Za-z_가-힣一-龯áéíóúÁÉÍÓÚüÜñÑ-]", "", label)
    return label or "none"


def _active_detail_metadata_warning(record: Any) -> str:
    job_id = _safe_candidate_identity(record) or "unknown"
    diagnostic = latest_active_detail_metadata_diagnostic()
    count = max(0, min(100, int(diagnostic.get("date_candidate_count", 0) or 0)))
    score = max(0, min(100000, int(diagnostic.get("selected_score", 0) or 0)))
    selected = _safe_active_detail_date_label(str(diagnostic.get("selected_date", "") or ""))
    if bool(diagnostic.get("date_verified", False)):
        status = (
            "within_24_hours"
            if bool(diagnostic.get("date_within_24_hours", False))
            else "outside_24_hours"
        )
    elif bool(diagnostic.get("date_detected", False)):
        status = "unparseable"
    else:
        status = "missing"
    return (
        f"active_detail_date_selected:{job_id}:{status}:"
        f"date_{selected}:candidates_{count}:score_{score}"
    )


def _normalize_top_card_identity_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().casefold()
    return re.sub(r"[^0-9a-z가-힣一-龯]+", " ", text).strip()


def _active_detail_top_card_date_identity_status(
    record: Any,
    active_title: str,
) -> tuple[bool, str]:
    """Validate active-detail date evidence against the visible top-card title.

    This intentionally does not use page URL, global hrefs, or left-list signals:
    those were proven to produce false positives when LinkedIn kept another
    detail panel active.
    """
    expected_title = _normalize_top_card_identity_text(
        getattr(record, "title", "")
    )
    visible_title = _normalize_top_card_identity_text(active_title)
    if not expected_title:
        return False, "candidate_title_missing"
    if not visible_title:
        return False, "top_card_title_missing"
    if expected_title == visible_title:
        return True, "top_card_title_match"

    expected_tokens = set(expected_title.split())
    visible_tokens = set(visible_title.split())
    if (
        len(expected_tokens) >= 4
        and expected_tokens.issubset(visible_tokens)
    ) or (
        len(visible_tokens) >= 4
        and visible_tokens.issubset(expected_tokens)
    ):
        return True, "top_card_title_compatible"

    # LinkedIn often expands concise card titles in the detail top-card, e.g.
    # "Machine Learning Engineer" -> "Machine Learning Engineer - AI (Remote)".
    # Accept that only for a bounded, ordered prefix with enough signal; this
    # keeps very short/generic titles from matching unrelated roles.
    expected_sequence = expected_title.split()
    visible_sequence = visible_title.split()
    if (
        len(expected_sequence) >= 3
        and len(visible_sequence) > len(expected_sequence)
        and visible_sequence[: len(expected_sequence)] == expected_sequence
    ):
        return True, "top_card_title_prefix_compatible"

    return False, "top_card_title_mismatch"


def _merge_active_detail_metadata(
    record: Any,
    page: Any,
    *,
    warnings: list[str] | None = None,
) -> Any:
    try:
        title, posted_text, published_at, confidence, within_24h = (
            _wait_for_active_detail_metadata(
                page,
                require_date=getattr(record, "published_at", None) is None,
            )
        )
    except Exception:
        return record
    if warnings is not None:
        warnings.append(_active_detail_metadata_warning(record))
    updates: dict[str, Any] = {}
    if not str(getattr(record, "title", "") or "").strip() and title:
        updates["title"] = title
    if getattr(record, "published_at", None) is None and published_at is not None:
        identity_matches, identity_reason = _active_detail_top_card_date_identity_status(
            record,
            title,
        )
        if warnings is not None:
            warnings.append(
                "active_detail_top_card_identity:"
                f"{_safe_candidate_identity(record) or 'unknown'}:"
                f"{identity_reason}"
            )
        if identity_matches:
            updates.update(
                {
                    "posted_at_text": posted_text,
                    "published_at": published_at,
                    "freshness_confidence": confidence,
                    "is_within_24_hours": within_24h,
                }
            )
    return record.model_copy(update=updates) if updates else record




def _activate_visible_search_card_date_evidence(
    page: Any,
    record: Any,
    *,
    warnings: list[str] | None = None,
) -> Any:
    """Activate the visible search card for one job and read verified date.

    This is intentionally isolated/removable. It only merges date evidence when
    the active detail top-card title is compatible with the candidate title.
    """
    job_id = _safe_candidate_identity(record)
    if not job_id:
        return record
    selectors = (
        f'[data-job-id$="{job_id}"]',
        f'[data-occludable-job-id$="{job_id}"]',
        f'a[href*="/jobs/view/{job_id}"]',
        f'a[href*="-{job_id}"]',
    )
    clicked = False
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if not locator.count():
                continue
            try:
                locator.scroll_into_view_if_needed(timeout=2000)
            except Exception:
                pass
            locator.click(timeout=5000)
            clicked = True
            break
        except Exception:
            continue
    if not clicked:
        if warnings is not None:
            warnings.append(f"card_activation_date_failed:{job_id}:card_not_found")
        return record
    try:
        title, posted_text, published_at, confidence, within_24h = (
            _wait_for_active_detail_metadata(page, require_date=True)
        )
    except Exception:
        if warnings is not None:
            warnings.append(f"card_activation_date_failed:{job_id}:metadata_error")
        return record
    if warnings is not None:
        warnings.append(_active_detail_metadata_warning(record))
    identity_matches, identity_reason = _active_detail_top_card_date_identity_status(
        record,
        title,
    )
    if warnings is not None:
        warnings.append(f"card_activation_date_identity:{job_id}:{identity_reason}")
    if not identity_matches:
        return record
    if published_at is None:
        if warnings is not None:
            warnings.append(f"card_activation_date_failed:{job_id}:missing_date")
        return record
    if warnings is not None:
        status = "within_24_hours" if within_24h else "outside_24_hours"
        warnings.append(f"card_activation_date_verified:{job_id}:{status}")
    return record.model_copy(
        update={
            "posted_at_text": posted_text,
            "published_at": published_at,
            "freshness_confidence": confidence,
            "is_within_24_hours": within_24h,
        }
    )


def _activate_visible_search_card_by_title_date_evidence(
    page: Any,
    record: Any,
    *,
    warnings: list[str] | None = None,
) -> Any:
    """Activate a visible search card by normalized title and read date evidence.

    This helper is a removable fallback for cases where job-id selectors click
    the wrong virtualized row. It does not persist or log row text, and it only
    merges date evidence after active top-card title validation.
    """
    job_id = _safe_candidate_identity(record)
    expected_title = _normalize_top_card_identity_text(getattr(record, "title", ""))
    if not job_id or not expected_title:
        if warnings is not None and job_id:
            warnings.append(f"card_title_activation_failed:{job_id}:missing_title")
        return record
    row_selectors = (
        'li[data-job-id], li[data-occludable-job-id], '
        '[role="listitem"], [role="option"], '
        '.job-card-container, .scaffold-layout__list-item'
    )
    clicked = False
    try:
        rows = page.locator(row_selectors)
        count = min(max(0, int(rows.count() or 0)), 80)
    except Exception:
        count = 0
    for index in range(count):
        try:
            row = rows.nth(index)
            try:
                text = row.inner_text(timeout=1000)
            except Exception:
                text = row.text_content(timeout=1000)
            row_title = _normalize_top_card_identity_text(text)
            if expected_title not in row_title:
                continue
            try:
                row.scroll_into_view_if_needed(timeout=2000)
            except Exception:
                pass
            try:
                link = row.locator('a[href*="/jobs/view/"]').first
                if link.count():
                    link.click(timeout=5000)
                else:
                    row.click(timeout=5000)
            except Exception:
                row.click(timeout=5000)
            clicked = True
            break
        except Exception:
            continue
    if not clicked:
        if warnings is not None:
            warnings.append(f"card_title_activation_failed:{job_id}:row_not_found")
        return record
    try:
        title, posted_text, published_at, confidence, within_24h = (
            _wait_for_active_detail_metadata(page, require_date=True)
        )
    except Exception:
        if warnings is not None:
            warnings.append(f"card_title_activation_failed:{job_id}:metadata_error")
        return record
    if warnings is not None:
        warnings.append(_active_detail_metadata_warning(record))
    identity_matches, identity_reason = _active_detail_top_card_date_identity_status(
        record,
        title,
    )
    if warnings is not None:
        warnings.append(f"card_title_activation_identity:{job_id}:{identity_reason}")
    if not identity_matches:
        return record
    if published_at is None:
        if warnings is not None:
            warnings.append(f"card_title_activation_failed:{job_id}:missing_date")
        return record
    if warnings is not None:
        status = "within_24_hours" if within_24h else "outside_24_hours"
        warnings.append(f"card_title_activation_verified:{job_id}:{status}")
    return record.model_copy(
        update={
            "posted_at_text": posted_text,
            "published_at": published_at,
            "freshness_confidence": confidence,
            "is_within_24_hours": within_24h,
        }
    )

def _capture_unverified_date_visual_evidence(
    visual_diagnostics: Any,
    page: Any,
    record: Any,
    *,
    warnings: list[str] | None = None,
) -> None:
    """Capture local-only screenshots for a candidate missing verified date.

    This helper is intentionally isolated so the diagnostic hook can be removed
    cleanly if it does not help. It never affects validation or acceptance.
    """
    if visual_diagnostics is None:
        return
    job_id = getattr(record, "linkedin_job_id", "") or _safe_candidate_identity(record)
    if not job_id:
        return
    captured = False
    if hasattr(visual_diagnostics, "capture_active_detail_date"):
        try:
            visual_diagnostics.capture_active_detail_date(
                page,
                job_id=job_id,
                reason="unverified_posted_date",
            )
            captured = True
        except Exception:
            pass
    if hasattr(visual_diagnostics, "capture_rejected_candidate_card"):
        try:
            visual_diagnostics.capture_rejected_candidate_card(
                page,
                job_id=job_id,
                reason="unverified_posted_date",
            )
            captured = True
        except Exception:
            pass
    if captured and warnings is not None:
        warnings.append(f"unverified_date_visual_debug:{job_id}")


def _merge_candidate_discovery_evidence(current: Any, incoming: Any) -> Any:
    updates: dict[str, Any] = {}
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
        incoming_value = getattr(incoming, field_name, None)
        if field_name in {"freshness_confidence", "is_within_24_hours"}:
            if getattr(current, "published_at", None) is None and getattr(incoming, "published_at", None) is not None:
                updates[field_name] = incoming_value
            continue
        if not current_value and incoming_value:
            updates[field_name] = incoming_value
    current_sources = set(getattr(current, "discovery_sources", None) or [])
    incoming_sources = set(getattr(incoming, "discovery_sources", None) or [])
    merged_sources = sorted(current_sources | incoming_sources)
    if merged_sources and merged_sources != list(getattr(current, "discovery_sources", None) or []):
        updates["discovery_sources"] = merged_sources
    if getattr(current, "candidate_metadata_incomplete", False) and _candidate_metadata_complete(
        current.model_copy(update=updates) if updates else current
    ):
        updates["candidate_metadata_incomplete"] = False
    return current.model_copy(update=updates) if updates else current


def _refresh_candidate_discovery_state(record: Any) -> Any:
    updates: dict[str, Any] = {}
    if getattr(record, "candidate_metadata_incomplete", False) and _candidate_metadata_complete(record):
        updates["candidate_metadata_incomplete"] = False
    if not getattr(record, "matched_terms", None):
        blob = " ".join(
            str(getattr(record, field, "") or "")
            for field in (
                "title",
                "company_name",
                "description_excerpt",
                "description_full_text",
            )
        ).casefold()
        matched = sorted({term for term in _LINKEDIN_MATCH_TERMS if term in blob})
        if matched:
            updates["matched_terms"] = matched
    return record.model_copy(update=updates) if updates else record


def _mark_diagnostics_after_query_dedupe(diagnostics: Any, *, new_count: int) -> Any:
    unique_count = int(
        getattr(diagnostics, "unique_candidate_count", 0)
        or getattr(diagnostics, "candidate_count", 0)
        or 0
    )
    return diagnostics.model_copy(
        update={
            "unique_candidate_count": unique_count,
            "new_candidate_count": new_count,
            "duplicate_candidate_count": max(0, unique_count - new_count),
        }
    )

def _normalize_location_marker_text(value: Any) -> str:
    return unquote_plus(str(value or "")).casefold()


def _has_any_marker(value: str, markers: tuple[str, ...]) -> bool:
    normalized = _normalize_location_marker_text(value)
    return any(marker in normalized for marker in markers)


def _record_country_signal_text(record: Any) -> str:
    return "\n".join(
        str(value or "")
        for value in (
            getattr(record, "title", ""),
            getattr(record, "company_name", ""),
            getattr(record, "description_excerpt", ""),
            getattr(record, "description_full_text", ""),
            getattr(record, "canonical_url", ""),
            getattr(record, "source_url", ""),
        )
    )


def _rejected_record_identity(record: Any) -> str:
    title = " ".join(
        str(getattr(record, "title", "") or "").casefold().split()
    )
    if title and any(marker in title for marker in ("remote", "remoto")):
        return f"title:{title}"
    source_url = str(getattr(record, "source_url", "") or "").strip().casefold()
    if "/jobs/view/" in source_url:
        return f"url:{source_url}"
    if title:
        return f"title:{title}"
    return ""


def _rejection_reason_rank(record: Any) -> int:
    reason = str(getattr(record, "reason", "") or "").casefold()
    return 0 if reason in {"duplicate", "duplicate_candidate_skipped_before_detail"} else 1


def _dedupe_rejected_records(
    rejected: list[Any],
) -> tuple[list[Any], list[str]]:
    deduped: list[Any] = []
    warnings: list[str] = []
    seen: dict[str, int] = {}
    for record in rejected:
        identity = _rejected_record_identity(record)
        if identity and identity in seen:
            existing_index = seen[identity]
            existing = deduped[existing_index]
            if _rejection_reason_rank(record) > _rejection_reason_rank(existing):
                deduped[existing_index] = record
                dropped = existing
            else:
                dropped = record
            warnings.append(
                "rejected_duplicate_dropped:"
                f"{str(getattr(dropped, 'title', '') or 'unknown')}:"
                f"{str(getattr(dropped, 'reason', '') or 'unknown')}"
            )
            continue
        if identity:
            seen[identity] = len(deduped)
        deduped.append(record)
    return deduped, warnings




_KOREA_REMOTE_ROLE_EVIDENCE_MARKERS = (
    "south korea",
    "corea del sur",
    "korea office",
    "korean office",
    "korea team",
    "korean team",
    "korea market",
    "korean market",
    "based in korea",
    "based in seoul",
    "located in korea",
    "located in seoul",
    "work from korea",
    "working from korea",
    "seoul office",
    "대한민국",
    "서울",
)
_JAPAN_REMOTE_ROLE_EVIDENCE_MARKERS = (
    "japan",
    "japón",
    "japan office",
    "japanese office",
    "japan team",
    "japanese team",
    "japan market",
    "japanese market",
    "based in japan",
    "based in tokyo",
    "located in japan",
    "located in tokyo",
    "work from japan",
    "working from japan",
    "tokyo office",
    "日本",
    "東京",
)


def _strip_linkedin_search_location_evidence(value: str) -> str:
    """Remove search-query location hints so APAC/remote needs role evidence."""
    return re.sub(
        r"(?:[?&]|^)location=[^&#\s]+",
        " ",
        str(value or ""),
        flags=re.IGNORECASE,
    )


def _remote_or_hybrid_location_matches_requested_country(
    visible_location: str,
    requested_location: str,
    country_signal_text: str = "",
) -> bool | None:
    """Resolve regional/remote spillover with bounded role/employer evidence.

    Return None when the location is not a regional/remote ambiguity. A LinkedIn
    search URL location is not enough proof: APAC/remote roles must mention the
    requested country in the role, employer, card, or detail evidence.
    """
    visible = _normalize_location_marker_text(visible_location)
    if not _has_any_marker(visible, _REGIONAL_REMOTE_LOCATION_MARKERS):
        return None

    requested = _normalize_location_marker_text(requested_location)
    signal_text = _normalize_location_marker_text(
        _strip_linkedin_search_location_evidence(country_signal_text)
    )
    if _has_any_marker(requested, _KOREA_LOCATION_MARKERS):
        return _has_any_marker(signal_text, _KOREA_REMOTE_ROLE_EVIDENCE_MARKERS)
    if _has_any_marker(requested, _JAPAN_LOCATION_MARKERS):
        return _has_any_marker(signal_text, _JAPAN_REMOTE_ROLE_EVIDENCE_MARKERS)
    return None









def _safe_linkedin_profile_url_from_href(href: Any) -> str:
    try:
        parsed = urlparse(urljoin("https://www.linkedin.com", str(href or "")))
    except Exception:
        return ""
    if parsed.scheme.lower() != "https":
        return ""
    if (parsed.hostname or "").lower() not in {"linkedin.com", "www.linkedin.com"}:
        return ""
    if parsed.username or parsed.password:
        return ""
    parts = [part for part in (parsed.path or "").split("/") if part]
    if len(parts) < 2 or parts[0] != "in":
        return ""
    slug = parts[1].strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", slug):
        return ""
    return urlunparse(("https", "www.linkedin.com", f"/in/{slug}", "", "", ""))


def _safe_recruiter_profile_url_from_active_detail(page: Any) -> str:
    try:
        hrefs = page.evaluate(
            """
            () => {
              const values = [];
              const seen = new Set();
              const pushLinks = (root) => {
                if (!root || seen.size >= 40) return;
                for (const link of Array.from(root.querySelectorAll('a[href*="/in/"]')).slice(0, 10)) {
                  const href = String(link.href || link.getAttribute('href') || '');
                  if (!href || seen.has(href)) continue;
                  seen.add(href);
                  values.push(href);
                }
              };
              const classRoots = Array.from(document.querySelectorAll(
                '[class*=hiring], [class*=recruiter], [class*=hirer], [class*=poster]'
              )).slice(0, 20);
              for (const root of classRoots) pushLinks(root);

              const markerPattern = /equipo de contrataci[oó]n|conoce al equipo de contrataci[oó]n|hiring team|recruiter|contratador|publicado por|published by/i;
              const textRoots = Array.from(document.querySelectorAll(
                'section, aside, div, li, article'
              )).slice(0, 500).filter((node) => {
                const text = String(node.innerText || node.textContent || '').replace(/\s+/g, ' ').trim();
                if (!text || text.length > 700) return false;
                if (!markerPattern.test(text)) return false;
                return Boolean(node.querySelector('a[href*="/in/"]'));
              }).slice(0, 20);
              for (const root of textRoots) pushLinks(root);

              return values.slice(0, 40);
            }
            """
        )
    except Exception:
        return ""
    if isinstance(hrefs, str):
        hrefs = [hrefs]
    if not isinstance(hrefs, list):
        return ""
    for href in hrefs:
        profile_url = _safe_linkedin_profile_url_from_href(href)
        if profile_url:
            return profile_url
    return ""


def _recruiter_location_country_signal(profile_snapshot: Any) -> str:
    if not isinstance(profile_snapshot, dict):
        return ""
    if bool(profile_snapshot.get("has_korea")):
        return "korea office"
    if bool(profile_snapshot.get("has_japan")):
        return "japan office"
    return ""


def _recruiter_location_evidence_signal(
    page: Any,
    record: Any,
    requested_location: str,
    visual_diagnostics: Any = None,
    *,
    warnings: list[str] | None = None,
) -> str:
    """Open a visible recruiter profile read-only and return positive country evidence.

    Recruiter evidence is additive only: missing or non-matching recruiter location
    never blocks a company-positive APAC/remote candidate.
    """
    if page is None:
        return ""
    if not _has_any_marker(
        str(getattr(record, "location", "") or ""),
        _REGIONAL_REMOTE_LOCATION_MARKERS,
    ):
        return ""
    job_id = getattr(record, "linkedin_job_id", "") or _safe_candidate_identity(record)
    try:
        active_title, *_ = _wait_for_active_detail_metadata(page, require_date=False)
    except Exception:
        if warnings is not None:
            warnings.append(f"recruiter_location_evidence:{job_id or 'unknown'}:metadata_error")
        return ""
    identity_matches, identity_reason = _active_detail_top_card_date_identity_status(
        record,
        active_title,
    )
    if not identity_matches:
        if warnings is not None:
            warnings.append(
                "recruiter_location_evidence:"
                f"{job_id or 'unknown'}:identity_{identity_reason}"
            )
        return ""
    profile_url = _safe_recruiter_profile_url_from_active_detail(page)
    if not profile_url:
        if warnings is not None:
            warnings.append(f"recruiter_location_evidence:{job_id or 'unknown'}:profile_url_missing")
        return ""
    recruiter_page = None
    try:
        recruiter_page = page.context.new_page()
        recruiter_page.goto(profile_url, wait_until="domcontentloaded", timeout=15000)
        try:
            recruiter_page.wait_for_timeout(800)
        except Exception:
            pass
        snapshot = recruiter_page.evaluate(
            """
            () => {
              const text = String(document.body && (document.body.innerText || document.body.textContent) || '').replace(/\s+/g, ' ').trim();
              return {
                has_korea: /south korea|corea del sur|republic of korea|seoul|대한민국|서울/i.test(text),
                has_japan: /japan|japón|tokyo|日本|東京/i.test(text),
                has_united_states: /united states|estados unidos|california|new york|menlo park|san francisco|usa|u\.s\./i.test(text),
                text_length: text.length,
              };
            }
            """
        )
        if visual_diagnostics is not None and hasattr(
            visual_diagnostics,
            "capture_recruiter_location",
        ):
            try:
                visual_diagnostics.capture_recruiter_location(
                    recruiter_page,
                    job_id=job_id or "unknown",
                    reason="remote_scope_review",
                )
            except Exception:
                pass
    except Exception:
        if warnings is not None:
            warnings.append(f"recruiter_location_evidence:{job_id or 'unknown'}:profile_fetch_error")
        return ""
    finally:
        if recruiter_page is not None:
            try:
                recruiter_page.close()
            except Exception:
                pass
    signal = _recruiter_location_country_signal(snapshot)
    if warnings is not None:
        if signal == "korea office":
            status = "recruiter_profile_korea"
        elif signal == "japan office":
            status = "recruiter_profile_japan"
        elif isinstance(snapshot, dict) and bool(snapshot.get("has_united_states")):
            status = "recruiter_profile_united_states"
        else:
            status = "recruiter_profile_no_country_match"
        warnings.append(f"recruiter_location_evidence:{job_id or 'unknown'}:{status}")
    requested = _normalize_location_marker_text(requested_location)
    if signal == "korea office" and _has_any_marker(requested, _KOREA_LOCATION_MARKERS):
        return signal
    if signal == "japan office" and _has_any_marker(requested, _JAPAN_LOCATION_MARKERS):
        return signal
    return ""

def _company_about_location_country_signal(about_snapshot: Any) -> str:
    if not isinstance(about_snapshot, dict):
        return ""
    if bool(about_snapshot.get("has_korea")):
        return "korea office"
    if bool(about_snapshot.get("has_japan")):
        return "japan office"
    return ""


def _safe_linkedin_company_slug_from_href(href: Any) -> str:
    try:
        parsed = urlparse(urljoin("https://www.linkedin.com", str(href or "")))
    except Exception:
        return ""
    if parsed.scheme.lower() != "https":
        return ""
    if (parsed.hostname or "").lower() not in {"linkedin.com", "www.linkedin.com"}:
        return ""
    if parsed.username or parsed.password:
        return ""
    parts = [part for part in (parsed.path or "").split("/") if part]
    if len(parts) < 2 or parts[0] != "company":
        return ""
    slug = parts[1].strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", slug):
        return ""
    return slug


def _safe_company_about_url_from_slug(slug: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", str(slug or "")):
        return ""
    try:
        return canonicalize_linkedin_url(
            urlunparse(("https", "www.linkedin.com", f"/company/{slug}/about", "", "", ""))
        )
    except Exception:
        return ""


def _safe_company_about_url_from_active_detail(page: Any) -> str:
    try:
        hrefs = page.evaluate(
            """
            () => {
              const selectors = [
                '.job-details-jobs-unified-top-card a[href*="/company/"]',
                '.jobs-unified-top-card a[href*="/company/"]',
                'main a[href*="/company/"]',
                'a[href*="/company/"]'
              ];
              const seen = new Set();
              const values = [];
              for (const selector of selectors) {
                for (const link of Array.from(document.querySelectorAll(selector)).slice(0, 20)) {
                  const href = String(link.href || link.getAttribute('href') || '');
                  if (!href || seen.has(href)) continue;
                  seen.add(href);
                  values.push(href);
                }
              }
              return values.slice(0, 20);
            }
            """
        )
    except Exception:
        return ""
    if isinstance(hrefs, str):
        hrefs = [hrefs]
    if not isinstance(hrefs, list):
        return ""
    for href in hrefs:
        slug = _safe_linkedin_company_slug_from_href(href)
        about_url = _safe_company_about_url_from_slug(slug) if slug else ""
        if about_url:
            return about_url
    return ""


def _company_about_location_evidence_signal(
    page: Any,
    record: Any,
    requested_location: str,
    visual_diagnostics: Any = None,
    *,
    warnings: list[str] | None = None,
) -> str:
    """Open the active company About page read-only and return country evidence.

    This is intentionally removable. It only runs for APAC/remote candidates and
    only after active-detail title identity matches the candidate.
    """
    if page is None:
        return ""
    if not _has_any_marker(
        str(getattr(record, "location", "") or ""),
        _REGIONAL_REMOTE_LOCATION_MARKERS,
    ):
        return ""
    job_id = getattr(record, "linkedin_job_id", "") or _safe_candidate_identity(record)
    try:
        active_title, *_ = _wait_for_active_detail_metadata(page, require_date=False)
    except Exception:
        if warnings is not None:
            warnings.append(f"company_about_location_evidence:{job_id or 'unknown'}:metadata_error")
        return ""
    identity_matches, identity_reason = _active_detail_top_card_date_identity_status(
        record,
        active_title,
    )
    if not identity_matches:
        if warnings is not None:
            warnings.append(
                "company_about_location_evidence:"
                f"{job_id or 'unknown'}:identity_{identity_reason}"
            )
        return ""
    about_url = _safe_company_about_url_from_active_detail(page)
    if not about_url:
        if warnings is not None:
            warnings.append(f"company_about_location_evidence:{job_id or 'unknown'}:company_url_missing")
        return ""
    company_page = None
    try:
        company_page = page.context.new_page()
        company_page.goto(about_url, wait_until="domcontentloaded", timeout=15000)
        try:
            company_page.wait_for_timeout(800)
        except Exception:
            pass
        snapshot = company_page.evaluate(
            """
            () => {
              const text = String(document.body && (document.body.innerText || document.body.textContent) || '').replace(/\\s+/g, ' ').trim();
              const hasKorea = /south korea|corea del sur|republic of korea|seoul|대한민국|서울/i.test(text);
              const hasJapan = /japan|japón|tokyo|日本|東京/i.test(text);
              const hasUnitedStates = /united states|estados unidos|california|new york|menlo park|san francisco|usa|u\\.s\\./i.test(text);
              return {
                has_korea: hasKorea,
                has_japan: hasJapan,
                has_united_states: hasUnitedStates,
                text_length: text.length,
              };
            }
            """
        )
        if visual_diagnostics is not None and hasattr(
            visual_diagnostics,
            "capture_company_about_location",
        ):
            try:
                visual_diagnostics.capture_company_about_location(
                    company_page,
                    job_id=job_id or "unknown",
                    reason="remote_scope_review",
                )
            except Exception:
                pass
    except Exception:
        if warnings is not None:
            warnings.append(f"company_about_location_evidence:{job_id or 'unknown'}:about_fetch_error")
        return ""
    finally:
        if company_page is not None:
            try:
                company_page.close()
            except Exception:
                pass
    signal = _company_about_location_country_signal(snapshot)
    if warnings is not None:
        if signal == "korea office":
            status = "company_about_korea"
        elif signal == "japan office":
            status = "company_about_japan"
        elif isinstance(snapshot, dict) and bool(snapshot.get("has_united_states")):
            status = "company_about_united_states"
        else:
            status = "company_about_no_country_match"
        warnings.append(f"company_about_location_evidence:{job_id or 'unknown'}:{status}")
    requested = _normalize_location_marker_text(requested_location)
    if signal == "korea office" and _has_any_marker(requested, _KOREA_LOCATION_MARKERS):
        return signal
    if signal == "japan office" and _has_any_marker(requested, _JAPAN_LOCATION_MARKERS):
        return signal
    return ""

def _remote_scope_page_country_evidence_signal(
    page: Any,
    record: Any,
    requested_location: str,
    *,
    warnings: list[str] | None = None,
) -> str:
    """Return a synthetic country marker when active company/recruiter evidence matches.

    This is intentionally conservative: the active detail title must match the
    candidate before any company/recruiter location evidence can affect scope.
    """
    if page is None:
        return ""
    if not _has_any_marker(
        str(getattr(record, "location", "") or ""),
        _REGIONAL_REMOTE_LOCATION_MARKERS,
    ):
        return ""
    job_id = getattr(record, "linkedin_job_id", "") or _safe_candidate_identity(record)
    try:
        active_title, *_ = _wait_for_active_detail_metadata(page, require_date=False)
    except Exception:
        if warnings is not None:
            warnings.append(f"remote_scope_structured_evidence:{job_id or 'unknown'}:metadata_error")
        return ""
    identity_matches, identity_reason = _active_detail_top_card_date_identity_status(
        record,
        active_title,
    )
    if not identity_matches:
        if warnings is not None:
            warnings.append(
                "remote_scope_structured_evidence:"
                f"{job_id or 'unknown'}:identity_{identity_reason}"
            )
        return ""
    try:
        evidence = page.evaluate(
            """
            () => {
              const normalize = (value) => String(value || '').toLowerCase();
              const textOf = (node) => normalize(node && (node.innerText || node.textContent) || '');
              const hasKorea = (text) => /south korea|corea del sur|korea office|korean office|korea team|korean team|korea market|korean market|based in korea|based in seoul|located in korea|located in seoul|work from korea|working from korea|seoul office|대한민국|서울/.test(text);
              const hasJapan = (text) => /japan|japón|japan office|japanese office|japan team|japanese team|japan market|japanese market|based in japan|based in tokyo|located in japan|located in tokyo|work from japan|working from japan|tokyo office|日本|東京/.test(text);
              const scan = (selector) => Array.from(document.querySelectorAll(selector))
                .slice(0, 40)
                .map(textOf)
                .filter((text) => text && text.length <= 1200);
              const companyTexts = scan('[class*=company], [class*=organization], a[href*="/company/"]');
              const recruiterTexts = scan('[class*=hiring], [class*=recruiter], [class*=hirer], [class*=poster]');
              const locationTexts = scan('[class*=location], [class*=primary-description], [class*=tertiary-description], [class*=top-card]');
              const any = (texts, fn) => texts.some(fn);
              return {
                company_has_korea: any(companyTexts, hasKorea),
                recruiter_has_korea: any(recruiterTexts, hasKorea),
                location_has_korea: any(locationTexts, hasKorea),
                company_has_japan: any(companyTexts, hasJapan),
                recruiter_has_japan: any(recruiterTexts, hasJapan),
                location_has_japan: any(locationTexts, hasJapan),
                company_node_count: companyTexts.length,
                recruiter_node_count: recruiterTexts.length,
                location_node_count: locationTexts.length,
              };
            }
            """
        )
    except Exception:
        if warnings is not None:
            warnings.append(f"remote_scope_structured_evidence:{job_id or 'unknown'}:evaluate_error")
        return ""
    requested = _normalize_location_marker_text(requested_location)
    if _has_any_marker(requested, _KOREA_LOCATION_MARKERS):
        source = ""
        if bool(evidence.get("company_has_korea")):
            source = "company_location_korea"
        elif bool(evidence.get("recruiter_has_korea")):
            source = "recruiter_location_korea"
        elif bool(evidence.get("location_has_korea")):
            source = "detail_location_korea"
        if source:
            if warnings is not None:
                warnings.append(
                    "remote_scope_structured_evidence:"
                    f"{job_id or 'unknown'}:{source}"
                )
            return "korea office"
    if _has_any_marker(requested, _JAPAN_LOCATION_MARKERS):
        source = ""
        if bool(evidence.get("company_has_japan")):
            source = "company_location_japan"
        elif bool(evidence.get("recruiter_has_japan")):
            source = "recruiter_location_japan"
        elif bool(evidence.get("location_has_japan")):
            source = "detail_location_japan"
        if source:
            if warnings is not None:
                warnings.append(
                    "remote_scope_structured_evidence:"
                    f"{job_id or 'unknown'}:{source}"
                )
            return "japan office"
    if warnings is not None:
        warnings.append(
            "remote_scope_structured_evidence:"
            f"{job_id or 'unknown'}:none:"
            f"company_nodes_{int(evidence.get('company_node_count', 0) or 0)}:"
            f"recruiter_nodes_{int(evidence.get('recruiter_node_count', 0) or 0)}"
        )
    return ""

def _capture_remote_scope_visual_evidence(
    visual_diagnostics: Any,
    page: Any,
    record: Any,
    *,
    reason: str = "remote_scope_review",
    warnings: list[str] | None = None,
) -> None:
    """Capture opt-in local evidence for APAC/remote company/recruiter scope."""
    if visual_diagnostics is None or page is None:
        return
    if not _has_any_marker(
        str(getattr(record, "location", "") or ""),
        _REGIONAL_REMOTE_LOCATION_MARKERS,
    ):
        return
    if not hasattr(visual_diagnostics, "capture_company_recruiter_location"):
        return
    job_id = getattr(record, "linkedin_job_id", "") or _safe_candidate_identity(record)
    if not job_id:
        return
    try:
        visual_diagnostics.capture_company_recruiter_location(
            page,
            job_id=job_id,
            reason=reason,
        )
        if warnings is not None:
            warnings.append(f"remote_scope_visual_debug:{job_id}:{reason}")
    except Exception:
        return

def _visible_location_matches_requested_scope(
    visible_location: str,
    requested_location: str,
    country_signal_text: str = "",
) -> bool:
    """Reject regional/remote spillover unless the job content names the country."""
    requested = (requested_location or "").casefold()
    visible = (visible_location or "").casefold()
    if not requested or not visible:
        return True

    if _has_any_marker(requested, _KOREA_LOCATION_MARKERS):
        if _has_any_marker(visible, _KOREA_LOCATION_MARKERS):
            return True
        if _has_any_marker(visible, _JAPAN_LOCATION_MARKERS):
            return False
        remote_scope = _remote_or_hybrid_location_matches_requested_country(
            visible,
            requested,
            country_signal_text,
        )
        if remote_scope is not None:
            return remote_scope
        return True

    if _has_any_marker(requested, _JAPAN_LOCATION_MARKERS):
        if _has_any_marker(visible, _JAPAN_LOCATION_MARKERS):
            return True
        if _has_any_marker(visible, _KOREA_LOCATION_MARKERS):
            return False
        remote_scope = _remote_or_hybrid_location_matches_requested_country(
            visible,
            requested,
            country_signal_text,
        )
        if remote_scope is not None:
            return remote_scope
        return True

    return True


def _normalized_numeric_job_id(value: str) -> str:
    match = re.search(r"(\d+)\s*$", str(value or "").strip())
    if not match:
        return ""
    return match.group(1).lstrip("0") or "0"


def _callable_accepts_keyword(callback: Callable[..., Any], keyword: str) -> bool:
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



def _search_hydration_outcome_for_row_activation(outcome: str) -> str:
    """Map row activation diagnostics to valid search hydration outcomes."""
    return "results" if outcome == "row_activation_success" else "failed"


def _date_stage_status(record: Any) -> str:
    if getattr(record, "published_at", None) is None:
        return "missing"
    return (
        "within_24_hours"
        if bool(getattr(record, "is_within_24_hours", False))
        else "outside_24_hours"
    )


def _date_stage_source(record: Any) -> str:
    sources = set(getattr(record, "discovery_sources", None) or [])
    if "visible_card" in sources:
        return "source_card"
    if "row_activation" in sources:
        return "source_row"
    if getattr(record, "published_at", None) is None:
        return "source_none"
    if sources:
        return "source_existing"
    return "source_unknown"


def _date_pipeline_stage_warning(record: Any, stage: str, *, reason: str = "") -> str:
    job_id = _safe_candidate_identity(record) or "unknown"
    safe_stage = re.sub(r"[^0-9A-Za-z_-]", "_", str(stage or "unknown"))[:40]
    if reason:
        safe_reason = re.sub(r"[^0-9A-Za-z_-]", "_", str(reason or "unknown"))[:50]
        return (
            f"date_pipeline_stage:{job_id}:{safe_stage}:"
            f"{safe_reason}:{_date_stage_source(record)}"
        )
    return (
        f"date_pipeline_stage:{job_id}:{safe_stage}:"
        f"{_date_stage_status(record)}:{_date_stage_source(record)}"
    )


def _apply_visible_card_date_evidence(
    record: Any,
    identity: str,
    visible_card_dates: dict[str, tuple[str, datetime, str, bool]],
) -> tuple[Any, str]:
    """Apply same-card visible date evidence without degrading accepted freshness.

    The search card is the authoritative source for search-time freshness when it
    is tied to the same job_id. It may fill a missing date or correct an existing
    outside_24_hours value. It must not overwrite an already within_24_hours
    candidate, because that could turn a valid candidate into a false reject.
    """
    if not identity or identity not in visible_card_dates:
        return record, ""
    if getattr(record, "published_at", None) is not None and bool(
        getattr(record, "is_within_24_hours", False)
    ):
        return record, ""

    previous_status = _date_stage_status(record)
    posted_text, published_at, confidence, within_24h = visible_card_dates[identity]
    updated = record.model_copy(
        update={
            "posted_at_text": posted_text,
            "published_at": published_at,
            "freshness_confidence": confidence,
            "is_within_24_hours": within_24h,
            "discovery_sources": sorted(
                set(getattr(record, "discovery_sources", None) or [])
                | {"visible_card"}
            ),
        }
    )
    next_status = "within_24_hours" if within_24h else "outside_24_hours"
    return updated, f"{previous_status}_to_{next_status}"


def _candidate_has_verified_fresh_date(record: Any) -> bool:
    return getattr(record, "published_at", None) is not None and bool(
        getattr(record, "is_within_24_hours", False)
    )


def _detail_priority_label(record: Any) -> str:
    sources = set(getattr(record, "discovery_sources", None) or [])
    if _candidate_has_verified_fresh_date(record) and "visible_card" in sources:
        return "verified_card_date"
    if _candidate_has_verified_fresh_date(record):
        return "verified_date"
    if str(getattr(record, "title", "") or "").strip() and getattr(
        record,
        "matched_terms",
        None,
    ):
        return "strong_metadata_missing_date"
    if str(getattr(record, "title", "") or "").strip():
        return "title_missing_date"
    return "incomplete_metadata"


def _detail_priority_rank(record: Any) -> int:
    return {
        "verified_card_date": 0,
        "verified_date": 1,
        "strong_metadata_missing_date": 2,
        "title_missing_date": 3,
        "incomplete_metadata": 4,
    }.get(_detail_priority_label(record), 9)


def _prioritize_candidates_for_detail(candidates: list[Any]) -> list[Any]:
    """Spend detail budget first on candidates most likely to pass validation."""
    return [
        record
        for _rank, _index, record in sorted(
            (
                (_detail_priority_rank(record), index, record)
                for index, record in enumerate(candidates)
            ),
            key=lambda item: (item[0], item[1]),
        )
    ]


def _row_activation_date_warning(outcome: Any) -> str:
    job_id = _normalized_numeric_job_id(getattr(outcome, "job_id", ""))
    if not job_id:
        return "row_date_unattributed"
    if bool(getattr(outcome, "date_verified", False)):
        freshness = (
            "within_24_hours"
            if bool(getattr(outcome, "date_within_24_hours", False))
            else "outside_24_hours"
        )
        return f"row_date_verified:{job_id}:{freshness}"
    if bool(getattr(outcome, "date_detected", False)):
        return f"row_date_unparseable:{job_id}"
    return f"row_date_missing:{job_id}"

def _wait_for_search_results_hydration_with_diagnostics(
    callback: Callable[..., str],
    page: Any,
    *,
    query: str,
    diagnostics: LinkedInSearchHydrationDiagnosticsCollector | None,
) -> str:
    kwargs: dict[str, Any] = {}
    if _callable_accepts_keyword(callback, "query"):
        kwargs["query"] = query
    if diagnostics and _callable_accepts_keyword(callback, "diagnostics"):
        kwargs["diagnostics"] = diagnostics
    return callback(page, **kwargs)


def _latest_search_hydration_event(
    diagnostics: LinkedInSearchHydrationDiagnosticsCollector | None,
    *,
    query: str,
) -> Any | None:
    if diagnostics is None:
        return None
    for event in reversed(diagnostics.events):
        if getattr(event, "query", "") == query:
            return event
    return None


def _is_timeout_without_search_signals(
    hydration_state: str,
    diagnostics: LinkedInSearchHydrationDiagnosticsCollector | None,
    *,
    query: str,
) -> bool:
    if hydration_state != "timeout":
        return False
    event = _latest_search_hydration_event(diagnostics, query=query)
    if event is None:
        return False
    return (
        getattr(event, "outcome", "") == "timeout"
        and int(getattr(event, "card_count", 0) or 0) == 0
        and int(getattr(event, "href_count", 0) or 0) == 0
        and not bool(getattr(event, "empty_state_visible", False))
        and not bool(getattr(event, "auth_checkpoint_visible", False))
    )


def _static_search_probe_outcome(probe: Any) -> str:
    if getattr(probe, "records", None):
        return "ok"
    detail = str(getattr(probe, "detail", "") or "")
    category = str(getattr(probe, "category", "") or "")
    if detail == "final_url_rejected":
        return "invalid_url"
    if category in {"query_access_rejected", "query_rate_limited"}:
        return "failed"
    return "no_candidates"


def _record_search_static_probe(
    diagnostics: LinkedInStaticProbeDiagnosticsCollector | None,
    *,
    query: str,
    probe: Any,
    accepted_count: int,
) -> None:
    if diagnostics is None:
        return
    diagnostics.record(
        kind="search_static_probe",
        query=query,
        status_code=int(getattr(probe, "status_code", 0) or 0),
        candidate_count=len(getattr(probe, "records", []) or []),
        accepted_count=accepted_count,
        outcome=_static_search_probe_outcome(probe),
    )


def _static_detail_probe_outcome(probe: Any) -> str:
    category = str(getattr(probe, "category", "") or "")
    detail = str(getattr(probe, "detail", "") or "")
    if category == "ok":
        return "ok"
    return {
        "missing_description": "missing_description",
        "incomplete_description": "incomplete_description",
        "missing_date": "missing_date",
        "outside_24_hours": "outside_24_hours",
        "final_url_rejected": "invalid_url",
        "job_id_mismatch": "invalid_url",
        "auth_checkpoint": "auth_checkpoint",
    }.get(detail, "failed")


def _record_detail_static_probe(
    diagnostics: LinkedInStaticProbeDiagnosticsCollector | None,
    *,
    probe: Any,
    record: Any,
) -> None:
    if diagnostics is None:
        return
    diagnostics.record(
        kind="detail_static_probe",
        job_id=getattr(record, "linkedin_job_id", ""),
        status_code=int(getattr(probe, "status_code", 0) or 0),
        candidate_count=1,
        accepted_count=1 if getattr(probe, "category", "") == "ok" else 0,
        outcome=_static_detail_probe_outcome(probe),
        body_source=str(getattr(probe, "body_source", "") or ""),
        description_length=int(getattr(probe, "description_length", 0) or 0),
        guest_status_code=int(getattr(probe, "guest_status_code", 0) or 0),
        guest_retry_count=int(getattr(probe, "guest_retry_count", 0) or 0),
        identity_consistent=bool(getattr(probe, "identity_consistent", True)),
    )


def _needs_static_detail_probe(
    record: Any,
    *,
    include_description: bool,
    is_incomplete_detail_body: Callable[[str], bool],
) -> bool:
    description = str(getattr(record, "description_full_text", "") or "")
    if include_description and not description.strip():
        return True
    if include_description and is_incomplete_detail_body(description):
        return True
    return getattr(record, "published_at", None) is None


def _has_plausible_detail_body(
    record: Any,
    *,
    include_description: bool,
    is_incomplete_detail_body: Callable[[str], bool],
) -> bool:
    if not include_description:
        return True
    description = str(getattr(record, "description_full_text", "") or "")
    return bool(description.strip()) and not is_incomplete_detail_body(description)


def _merge_detail_evidence(
    current: Any,
    incoming: Any,
    *,
    include_description: bool,
    is_incomplete_detail_body: Callable[[str], bool],
) -> Any:
    """Preserve safe detail evidence when a later attempt is less complete."""

    if incoming is current:
        return current
    if not include_description:
        return incoming
    if not _has_plausible_detail_body(
        current,
        include_description=include_description,
        is_incomplete_detail_body=is_incomplete_detail_body,
    ):
        return incoming
    if _has_plausible_detail_body(
        incoming,
        include_description=include_description,
        is_incomplete_detail_body=is_incomplete_detail_body,
    ):
        return incoming

    preserved: dict[str, str] = {
        "description_full_text": str(
            getattr(current, "description_full_text", "") or ""
        )
    }
    incoming_excerpt = str(getattr(incoming, "description_excerpt", "") or "")
    current_excerpt = str(getattr(current, "description_excerpt", "") or "")
    if current_excerpt and not incoming_excerpt:
        preserved["description_excerpt"] = current_excerpt
    return incoming.model_copy(update=preserved)


def _static_detail_probe_preserves_body_but_misses_date(probe: Any, record: Any) -> bool:
    return (
        str(getattr(probe, "category", "") or "") == "detail_incomplete"
        and str(getattr(probe, "detail", "") or "") == "missing_date"
        and bool(str(getattr(record, "description_full_text", "") or "").strip())
        and getattr(record, "published_at", None) is None
    )


def _session_has_authenticated_request(session: Any) -> bool:
    context = getattr(session, "context", None)
    request = getattr(context, "request", None)
    return callable(getattr(request, "get", None))


def _run_guarded_direct_detail_fallback(
    page,
    candidate: Any,
    *,
    source_url: str,
    include_description: bool,
    warnings: list[str],
    validate_jobs_url: Callable[[str], str],
    validate_authenticated_page: Callable[[object], None],
    wait_for_search_results_hydration: Callable[..., str],
    enrich_job_detail: Callable[..., Any],
    safe_error_label: Callable[[Exception], str],
    terminal_error_types: tuple[type[Exception], ...],
    diagnostics: LinkedInDetailDiagnosticsCollector | None = None,
) -> Any:
    """Open one exact detail URL and restore the original hydrated search route."""

    job_id = _normalized_numeric_job_id(
        getattr(candidate, "linkedin_job_id", "")
    )
    if not job_id:
        raise ValueError("direct_detail_job_id_missing")
    original_search_url = validate_jobs_url(source_url)
    candidate_url = str(getattr(candidate, "canonical_url", "") or "").strip()
    direct_url = validate_jobs_url(
        candidate_url or f"https://www.linkedin.com/jobs/view/{job_id}"
    )
    direct_url_match = re.search(
        r"/jobs/view/(?:[^/?#]*-)?(\d+)/?(?:[?#]|$)",
        direct_url,
    )
    direct_url_job_id = (
        direct_url_match.group(1).lstrip("0") or "0"
        if direct_url_match
        else ""
    )
    if direct_url_job_id != job_id:
        raise ValueError("direct_detail_job_id_mismatch")
    direct_candidate = candidate.model_copy(
        update={
            "linkedin_job_id": job_id,
            "canonical_url": direct_url,
        }
    )
    warnings.append(f"direct_detail_fallback_used:{job_id}")
    if diagnostics:
        diagnostics.record(
            direct_candidate,
            phase="fallback",
            mode="direct",
            outcome="started",
            include_description=include_description,
            date_ready=direct_candidate.published_at is not None,
        )
    try:
        enrichment_kwargs: dict[str, Any] = {
            "include_description": include_description,
            "now": datetime.now(timezone.utc),
        }
        if diagnostics and _callable_accepts_keyword(enrich_job_detail, "diagnostics"):
            enrichment_kwargs["diagnostics"] = diagnostics
        enriched = enrich_job_detail(page, direct_candidate, **enrichment_kwargs)
        return _merge_active_detail_metadata(enriched, page, warnings=warnings)
    except Exception:
        if diagnostics:
            diagnostics.record(
                direct_candidate,
                phase="fallback",
                mode="direct",
                outcome="failed",
                include_description=include_description,
                date_ready=direct_candidate.published_at is not None,
            )
        raise
    finally:
        try:
            page.goto(
                original_search_url,
                wait_until="domcontentloaded",
                timeout=30000,
            )
            validate_authenticated_page(page)
            hydration_state = (
                _wait_for_search_results_hydration_with_diagnostics(
                    wait_for_search_results_hydration,
                    page,
                    query=f"restore:{job_id}",
                    diagnostics=get_active_search_hydration_diagnostics(),
                )
            )
            if hydration_state != "results":
                warnings.append(f"list_not_hydrated:{job_id}")
                warnings.append(
                    f"search_restore_failed:{job_id}:list_not_hydrated"
                )
        except terminal_error_types:
            warnings.append(f"search_restore_failed:{job_id}:auth_or_blocked")
            raise
        except Exception as exc:
            warnings.append(
                f"search_restore_failed:{job_id}:{safe_error_label(exc)}"
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
    probe_detail_with_authenticated_request: Callable[..., Any]
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
            probe_detail_with_authenticated_request=module._probe_linkedin_detail_with_authenticated_request,
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
    _probe_linkedin_detail_with_authenticated_request = deps.probe_detail_with_authenticated_request
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
    detail_diagnostics = get_active_detail_diagnostics()
    search_hydration_diagnostics = get_active_search_hydration_diagnostics()
    static_probe_diagnostics = get_active_static_probe_diagnostics()
    visual_diagnostics = get_active_visual_diagnostics()

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
    seen_candidate_indexes: dict[str, int] = {}
    standalone_candidate_keys: set[str] = set()
    static_probe_candidate_keys: set[str] = set()
    rejected: list["LinkedInRejectedRecord"] = []
    timings: list["LinkedInQueryTiming"] = []
    warnings: list[str] = []
    query_urls: list[str] = []
    queued_for_detail_by_source: dict[str, int] = {}
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
                visual_run = (
                    visual_diagnostics.start_run(page, query=role)
                    if visual_diagnostics is not None
                    else None
                )
                try:
                    page.goto(
                        search_url,
                        wait_until="domcontentloaded",
                        timeout=30000,
                    )
                    _validate_authenticated_page(page)
                    hydration_state = (
                        _wait_for_search_results_hydration_with_diagnostics(
                            _wait_for_search_results_hydration,
                            page,
                            query=role,
                            diagnostics=search_hydration_diagnostics,
                        )
                    )
                    if visual_run is not None:
                        if hasattr(visual_run, "capture_after_hydration"):
                            visual_run.capture_after_hydration(page)
                        else:
                            visual_run.capture_before(page)
                    if hydration_state == "empty":
                        warnings.append(f"query_empty_results_explicit:{role}")
                    elif hydration_state == "timeout":
                        warnings.append(f"list_not_hydrated:{role}")
                        warnings.append(
                            f"query_hydration_timeout:no_terminal_signal:{role}"
                        )
                    parser_kwargs: dict[str, Any] = {
                        "source_url": search_url,
                        "now": datetime.now(timezone.utc),
                    }
                    if hydration_state == "timeout" and _callable_accepts_keyword(
                        _parse_linkedin_jobs_html_with_diagnostics,
                        "allow_standalone_fallback",
                    ):
                        parser_kwargs["allow_standalone_fallback"] = True
                    discovered, diagnostics = (
                        _parse_linkedin_jobs_html_with_diagnostics(
                            page.content(),
                            **parser_kwargs,
                        )
                    )
                    visible_card_dates = collect_visible_search_card_dates(page)
                    if visible_card_dates:
                        updated_discovered = []
                        for record in discovered:
                            identity = _safe_candidate_identity(record)
                            record, card_date_transition = (
                                _apply_visible_card_date_evidence(
                                    record,
                                    identity or "",
                                    visible_card_dates,
                                )
                            )
                            if card_date_transition:
                                warnings.append(
                                    "card_date_verified:"
                                    f"{identity}:"
                                    f"{_date_stage_status(record)}"
                                )
                                if card_date_transition.startswith(
                                    "outside_24_hours_to_"
                                ):
                                    warnings.append(
                                        "card_date_overrode_existing:"
                                        f"{identity}:{card_date_transition}"
                                    )
                                warnings.append(
                                    _date_pipeline_stage_warning(
                                        record,
                                        "visible_card",
                                    )
                                )
                            updated_discovered.append(record)
                        discovered = updated_discovered
                    normal_dom_records = list(discovered)
                    timeout_without_search_signals = (
                        _is_timeout_without_search_signals(
                            hydration_state,
                            search_hydration_diagnostics,
                            query=role,
                        )
                    )
                    static_search_probe = None
                    static_probe_keys_for_query: set[str] = set()
                    accepted_from_static_probe = 0
                    if timeout_without_search_signals:
                        warnings.append(
                            "search_static_probe_attempt:"
                            f"{query_location or 'unspecified'}"
                        )
                        static_probe_kwargs: dict[str, Any] = {
                            "source_url": search_url,
                            "now": datetime.now(timezone.utc),
                        }
                        if _callable_accepts_keyword(
                            _probe_linkedin_search_with_authenticated_request,
                            "allow_standalone_fallback",
                        ):
                            static_probe_kwargs[
                                "allow_standalone_fallback"
                            ] = True
                        static_search_probe = (
                            _probe_linkedin_search_with_authenticated_request(
                                session,
                                **static_probe_kwargs,
                            )
                        )
                        static_records = list(
                            getattr(static_search_probe, "records", []) or []
                        )
                        static_warning = (
                            "search_static_probe_result:"
                            f"{query_location or 'unspecified'}:"
                            f"status_{getattr(static_search_probe, 'status_code', 0)}:"
                            f"{getattr(static_search_probe, 'category', 'unknown')}:"
                            f"candidates_{len(static_records)}"
                        )
                        if getattr(static_search_probe, "detail", ""):
                            static_warning += (
                                f":{getattr(static_search_probe, 'detail')}"
                            )
                        warnings.append(static_warning)
                        if static_records:
                            diagnostics = static_search_probe.diagnostics
                            for record in static_records:
                                static_probe_keys_for_query.add(_record_key(record))
                            discovered.extend(static_records)
                    row_discovery = None
                    dom_records_before_rows = list(discovered)
                    if visual_run is not None and hasattr(
                        visual_run, "capture_before_scroll"
                    ):
                        visual_run.capture_before_scroll(page)
                    dom_job_ids = {
                        _safe_candidate_identity(record)
                        for record in normal_dom_records
                        if _safe_candidate_identity(record)
                    }
                    if len(dom_job_ids) < MIN_EXPECTED_DISCOVERY_CANDIDATES:
                        try:
                            row_activation_kwargs: dict[str, Any] = {
                                "source_url": search_url,
                                "existing_job_ids": set(dom_job_ids)
                                | {
                                    _safe_candidate_identity(record)
                                    for record in candidates
                                    if _safe_candidate_identity(record)
                                },
                            }
                            if (
                                visual_run is not None
                                and not hasattr(visual_run, "capture_detail")
                                and _callable_accepts_keyword(
                                discover_job_rows_via_activation,
                                "diagnostic_capture",
                                )
                            ):
                                row_activation_kwargs["diagnostic_capture"] = (
                                    visual_run.capture_activation
                                )
                            if visual_run is not None and _callable_accepts_keyword(
                                discover_job_rows_via_activation,
                                "diagnostic_detail_capture",
                            ):
                                row_activation_kwargs["diagnostic_detail_capture"] = (
                                    visual_run.capture_detail
                                )
                            if visual_run is not None and _callable_accepts_keyword(
                                discover_job_rows_via_activation,
                                "diagnostic_scroll",
                            ):
                                row_activation_kwargs["diagnostic_scroll"] = (
                                    visual_run.capture_after_scroll
                                )
                            row_discovery = discover_job_rows_via_activation(
                                page,
                                **row_activation_kwargs,
                            )
                        except Exception as row_exc:
                            warnings.append(
                                "row_activation_failed:"
                                f"{query_location or 'unspecified'}:"
                                f"{_safe_error_label(row_exc)}"
                            )
                        else:
                            valid_dom_records = [
                                record
                                for record in dom_records_before_rows
                                if _safe_candidate_identity(record)
                            ]
                            discovered, dom_contributed, row_contributed = (
                                merge_row_activation_records(
                                    valid_dom_records,
                                    row_discovery.records,
                                )
                            )
                            discovered.extend(
                                record
                                for record in dom_records_before_rows
                                if not _safe_candidate_identity(record)
                            )
                            if row_contributed or row_discovery.structural_rows_found > 0:
                                diagnostics = diagnostics.model_copy(
                                    update={
                                        "discovery_mode": discovery_mode_for_sources(
                                            dom_contributed=dom_contributed,
                                            row_contributed=row_contributed,
                                            structural_rows_found=(
                                                row_discovery.structural_rows_found > 0
                                            ),
                                            row_job_ids_resolved=row_discovery.job_ids_resolved,
                                        ),
                                        "discovery_degraded": True,
                                    }
                                )
                            diagnostics = diagnostics.model_copy(
                                update={
                                    "row_activation_count": row_discovery.activation_count,
                                    "row_activation_success_count": row_discovery.success_count,
                                    "row_activation_no_change_count": row_discovery.no_change_count,
                                    "row_activation_no_job_id_count": row_discovery.no_job_id_count,
                                    "row_activation_duplicate_count": row_discovery.duplicate_count,
                                    "row_activation_scroll_count": row_discovery.scroll_count,
                                    "selected_row_container_score": row_discovery.selected_row_container_score,
                                    "row_candidate_count": row_discovery.structural_rows_found,
                                    "row_interactive_count": row_discovery.row_interactive_count,
                                    "row_job_ids_resolved": row_discovery.job_ids_resolved,
                                    "row_activation_stop_reason": row_discovery.stop_reason,
                                    "candidate_count": len({
                                        _safe_candidate_identity(record)
                                        for record in discovered
                                        if _safe_candidate_identity(record)
                                    }),
                                    "unique_candidate_count": len({
                                        _safe_candidate_identity(record)
                                        for record in discovered
                                        if _safe_candidate_identity(record)
                                    }),
                                }
                            )
                            for outcome in row_discovery.outcomes:
                                if search_hydration_diagnostics is not None:
                                    search_hydration_diagnostics.record(
                                        query=role,
                                        elapsed_ms=0,
                                        row_activation_count=1,
                                        row_activation_success_count=(
                                            1
                                            if outcome.outcome
                                            == "row_activation_success"
                                            else 0
                                        ),
                                        row_activation_no_change_count=(
                                            1
                                            if outcome.outcome
                                            == "row_activation_no_change"
                                            else 0
                                        ),
                                        row_activation_no_job_id_count=(
                                            1
                                            if outcome.outcome
                                            == "row_activation_no_job_id"
                                            else 0
                                        ),
                                        row_activation_duplicate_count=(
                                            1
                                            if outcome.outcome
                                            == "row_activation_duplicate"
                                            else 0
                                        ),
                                        row_activation_scroll_count=(
                                            row_discovery.scroll_count
                                        ),
                                        selected_row_container_score=(
                                            row_discovery.selected_row_container_score
                                        ),
                                        row_candidate_count=(
                                            row_discovery.structural_rows_found
                                        ),
                                        row_interactive_count=(
                                            row_discovery.row_interactive_count
                                        ),
                                        row_job_ids_resolved=(
                                            row_discovery.job_ids_resolved
                                        ),
                                        row_activation_stop_reason=(
                                            row_discovery.stop_reason
                                        ),
                                        outcome=_search_hydration_outcome_for_row_activation(
                                            outcome.outcome
                                        ),
                                    )
                                if outcome.outcome in {
                                    "row_activation_success",
                                    "row_activation_duplicate",
                                    "row_activation_no_job_id",
                                }:
                                    warnings.append(
                                        _row_activation_date_warning(outcome)
                                    )
                                if outcome.outcome != "row_activation_success":
                                    warnings.append(outcome.outcome)
                    if visual_run is not None:
                        # Capture after bounded row discovery so the panel snapshot
                        # includes the post-scroll virtualized state.
                        visual_run.capture_after(page)
                    search_navigation_state.active_source_url = search_url
                    discovered_count = len(discovered)
                    standalone_fallback_count = (
                        diagnostics.discard_reasons.get(
                            "standalone_link_fallback",
                            0,
                        )
                    )
                    discovered_from_standalone_fallback = (
                        hydration_state == "timeout"
                        and standalone_fallback_count > 0
                    )
                    queued_from_query = 0
                    for record in discovered:
                        identity = _safe_candidate_identity(record)
                        if not identity:
                            rejected.append(
                                LinkedInRejectedRecord(
                                    source_url=getattr(record, "canonical_url", ""),
                                    title=getattr(record, "title", ""),
                                    reason="missing_job_identity",
                                )
                            )
                            continue
                        record = record.model_copy(
                            update={"linkedin_job_id": identity}
                        )
                        warnings.append(_date_pipeline_stage_warning(record, "before_dedupe"))
                        dedupe_key = _record_key(record)
                        if dedupe_key in seen_candidate_keys:
                            existing_index = seen_candidate_indexes.get(dedupe_key)
                            if existing_index is not None:
                                candidates[existing_index] = _merge_candidate_discovery_evidence(
                                    candidates[existing_index],
                                    record,
                                )
                                warnings.append(
                                    _date_pipeline_stage_warning(
                                        candidates[existing_index],
                                        "after_dedupe_merge",
                                    )
                                )
                            warnings.append(
                                "duplicate_candidate_skipped_before_detail:"
                                f"{identity}"
                            )
                            continue
                        seen_candidate_keys.add(dedupe_key)
                        seen_candidate_indexes[dedupe_key] = len(candidates)
                        queued_from_query += 1
                        if discovered_from_standalone_fallback:
                            standalone_candidate_keys.add(dedupe_key)
                        if dedupe_key in static_probe_keys_for_query:
                            static_probe_candidate_keys.add(dedupe_key)
                            accepted_from_static_probe += 1
                        candidate_locations[dedupe_key] = query_location
                        candidates.append(record)
                        warnings.append(_date_pipeline_stage_warning(record, "after_queue"))
                    queued_for_detail_by_source[search_url] = (
                        queued_for_detail_by_source.get(search_url, 0)
                        + queued_from_query
                    )
                    diagnostics = _mark_diagnostics_after_query_dedupe(
                        diagnostics,
                        new_count=queued_from_query,
                    )
                    if static_search_probe is not None:
                        _record_search_static_probe(
                            static_probe_diagnostics,
                            query=role,
                            probe=static_search_probe,
                            accepted_count=accepted_from_static_probe,
                        )
                    consecutive_query_errors_by_location[query_location] = 0
                    recoverable_query_errors_by_location[query_location] = 0
                    error = ""
                    break
                except (LinkedInAuthRequiredError, LinkedInBlockedError):
                    if visual_run is not None:
                        visual_run.capture_after(page)
                    raise
                except Exception as exc:
                    if visual_run is not None:
                        visual_run.capture_after(page)
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
                            queued_from_query = 0
                            for record in probe.records:
                                identity = _safe_candidate_identity(record)
                                if not identity:
                                    rejected.append(
                                        LinkedInRejectedRecord(
                                            source_url=getattr(record, "canonical_url", ""),
                                            title=getattr(record, "title", ""),
                                            reason="missing_job_identity",
                                        )
                                    )
                                    continue
                                record = record.model_copy(
                                    update={"linkedin_job_id": identity}
                                )
                                dedupe_key = _record_key(record)
                                if dedupe_key in seen_candidate_keys:
                                    existing_index = seen_candidate_indexes.get(dedupe_key)
                                    if existing_index is not None:
                                        candidates[existing_index] = _merge_candidate_discovery_evidence(
                                            candidates[existing_index],
                                            record,
                                        )
                                    warnings.append(
                                        "duplicate_candidate_skipped_before_detail:"
                                        f"{identity}"
                                    )
                                    continue
                                seen_candidate_keys.add(dedupe_key)
                                seen_candidate_indexes[dedupe_key] = len(candidates)
                                queued_from_query += 1
                                candidate_locations[dedupe_key] = (
                                    query_location
                                )
                                candidates.append(record)
                            queued_for_detail_by_source[search_url] = (
                                queued_for_detail_by_source.get(search_url, 0)
                                + queued_from_query
                            )
                            diagnostics = _mark_diagnostics_after_query_dedupe(
                                diagnostics,
                                new_count=queued_from_query,
                            )
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
                    if visual_run is not None:
                        visual_run.stop_trace(page)
                        visual_run.finalize()
                    search_navigation_state.completed_at = time.monotonic()
            completed = datetime.now(timezone.utc)
            timings.append(
                LinkedInQueryTiming(
                    query=role,
                    started_at=started,
                    completed_at=completed,
                    elapsed_ms=int((time.monotonic() - started_monotonic) * 1000),
                    discovered_count=discovered_count,
                    retained_count=queued_for_detail_by_source.get(search_url, 0),
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
            if (
                not candidate.matched_terms
                and not getattr(candidate, "candidate_metadata_incomplete", False)
            ):
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
        if request.include_description:
            shortlist = _prioritize_candidates_for_detail(shortlist)
            for candidate in shortlist:
                if _candidate_has_verified_fresh_date(candidate):
                    warnings.append(
                        "detail_priority_queue:"
                        f"{_safe_candidate_identity(candidate) or 'unknown'}:"
                        f"{_detail_priority_label(candidate)}"
                    )
        last_detail_click_at: float | None = None
        detail_attempts = 0
        detail_attempts_by_location: dict[str, int] = {}
        detail_network_circuit_locations: set[str] = set()
        direct_detail_fallback_disabled_locations: set[str] = set()
        direct_detail_fallback_attempted_candidates: set[str] = set()
        static_detail_probe_attempted_candidates: set[str] = set()
        consecutive_detail_network_failures_by_location: dict[str, int] = {}
        if not _session_page_is_alive(session):
            warnings.append(
                "detail_runtime_session_unavailable:"
                f"{next(iter(query_circuit_reasons_by_location.values()), 'page_unusable')}"
            )

        enriched_records: list["LinkedInVacancyRecord"] = []
        should_balance_locations = len([location for location in normalized_locations if location]) > 1

        def record_detail_rejection(record: Any, reason: str, mode: str) -> None:
            if not detail_diagnostics:
                return
            rejection = {
                "missing_description_full_text": "missing_description",
                "detail_incomplete_body": "incomplete_description",
                "unverified_posted_date": "missing_date",
                "outside_24_hours": "outside_24_hours",
                "location_scope_mismatch": "location_mismatch",
            }.get(reason, "detail_failure")
            detail_diagnostics.record(
                record,
                phase="validation",
                mode=mode,
                outcome="rejected",
                include_description=request.include_description,
                description_ready=bool(getattr(record, "description_full_text", "")),
                date_ready=getattr(record, "published_at", None) is not None,
                rejection=rejection,
            )

        for candidate in shortlist:
            candidate_key = _record_key(candidate)
            candidate_location = candidate_locations.get(candidate_key, "")
            verified = candidate
            needs_enrichment = _needs_detail_enrichment(candidate)
            detail_reason = ""
            detail_mode = "none"
            source_url = candidate.source_url
            is_standalone_candidate = candidate_key in standalone_candidate_keys
            is_static_probe_candidate = candidate_key in static_probe_candidate_keys
            is_direct_url_candidate = (
                is_standalone_candidate or is_static_probe_candidate
            )

            if (
                needs_enrichment
                and _session_has_authenticated_request(session)
                and candidate_key not in static_detail_probe_attempted_candidates
                and _needs_static_detail_probe(
                    verified,
                    include_description=request.include_description,
                    is_incomplete_detail_body=_is_incomplete_detail_body,
                )
                and detail_attempts < detail_budget
                and detail_attempts_by_location.get(candidate_location, 0)
                < detail_quota_per_location
            ):
                static_detail_probe_attempted_candidates.add(candidate_key)
                detail_attempts += 1
                detail_attempts_by_location[candidate_location] = (
                    detail_attempts_by_location.get(candidate_location, 0) + 1
                )
                job_id = verified.linkedin_job_id or "unknown"
                warnings.append(f"detail_static_probe_attempt:{job_id}")
                try:
                    static_detail_probe = (
                        _probe_linkedin_detail_with_authenticated_request(
                            session,
                            verified,
                            include_description=request.include_description,
                            now=datetime.now(timezone.utc),
                        )
                    )
                except (LinkedInAuthRequiredError, LinkedInBlockedError):
                    raise
                except Exception as exc:
                    warnings.append(
                        "detail_static_probe_failed:"
                        f"{job_id}:{_safe_error_label(exc)}"
                    )
                else:
                    _record_detail_static_probe(
                        static_probe_diagnostics,
                        probe=static_detail_probe,
                        record=verified,
                    )
                    static_detail_warning = (
                        "detail_static_probe_result:"
                        f"{job_id}:"
                        f"status_{getattr(static_detail_probe, 'status_code', 0)}:"
                        f"{getattr(static_detail_probe, 'category', 'unknown')}"
                    )
                    if getattr(static_detail_probe, "detail", ""):
                        static_detail_warning += (
                            f":{getattr(static_detail_probe, 'detail')}"
                        )
                    warnings.append(static_detail_warning)
                    probe_record = getattr(static_detail_probe, "record", verified)
                    verified = _merge_detail_evidence(
                        verified,
                        probe_record,
                        include_description=request.include_description,
                        is_incomplete_detail_body=_is_incomplete_detail_body,
                    )
                    warnings.append(_date_pipeline_stage_warning(verified, "after_static_probe"))
                    if _static_detail_probe_preserves_body_but_misses_date(
                        static_detail_probe,
                        verified,
                    ):
                        detail_reason = "unverified_posted_date"
                    if getattr(static_detail_probe, "category", "") == "ok":
                        verified = _merge_detail_evidence(
                            verified,
                            probe_record,
                            include_description=request.include_description,
                            is_incomplete_detail_body=_is_incomplete_detail_body,
                        )
                        needs_enrichment = False
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
                    if _candidate_has_verified_fresh_date(verified):
                        warnings.append(
                            "detail_budget_deferred_verified_candidate:"
                            f"{verified.linkedin_job_id or 'unknown'}"
                        )
                elif is_direct_url_candidate:
                    if not direct_detail_fallback:
                        warning_prefix = (
                            "standalone_direct_detail_fallback_disabled"
                            if is_standalone_candidate
                            else "static_probe_direct_detail_fallback_disabled"
                        )
                        warnings.append(
                            f"{warning_prefix}:"
                            f"{candidate.linkedin_job_id or 'unknown'}"
                        )
                    else:
                        direct_detail_fallback_attempted_candidates.add(
                            candidate_key
                        )
                        detail_attempts += 1
                        detail_attempts_by_location[candidate_location] = (
                            location_detail_attempts + 1
                        )
                        try:
                            search_navigation_state.invalidate_source()
                            detail_mode = "direct"
                            direct_record = _run_guarded_direct_detail_fallback(
                                page,
                                candidate,
                                source_url=source_url,
                                include_description=request.include_description,
                                warnings=warnings,
                                validate_jobs_url=validate_linkedin_jobs_url,
                                validate_authenticated_page=_validate_authenticated_page,
                                wait_for_search_results_hydration=_wait_for_search_results_hydration,
                                enrich_job_detail=_enrich_job_detail,
                                safe_error_label=_safe_error_label,
                                terminal_error_types=(
                                    LinkedInAuthRequiredError,
                                    LinkedInBlockedError,
                                ),
                                diagnostics=detail_diagnostics,
                            )
                            verified = _merge_detail_evidence(
                                verified,
                                direct_record,
                                include_description=request.include_description,
                                is_incomplete_detail_body=_is_incomplete_detail_body,
                            )
                            detail_reason = ""
                        except (LinkedInAuthRequiredError, LinkedInBlockedError):
                            raise
                        except Exception as exc:
                            detail_error = _safe_error_label(exc)
                            detail_reason = (
                                "detail_network_failure"
                                if _is_page_recoverable_error(detail_error)
                                else "detail_fetch_failed"
                            )
                            warnings.append(
                                f"detail_fallback_failed:"
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
                                    "selector_drift/card_missing"
                                )
                            _respect_detail_click_cadence(
                                page,
                                last_detail_click_at=last_detail_click_at,
                                interval_ms=detail_click_interval_ms,
                            )
                            detail_mode = "panel"
                            panel_record = _enrich_job_detail_via_panel(
                                page,
                                candidate,
                                card_link=card_link,
                                include_description=request.include_description,
                                now=datetime.now(timezone.utc),
                                diagnostics=detail_diagnostics,
                            )
                            verified = _merge_detail_evidence(
                                verified,
                                panel_record,
                                include_description=request.include_description,
                                is_incomplete_detail_body=_is_incomplete_detail_body,
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
                                            "selector_drift/card_missing"
                                        )
                                    _respect_detail_click_cadence(
                                        page,
                                        last_detail_click_at=last_detail_click_at,
                                        interval_ms=detail_click_interval_ms,
                                    )
                                    detail_mode = "panel"
                                    retry_panel_record = _enrich_job_detail_via_panel(
                                        page,
                                        candidate,
                                        card_link=retry_card_link,
                                        include_description=request.include_description,
                                        now=datetime.now(timezone.utc),
                                        diagnostics=detail_diagnostics,
                                    )
                                    verified = _merge_detail_evidence(
                                        verified,
                                        retry_panel_record,
                                        include_description=request.include_description,
                                        is_incomplete_detail_body=_is_incomplete_detail_body,
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
                            and candidate_key
                            not in direct_detail_fallback_attempted_candidates
                            and candidate_location
                            not in direct_detail_fallback_disabled_locations
                            and detail_attempts < detail_budget
                            and detail_attempts_by_location.get(
                                candidate_location,
                                0,
                            )
                            < detail_quota_per_location
                            and detail_reason
                            not in {
                                "detail_budget_exhausted",
                                "detail_network_failure",
                            }
                        )
                        if fallback_allowed:
                            direct_detail_fallback_attempted_candidates.add(
                                candidate_key
                            )
                            detail_attempts += 1
                            detail_attempts_by_location[candidate_location] = (
                                detail_attempts_by_location.get(
                                    candidate_location,
                                    0,
                                )
                                + 1
                            )
                            try:
                                search_navigation_state.invalidate_source()
                                detail_mode = "direct"
                                direct_record = _run_guarded_direct_detail_fallback(
                                    page,
                                    candidate,
                                    source_url=source_url,
                                    include_description=request.include_description,
                                    warnings=warnings,
                                    validate_jobs_url=validate_linkedin_jobs_url,
                                    validate_authenticated_page=_validate_authenticated_page,
                                    wait_for_search_results_hydration=_wait_for_search_results_hydration,
                                    enrich_job_detail=_enrich_job_detail,
                                    safe_error_label=_safe_error_label,
                                    terminal_error_types=(
                                        LinkedInAuthRequiredError,
                                        LinkedInBlockedError,
                                    ),
                                    diagnostics=detail_diagnostics,
                                )
                                verified = _merge_detail_evidence(
                                    verified,
                                    direct_record,
                                    include_description=request.include_description,
                                    is_incomplete_detail_body=_is_incomplete_detail_body,
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
                not detail_reason
                and _session_has_authenticated_request(session)
                and candidate_key not in static_detail_probe_attempted_candidates
                and _needs_static_detail_probe(
                    verified,
                    include_description=request.include_description,
                    is_incomplete_detail_body=_is_incomplete_detail_body,
                )
                and detail_attempts < detail_budget
                and detail_attempts_by_location.get(candidate_location, 0)
                < detail_quota_per_location
            ):
                static_detail_probe_attempted_candidates.add(candidate_key)
                detail_attempts += 1
                detail_attempts_by_location[candidate_location] = (
                    detail_attempts_by_location.get(candidate_location, 0) + 1
                )
                job_id = verified.linkedin_job_id or "unknown"
                warnings.append(f"detail_static_probe_attempt:{job_id}")
                try:
                    static_detail_probe = (
                        _probe_linkedin_detail_with_authenticated_request(
                            session,
                            verified,
                            include_description=request.include_description,
                            now=datetime.now(timezone.utc),
                        )
                    )
                except (LinkedInAuthRequiredError, LinkedInBlockedError):
                    raise
                except Exception as exc:
                    if not detail_reason:
                        detail_reason = (
                            "detail_network_failure"
                            if _is_page_recoverable_error(_safe_error_label(exc))
                            else "detail_fetch_failed"
                        )
                    warnings.append(
                        "detail_static_probe_failed:"
                        f"{job_id}:{_safe_error_label(exc)}"
                    )
                else:
                    _record_detail_static_probe(
                        static_probe_diagnostics,
                        probe=static_detail_probe,
                        record=verified,
                    )
                    static_detail_warning = (
                        "detail_static_probe_result:"
                        f"{job_id}:"
                        f"status_{getattr(static_detail_probe, 'status_code', 0)}:"
                        f"{getattr(static_detail_probe, 'category', 'unknown')}"
                    )
                    if getattr(static_detail_probe, "detail", ""):
                        static_detail_warning += (
                            f":{getattr(static_detail_probe, 'detail')}"
                        )
                    warnings.append(static_detail_warning)
                    probe_record = getattr(static_detail_probe, "record", verified)
                    verified = _merge_detail_evidence(
                        verified,
                        probe_record,
                        include_description=request.include_description,
                        is_incomplete_detail_body=_is_incomplete_detail_body,
                    )
                    warnings.append(_date_pipeline_stage_warning(verified, "after_static_probe"))
                    if _static_detail_probe_preserves_body_but_misses_date(
                        static_detail_probe,
                        verified,
                    ):
                        detail_reason = "unverified_posted_date"
                    if getattr(static_detail_probe, "category", "") == "ok":
                        verified = _merge_detail_evidence(
                            verified,
                            probe_record,
                            include_description=request.include_description,
                            is_incomplete_detail_body=_is_incomplete_detail_body,
                        )
                        detail_reason = ""
                    elif getattr(
                        static_detail_probe,
                        "category",
                        "",
                    ) in {
                        "detail_navigation_failure",
                        "detail_access_rejected",
                        "detail_rate_limited",
                        "detail_upstream_failure",
                    }:
                        detail_reason = "detail_fetch_failed"

            if (
                direct_detail_fallback
                and getattr(verified, "published_at", None) is None
                and bool(str(getattr(verified, "description_full_text", "") or "").strip())
                and candidate_key not in direct_detail_fallback_attempted_candidates
                and candidate_location not in direct_detail_fallback_disabled_locations
                and detail_attempts < detail_budget
                and detail_attempts_by_location.get(candidate_location, 0)
                < detail_quota_per_location
            ):
                direct_detail_fallback_attempted_candidates.add(candidate_key)
                detail_attempts += 1
                detail_attempts_by_location[candidate_location] = (
                    detail_attempts_by_location.get(candidate_location, 0) + 1
                )
                warnings.append(
                    "date_direct_detail_fallback_used:"
                    f"{candidate.linkedin_job_id or 'unknown'}"
                )
                try:
                    search_navigation_state.invalidate_source()
                    detail_mode = "direct"
                    direct_record = _run_guarded_direct_detail_fallback(
                        page,
                        verified,
                        source_url=source_url,
                        include_description=request.include_description,
                        warnings=warnings,
                        validate_jobs_url=validate_linkedin_jobs_url,
                        validate_authenticated_page=_validate_authenticated_page,
                        wait_for_search_results_hydration=_wait_for_search_results_hydration,
                        enrich_job_detail=_enrich_job_detail,
                        safe_error_label=_safe_error_label,
                        terminal_error_types=(
                            LinkedInAuthRequiredError,
                            LinkedInBlockedError,
                        ),
                        diagnostics=detail_diagnostics,
                    )
                    verified = _merge_detail_evidence(
                        verified,
                        direct_record,
                        include_description=request.include_description,
                        is_incomplete_detail_body=_is_incomplete_detail_body,
                    )
                    warnings.append(_date_pipeline_stage_warning(verified, "after_direct_fallback"))
                    if getattr(verified, "published_at", None) is not None:
                        detail_reason = ""
                        if (
                            not bool(getattr(verified, "is_within_24_hours", False))
                            and visual_diagnostics is not None
                            and hasattr(visual_diagnostics, "capture_active_detail_date")
                        ):
                            try:
                                visual_diagnostics.capture_active_detail_date(
                                    page,
                                    job_id=verified.linkedin_job_id
                                    or _safe_candidate_identity(verified),
                                    reason="outside_24_hours",
                                )
                            except Exception:
                                pass
                except (LinkedInAuthRequiredError, LinkedInBlockedError):
                    raise
                except Exception as exc:
                    warnings.append(
                        "date_direct_detail_fallback_failed:"
                        f"{candidate.linkedin_job_id or 'unknown'}:"
                        f"{_safe_error_label(exc)}"
                    )

            verified = _refresh_candidate_discovery_state(verified)

            if not verified.matched_terms:
                rejected.append(
                    LinkedInRejectedRecord(
                        source_url=verified.canonical_url,
                        title=verified.title,
                        reason="low_topic_relevance",
                    )
                )
                record_detail_rejection(verified, "low_topic_relevance", detail_mode)
                continue

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
                rejection_reason = detail_reason or "missing_description_full_text"
                record_detail_rejection(verified, rejection_reason, detail_mode)
                warnings.append(
                    "metadata_enrichment_incomplete:"
                    f"{verified.linkedin_job_id or 'unknown'}:"
                    f"{rejection_reason}"
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
                record_detail_rejection(verified, "detail_incomplete_body", detail_mode)
                warnings.append(
                    "metadata_enrichment_incomplete:"
                    f"{verified.linkedin_job_id or 'unknown'}:detail_incomplete_body"
                )
                continue

            if verified.published_at is None:
                verified = _activate_visible_search_card_date_evidence(
                    page,
                    verified,
                    warnings=warnings,
                )
                warnings.append(_date_pipeline_stage_warning(verified, "after_card_activation"))
            if verified.published_at is None:
                verified = _activate_visible_search_card_by_title_date_evidence(
                    page,
                    verified,
                    warnings=warnings,
                )
                warnings.append(
                    _date_pipeline_stage_warning(
                        verified,
                        "after_card_title_activation",
                    )
                )
            if verified.published_at is None:
                warnings.append(
                    _date_pipeline_stage_warning(
                        verified,
                        "final",
                        reason=detail_reason or "unverified_posted_date",
                    )
                )
                _capture_unverified_date_visual_evidence(
                    visual_diagnostics,
                    page,
                    verified,
                    warnings=warnings,
                )
                rejected.append(
                    LinkedInRejectedRecord(
                        source_url=verified.canonical_url,
                        title=verified.title,
                        reason=detail_reason or "unverified_posted_date",
                    )
                )
                record_detail_rejection(
                    verified,
                    detail_reason or "unverified_posted_date",
                    detail_mode,
                )
                continue
            if not verified.is_within_24_hours:
                warnings.append(
                    _date_pipeline_stage_warning(
                        verified,
                        "final",
                        reason="outside_24_hours",
                    )
                )
                if visual_diagnostics is not None and hasattr(
                    visual_diagnostics,
                    "capture_rejected_candidate_card",
                ):
                    try:
                        visual_diagnostics.capture_rejected_candidate_card(
                            page,
                            job_id=verified.linkedin_job_id or _safe_candidate_identity(verified),
                            reason="outside_24_hours",
                        )
                    except Exception:
                        pass
                rejected.append(
                    LinkedInRejectedRecord(
                        source_url=verified.canonical_url,
                        title=verified.title,
                        reason="outside_24_hours",
                    )
                )
                record_detail_rejection(verified, "outside_24_hours", detail_mode)
                continue
            _capture_remote_scope_visual_evidence(
                visual_diagnostics,
                page,
                verified,
                warnings=warnings,
            )
            company_about_signal = _company_about_location_evidence_signal(
                page,
                verified,
                candidate_location,
                visual_diagnostics,
                warnings=warnings,
            )
            recruiter_location_signal = _recruiter_location_evidence_signal(
                page,
                verified,
                candidate_location,
                visual_diagnostics,
                warnings=warnings,
            )
            remote_scope_page_signal = _remote_scope_page_country_evidence_signal(
                page,
                verified,
                candidate_location,
                warnings=warnings,
            )
            country_signal_text = _record_country_signal_text(verified)
            for scope_signal in (company_about_signal, recruiter_location_signal, remote_scope_page_signal):
                if scope_signal:
                    country_signal_text = f"{country_signal_text}\n{scope_signal}"
            if not _visible_location_matches_requested_scope(
                verified.location,
                candidate_location,
                country_signal_text,
            ):
                rejected.append(
                    LinkedInRejectedRecord(
                        source_url=verified.canonical_url,
                        title=verified.title,
                        reason="location_scope_mismatch",
                    )
                )
                record_detail_rejection(verified, "location_scope_mismatch", detail_mode)
                warnings.append(
                    "location_scope_mismatch:"
                    f"{verified.linkedin_job_id or 'unknown'}:"
                    f"requested={candidate_location or 'unspecified'}:"
                    f"visible={verified.location or 'unspecified'}"
                )
                continue
            if detail_reason:
                warnings.append(
                    "metadata_enrichment_incomplete:"
                    f"{verified.linkedin_job_id or 'unknown'}:"
                    f"{detail_reason}"
                )
            warnings.append(_date_pipeline_stage_warning(verified, "final", reason="accepted"))
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
        rejected, rejected_duplicate_warnings = _dedupe_rejected_records(rejected)
        warnings.extend(rejected_duplicate_warnings)
        if (
            not records
            and any(
                warning.startswith("query_hydration_timeout:")
                for warning in warnings
            )
            and any(
                warning.startswith(
                    ("search_static_probe_result:", "detail_static_probe_result:")
                )
                for warning in warnings
            )
            and any(
                item.reason
                in {
                    "missing_description_full_text",
                    "detail_incomplete_body",
                    "unverified_posted_date",
                    "detail_fetch_failed",
                    "detail_network_failure",
                    "detail_budget_exhausted",
                }
                for item in rejected
            )
            and "linkedin_hydration_unusable" not in warnings
        ):
            warnings.append("linkedin_hydration_unusable")

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

        timings = [
            timing.model_copy(
                update={"retained_count": queued_for_detail_by_source.get(query_url, 0)}
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
