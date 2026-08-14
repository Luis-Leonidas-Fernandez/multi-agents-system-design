"""Opt-in local visual diagnostics for LinkedIn search discovery."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
from typing import Iterator
from urllib.parse import urlsplit, urlunsplit

from features.web_scraping.domain.linkedin_models import (
    LinkedInVisualDiagnosticArtifact,
)
from features.web_scraping.infrastructure.linkedin_card_structure_debug import (
    CARD_STRUCTURE_DEBUG_FILENAME,
    capture_visible_linkedin_card_structure_debug,
)


_ENABLED_ENV = "LINKEDIN_SEARCH_VISUAL_DIAGNOSTICS"
_VISUAL_DIR = "visual-diagnostics"
_ALLOWED_DATA_ATTRIBUTES = (
    "data-job-id",
    "data-occludable-job-id",
    "data-entity-urn",
)
MAX_VISUAL_DIAGNOSTIC_QUERIES = 1
MAX_PANEL_HTML_CAPTURES = 2
MAX_ACTIVATION_HTML_CAPTURES = 3
MAX_HTML_SOURCE_CHARS = 4_000_000
MAX_SANITIZED_HTML_CHARS = 120_000
HTML_DIAGNOSTICS_ENV = "LINKEDIN_HTML_DIAGNOSTICS"
MAX_HTML_DIAGNOSTIC_ACTIVATIONS = 3
MAX_HTML_DIAGNOSTIC_SCROLL_CAPTURES = 3
MAX_REJECTED_CARD_VISUAL_CAPTURES = 12
MAX_ACTIVE_DETAIL_DATE_VISUAL_CAPTURES = 12
MAX_COMPANY_RECRUITER_LOCATION_VISUAL_CAPTURES = 12
MAX_COMPANY_ABOUT_LOCATION_VISUAL_CAPTURES = 12
MAX_RECRUITER_LOCATION_VISUAL_CAPTURES = 12
_HTML_DROP_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
    "iframe",
    "object",
    "embed",
    "canvas",
    "svg",
}
_HTML_ALLOWED_ARIA = {
    "aria-selected",
    "aria-current",
    "aria-expanded",
}
_HTML_SAFE_TAG = re.compile(r"^[a-z][a-z0-9]{0,20}$")
_HTML_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_HTML_SAFE_VALUE = re.compile(r"^[A-Za-z0-9_:/.-]{1,120}$")
_STRUCTURE_SCRIPT = """
() => {
  const allowedAttrs = ["data-job-id", "data-occludable-job-id", "data-entity-urn"];
  const jobSelector = "a[href*='/jobs/view/'], [data-entity-urn*='jobPosting'], [data-job-id], [data-occludable-job-id]";
  const allNodes = Array.from(document.querySelectorAll("body *"));
  const safeClasses = (node) => Array.from(node.classList || [])
    .slice(0, 12)
    .map((token) => String(token).slice(0, 80));
  const allowedAttributes = (node) => {
    const attrs = {};
    for (const name of allowedAttrs) {
      if (node.hasAttribute && node.hasAttribute(name)) {
        attrs[name] = String(node.getAttribute(name) || "").slice(0, 120);
      }
    }
    return attrs;
  };
  const descriptor = (node) => {
    const scrollHeight = Math.max(0, Math.trunc(node.scrollHeight || 0));
    const clientHeight = Math.max(0, Math.trunc(node.clientHeight || 0));
    const scrollTop = Math.max(0, Math.trunc(node.scrollTop || 0));
    const anchorCount = node.querySelectorAll ? node.querySelectorAll("a").length : 0;
    const jobsViewCount = node.querySelectorAll ? node.querySelectorAll("a[href*='/jobs/view/']").length : 0;
    const urnCount = node.querySelectorAll ? node.querySelectorAll("[data-entity-urn*='jobPosting']").length : 0;
    const dataJobIdCount = node.querySelectorAll ? node.querySelectorAll("[data-job-id]").length : 0;
    const dataOccludableJobIdCount = node.querySelectorAll ? node.querySelectorAll("[data-occludable-job-id]").length : 0;
    return {
      container_index: allNodes.indexOf(node),
      tag: String(node.tagName || "").toLowerCase(),
      class_tokens: safeClasses(node),
      attributes: allowedAttributes(node),
      contains_main: Boolean(node.matches && node.matches("main")) || Boolean(node.querySelector && node.querySelector("main")),
      scrollHeight,
      clientHeight,
      scrollTop,
      anchor_count: anchorCount,
      jobs_view_count: jobsViewCount,
      urn_count: urnCount,
      data_job_id_count: dataJobIdCount,
      data_occludable_job_id_count: dataOccludableJobIdCount,
      job_signal_count: jobsViewCount + urnCount + dataJobIdCount + dataOccludableJobIdCount
    };
  };
  const scrollables = allNodes
    .filter((node) => Math.max(0, Math.trunc(node.scrollHeight || 0)) > Math.max(0, Math.trunc(node.clientHeight || 0)) + 8)
    .map(descriptor)
    .sort((left, right) =>
      (right.job_signal_count - left.job_signal_count)
      || ((right.scrollHeight - right.clientHeight) - (left.scrollHeight - left.clientHeight))
      || (left.container_index - right.container_index)
    )
    .slice(0, 10);
  const structural = allNodes
    .filter((node) =>
      (node.matches && node.matches("main, section, article, [role='main'], [role='list'], [role='listbox'], [role='listitem'], [data-job-id], [data-occludable-job-id], [data-entity-urn*='jobPosting']"))
      || (node.querySelectorAll && node.querySelectorAll(jobSelector).length > 0)
    )
    .map(descriptor)
    .slice(0, 80);
  return {
    schema_version: "1.0",
    node_count: allNodes.length,
    frame_count: window.frames ? window.frames.length : 0,
    body_scrollHeight: Math.max(0, Math.trunc(document.body ? document.body.scrollHeight || 0 : 0)),
    viewport: {
      width: Math.max(0, Math.trunc(window.innerWidth || 0)),
      height: Math.max(0, Math.trunc(window.innerHeight || 0))
    },
    signal_counts: {
      anchors: document.querySelectorAll("a").length,
      jobs_view: document.querySelectorAll("a[href*='/jobs/view/']").length,
      urn: document.querySelectorAll("[data-entity-urn*='jobPosting']").length,
      data_job_id: document.querySelectorAll("[data-job-id]").length,
      data_occludable_job_id: document.querySelectorAll("[data-occludable-job-id]").length
    },
    scrollables,
    structural
  };
}
"""


_REJECTED_CARD_FOCUS_SCRIPT = r"""
(jobId) => {
  const safeJobId = String(jobId || '').replace(/\D+/g, '');
  const patterns = [
    /\bhace\s+(?:unos?\s+segundos?|un\s+momento|\d+\s+(?:minutos?|horas?|d[ií]as?))\b/i,
    /\b(?:just now|moments ago|a few seconds ago|\d+\s+(?:minutes?|hours?|days?)\s+ago)\b/i,
    /\b(?:hoy|today)\b/i,
    /\b\d+\s*(?:시간|분|일)\s*전\b/i,
    /\b\d+\s*(?:時間|分|日)前\b/i,
  ];
  const textOf = (node) => String(node && (node.innerText || node.textContent) || '').replace(/\s+/g, ' ').trim();
  const bbox = (node) => {
    const rect = node && node.getBoundingClientRect ? node.getBoundingClientRect() : null;
    return rect ? {
      x: Math.round(rect.left || 0),
      y: Math.round(rect.top || 0),
      width: Math.round(rect.width || 0),
      height: Math.round(rect.height || 0),
    } : {x: 0, y: 0, width: 0, height: 0};
  };
  const dateTexts = (node) => {
    const found = [];
    const nodes = [node, ...Array.from(node && node.querySelectorAll ? node.querySelectorAll('*') : []).slice(0, 180)];
    for (const item of nodes) {
      const text = textOf(item);
      if (!text || text.length > 360) continue;
      for (const pattern of patterns) {
        const match = text.match(pattern);
        if (match && match[0] && !found.includes(match[0])) found.push(match[0].slice(0, 80));
      }
      if (found.length >= 5) break;
    }
    return found;
  };
  const ancestorDescriptors = (node) => {
    const chain = [];
    let current = node;
    for (let depth = 0; current && current !== document.body && depth < 7; depth += 1) {
      chain.push({
        depth,
        tag: String(current.tagName || '').toLowerCase().slice(0, 32),
        class_tokens: Array.from(current.classList || []).slice(0, 12).map((token) => String(token).slice(0, 80)),
        has_href: Boolean(current.querySelector && current.querySelector('a[href*="/jobs/view/"]')),
        has_data_job_id: Boolean(current.hasAttribute && current.hasAttribute('data-job-id')),
        has_data_occludable_job_id: Boolean(current.hasAttribute && current.hasAttribute('data-occludable-job-id')),
        bbox: bbox(current),
        date_texts: dateTexts(current),
        child_count: current.children ? current.children.length : 0,
      });
      current = current.parentElement;
    }
    return chain;
  };
  const hrefSelector = `a[href*="/jobs/view/${safeJobId}"], a[href*="-${safeJobId}"]`;
  const attrSelector = `[data-job-id$="${safeJobId}"], [data-occludable-job-id$="${safeJobId}"], [data-entity-urn$="${safeJobId}"]`;
  let node = document.querySelector(attrSelector);
  let selector_kind = node ? 'attribute' : '';
  if (!node) {
    const link = document.querySelector(hrefSelector);
    if (link) {
      selector_kind = 'href';
      node = link.closest('li, [role="listitem"], [role="option"], .scaffold-layout__list-item, .job-card-container') || link;
    }
  }
  if (!node) {
    return {found: false, job_id: safeJobId, selector_kind: 'none'};
  }
  try { node.scrollIntoView({block: 'center', inline: 'nearest'}); } catch (_) {}
  return {
    found: true,
    job_id: safeJobId,
    selector_kind,
    tag: String(node.tagName || '').toLowerCase().slice(0, 32),
    class_tokens: Array.from(node.classList || []).slice(0, 12).map((token) => String(token).slice(0, 80)),
    has_href: Boolean(node.querySelector && node.querySelector('a[href*="/jobs/view/"]')),
    has_data_job_id: Boolean(node.hasAttribute && node.hasAttribute('data-job-id')),
    has_data_occludable_job_id: Boolean(node.hasAttribute && node.hasAttribute('data-occludable-job-id')),
    title_present: Boolean(node.querySelector && node.querySelector('a[href*="/jobs/view/"], .job-card-list__title, .job-card-container__link')),
    date_texts: dateTexts(node),
    bbox: bbox(node),
    ancestor_chain: ancestorDescriptors(node),
  };
}
"""


_ACTIVE_DETAIL_DATE_FOCUS_SCRIPT = r"""
() => {
  const patterns = [
    /\bhace\s+(?:unos?\s+segundos?|un\s+momento|\d+\s+(?:minutos?|horas?|d[ií]as?))\b/i,
    /\b(?:just now|moments ago|a few seconds ago|\d+\s+(?:minutes?|hours?|days?)\s+ago)\b/i,
    /\b(?:hoy|today)\b/i,
    /\b\d+\s*(?:시간|분|일)\s*전\b/i,
    /\b\d+\s*(?:時間|分|日)前\b/i,
  ];
  const bbox = (node) => {
    const rect = node && node.getBoundingClientRect ? node.getBoundingClientRect() : null;
    return rect ? {
      x: Math.round(rect.left || 0),
      y: Math.round(rect.top || 0),
      width: Math.round(rect.width || 0),
      height: Math.round(rect.height || 0),
    } : {x: 0, y: 0, width: 0, height: 0};
  };
  const visible = (node) => {
    const rect = node && node.getBoundingClientRect ? node.getBoundingClientRect() : null;
    return Boolean(rect && rect.width > 0 && rect.height > 0 && rect.bottom >= 0 && rect.right >= 0 && rect.top <= (window.innerHeight || 0));
  };
  const textOf = (node) => String(node && (node.innerText || node.textContent) || '').replace(/\s+/g, ' ').trim();
  const matchDate = (text) => {
    for (const pattern of patterns) {
      const match = String(text || '').match(pattern);
      if (match && match[0]) return match[0].slice(0, 80);
    }
    return "";
  };
  const main = document.querySelector('main') || document.body;
  const heading = Array.from(document.querySelectorAll('main h1, h1')).find(visible) || null;
  const detailRoot = heading ? (heading.closest('section, article, main, .jobs-search__job-details--container, .jobs-details') || main) : main;
  const candidates = [];
  const nodes = Array.from(detailRoot.querySelectorAll ? detailRoot.querySelectorAll('*') : []).slice(0, 900);
  for (const node of nodes) {
    if (!visible(node)) continue;
    const text = textOf(node);
    if (!text || text.length > 360) continue;
    const date = matchDate(text);
    if (!date) continue;
    const rect = node.getBoundingClientRect();
    candidates.push({
      date_text: date,
      tag: String(node.tagName || '').toLowerCase().slice(0, 32),
      class_tokens: Array.from(node.classList || []).slice(0, 12).map((token) => String(token).slice(0, 80)),
      bbox: bbox(node),
      distance_from_title: heading ? Math.round(Math.abs(rect.top - heading.getBoundingClientRect().top)) : 0,
    });
    if (candidates.length >= 12) break;
  }
  return {
    found: Boolean(detailRoot),
    title_present: Boolean(heading),
    title_bbox: heading ? bbox(heading) : {x: 0, y: 0, width: 0, height: 0},
    detail_bbox: bbox(detailRoot),
    date_candidates: candidates,
  };
}
"""


def visual_diagnostics_enabled() -> bool:
    return str(os.getenv(_ENABLED_ENV, "")).strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def html_diagnostics_enabled() -> bool:
    """Return the secure HTML diagnostics opt-in state."""
    return str(os.getenv(HTML_DIAGNOSTICS_ENV, "false")).strip().casefold() in {
        "1", "true", "yes", "on"
    }


def _sanitize_query(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    folded = text.casefold()
    if "://" in folded or "linkedin.com" in folded or ("<" in text and ">" in text):
        return "redacted"
    return text[:180] or "unknown"


def _relative(path: Path, *, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.name


def _safe_write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _safe_job_id(value: object) -> str:
    match = re.search(r"(\d{3,})", str(value or ""))
    return match.group(1).lstrip("0") if match else ""


def _safe_reason_label(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z_-]", "_", str(value or "unknown"))[:50] or "unknown"


class _SanitizedHTMLParser(HTMLParser):
    """Emit a bounded DOM skeleton without page text or sensitive attributes."""

    def __init__(self, *, max_chars: int) -> None:
        super().__init__(convert_charrefs=False)
        self.max_chars = max_chars
        self.parts: list[str] = []
        self.open_tags: list[str] = []
        self.dropped_depth = 0
        self.removed_tags: set[str] = set()
        self.removed_attributes = 0
        self.event_attributes_removed = 0
        self.urls_canonicalized = 0
        self.query_strings_removed = 0
        self.nodes_serialized = 0
        self.truncated = False

    @property
    def output(self) -> str:
        return "".join(self.parts)

    def _append(self, value: str) -> None:
        if self.truncated or len(self.output) + len(value) > self.max_chars:
            self.truncated = True
            return
        self.parts.append(value)

    def _safe_attributes(self, attrs: list[tuple[str, str | None]]) -> str:
        safe: list[str] = []
        for raw_name, raw_value in attrs:
            name = str(raw_name or "").casefold()
            value = str(raw_value or "").strip()
            if name == "class":
                tokens = [
                    re.sub(r"[^A-Za-z0-9_-]", "", token)[:80]
                    for token in value.split()
                    if re.sub(r"[^A-Za-z0-9_-]", "", token)[:80]
                ][:20]
                if tokens:
                    safe.append(
                        f' class="{escape(" ".join(token[:80] for token in tokens), quote=True)}"'
                    )
                continue
            if name.startswith("on"):
                self.event_attributes_removed += 1
                self.removed_attributes += 1
                continue
            if name == "href":
                parts = urlsplit(value)
                if parts.scheme and parts.scheme.casefold() not in {"http", "https"}:
                    canonical = ""
                else:
                    # Keep only the canonical path: no host, query, fragment, or token.
                    canonical = urlunsplit(("", "", parts.path, "", ""))
                if canonical and len(canonical) <= 240:
                    safe.append(f' href="{escape(canonical, quote=True)}"')
                    self.urls_canonicalized += 1
                    if parts.query or parts.fragment:
                        self.query_strings_removed += 1
                else:
                    self.removed_attributes += 1
                continue
            if name == "role" or name == "tabindex" or name in _HTML_ALLOWED_ARIA:
                if value and len(value) <= 80 and _HTML_SAFE_VALUE.fullmatch(value):
                    safe.append(f' {name}="{escape(value, quote=True)}"')
                else:
                    self.removed_attributes += 1
                continue
            if name in _ALLOWED_DATA_ATTRIBUTES and _HTML_SAFE_VALUE.fullmatch(value):
                safe.append(f' {name}="{escape(value, quote=True)}"')
                continue
            self.removed_attributes += 1
        return "".join(safe)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        name = str(tag or "").casefold()
        if name in _HTML_DROP_TAGS:
            self.removed_tags.add(name)
            self.dropped_depth += 1
            return
        if self.dropped_depth or not _HTML_SAFE_TAG.fullmatch(name):
            return
        self._append(f"<{name}{self._safe_attributes(attrs)}>")
        if not self.truncated:
            self.open_tags.append(name)
            self.nodes_serialized += 1

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        name = str(tag or "").casefold()
        if name in _HTML_DROP_TAGS:
            self.removed_tags.add(name)
            return
        if self.dropped_depth or not _HTML_SAFE_TAG.fullmatch(name):
            return
        self._append(f"<{name}{self._safe_attributes(attrs)}/>")
        if not self.truncated:
            self.nodes_serialized += 1

    def handle_endtag(self, tag: str) -> None:
        name = str(tag or "").casefold()
        if name in _HTML_DROP_TAGS:
            self.dropped_depth = max(0, self.dropped_depth - 1)
            return
        if self.dropped_depth or name not in self.open_tags:
            return
        while self.open_tags:
            current = self.open_tags.pop()
            self._append(f"</{current}>")
            if current == name:
                break

    def handle_data(self, _data: str) -> None:
        # Text can contain names, descriptions, tokens, and prompt injection.
        return

    def handle_comment(self, _data: str) -> None:
        return

    def handle_decl(self, _decl: str) -> None:
        return


def sanitize_linkedin_html(
    raw_html: object,
    *,
    max_source_chars: int = MAX_HTML_SOURCE_CHARS,
    max_output_chars: int = MAX_SANITIZED_HTML_CHARS,
) -> tuple[str, dict[str, object]]:
    """Return a bounded, text-free HTML skeleton and a safe sanitization report."""

    source = str(raw_html or "")
    report: dict[str, object] = {
        "source_chars": min(len(source), max_source_chars),
        "sanitized_chars": 0,
        "removed_text": True,
        "removed_tags": [],
        "removed_attributes": 0,
        "truncated": False,
        "status": "ok",
    }
    if not source or len(source) > max_source_chars:
        report["status"] = "rejected_source_limit" if source else "empty"
        return "", report
    parser = _SanitizedHTMLParser(max_chars=max_output_chars)
    try:
        parser.feed(source)
        parser.close()
    except Exception:
        report["status"] = "parse_failed"
        return "", report
    while parser.open_tags:
        parser._append(f"</{parser.open_tags.pop()}>")
    sanitized = parser.output
    report.update(
        {
            "sanitized_chars": len(sanitized),
            "removed_tags": sorted(parser.removed_tags),
            "removed_attributes": parser.removed_attributes,
            "attributes_removed": parser.removed_attributes,
            "scripts_removed": int("script" in parser.removed_tags),
            "styles_removed": int("style" in parser.removed_tags),
            "event_attributes_removed": parser.event_attributes_removed,
            "urls_canonicalized": parser.urls_canonicalized,
            "query_strings_removed": parser.query_strings_removed,
            "nodes_serialized": parser.nodes_serialized,
            "truncated": parser.truncated,
        }
    )
    if not sanitized:
        report["status"] = "empty_after_sanitization"
    elif parser.truncated:
        report["status"] = "bounded"
    return sanitized, report


def _enrich_after_scrollables(before: dict, after: dict) -> dict:
    before_scrollables = {
        item.get("container_index"): item
        for item in before.get("scrollables", [])
        if isinstance(item, dict)
    }
    for item in after.get("scrollables", []):
        if not isinstance(item, dict):
            continue
        before_item = before_scrollables.get(item.get("container_index"))
        if not isinstance(before_item, dict):
            continue
        before_top = int(before_item.get("scrollTop", 0) or 0)
        after_top = int(item.get("scrollTop", 0) or 0)
        before_signal = int(before_item.get("job_signal_count", 0) or 0)
        after_signal = int(item.get("job_signal_count", 0) or 0)
        item["scrollTopBefore"] = before_top
        item["scrollTopAfter"] = after_top
        item["scroll_delta"] = max(0, after_top - before_top)
        item["job_signal_count_before"] = before_signal
        item["job_signal_count_after"] = after_signal
    return after


def _bounded_int(value: object, *, maximum: int) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, min(number, maximum))


def _sanitize_descriptor(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    sanitized: dict[str, object] = {}
    numeric_keys = {
        "container_index",
        "scrollHeight",
        "clientHeight",
        "scrollTop",
        "anchor_count",
        "jobs_view_count",
        "urn_count",
        "data_job_id_count",
        "data_occludable_job_id_count",
        "job_signal_count",
        "scrollTopBefore",
        "scrollTopAfter",
        "scroll_delta",
        "job_signal_count_before",
        "job_signal_count_after",
    }
    for key in numeric_keys:
        if key in value:
            sanitized[key] = _bounded_int(value[key], maximum=2_000_000)
    tag = str(value.get("tag") or "").casefold()
    if _HTML_SAFE_TAG.fullmatch(tag):
        sanitized["tag"] = tag
    if isinstance(value.get("contains_main"), bool):
        sanitized["contains_main"] = value["contains_main"]
    attributes = value.get("attributes")
    if isinstance(attributes, dict):
        sanitized["attributes"] = {
            key: str(attributes.get(key) or "")[:120]
            for key in _ALLOWED_DATA_ATTRIBUTES
            if key in attributes
            and _HTML_SAFE_VALUE.fullmatch(str(attributes.get(key) or ""))
        }
    if isinstance(value.get("class_tokens"), list):
        sanitized["class_tokens"] = [
            str(item)[:80]
            for item in value.get("class_tokens", [])[:12]
            if _HTML_SAFE_TOKEN.fullmatch(str(item))
        ]
    return sanitized


def _sanitize_structure(value: object) -> dict:
    if not isinstance(value, dict):
        return {"schema_version": "1.0", "capture_error": "invalid_structure"}
    signal_counts = value.get("signal_counts")
    viewport = value.get("viewport")
    safe_viewport = {
        key: _bounded_int(viewport.get(key), maximum=10_000)
        for key in ("width", "height")
    } if isinstance(viewport, dict) else {}
    safe_signal_counts = {
        key: _bounded_int(signal_counts.get(key), maximum=10_000)
        for key in (
            "anchors",
            "jobs_view",
            "urn",
            "data_job_id",
            "data_occludable_job_id",
        )
    } if isinstance(signal_counts, dict) else {}
    return {
        "schema_version": "1.0",
        "node_count": _bounded_int(value.get("node_count"), maximum=10_000),
        "frame_count": _bounded_int(value.get("frame_count"), maximum=10_000),
        "body_scrollHeight": _bounded_int(
            value.get("body_scrollHeight"), maximum=2_000_000
        ),
        "viewport": safe_viewport,
        "signal_counts": safe_signal_counts,
        "scrollables": [
            _sanitize_descriptor(item)
            for item in (value.get("scrollables") or [])
            if isinstance(item, dict)
        ][:10],
        "structural": [
            _sanitize_descriptor(item)
            for item in (value.get("structural") or [])
            if isinstance(item, dict)
        ][:80],
    }


def _sanitize_card_focus_snapshot(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {"schema_version": "1.0", "found": False, "capture_error": "invalid"}
    safe: dict[str, object] = {
        "schema_version": "1.0",
        "found": bool(value.get("found")),
        "job_id": _safe_job_id(value.get("job_id")),
        "selector_kind": _safe_reason_label(value.get("selector_kind")),
    }
    for key in (
        "tag",
        "has_href",
        "has_data_job_id",
        "has_data_occludable_job_id",
        "title_present",
    ):
        if key in value:
            safe[key] = bool(value[key]) if key.startswith("has_") or key == "title_present" else str(value[key])[:32]
    if isinstance(value.get("class_tokens"), list):
        safe["class_tokens"] = [
            str(token)[:80]
            for token in value["class_tokens"][:12]
            if _HTML_SAFE_TOKEN.fullmatch(str(token))
        ]
    if isinstance(value.get("date_texts"), list):
        safe["date_texts"] = [
            re.sub(r"[^0-9A-Za-z_ .:/()\-가-힣一-龯áéíóúÁÉÍÓÚüÜñÑ]", "", str(text or ""))[:80]
            for text in value["date_texts"][:5]
        ]
    if isinstance(value.get("bbox"), dict):
        safe["bbox"] = {
            key: max(-100000, min(100000, int(value["bbox"].get(key, 0) or 0)))
            for key in ("x", "y", "width", "height")
        }
    ancestors = []
    for raw in value.get("ancestor_chain", []) if isinstance(value.get("ancestor_chain"), list) else []:
        if not isinstance(raw, dict):
            continue
        ancestors.append(
            {
                "depth": _bounded_int(raw.get("depth"), maximum=8),
                "tag": str(raw.get("tag") or "")[:32],
                "class_tokens": [
                    str(token)[:80]
                    for token in (raw.get("class_tokens") or [])[:12]
                    if _HTML_SAFE_TOKEN.fullmatch(str(token))
                ],
                "has_href": bool(raw.get("has_href")),
                "has_data_job_id": bool(raw.get("has_data_job_id")),
                "has_data_occludable_job_id": bool(raw.get("has_data_occludable_job_id")),
                "bbox": {
                    key: max(-100000, min(100000, int((raw.get("bbox") or {}).get(key, 0) or 0)))
                    for key in ("x", "y", "width", "height")
                },
                "date_texts": [
                    re.sub(r"[^0-9A-Za-z_ .:/()\-가-힣一-龯áéíóúÁÉÍÓÚüÜñÑ]", "", str(text or ""))[:80]
                    for text in (raw.get("date_texts") or [])[:5]
                ],
                "child_count": _bounded_int(raw.get("child_count"), maximum=500),
            }
        )
    safe["ancestor_chain"] = ancestors[:7]
    return safe


def _safe_bbox(value: object) -> dict[str, int]:
    raw = value if isinstance(value, dict) else {}
    return {
        key: max(-100000, min(100000, int(raw.get(key, 0) or 0)))
        for key in ("x", "y", "width", "height")
    }


def _safe_date_text(value: object) -> str:
    return re.sub(
        r"[^0-9A-Za-z_ .:/()\-가-힣一-龯áéíóúÁÉÍÓÚüÜñÑ]",
        "",
        str(value or ""),
    )[:80]


def _sanitize_active_detail_date_snapshot(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {"schema_version": "1.0", "found": False, "capture_error": "invalid"}
    candidates = []
    for raw in value.get("date_candidates", []) if isinstance(value.get("date_candidates"), list) else []:
        if not isinstance(raw, dict):
            continue
        candidates.append(
            {
                "date_text": _safe_date_text(raw.get("date_text")),
                "tag": str(raw.get("tag") or "")[:32],
                "class_tokens": [
                    str(token)[:80]
                    for token in (raw.get("class_tokens") or [])[:12]
                    if _HTML_SAFE_TOKEN.fullmatch(str(token))
                ],
                "bbox": _safe_bbox(raw.get("bbox")),
                "distance_from_title": _bounded_int(
                    raw.get("distance_from_title"),
                    maximum=100_000,
                ),
            }
        )
    return {
        "schema_version": "1.0",
        "found": bool(value.get("found")),
        "title_present": bool(value.get("title_present")),
        "title_bbox": _safe_bbox(value.get("title_bbox")),
        "detail_bbox": _safe_bbox(value.get("detail_bbox")),
        "date_candidates": candidates[:12],
    }


def capture_rejected_job_card_visual_debug(
    page,
    output_dir: Path,
    *,
    job_id: object,
    reason: object,
) -> list[str]:
    """Capture local-only visual evidence for one rejected search card.

    This is diagnostics-only and intentionally removable: it writes a focused
    screenshot plus a bounded structural JSON, with no HTML or full text.
    """
    safe_job_id = _safe_job_id(job_id)
    if not safe_job_id:
        return []
    safe_reason = _safe_reason_label(reason)
    prefix = f"rejected-card-{safe_job_id}-{safe_reason}"
    created: list[str] = []
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        raw = page.evaluate(_REJECTED_CARD_FOCUS_SCRIPT, safe_job_id)
        snapshot = _sanitize_card_focus_snapshot(raw)
        json_path = output_dir / f"{prefix}.json"
        _safe_write_json(json_path, snapshot)
        created.append(json_path.name)
        if not snapshot.get("found"):
            return created
        selectors = [
            f'[data-job-id$="{safe_job_id}"]',
            f'[data-occludable-job-id$="{safe_job_id}"]',
            f'a[href*="/jobs/view/{safe_job_id}"]',
            f'a[href*="-{safe_job_id}"]',
        ]
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                if not locator.count():
                    continue
                locator.screenshot(path=str(output_dir / f"{prefix}.png"))
                created.append(f"{prefix}.png")
                break
            except Exception:
                continue
    except Exception:
        return created
    return created


def capture_active_detail_date_visual_debug(
    page,
    output_dir: Path,
    *,
    job_id: object,
    reason: object,
) -> list[str]:
    """Capture local-only evidence for the active detail date selection."""
    safe_job_id = _safe_job_id(job_id)
    if not safe_job_id:
        return []
    safe_reason = _safe_reason_label(reason)
    prefix = f"active-detail-date-{safe_job_id}-{safe_reason}"
    created: list[str] = []
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        raw = page.evaluate(_ACTIVE_DETAIL_DATE_FOCUS_SCRIPT)
        snapshot = _sanitize_active_detail_date_snapshot(raw)
        json_path = output_dir / f"{prefix}.json"
        _safe_write_json(json_path, snapshot)
        created.append(json_path.name)
        try:
            locator = page.locator("main").first
            if locator.count():
                locator.screenshot(path=str(output_dir / f"{prefix}.png"))
                created.append(f"{prefix}.png")
            else:
                page.screenshot(path=str(output_dir / f"{prefix}.png"), full_page=True)
                created.append(f"{prefix}.png")
        except Exception:
            return created
    except Exception:
        return created
    return created


_COMPANY_RECRUITER_LOCATION_SCRIPT = r"""
() => {
  const safeClasses = (node) => Array.from(node && node.classList || [])
    .slice(0, 12)
    .map((token) => String(token).slice(0, 80));
  const bbox = (node) => {
    const rect = node && node.getBoundingClientRect ? node.getBoundingClientRect() : null;
    return rect ? {
      x: Math.round(rect.left || 0),
      y: Math.round(rect.top || 0),
      width: Math.round(rect.width || 0),
      height: Math.round(rect.height || 0),
    } : {x: 0, y: 0, width: 0, height: 0};
  };
  const descriptor = (selector) => {
    const node = document.querySelector(selector);
    return node ? {
      found: true,
      tag: String(node.tagName || '').toLowerCase().slice(0, 32),
      class_tokens: safeClasses(node),
      bbox: bbox(node),
      anchor_count: node.querySelectorAll ? node.querySelectorAll('a').length : 0,
    } : {found: false};
  };
  const recruiterNodes = Array.from(document.querySelectorAll('[class*="hiring"], [class*="recruiter"], [class*="hirer"], [class*="poster"]')).slice(0, 12);
  const locationNodes = Array.from(document.querySelectorAll('[class*="top-card"], [class*="primary-description"], [class*="tertiary-description"], [class*="location"]')).slice(0, 24);
  return {
    schema_version: '1.0',
    top_card: descriptor('main, .jobs-search__job-details, .job-details-jobs-unified-top-card, .jobs-unified-top-card'),
    company: descriptor('.job-details-jobs-unified-top-card__company-name, .jobs-unified-top-card__company-name, a[href*="/company/"]'),
    recruiter_candidate_count: recruiterNodes.length,
    recruiter_candidates: recruiterNodes.map((node) => ({
      tag: String(node.tagName || '').toLowerCase().slice(0, 32),
      class_tokens: safeClasses(node),
      bbox: bbox(node),
      anchor_count: node.querySelectorAll ? node.querySelectorAll('a').length : 0,
    })),
    location_candidate_count: locationNodes.length,
    location_candidates: locationNodes.map((node) => ({
      tag: String(node.tagName || '').toLowerCase().slice(0, 32),
      class_tokens: safeClasses(node),
      bbox: bbox(node),
      anchor_count: node.querySelectorAll ? node.querySelectorAll('a').length : 0,
    })),
  };
}
"""


def capture_company_recruiter_location_visual_debug(
    page,
    output_dir: Path,
    *,
    job_id: object,
    reason: object,
) -> list[str]:
    """Capture local-only screenshots for company/location/recruiter evidence.

    This is diagnostics-only and intentionally removable. Screenshots can contain
    personal data, so artifacts stay local and are referenced only by relative
    manifest paths.
    """
    safe_job_id = _safe_job_id(job_id)
    if not safe_job_id:
        return []
    safe_reason = _safe_reason_label(reason)
    prefix = f"company-recruiter-location-{safe_job_id}-{safe_reason}"
    created: list[str] = []
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot = page.evaluate(_COMPANY_RECRUITER_LOCATION_SCRIPT)
        except Exception:
            snapshot = {"schema_version": "1.0", "evaluation_failed": True}
        json_path = output_dir / f"{prefix}.json"
        _safe_write_json(json_path, snapshot if isinstance(snapshot, dict) else {"schema_version": "1.0"})
        created.append(json_path.name)
        selectors = (
            ("top-card", "main, .jobs-search__job-details, .job-details-jobs-unified-top-card, .jobs-unified-top-card"),
            ("company", '.job-details-jobs-unified-top-card__company-name, .jobs-unified-top-card__company-name, a[href*="/company/"]'),
            ("recruiter", '[class*="hiring"], [class*="recruiter"], [class*="hirer"], [class*="poster"]'),
        )
        captured_any = False
        for label, selector in selectors:
            try:
                locator = page.locator(selector).first
                if not locator.count():
                    continue
                locator.screenshot(path=str(output_dir / f"{prefix}-{label}.png"))
                created.append(f"{prefix}-{label}.png")
                captured_any = True
            except Exception:
                continue
        try:
            page.screenshot(path=str(output_dir / f"{prefix}-page.png"), full_page=False)
            created.append(f"{prefix}-page.png")
        except Exception:
            pass
    except Exception:
        return created
    return created


_COMPANY_ABOUT_LOCATION_SCRIPT = r"""
() => {
  const text = String(document.body && (document.body.innerText || document.body.textContent) || '').replace(/\s+/g, ' ').trim();
  const korea = /south korea|corea del sur|republic of korea|seoul|대한민국|서울/i.test(text);
  const japan = /japan|japón|tokyo|日本|東京/i.test(text);
  const unitedStates = /united states|estados unidos|california|new york|menlo park|san francisco|usa|u\.s\./i.test(text);
  const headquartersMatch = text.match(/(?:headquarters|sede(?: central)?|ubicaci[oó]n(?: principal)?|본사)[:\s]+(.{0,180})/i);
  const locationMatch = text.match(/(?:menlo park|california|san francisco|new york|seoul|south korea|corea del sur|tokyo|japan|대한민국|서울|日本|東京).{0,120}/i);
  return {
    schema_version: '1.0',
    page_path: window.location ? String(window.location.pathname || '').slice(0, 160) : '',
    has_korea: korea,
    has_japan: japan,
    has_united_states: unitedStates,
    headquarters_text: headquartersMatch && headquartersMatch[1] ? headquartersMatch[1].slice(0, 180) : '',
    location_text: locationMatch && locationMatch[0] ? locationMatch[0].slice(0, 180) : '',
    text_length: text.length,
  };
}
"""


def _sanitize_company_about_location_snapshot(value: object) -> dict[str, object]:
    raw = value if isinstance(value, dict) else {}
    return {
        "schema_version": "1.0",
        "page_path": re.sub(r"[^0-9A-Za-z_./-]", "", str(raw.get("page_path", "") or ""))[:160],
        "has_korea": bool(raw.get("has_korea")),
        "has_japan": bool(raw.get("has_japan")),
        "has_united_states": bool(raw.get("has_united_states")),
        "headquarters_text": re.sub(r"\s+", " ", str(raw.get("headquarters_text", "") or "")).strip()[:180],
        "location_text": re.sub(r"\s+", " ", str(raw.get("location_text", "") or "")).strip()[:180],
        "text_length": _bounded_int(raw.get("text_length"), maximum=1_000_000),
    }


def capture_company_about_location_visual_debug(
    page,
    output_dir: Path,
    *,
    job_id: object,
    reason: object,
) -> list[str]:
    """Capture local-only evidence from a LinkedIn company About page."""
    safe_job_id = _safe_job_id(job_id)
    if not safe_job_id:
        return []
    safe_reason = _safe_reason_label(reason)
    prefix = f"company-about-location-{safe_job_id}-{safe_reason}"
    created: list[str] = []
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot = _sanitize_company_about_location_snapshot(
                page.evaluate(_COMPANY_ABOUT_LOCATION_SCRIPT)
            )
        except Exception:
            snapshot = {"schema_version": "1.0", "evaluation_failed": True}
        json_path = output_dir / f"{prefix}.json"
        _safe_write_json(json_path, snapshot)
        created.append(json_path.name)
        try:
            page.screenshot(path=str(output_dir / f"{prefix}.png"), full_page=False)
            created.append(f"{prefix}.png")
        except Exception:
            pass
    except Exception:
        return created
    return created


_RECRUITER_LOCATION_SCRIPT = r"""
() => {
  const text = String(document.body && (document.body.innerText || document.body.textContent) || '').replace(/\s+/g, ' ').trim();
  const korea = /south korea|corea del sur|republic of korea|seoul|대한민국|서울/i.test(text);
  const japan = /japan|japón|tokyo|日本|東京/i.test(text);
  const unitedStates = /united states|estados unidos|california|new york|menlo park|san francisco|usa|u\.s\./i.test(text);
  const locationMatch = text.match(/(?:menlo park|california|san francisco|new york|seoul|south korea|corea del sur|tokyo|japan|대한민국|서울|日本|東京).{0,120}/i);
  return {
    schema_version: '1.0',
    page_path: window.location ? String(window.location.pathname || '').slice(0, 160) : '',
    has_korea: korea,
    has_japan: japan,
    has_united_states: unitedStates,
    location_text: locationMatch && locationMatch[0] ? locationMatch[0].slice(0, 180) : '',
    text_length: text.length,
  };
}
"""


def _sanitize_recruiter_location_snapshot(value: object) -> dict[str, object]:
    raw = value if isinstance(value, dict) else {}
    return {
        "schema_version": "1.0",
        "page_path": re.sub(r"[^0-9A-Za-z_./-]", "", str(raw.get("page_path", "") or ""))[:160],
        "has_korea": bool(raw.get("has_korea")),
        "has_japan": bool(raw.get("has_japan")),
        "has_united_states": bool(raw.get("has_united_states")),
        "location_text": re.sub(r"\s+", " ", str(raw.get("location_text", "") or "")).strip()[:180],
        "text_length": _bounded_int(raw.get("text_length"), maximum=1_000_000),
    }


def capture_recruiter_location_visual_debug(
    page,
    output_dir: Path,
    *,
    job_id: object,
    reason: object,
) -> list[str]:
    """Capture local-only evidence from a recruiter profile page."""
    safe_job_id = _safe_job_id(job_id)
    if not safe_job_id:
        return []
    safe_reason = _safe_reason_label(reason)
    prefix = f"recruiter-location-{safe_job_id}-{safe_reason}"
    created: list[str] = []
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot = _sanitize_recruiter_location_snapshot(
                page.evaluate(_RECRUITER_LOCATION_SCRIPT)
            )
        except Exception:
            snapshot = {"schema_version": "1.0", "evaluation_failed": True}
        json_path = output_dir / f"{prefix}.json"
        _safe_write_json(json_path, snapshot)
        created.append(json_path.name)
        try:
            page.screenshot(path=str(output_dir / f"{prefix}.png"), full_page=False)
            created.append(f"{prefix}.png")
        except Exception:
            pass
    except Exception:
        return created
    return created


def _append_manifest_artifacts(base_dir: Path, filenames: list[str]) -> None:
    if not filenames:
        return
    manifest_path = base_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    for filename in filenames:
        if re.fullmatch(
            r"(?:rejected-card|active-detail-date|company-recruiter-location|company-about-location|recruiter-location)-[0-9]+-[0-9A-Za-z_-]+(?:-[0-9A-Za-z_-]+)?\.(?:json|png)",
            filename,
        ):
            artifacts[filename] = filename
    manifest["artifacts"] = artifacts
    try:
        _safe_write_json(manifest_path, manifest)
    except Exception:
        return


@dataclass
class _VisualSearchRun:
    collector: "LinkedInVisualDiagnosticsCollector"
    query: str
    trace_started: bool = False
    before_structure: dict = field(default_factory=dict)
    before_main_capture: bool = False
    after_main_capture: bool = False
    panel_html_captures: int = 0
    activation_html_captures: int = 0
    html_reports: list[dict[str, object]] = field(default_factory=list)

    @property
    def base_dir(self) -> Path:
        return self.collector.base_dir

    def start_trace(self, page) -> None:
        try:
            tracing = page.context.tracing
            tracing.start(screenshots=True, snapshots=True, sources=False)
            if hasattr(tracing, "start_chunk"):
                tracing.start_chunk(title="linkedin-search-visual-diagnostics")
            self.trace_started = True
        except Exception:
            self.trace_started = False

    def capture_before(self, page) -> None:
        self._capture_full(page, "before-full.png")
        self.before_main_capture = self._capture_main(page, "before-main.png")
        self.before_structure = self._capture_structure(page)
        _safe_write_json(self.base_dir / "structure-before.json", self.before_structure)
        self._capture_html(page, "panel-before.html", kind="panel")

    def capture_after(self, page) -> None:
        self._capture_full(page, "after-full.png")
        self.after_main_capture = self._capture_main(page, "after-main.png")
        after_structure = self._capture_structure(page)
        after_structure = _enrich_after_scrollables(self.before_structure, after_structure)
        _safe_write_json(self.base_dir / "structure-after.json", after_structure)
        self._capture_html(page, "panel-after.html", kind="panel")

    def capture_activation(self, page, outcome: str) -> None:
        """Capture at most three sanitized activation states for local debugging."""

        if self.activation_html_captures >= MAX_ACTIVATION_HTML_CAPTURES:
            return
        safe_outcome = (
            outcome
            if outcome
            in {
                "row_activation_success",
                "row_activation_no_change",
                "row_activation_no_job_id",
                "row_activation_duplicate",
            }
            else "unknown"
        )
        index = self.activation_html_captures + 1
        self._capture_html(
            page,
            f"activation-{index:02d}-{safe_outcome}.html",
            kind="activation",
        )

    def stop_trace(self, page) -> None:
        if not self.trace_started:
            return
        try:
            tracing = page.context.tracing
            trace_path = str(self.base_dir / "trace.zip")
            if hasattr(tracing, "stop_chunk"):
                tracing.stop_chunk(path=trace_path)
                if hasattr(tracing, "stop"):
                    tracing.stop()
            else:
                tracing.stop(path=trace_path)
        except Exception:
            return

    def finalize(self) -> None:
        def artifact_ref(filename: str) -> str | None:
            return filename if (self.base_dir / filename).is_file() else None

        manifest = {
            "schema_version": "1.0",
            "query": _sanitize_query(self.query),
            "visual_diagnostics_enabled": True,
            "local_only": True,
            "main_capture": {
                "before": self.before_main_capture,
                "after": self.after_main_capture,
            },
            "artifacts": {
                "before_full": artifact_ref("before-full.png"),
                "before_main": (
                    artifact_ref("before-main.png")
                    if self.before_main_capture
                    else None
                ),
                "structure_before": artifact_ref("structure-before.json"),
                "after_full": artifact_ref("after-full.png"),
                "after_main": (
                    artifact_ref("after-main.png")
                    if self.after_main_capture
                    else None
                ),
                "structure_after": artifact_ref("structure-after.json"),
                "panel_before": artifact_ref("panel-before.html"),
                "panel_after": artifact_ref("panel-after.html"),
                "activation_html": [
                    path.name
                    for path in sorted(self.base_dir.glob("activation-*.html"))
                    if path.is_file()
                ],
            },
            "html": {
                "local_only": True,
                "text_removed": True,
                "panel_capture_cap": MAX_PANEL_HTML_CAPTURES,
                "activation_capture_cap": MAX_ACTIVATION_HTML_CAPTURES,
                "sanitization_reports": list(self.html_reports),
            },
            "trace": {
                "path": "trace.zip",
                "sensitive_local_artifact": True,
                "screenshots": True,
                "snapshots": True,
                "sources": False,
                "share_policy": "never_attach_or_upload_automatically",
            },
        }
        _safe_write_json(self.base_dir / "manifest.json", manifest)
        self.collector.record(
            LinkedInVisualDiagnosticArtifact(
                query=_sanitize_query(self.query),
                manifest_path=_relative(
                    self.base_dir / "manifest.json",
                    base=self.collector.audit_dir,
                ),
                sensitive_local_artifact=True,
            )
        )

    def _capture_full(self, page, filename: str) -> None:
        try:
            page.screenshot(path=str(self.base_dir / filename), full_page=True)
        except Exception:
            return

    def _capture_main(self, page, filename: str) -> bool:
        try:
            locator = page.locator("main").first
            if not locator.count():
                return False
            locator.screenshot(path=str(self.base_dir / filename))
            return True
        except Exception:
            return False

    def _capture_structure(self, page) -> dict:
        try:
            value = page.evaluate(_STRUCTURE_SCRIPT)
        except Exception:
            return {"schema_version": "1.0", "capture_error": "evaluate_failed"}
        return _sanitize_structure(value)

    def _capture_html(self, page, filename: str, *, kind: str) -> bool:
        if kind == "panel" and self.panel_html_captures >= MAX_PANEL_HTML_CAPTURES:
            return False
        if kind == "activation" and (
            self.activation_html_captures >= MAX_ACTIVATION_HTML_CAPTURES
        ):
            return False
        if kind == "panel":
            self.panel_html_captures += 1
        else:
            self.activation_html_captures += 1
        try:
            raw_html = page.content()
        except Exception:
            self.html_reports.append(
                {"kind": kind, "status": "content_failed", "path": filename}
            )
            return False
        sanitized, report = sanitize_linkedin_html(raw_html)
        report = {"kind": kind, "path": filename, **report}
        self.html_reports.append(report)
        if not sanitized:
            return False
        try:
            (self.base_dir / filename).write_text(sanitized, encoding="utf-8")
        except Exception:
            return False
        return True


_HTML_ROW_STRUCTURE_SCRIPT = r"""
() => {
  const viewportHeight = Math.max(0, Math.trunc(window.innerHeight || 0));
  const viewportWidth = Math.max(0, Math.trunc(window.innerWidth || 0));
  const nodes = Array.from(document.querySelectorAll("body *")).slice(0, 2500);
  const path = (node) => {
    const parts = [];
    let current = node;
    while (current && current !== document.body && parts.length < 8) {
      const parent = current.parentElement;
      if (!parent) break;
      const sameTag = Array.from(parent.children).filter((child) => child.tagName === current.tagName);
      parts.unshift(`${String(current.tagName || '').toLowerCase()}:${Math.max(0, sameTag.indexOf(current))}`);
      current = parent;
    }
    return parts;
  };
  const attrs = (node) => {
    const result = {};
    for (const name of ["data-job-id", "data-occludable-job-id", "data-entity-urn"]) {
      if (node.hasAttribute && node.hasAttribute(name)) result[name] = String(node.getAttribute(name) || '').slice(0, 120);
    }
    return result;
  };
  const canonicalHref = (node) => {
    const link = node.matches && node.matches("a") ? node : (node.querySelector && node.querySelector("a[href]"));
    if (!link) return "";
    try {
      const parsed = new URL(String(link.getAttribute("href") || ""), window.location.origin);
      parsed.search = "";
      parsed.hash = "";
      return `${parsed.pathname}`.slice(0, 240);
    } catch (_) { return ""; }
  };
  const visible = (rect) => rect.width > 0 && rect.height > 0 && rect.bottom >= 0 && rect.right >= 0 && rect.top <= viewportHeight && rect.left <= viewportWidth;
  const rows = nodes.map((node, index) => {
    const rect = node.getBoundingClientRect();
    const role = String(node.getAttribute && node.getAttribute("role") || "").slice(0, 40);
    const tabindex = node.hasAttribute && node.hasAttribute("tabindex") ? String(node.getAttribute("tabindex") || "").slice(0, 20) : "";
    const classTokens = Array.from(node.classList || []).slice(0, 20).map((token) => String(token).slice(0, 80));
    const hasRowSignal = Boolean(node.matches && node.matches("[data-job-id], [data-occludable-job-id], [role='listitem'], [role='option'], li")) || Boolean(node.querySelector && node.querySelector("a[href*='/jobs/view/'], [data-job-id], [data-occludable-job-id]"));
    const isVisible = visible(rect);
    return {
      index,
      tag: String(node.tagName || "").toLowerCase(),
      class_tokens: classTokens,
      role,
      tabindex,
      aria_selected: node.getAttribute && node.getAttribute("aria-selected") || "",
      aria_current: node.getAttribute && node.getAttribute("aria-current") || "",
      aria_expanded: node.getAttribute && node.getAttribute("aria-expanded") || "",
      allowlisted_attrs: attrs(node),
      href: canonicalHref(node),
      bounds: {x: Math.round(rect.left), y: Math.round(rect.top), width: Math.round(rect.width), height: Math.round(rect.height)},
      vertical_band: Math.round(rect.top / 10) * 10,
      structural_path: path(node),
      visible: isVisible,
      row_candidate: Boolean(isVisible && hasRowSignal),
    };
  }).filter((item) => item.row_candidate).slice(0, 500);
  return {row_count: rows.length, viewport: {width: viewportWidth, height: viewportHeight}, rows};
}
"""


def _safe_row_snapshot(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {"schema_version": "1.0", "row_count": 0, "rows": [], "capture_error": "invalid"}
    rows: list[dict[str, object]] = []
    for raw in value.get("rows", []) if isinstance(value.get("rows"), list) else []:
        if not isinstance(raw, dict):
            continue
        item: dict[str, object] = {}
        for key in ("index", "tag", "role", "tabindex", "aria_selected", "aria_current", "aria_expanded", "href", "vertical_band", "visible", "row_candidate"):
            if key in raw:
                item[key] = raw[key] if key not in {"tag", "role", "tabindex", "aria_selected", "aria_current", "aria_expanded", "href"} else str(raw[key] or "")[:240]
        if isinstance(raw.get("class_tokens"), list):
            item["class_tokens"] = [str(token)[:80] for token in raw["class_tokens"][:20]]
        if isinstance(raw.get("allowlisted_attrs"), dict):
            item["allowlisted_attrs"] = {
                key: str(raw["allowlisted_attrs"].get(key) or "")[:120]
                for key in _ALLOWED_DATA_ATTRIBUTES
                if key in raw["allowlisted_attrs"]
            }
        if isinstance(raw.get("bounds"), dict):
            item["bounds"] = {
                key: max(-100000, min(100000, int(raw["bounds"].get(key, 0) or 0)))
                for key in ("x", "y", "width", "height")
            }
        if isinstance(raw.get("structural_path"), list):
            item["structural_path"] = [str(part)[:80] for part in raw["structural_path"][:8]]
        rows.append(item)
    return {
        "schema_version": "1.0",
        "row_count": max(
            len(rows),
            min(10_000, max(0, int(value.get("row_count", 0) or 0))),
        ),
        "viewport": value.get("viewport") if isinstance(value.get("viewport"), dict) else {},
        "rows": rows[:500],
    }


class _HTMLDiagnosticRun:
    """Bounded, sanitized HTML evidence for one search query."""

    def __init__(self, collector: "LinkedInHTMLDiagnosticsCollector", query: str) -> None:
        self.collector = collector
        self.query = _sanitize_query(query)
        self.panel_count = 0
        self.activation_count = 0
        self.scroll_index = 0
        self.before_rows: dict[str, object] | None = None
        self.last_rows: dict[str, object] | None = None
        self.reports: list[dict[str, object]] = []
        self.rejected_card_captures = 0
        self.active_detail_date_captures = 0

    @property
    def base_dir(self) -> Path:
        return self.collector.base_dir

    def _content(self, page, filename: str) -> bool:
        try:
            raw_html = page.content()
        except Exception:
            self.reports.append({"path": filename, "status": "content_failed"})
            return False
        try:
            sanitized, report = sanitize_linkedin_html(raw_html)
        except Exception:
            self.reports.append({"path": filename, "status": "sanitize_failed"})
            return False
        report = {"path": filename, "attributes_removed": report.get("attributes_removed", report.get("removed_attributes", 0)), **{key: report.get(key, 0) for key in (
            "scripts_removed", "styles_removed", "event_attributes_removed",
            "urls_canonicalized", "query_strings_removed", "nodes_serialized",
        )}, "status": report.get("status", "unknown")}
        self.reports.append(report)
        if not sanitized or report["status"] not in {"ok", "bounded"}:
            return False
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            (self.base_dir / filename).write_text(sanitized, encoding="utf-8")
            return True
        except Exception:
            return False

    def _rows(self, page) -> dict[str, object]:
        try:
            return _safe_row_snapshot(page.evaluate(_HTML_ROW_STRUCTURE_SCRIPT))
        except Exception:
            return {"schema_version": "1.0", "row_count": 0, "rows": [], "capture_error": "evaluate_failed"}

    def _write_rows(self, filename: str, rows: dict[str, object]) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        _safe_write_json(self.base_dir / filename, rows)

    def capture_after_hydration(self, page) -> None:
        self._content(page, "search-after-hydration.html")

    def capture_before_scroll(self, page) -> None:
        if self.before_rows is not None:
            return
        self._content(page, "results-panel-before-scroll.html")
        capture_visible_linkedin_card_structure_debug(page, self.base_dir)
        self.before_rows = self._rows(page)
        self.last_rows = self.before_rows
        self._write_rows("row-candidates-before.json", self.before_rows)
        _safe_write_json(self.base_dir / "structure-before.json", self.before_rows)

    def capture_after_scroll(self, page, scroll_index: int = 0) -> None:
        if self.before_rows is None:
            self.capture_before_scroll(page)
        current = self._rows(page)
        if current == self.last_rows:
            return
        if self.panel_count >= MAX_HTML_DIAGNOSTIC_SCROLL_CAPTURES:
            self.last_rows = current
            return
        self.panel_count += 1
        self.scroll_index += 1
        suffix = f"{self.panel_count:02d}"
        self._content(page, f"results-panel-after-scroll-{suffix}.html")
        self._write_rows(f"row-candidates-after-scroll-{suffix}.json", current)
        self.last_rows = current

    def capture_detail(self, page, phase: str) -> None:
        if phase == "before_click":
            self._content(page, "detail-before-click.html")
            return
        self.capture_activation(page, phase)

    def capture_activation(self, page, _outcome: str = "") -> None:
        if self.activation_count >= MAX_HTML_DIAGNOSTIC_ACTIVATIONS - 1:
            return
        self.activation_count += 1
        self._content(page, f"detail-after-click-{self.activation_count:02d}.html")

    def capture_before(self, page) -> None:
        """Compatibility hook; new captures are deliberately phase-specific."""
        return

    def capture_after(self, page) -> None:
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(self.base_dir / "screenshot-after.png"), full_page=True)
        except Exception:
            return

    def capture_rejected_candidate_card(self, page, *, job_id: object, reason: object) -> None:
        if self.rejected_card_captures >= MAX_REJECTED_CARD_VISUAL_CAPTURES:
            return
        created = capture_rejected_job_card_visual_debug(
            page,
            self.base_dir,
            job_id=job_id,
            reason=reason,
        )
        if created:
            self.rejected_card_captures += 1

    def capture_active_detail_date(self, page, *, job_id: object, reason: object) -> None:
        if self.active_detail_date_captures >= MAX_ACTIVE_DETAIL_DATE_VISUAL_CAPTURES:
            return
        created = capture_active_detail_date_visual_debug(
            page,
            self.base_dir,
            job_id=job_id,
            reason=reason,
        )
        if created:
            self.active_detail_date_captures += 1

    def stop_trace(self, _page) -> None:
        """Keep the shared pipeline lifecycle compatible without tracing HTML mode."""
        return

    def finalize(self) -> None:
        if self.before_rows is not None:
            _safe_write_json(self.base_dir / "structure-after.json", self.last_rows or self.before_rows)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        allowed_names = {
            "search-after-hydration.html", "results-panel-before-scroll.html",
            "detail-before-click.html", "structure-before.json", "structure-after.json",
            "screenshot-after.png",
            *(f"results-panel-after-scroll-{index:02d}.html" for index in range(1, 4)),
            *(f"row-candidates-after-scroll-{index:02d}.json" for index in range(1, 4)),
            *(f"detail-after-click-{index:02d}.html" for index in range(1, 3)),
            "row-candidates-before.json", CARD_STRUCTURE_DEBUG_FILENAME,
        }
        artifacts = {
            name: name for name in sorted(allowed_names)
            if (self.base_dir / name).is_file()
        }
        artifacts.update(
            {
                path.name: path.name
                for path in sorted(
                    list(self.base_dir.glob("rejected-card-*"))
                    + list(self.base_dir.glob("active-detail-date-*"))
                    + list(self.base_dir.glob("company-recruiter-location-*"))
                    + list(self.base_dir.glob("company-about-location-*"))
                    + list(self.base_dir.glob("recruiter-location-*"))
                )
                if path.is_file()
            }
        )
        counts = {key: 0 for key in (
            "scripts_removed", "styles_removed", "event_attributes_removed", "attributes_removed",
            "urls_canonicalized", "query_strings_removed", "nodes_serialized",
        )}
        for report in self.reports:
            for key in counts:
                counts[key] += int(report.get(key, 0) or 0)
        manifest = {
            "schema_version": "1.0",
            "query": self.query,
            "local_only": True,
            "sensitive_local_artifacts": True,
            "artifacts": artifacts,
            "sanitization_report": counts,
            "sanitization_attempts": self.reports,
        }
        _safe_write_json(self.base_dir / "manifest.json", manifest)
        self.collector.record(LinkedInVisualDiagnosticArtifact(
            query=self.query,
            manifest_path=f"data/private/linkedin/diagnostics/{self.collector.job_uid}/manifest.json",
            sensitive_local_artifact=True,
        ))


class LinkedInHTMLDiagnosticsCollector:
    def __init__(self, output_root: Path | None = None, *, job_uid: str = "local") -> None:
        self.job_uid = re.sub(r"[^A-Za-z0-9_-]", "-", job_uid or "local")[:120] or "local"
        self.base_dir = Path(output_root or Path("data/private/linkedin/diagnostics")) / self.job_uid
        self._events: list[LinkedInVisualDiagnosticArtifact] = []
        self._captured = False
        self._rejected_card_captures = 0
        self._active_detail_date_captures = 0
        self._company_recruiter_location_captures = 0
        self._company_about_location_captures = 0
        self._recruiter_location_captures = 0

    @property
    def events(self) -> list[LinkedInVisualDiagnosticArtifact]:
        return list(self._events)

    def should_capture(self) -> bool:
        return html_diagnostics_enabled() and not self._captured

    def start_run(self, _page, *, query: str):
        if not self.should_capture():
            return None
        self._captured = True
        return _HTMLDiagnosticRun(self, query)

    def record(self, event: LinkedInVisualDiagnosticArtifact) -> None:
        self._events.append(event)

    def capture_rejected_candidate_card(self, page, *, job_id: object, reason: object) -> None:
        if self._rejected_card_captures >= MAX_REJECTED_CARD_VISUAL_CAPTURES:
            return
        created = capture_rejected_job_card_visual_debug(
            page,
            self.base_dir,
            job_id=job_id,
            reason=reason,
        )
        if created:
            self._rejected_card_captures += 1
            _append_manifest_artifacts(self.base_dir, created)

    def capture_active_detail_date(self, page, *, job_id: object, reason: object) -> None:
        if self._active_detail_date_captures >= MAX_ACTIVE_DETAIL_DATE_VISUAL_CAPTURES:
            return
        created = capture_active_detail_date_visual_debug(
            page,
            self.base_dir,
            job_id=job_id,
            reason=reason,
        )
        if created:
            self._active_detail_date_captures += 1
            _append_manifest_artifacts(self.base_dir, created)

    def capture_company_recruiter_location(self, page, *, job_id: object, reason: object) -> None:
        if self._company_recruiter_location_captures >= MAX_COMPANY_RECRUITER_LOCATION_VISUAL_CAPTURES:
            return
        created = capture_company_recruiter_location_visual_debug(
            page,
            self.base_dir,
            job_id=job_id,
            reason=reason,
        )
        if created:
            self._company_recruiter_location_captures += 1
            _append_manifest_artifacts(self.base_dir, created)

    def capture_company_about_location(self, page, *, job_id: object, reason: object) -> None:
        if self._company_about_location_captures >= MAX_COMPANY_ABOUT_LOCATION_VISUAL_CAPTURES:
            return
        created = capture_company_about_location_visual_debug(
            page,
            self.base_dir,
            job_id=job_id,
            reason=reason,
        )
        if created:
            self._company_about_location_captures += 1
            _append_manifest_artifacts(self.base_dir, created)

    def capture_recruiter_location(self, page, *, job_id: object, reason: object) -> None:
        if self._recruiter_location_captures >= MAX_RECRUITER_LOCATION_VISUAL_CAPTURES:
            return
        created = capture_recruiter_location_visual_debug(
            page,
            self.base_dir,
            job_id=job_id,
            reason=reason,
        )
        if created:
            self._recruiter_location_captures += 1
            _append_manifest_artifacts(self.base_dir, created)


class LinkedInVisualDiagnosticsCollector:
    """Collect local-only visual diagnostic artifact references."""

    def __init__(self, audit_dir: Path) -> None:
        self.audit_dir = Path(audit_dir)
        self.base_dir = self.audit_dir / _VISUAL_DIR
        self._events: list[LinkedInVisualDiagnosticArtifact] = []
        self._captured = False
        self._rejected_card_captures = 0
        self._active_detail_date_captures = 0
        self._company_recruiter_location_captures = 0
        self._company_about_location_captures = 0
        self._recruiter_location_captures = 0

    @property
    def events(self) -> list[LinkedInVisualDiagnosticArtifact]:
        return list(self._events)

    def should_capture(self) -> bool:
        return visual_diagnostics_enabled() and not self._captured

    def start_run(self, page, *, query: str) -> _VisualSearchRun | None:
        if not self.should_capture() or len(self._events) >= MAX_VISUAL_DIAGNOSTIC_QUERIES:
            return None
        self._captured = True
        self.base_dir.mkdir(parents=True, exist_ok=True)
        run = _VisualSearchRun(self, query=query)
        run.start_trace(page)
        return run

    def record(self, event: LinkedInVisualDiagnosticArtifact) -> None:
        self._events.append(event)

    def capture_rejected_candidate_card(self, page, *, job_id: object, reason: object) -> None:
        if self._rejected_card_captures >= MAX_REJECTED_CARD_VISUAL_CAPTURES:
            return
        created = capture_rejected_job_card_visual_debug(
            page,
            self.base_dir,
            job_id=job_id,
            reason=reason,
        )
        if created:
            self._rejected_card_captures += 1
            _append_manifest_artifacts(self.base_dir, created)

    def capture_active_detail_date(self, page, *, job_id: object, reason: object) -> None:
        if self._active_detail_date_captures >= MAX_ACTIVE_DETAIL_DATE_VISUAL_CAPTURES:
            return
        created = capture_active_detail_date_visual_debug(
            page,
            self.base_dir,
            job_id=job_id,
            reason=reason,
        )
        if created:
            self._active_detail_date_captures += 1
            _append_manifest_artifacts(self.base_dir, created)

    def capture_company_recruiter_location(self, page, *, job_id: object, reason: object) -> None:
        if self._company_recruiter_location_captures >= MAX_COMPANY_RECRUITER_LOCATION_VISUAL_CAPTURES:
            return
        created = capture_company_recruiter_location_visual_debug(
            page,
            self.base_dir,
            job_id=job_id,
            reason=reason,
        )
        if created:
            self._company_recruiter_location_captures += 1
            _append_manifest_artifacts(self.base_dir, created)

    def capture_company_about_location(self, page, *, job_id: object, reason: object) -> None:
        if self._company_about_location_captures >= MAX_COMPANY_ABOUT_LOCATION_VISUAL_CAPTURES:
            return
        created = capture_company_about_location_visual_debug(
            page,
            self.base_dir,
            job_id=job_id,
            reason=reason,
        )
        if created:
            self._company_about_location_captures += 1
            _append_manifest_artifacts(self.base_dir, created)

    def capture_recruiter_location(self, page, *, job_id: object, reason: object) -> None:
        if self._recruiter_location_captures >= MAX_RECRUITER_LOCATION_VISUAL_CAPTURES:
            return
        created = capture_recruiter_location_visual_debug(
            page,
            self.base_dir,
            job_id=job_id,
            reason=reason,
        )
        if created:
            self._recruiter_location_captures += 1
            _append_manifest_artifacts(self.base_dir, created)


_ACTIVE_VISUAL_DIAGNOSTICS: ContextVar[
    LinkedInVisualDiagnosticsCollector | None
] = ContextVar("linkedin_visual_diagnostics", default=None)


def get_active_visual_diagnostics() -> LinkedInVisualDiagnosticsCollector | None:
    return _ACTIVE_VISUAL_DIAGNOSTICS.get()


@contextmanager
def visual_diagnostics_context(
    audit_dir: Path,
    *,
    job_uid: str = "local",
    diagnostics_root: Path | None = None,
) -> Iterator[LinkedInVisualDiagnosticsCollector | LinkedInHTMLDiagnosticsCollector]:
    if html_diagnostics_enabled():
        collector = LinkedInHTMLDiagnosticsCollector(
            diagnostics_root or audit_dir,
            job_uid=job_uid,
        )
    else:
        collector = LinkedInVisualDiagnosticsCollector(audit_dir)
    token = _ACTIVE_VISUAL_DIAGNOSTICS.set(collector)
    try:
        yield collector
    finally:
        _ACTIVE_VISUAL_DIAGNOSTICS.reset(token)


__all__ = [
    "MAX_ACTIVATION_HTML_CAPTURES",
    "MAX_HTML_DIAGNOSTIC_ACTIVATIONS",
    "HTML_DIAGNOSTICS_ENV",
    "LinkedInHTMLDiagnosticsCollector",
    "MAX_PANEL_HTML_CAPTURES",
    "MAX_VISUAL_DIAGNOSTIC_QUERIES",
    "LinkedInVisualDiagnosticsCollector",
    "MAX_REJECTED_CARD_VISUAL_CAPTURES",
    "capture_rejected_job_card_visual_debug",
    "capture_company_recruiter_location_visual_debug",
    "capture_company_about_location_visual_debug",
    "capture_recruiter_location_visual_debug",
    "get_active_visual_diagnostics",
    "sanitize_linkedin_html",
    "visual_diagnostics_context",
    "visual_diagnostics_enabled",
    "html_diagnostics_enabled",
]
