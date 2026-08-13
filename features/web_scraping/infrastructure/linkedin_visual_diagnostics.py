"""Opt-in local visual diagnostics for LinkedIn search discovery."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Iterator

from features.web_scraping.domain.linkedin_models import (
    LinkedInVisualDiagnosticArtifact,
)


_ENABLED_ENV = "LINKEDIN_SEARCH_VISUAL_DIAGNOSTICS"
_VISUAL_DIR = "visual-diagnostics"
_ALLOWED_DATA_ATTRIBUTES = (
    "data-job-id",
    "data-occludable-job-id",
    "data-entity-urn",
)
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


def visual_diagnostics_enabled() -> bool:
    return str(os.getenv(_ENABLED_ENV, "")).strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
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


def _sanitize_descriptor(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    sanitized: dict[str, object] = {}
    for key in (
        "container_index",
        "tag",
        "class_tokens",
        "contains_main",
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
    ):
        if key in value:
            sanitized[key] = value[key]
    attributes = value.get("attributes")
    if isinstance(attributes, dict):
        sanitized["attributes"] = {
            key: str(attributes.get(key) or "")[:120]
            for key in _ALLOWED_DATA_ATTRIBUTES
            if key in attributes
        }
    if isinstance(value.get("class_tokens"), list):
        sanitized["class_tokens"] = [
            str(item)[:80] for item in value.get("class_tokens", [])[:12]
        ]
    return sanitized


def _sanitize_structure(value: object) -> dict:
    if not isinstance(value, dict):
        return {"schema_version": "1.0", "capture_error": "invalid_structure"}
    signal_counts = value.get("signal_counts")
    viewport = value.get("viewport")
    return {
        "schema_version": str(value.get("schema_version") or "1.0")[:20],
        "node_count": int(value.get("node_count", 0) or 0),
        "frame_count": int(value.get("frame_count", 0) or 0),
        "body_scrollHeight": int(value.get("body_scrollHeight", 0) or 0),
        "viewport": viewport if isinstance(viewport, dict) else {},
        "signal_counts": signal_counts if isinstance(signal_counts, dict) else {},
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


@dataclass
class _VisualSearchRun:
    collector: "LinkedInVisualDiagnosticsCollector"
    query: str
    trace_started: bool = False
    before_structure: dict = field(default_factory=dict)
    before_main_capture: bool = False
    after_main_capture: bool = False

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

    def capture_after(self, page) -> None:
        self._capture_full(page, "after-full.png")
        self.after_main_capture = self._capture_main(page, "after-main.png")
        after_structure = self._capture_structure(page)
        after_structure = _enrich_after_scrollables(self.before_structure, after_structure)
        _safe_write_json(self.base_dir / "structure-after.json", after_structure)

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
                "before_full": "before-full.png",
                "before_main": "before-main.png",
                "structure_before": "structure-before.json",
                "after_full": "after-full.png",
                "after_main": "after-main.png",
                "structure_after": "structure-after.json",
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


class LinkedInVisualDiagnosticsCollector:
    """Collect local-only visual diagnostic artifact references."""

    def __init__(self, audit_dir: Path) -> None:
        self.audit_dir = Path(audit_dir)
        self.base_dir = self.audit_dir / _VISUAL_DIR
        self._events: list[LinkedInVisualDiagnosticArtifact] = []
        self._captured = False

    @property
    def events(self) -> list[LinkedInVisualDiagnosticArtifact]:
        return list(self._events)

    def should_capture(self) -> bool:
        return visual_diagnostics_enabled() and not self._captured

    def start_run(self, page, *, query: str) -> _VisualSearchRun | None:
        if not self.should_capture():
            return None
        self._captured = True
        self.base_dir.mkdir(parents=True, exist_ok=True)
        run = _VisualSearchRun(self, query=query)
        run.start_trace(page)
        return run

    def record(self, event: LinkedInVisualDiagnosticArtifact) -> None:
        self._events.append(event)


_ACTIVE_VISUAL_DIAGNOSTICS: ContextVar[
    LinkedInVisualDiagnosticsCollector | None
] = ContextVar("linkedin_visual_diagnostics", default=None)


def get_active_visual_diagnostics() -> LinkedInVisualDiagnosticsCollector | None:
    return _ACTIVE_VISUAL_DIAGNOSTICS.get()


@contextmanager
def visual_diagnostics_context(audit_dir: Path) -> Iterator[
    LinkedInVisualDiagnosticsCollector
]:
    collector = LinkedInVisualDiagnosticsCollector(audit_dir)
    token = _ACTIVE_VISUAL_DIAGNOSTICS.set(collector)
    try:
        yield collector
    finally:
        _ACTIVE_VISUAL_DIAGNOSTICS.reset(token)


__all__ = [
    "LinkedInVisualDiagnosticsCollector",
    "get_active_visual_diagnostics",
    "visual_diagnostics_context",
    "visual_diagnostics_enabled",
]
