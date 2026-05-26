"""Persistencia temporal de artefactos Moodle para revisión y sync."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from application.services.request_runtime import get_request_runtime_config
from features.web_scraping.domain.moodle_models import MoodleAssignment, ValidatedMoodleAssignments


_REQUIRED_ROOT_KEYS = {"valid", "invalid", "issues", "meta"}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "session"


@dataclass(frozen=True)
class MoodleArtifactPaths:
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class MoodleArtifactSummary:
    json_path: str
    markdown_path: str
    structure_valid: bool
    approved: bool
    valid_count: int
    invalid_count: int
    issue_count: int
    issues: list[dict[str, str]]
    json_content: str
    markdown_content: str
    sync_created_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "jsonPath": self.json_path,
            "markdownPath": self.markdown_path,
            "structureValid": self.structure_valid,
            "approved": self.approved,
            "validCount": self.valid_count,
            "invalidCount": self.invalid_count,
            "issueCount": self.issue_count,
            "issues": self.issues,
            "jsonContent": self.json_content,
            "markdownContent": self.markdown_content,
            "syncCreatedCount": self.sync_created_count,
        }


def _session_artifact_dir() -> Path:
    config = get_request_runtime_config()
    session_id = _slug(config.session_id or "local")
    request_id = _slug(config.request_id or "manual")
    return Path("data") / "sessions" / session_id / "moodle" / request_id


def _artifact_paths() -> MoodleArtifactPaths:
    base_dir = _session_artifact_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    return MoodleArtifactPaths(
        json_path=base_dir / "moodle_tasks.json",
        markdown_path=base_dir / "moodle_tasks.md",
    )


def _artifact_structure_valid(payload: dict[str, Any]) -> bool:
    if not _REQUIRED_ROOT_KEYS.issubset(payload.keys()):
        return False
    valid_items = payload.get("valid")
    invalid_items = payload.get("invalid")
    issues = payload.get("issues")
    meta = payload.get("meta")
    if not isinstance(valid_items, list) or not isinstance(invalid_items, list) or not isinstance(issues, list) or not isinstance(meta, dict):
        return False
    if not valid_items:
        return False
    return not any(str(issue.get("severity", "")).lower() == "error" for issue in issues if isinstance(issue, dict))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def persist_moodle_artifacts(validated: ValidatedMoodleAssignments, review_markdown: str) -> MoodleArtifactPaths:
    paths = _artifact_paths()
    payload = validated.to_dict()
    payload["meta"] = {
        "session_id": get_request_runtime_config().session_id,
        "request_id": get_request_runtime_config().request_id,
        "approved": False,
        "sync_created_count": 0,
    }
    _write_json(paths.json_path, payload)
    paths.markdown_path.write_text(review_markdown, encoding="utf-8")
    return paths


def load_validated_moodle_artifact(json_path: str | Path) -> dict[str, Any]:
    path = Path(json_path)
    return _read_json(path)


def load_valid_moodle_assignments_from_artifact(json_path: str | Path) -> list[MoodleAssignment]:
    payload = load_validated_moodle_artifact(json_path)
    valid_items = payload.get("valid") if isinstance(payload.get("valid"), list) else []
    assignments: list[MoodleAssignment] = []
    for item in valid_items:
        if not isinstance(item, dict):
            continue
        assignments.append(
            MoodleAssignment(
                title=str(item.get("title") or "").strip(),
                course=str(item.get("course") or "").strip(),
                due_date=str(item.get("due_date") or "").strip(),
                url=str(item.get("url") or "").strip(),
                status=str(item.get("status") or "unknown"),  # type: ignore[arg-type]
                source=str(item.get("source") or "moodle"),
                raw_date_text=str(item.get("raw_date_text") or ""),
            )
        )
    return assignments


def approve_moodle_artifact(json_path: str | Path, approved: bool = True) -> dict[str, Any]:
    path = Path(json_path)
    payload = _read_json(path)
    if approved and not _artifact_structure_valid(payload):
        raise ValueError("El artifact no respeta la estructura esperada y no puede aprobarse.")
    meta = payload.setdefault("meta", {})
    meta["approved"] = bool(approved)
    _write_json(path, payload)
    return payload


def mark_moodle_artifact_calendar_sync(json_path: str | Path, created_count: int) -> dict[str, Any]:
    path = Path(json_path)
    payload = _read_json(path)
    meta = payload.setdefault("meta", {})
    meta["sync_created_count"] = max(0, int(created_count))
    _write_json(path, payload)
    return payload


def mark_moodle_artifact_notion_sync(
    json_path: str | Path,
    *,
    created_count: int,
    updated_count: int,
    skipped_count: int,
    error_count: int,
) -> dict[str, Any]:
    path = Path(json_path)
    payload = _read_json(path)
    meta = payload.setdefault("meta", {})
    meta["notion_sync"] = {
        "created_count": max(0, int(created_count)),
        "updated_count": max(0, int(updated_count)),
        "skipped_count": max(0, int(skipped_count)),
        "error_count": max(0, int(error_count)),
    }
    _write_json(path, payload)
    return payload


def delete_moodle_artifact(json_path: str | Path) -> None:
    path = Path(json_path)
    markdown_path = path.with_name("moodle_tasks.md")
    if path.exists():
        path.unlink()
    if markdown_path.exists():
        markdown_path.unlink()
    parent = path.parent
    try:
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        pass


def summarize_moodle_artifact(json_path: str | Path) -> MoodleArtifactSummary:
    path = Path(json_path)
    payload = _read_json(path)
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    markdown_path = path.with_name("moodle_tasks.md")
    markdown_content = markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else ""
    valid_items = payload.get("valid") if isinstance(payload.get("valid"), list) else []
    invalid_items = payload.get("invalid") if isinstance(payload.get("invalid"), list) else []
    issues = [issue for issue in payload.get("issues", []) if isinstance(issue, dict)] if isinstance(payload.get("issues"), list) else []
    return MoodleArtifactSummary(
        json_path=str(path),
        markdown_path=str(markdown_path),
        structure_valid=_artifact_structure_valid(payload),
        approved=bool(meta.get("approved")),
        valid_count=len(valid_items),
        invalid_count=len(invalid_items),
        issue_count=len(issues),
        issues=[{k: str(v) for k, v in issue.items()} for issue in issues],
        json_content=json.dumps(payload, ensure_ascii=False, indent=2),
        markdown_content=markdown_content,
        sync_created_count=int(meta.get("sync_created_count") or 0),
    )


def list_session_moodle_artifacts(session_id: str) -> list[MoodleArtifactSummary]:
    session_slug = _slug(session_id or "local")
    base_dir = Path("data") / "sessions" / session_slug / "moodle"
    if not base_dir.exists():
        return []
    json_paths = sorted(base_dir.glob("*/moodle_tasks.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    summaries: list[MoodleArtifactSummary] = []
    for path in json_paths:
        try:
            summaries.append(summarize_moodle_artifact(path))
        except Exception:
            continue
    return summaries


__all__ = [
    "MoodleArtifactPaths",
    "MoodleArtifactSummary",
    "approve_moodle_artifact",
    "delete_moodle_artifact",
    "list_session_moodle_artifacts",
    "load_valid_moodle_assignments_from_artifact",
    "load_validated_moodle_artifact",
    "mark_moodle_artifact_calendar_sync",
    "mark_moodle_artifact_notion_sync",
    "persist_moodle_artifacts",
    "summarize_moodle_artifact",
]
