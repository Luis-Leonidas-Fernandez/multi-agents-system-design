"""Modelos del flujo Moodle → revisión humana → Google Calendar."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


MoodleAssignmentStatus = Literal["pending", "due_today", "overdue", "submitted", "unknown"]
ValidationSeverity = Literal["error", "warning"]


@dataclass(frozen=True)
class MoodleAssignment:
    title: str
    course: str = ""
    due_date: str = ""
    url: str = ""
    status: MoodleAssignmentStatus = "unknown"
    source: str = "moodle"
    raw_date_text: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class MoodleAssignmentValidationIssue:
    assignment_title: str
    severity: ValidationSeverity
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ValidatedMoodleAssignments:
    valid: list[MoodleAssignment] = field(default_factory=list)
    invalid: list[MoodleAssignment] = field(default_factory=list)
    issues: list[MoodleAssignmentValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": [item.to_dict() for item in self.valid],
            "invalid": [item.to_dict() for item in self.invalid],
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class CalendarDraftEvent:
    summary: str
    start: str
    end: str
    description: str = ""
    location: str = ""
    source_title: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


__all__ = [
    "CalendarDraftEvent",
    "MoodleAssignment",
    "MoodleAssignmentStatus",
    "MoodleAssignmentValidationIssue",
    "ValidatedMoodleAssignments",
    "ValidationSeverity",
]
