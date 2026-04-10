"""Persistencia de audit trail para invocaciones de tools."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, cast


_AUDIT_DIR = Path("sessions")


class ToolAuditStore:
    """Store de eventos de tool audit en JSONL por sesión."""

    def __init__(self, audit_dir: Path | None = None) -> None:
        self._audit_dir = audit_dir or _AUDIT_DIR

    def _session_path(self, session_id: str) -> Path:
        return self._audit_dir / session_id / "TOOL_AUDIT.jsonl"

    def append_event(self, event: Any) -> None:
        if not is_dataclass(event):
            raise TypeError("event must be a dataclass instance")
        event_obj = cast(Any, event)
        path = self._session_path(event_obj.session_id or event_obj.request_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(event_obj), ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(payload + "\n")

    def load_events(self, session_id: str) -> list[dict[str, Any]]:
        path = self._session_path(session_id)
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return events

    def list_sessions(self) -> list[str]:
        if not self._audit_dir.exists():
            return []
        sessions: list[str] = []
        for session_dir in self._audit_dir.iterdir():
            if session_dir.is_dir() and (session_dir / "TOOL_AUDIT.jsonl").exists():
                sessions.append(session_dir.name)
        return sorted(sessions)

    def find_events(
        self,
        session_id: str,
        *,
        request_id: str | None = None,
        trace_id: str | None = None,
        tool_name: str | None = None,
    ) -> list[dict[str, Any]]:
        events = self.load_events(session_id)
        filtered: list[dict[str, Any]] = []
        for event in events:
            if request_id and event.get("request_id") != request_id:
                continue
            if trace_id and event.get("trace_id") != trace_id:
                continue
            if tool_name and event.get("tool_name") != tool_name:
                continue
            filtered.append(event)
        return filtered


tool_audit_store = ToolAuditStore()


__all__ = ["ToolAuditStore", "tool_audit_store"]
