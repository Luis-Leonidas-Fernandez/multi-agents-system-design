"""Checkpoints/bookmarks persistidos por sesión."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import re
import time
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SessionBookmark:
    checkpoint_id: str
    session_id: str
    label: str
    created_at_ms: int
    note: str
    message_count: int
    has_memory: bool
    artifact_path: str
    replay_item_count: int
    context_budget: dict[str, Any]
    prompt_agents: list[str]


class SessionBookmarkStore:
    """Persistencia simple de checkpoints por sesión."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or Path("sessions")

    def _session_dir(self, session_id: str) -> Path:
        return self._base_dir / session_id

    def _index_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "CHECKPOINTS.json"

    def save(self, bookmark: SessionBookmark) -> None:
        path = self._index_path(bookmark.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        bookmarks = self.list(bookmark.session_id)
        bookmarks = [item for item in bookmarks if item.get("checkpoint_id") != bookmark.checkpoint_id]
        bookmarks.append(asdict(bookmark))
        path.write_text(json.dumps(bookmarks, ensure_ascii=False, indent=2), encoding="utf-8")

    def list(self, session_id: str) -> list[dict[str, Any]]:
        path = self._index_path(session_id)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return payload if isinstance(payload, list) else []

    def load(self, session_id: str, checkpoint_id: str) -> dict[str, Any] | None:
        for item in self.list(session_id):
            if str(item.get("checkpoint_id")) == checkpoint_id:
                return item
        return None

    def list_sessions(self) -> list[str]:
        if not self._base_dir.exists():
            return []
        sessions: list[str] = []
        for session_dir in self._base_dir.iterdir():
            if session_dir.is_dir() and (session_dir / "CHECKPOINTS.json").exists():
                sessions.append(session_dir.name)
        return sorted(sessions)


class SessionBookmarkService:
    """Crea y consulta bookmarks/checkpoints de sesión."""

    def __init__(self, store: SessionBookmarkStore | None = None) -> None:
        self._store = store or SessionBookmarkStore()

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
        return slug or "checkpoint"

    def create(
        self,
        *,
        session_id: str,
        label: str | None,
        artifact_path: str,
        message_count: int,
        has_memory: bool,
        replay_item_count: int,
        context_budget: dict[str, Any],
        prompt_agents: list[str],
        note: str = "",
    ) -> SessionBookmark:
        created_at_ms = int(time.time() * 1000)
        base_label = label.strip() if label else f"checkpoint-{message_count}"
        checkpoint_id = f"{self._slugify(base_label)}-{created_at_ms}"
        bookmark = SessionBookmark(
            checkpoint_id=checkpoint_id,
            session_id=session_id,
            label=base_label,
            created_at_ms=created_at_ms,
            note=note.strip(),
            message_count=message_count,
            has_memory=has_memory,
            artifact_path=artifact_path,
            replay_item_count=replay_item_count,
            context_budget=context_budget,
            prompt_agents=sorted(set(prompt_agents)),
        )
        self._store.save(bookmark)
        return bookmark

    def list(self, session_id: str) -> list[dict[str, Any]]:
        return self._store.list(session_id)

    def describe(self, session_id: str, checkpoint_id: str) -> dict[str, Any] | None:
        return self._store.load(session_id, checkpoint_id)

    def list_sessions(self) -> list[str]:
        return self._store.list_sessions()


session_bookmark_store = SessionBookmarkStore()
session_bookmark_service = SessionBookmarkService()


__all__ = [
    "SessionBookmark",
    "SessionBookmarkService",
    "SessionBookmarkStore",
    "session_bookmark_service",
    "session_bookmark_store",
]
