"""Retention helpers for local session artifacts."""
from __future__ import annotations

import shutil
from pathlib import Path

DEFAULT_SESSION_RETENTION_COUNT = 3


def enforce_recent_session_retention(
    root: Path | str = Path("data") / "sessions",
    *,
    keep: int = DEFAULT_SESSION_RETENTION_COUNT,
) -> list[Path]:
    """Keep only the newest local session directories under ``root``.

    Root files such as sessions.db or jsonl logs are preserved. Directories are
    sorted by modification time descending, which matches how audit/artifact
    writes update the active session directory.
    """
    root_path = Path(root)
    if keep <= 0 or not root_path.exists():
        return []
    session_dirs = [path for path in root_path.iterdir() if path.is_dir()]
    session_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    removed: list[Path] = []
    for old_dir in session_dirs[keep:]:
        shutil.rmtree(old_dir)
        removed.append(old_dir)
    return removed


__all__ = [
    "DEFAULT_SESSION_RETENTION_COUNT",
    "enforce_recent_session_retention",
]
