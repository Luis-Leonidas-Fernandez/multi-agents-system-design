"""Búsqueda y ranking de memoria destilada por sesión.

Esto hace que la memoria deje de ser solo un archivo plano por sesión y
se convierta en un corpus consultable para CLI y runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import re
import time
from pathlib import Path
from typing import Any


_SESSIONS_DIR = Path("infra") / "sessions"


@dataclass(frozen=True)
class MemorySearchHit:
    session_id: str
    score: float
    memory_path: str
    excerpt: str
    matched_terms: list[str]
    line_count: int
    char_count: int
    modified_at_ms: int


class MemoryRetrievalService:
    def __init__(self, sessions_dir: Path | None = None) -> None:
        self._sessions_dir = sessions_dir or _SESSIONS_DIR

    def list_sessions(self) -> list[str]:
        if not self._sessions_dir.exists():
            return []
        session_ids: set[str] = set()
        for path in self._sessions_dir.iterdir():
            if path.is_dir() and (path / "MEMORY.md").exists():
                session_ids.add(path.name)
        return sorted(session_ids)

    def load_memory(self, session_id: str) -> str:
        path = self._sessions_dir / session_id / "MEMORY.md"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        session_id: str | None = None,
    ) -> list[MemorySearchHit]:
        terms = self._normalize_terms(query)
        if not terms:
            return []

        session_ids = [session_id] if session_id else self.list_sessions()
        hits: list[MemorySearchHit] = []
        for candidate in session_ids:
            if candidate is None:
                continue
            hit = self._score_session(candidate, terms)
            if hit is not None:
                hits.append(hit)

        hits.sort(key=lambda item: (-item.score, -item.modified_at_ms, item.session_id))
        return hits[:limit]

    def summarize(self, query: str, *, limit: int = 5, session_id: str | None = None) -> dict[str, Any]:
        hits = self.search(query, limit=limit, session_id=session_id)
        return {
            "query": query,
            "total": len(hits),
            "hits": [asdict(hit) for hit in hits],
        }

    def _score_session(self, session_id: str, terms: list[str]) -> MemorySearchHit | None:
        path = self._sessions_dir / session_id / "MEMORY.md"
        if not path.exists():
            return None

        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        matched_terms = [term for term in terms if term in lowered]
        if not matched_terms:
            return None

        lines = [line.strip(" -•\t") for line in text.splitlines() if line.strip()]
        excerpt = self._pick_excerpt(lines, terms)
        hit_count = sum(lowered.count(term) for term in terms)
        bullet_bonus = sum(1 for line in lines if any(term in line.lower() for term in terms))
        recency_bonus = self._recency_bonus(path.stat().st_mtime)
        score = float(hit_count * 10 + bullet_bonus * 3 + recency_bonus)
        return MemorySearchHit(
            session_id=session_id,
            score=score,
            memory_path=str(path),
            excerpt=excerpt,
            matched_terms=matched_terms,
            line_count=len(lines),
            char_count=len(text),
            modified_at_ms=int(path.stat().st_mtime * 1000),
        )

    def _pick_excerpt(self, lines: list[str], terms: list[str]) -> str:
        for line in lines:
            lower = line.lower()
            if any(term in lower for term in terms):
                return line[:180]
        return lines[0][:180] if lines else ""

    def _normalize_terms(self, query: str) -> list[str]:
        tokens = re.findall(r"[\wáéíóúñÁÉÍÓÚÑ.-]+", query.lower())
        stop_words = {"de", "la", "el", "y", "o", "a", "en", "por", "para", "con", "un", "una", "los", "las"}
        return [token for token in tokens if len(token) > 1 and token not in stop_words]

    def _recency_bonus(self, mtime_seconds: float) -> int:
        age_days = max(0.0, (time.time() - mtime_seconds) / 86400.0)
        return max(0, int(50 - age_days * 5))


memory_retrieval_service = MemoryRetrievalService()


__all__ = ["MemorySearchHit", "MemoryRetrievalService", "memory_retrieval_service"]
