"""Helpers compartidos para la capa de búsqueda web."""
from __future__ import annotations

import os


def _web_search_debug_enabled() -> bool:
    return (os.getenv("WEB_DEBUG") or "").strip().lower() in {"1", "true", "yes", "on"}


def _web_search_debug(label: str, **data) -> None:
    if not _web_search_debug_enabled():
        return
    payload = " ".join(f"{key}={repr(value)}" for key, value in data.items())
    print(f"[WEB_DEBUG] {label}{(' ' + payload) if payload else ''}", flush=True)


def _format_search_results(query: str, hits: list[dict[str, str]]) -> str:
    from domain.web_classifier import _is_specific_article_hit

    if not hits:
        return "No results found."

    lines = []
    for idx, hit in enumerate(hits[:8], start=1):
        title = hit.get("title") or hit.get("url") or "result"
        link = hit.get("url") or ""
        snippet = (hit.get("content") or "").strip()
        tag = "[article]" if _is_specific_article_hit(hit) else "[hub]"
        if link:
            lines.append(f"{idx}. {tag} [{title}]({link})")
        else:
            lines.append(f"{idx}. {tag} {title}")
        if snippet:
            lines.append(f"   {snippet[:120].rstrip()}{'…' if len(snippet) > 120 else ''}")

    lines.append("")
    lines.append("Call web_fetch on [article] URLs (not [hub]) to read full content before writing your summary.")
    return "\n".join(lines).strip()


__all__ = ["_web_search_debug_enabled", "_web_search_debug", "_format_search_results"]
