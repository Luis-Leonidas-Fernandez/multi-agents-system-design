"""Safe local-only structure diagnostics for visible LinkedIn job cards."""
from __future__ import annotations

from pathlib import Path
from typing import Any


CARD_STRUCTURE_DEBUG_ENV = "LINKEDIN_CARD_STRUCTURE_DEBUG"
CARD_STRUCTURE_DEBUG_FILENAME = "card-structure-visible.json"


def card_structure_debug_enabled() -> bool:
    import os

    return str(os.getenv(CARD_STRUCTURE_DEBUG_ENV, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _safe_int(value: Any, *, minimum: int = 0, maximum: int = 100_000) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return minimum
    return max(minimum, min(maximum, number))


def _safe_string(value: Any, *, limit: int = 120) -> str:
    import re

    text = re.sub(r"\s+", " ", str(value or "")).strip()[:limit]
    return re.sub(r"[^0-9A-Za-z_ .:/()\-가-힣一-龯áéíóúÁÉÍÓÚüÜñÑ]", "", text)


def _safe_card_structure_snapshot(value: Any) -> dict[str, object]:
    if not isinstance(value, dict):
        return {"schema_version": "1.0", "cards": [], "card_count": 0}
    cards = []
    for raw_card in value.get("cards") or []:
        if not isinstance(raw_card, dict):
            continue
        ancestors = []
        for raw_ancestor in raw_card.get("ancestor_chain") or []:
            if not isinstance(raw_ancestor, dict):
                continue
            ancestors.append(
                {
                    "depth": _safe_int(raw_ancestor.get("depth"), maximum=8),
                    "tag": _safe_string(raw_ancestor.get("tag"), limit=32),
                    "class_tokens": [
                        _safe_string(token, limit=64)
                        for token in (raw_ancestor.get("class_tokens") or [])[:12]
                    ],
                    "has_href": bool(raw_ancestor.get("has_href")),
                    "has_data_job_id": bool(raw_ancestor.get("has_data_job_id")),
                    "has_data_occludable_job_id": bool(
                        raw_ancestor.get("has_data_occludable_job_id")
                    ),
                    "bbox": {
                        "x": _safe_int((raw_ancestor.get("bbox") or {}).get("x")),
                        "y": _safe_int((raw_ancestor.get("bbox") or {}).get("y")),
                        "width": _safe_int((raw_ancestor.get("bbox") or {}).get("width")),
                        "height": _safe_int((raw_ancestor.get("bbox") or {}).get("height")),
                    },
                    "date_texts": [
                        _safe_string(text, limit=80)
                        for text in (raw_ancestor.get("date_texts") or [])[:5]
                    ],
                    "child_count": _safe_int(raw_ancestor.get("child_count"), maximum=500),
                }
            )
        cards.append(
            {
                "card_index": _safe_int(raw_card.get("card_index"), maximum=200),
                "resolved_job_id": _safe_string(raw_card.get("resolved_job_id"), limit=32),
                "title_present": bool(raw_card.get("title_present")),
                "date_texts": [
                    _safe_string(text, limit=80)
                    for text in (raw_card.get("date_texts") or [])[:8]
                ],
                "has_href": bool(raw_card.get("has_href")),
                "has_data_job_id": bool(raw_card.get("has_data_job_id")),
                "has_data_occludable_job_id": bool(raw_card.get("has_data_occludable_job_id")),
                "bbox": {
                    "x": _safe_int((raw_card.get("bbox") or {}).get("x")),
                    "y": _safe_int((raw_card.get("bbox") or {}).get("y")),
                    "width": _safe_int((raw_card.get("bbox") or {}).get("width")),
                    "height": _safe_int((raw_card.get("bbox") or {}).get("height")),
                },
                "ancestor_chain": ancestors[:8],
            }
        )
    return {
        "schema_version": "1.0",
        "card_count": _safe_int(value.get("card_count"), maximum=200),
        "captured_count": len(cards),
        "viewport": value.get("viewport") if isinstance(value.get("viewport"), dict) else {},
        "cards": cards[:24],
    }


def capture_visible_linkedin_card_structure_debug(page: Any, output_dir: Path) -> Path | None:
    """Capture bounded, sanitized structure for visible job cards.

    This is diagnostics-only: no HTML, cookies, tokens, full body text, or query strings.
    """
    if not card_structure_debug_enabled():
        return None
    try:
        raw = page.evaluate(
            r"""
            () => {
              const patterns = [
                /\bhace\s+(?:unos?\s+segundos?|un\s+momento|\d+\s+(?:minutos?|horas?|d[ií]as?))\b/i,
                /\b(?:just now|moments ago|a few seconds ago|\d+\s+(?:minutes?|hours?|days?)\s+ago)\b/i,
                /\b(?:hoy|today)\b/i,
                /\b\d+\s*(?:시간|분|일)\s*전\b/i,
                /\b\d+\s*(?:時間|分|日)前\b/i,
              ];
              const viewportWidth = Math.max(0, Number(window.innerWidth || 0));
              const textOf = (node) => String(node && (node.innerText || node.textContent) || '').replace(/\s+/g, ' ').trim();
              const dateTexts = (node) => {
                const found = [];
                const nodes = [node, ...Array.from(node && node.querySelectorAll ? node.querySelectorAll('*') : []).slice(0, 240)];
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
              const jobIdFromHref = (href) => {
                const match = String(href || '').match(/\/jobs\/view\/(?:[^/?#]*-)?(\d+)\/?(?:[?#]|$)/);
                return match && match[1] ? match[1].replace(/^0+/, '') || '0' : '';
              };
              const jobIdFromNode = (node) => {
                for (const attr of ['data-job-id', 'data-occludable-job-id']) {
                  const value = node && node.getAttribute ? node.getAttribute(attr) : '';
                  const match = String(value || '').match(/(\d{3,})$/);
                  if (match && match[1]) return match[1].replace(/^0+/, '') || '0';
                }
                const link = node && node.querySelector ? node.querySelector('a[href*="/jobs/view/"]') : null;
                return link ? jobIdFromHref(link.getAttribute('href') || '') : '';
              };
              const bbox = (node) => {
                const rect = node && node.getBoundingClientRect ? node.getBoundingClientRect() : null;
                return rect ? {
                  x: Math.max(0, Math.round(rect.left || 0)),
                  y: Math.max(0, Math.round(rect.top || 0)),
                  width: Math.max(0, Math.round(rect.width || 0)),
                  height: Math.max(0, Math.round(rect.height || 0)),
                } : {x: 0, y: 0, width: 0, height: 0};
              };
              const descriptor = (node, depth) => ({
                depth,
                tag: String(node && node.tagName || '').toLowerCase().slice(0, 32),
                class_tokens: String(node && node.className || '').split(/\s+/).filter(Boolean).slice(0, 12),
                has_href: Boolean(node && node.querySelector && node.querySelector('a[href*="/jobs/view/"]')),
                has_data_job_id: Boolean(node && node.hasAttribute && node.hasAttribute('data-job-id')),
                has_data_occludable_job_id: Boolean(node && node.hasAttribute && node.hasAttribute('data-occludable-job-id')),
                bbox: bbox(node),
                date_texts: dateTexts(node),
                child_count: node && node.children ? node.children.length : 0,
              });
              const candidates = Array.from(document.querySelectorAll(
                'li[data-job-id], li[data-occludable-job-id], [role="listitem"][data-job-id], [role="option"][data-job-id], [role="listitem"][data-occludable-job-id], [role="option"][data-occludable-job-id], .job-card-container, .scaffold-layout__list-item, a[href*="/jobs/view/"]'
              )).slice(0, 180);
              const cards = [];
              const seen = new Set();
              for (const candidate of candidates) {
                let node = candidate;
                if (String(candidate.tagName || '').toLowerCase() === 'a') {
                  let current = candidate;
                  for (let depth = 0; current && current !== document.body && depth < 7; depth += 1) {
                    if (jobIdFromNode(current) || String(current.tagName || '').toLowerCase() === 'li') {
                      node = current;
                      break;
                    }
                    current = current.parentElement;
                  }
                }
                const rect = node && node.getBoundingClientRect ? node.getBoundingClientRect() : null;
                if (!rect || rect.width <= 0 || rect.height <= 0 || rect.top < 70) continue;
                if (viewportWidth > 0 && rect.left > viewportWidth * 0.58) continue;
                const jobId = jobIdFromNode(node);
                const key = jobId || `${Math.round(rect.left)}:${Math.round(rect.top)}:${Math.round(rect.width)}:${Math.round(rect.height)}`;
                if (seen.has(key)) continue;
                seen.add(key);
                const chain = [];
                let current = node;
                for (let depth = 0; current && current !== document.body && depth < 8; depth += 1) {
                  chain.push(descriptor(current, depth));
                  current = current.parentElement;
                }
                cards.push({
                  card_index: cards.length,
                  resolved_job_id: jobId,
                  title_present: Boolean(node.querySelector && node.querySelector('a[href*="/jobs/view/"], .job-card-list__title, .job-card-container__link')),
                  date_texts: dateTexts(node),
                  has_href: Boolean(node.querySelector && node.querySelector('a[href*="/jobs/view/"]')),
                  has_data_job_id: Boolean(node.hasAttribute && node.hasAttribute('data-job-id')),
                  has_data_occludable_job_id: Boolean(node.hasAttribute && node.hasAttribute('data-occludable-job-id')),
                  bbox: bbox(node),
                  ancestor_chain: chain,
                });
                if (cards.length >= 24) break;
              }
              return {
                card_count: cards.length,
                viewport: {width: Math.round(window.innerWidth || 0), height: Math.round(window.innerHeight || 0)},
                cards,
              };
            }
            """
        )
    except Exception:
        return None
    snapshot = _safe_card_structure_snapshot(raw)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / CARD_STRUCTURE_DEBUG_FILENAME
        import json

        path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
    except Exception:
        return None


__all__ = [
    "CARD_STRUCTURE_DEBUG_ENV",
    "CARD_STRUCTURE_DEBUG_FILENAME",
    "capture_visible_linkedin_card_structure_debug",
    "card_structure_debug_enabled",
]
