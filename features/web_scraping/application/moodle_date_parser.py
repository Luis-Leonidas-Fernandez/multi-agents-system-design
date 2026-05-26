"""Parseo dedicado de fechas Moodle hacia ISO 8601."""
from __future__ import annotations

import re
from datetime import date, datetime


_MONTH_ALIASES = {
    "ene": 1,
    "enero": 1,
    "feb": 2,
    "febrero": 2,
    "mar": 3,
    "marzo": 3,
    "abr": 4,
    "abril": 4,
    "may": 5,
    "mayo": 5,
    "jun": 6,
    "junio": 6,
    "jul": 7,
    "julio": 7,
    "ago": 8,
    "agosto": 8,
    "sep": 9,
    "sept": 9,
    "septiembre": 9,
    "set": 9,
    "setiembre": 9,
    "oct": 10,
    "octubre": 10,
    "nov": 11,
    "noviembre": 11,
    "dic": 12,
    "diciembre": 12,
}

_WEEKDAY_PREFIXES = (
    "lunes",
    "martes",
    "miércoles",
    "miercoles",
    "jueves",
    "viernes",
    "sábado",
    "sabado",
    "domingo",
)


def _normalize_date_text(value: str) -> str:
    normalized = re.sub(r"\s+", " ", (value or "").strip().lower())
    normalized = normalized.replace(" a las ", " ").replace(" hs", "").replace(" h", "")
    normalized = normalized.split("—", 1)[0].strip()
    normalized = normalized.split("|", 1)[0].strip()
    normalized = re.sub(r",\s*\d{1,2}:\d{2}$", "", normalized)
    normalized = re.sub(r"\s+\d{1,2}:\d{2}$", "", normalized)
    normalized = re.sub(
        r"^(?:" + "|".join(_WEEKDAY_PREFIXES) + r")\s*,\s*",
        "",
        normalized,
    )
    return normalized.strip(" ,.;")


def _parse_known_formats(normalized: str) -> date | None:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(normalized[:10], fmt).date()
        except ValueError:
            continue
    return None


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _parse_spanish_date(normalized: str, *, reference_date: date) -> date | None:
    full_match = re.search(r"(?P<day>\d{1,2})\s+de\s+(?P<month>[a-záéíóú]+)\s+de\s+(?P<year>\d{4})", normalized)
    if full_match:
        month = _MONTH_ALIASES.get(full_match.group("month").rstrip("."))
        if month is None:
            return None
        return _safe_date(int(full_match.group("year")), month, int(full_match.group("day")))

    short_match = re.search(r"(?P<day>\d{1,2})\s+de\s+(?P<month>[a-záéíóú.]+)$", normalized)
    if short_match:
        month = _MONTH_ALIASES.get(short_match.group("month").rstrip("."))
        if month is None:
            return None
        return _safe_date(reference_date.year, month, int(short_match.group("day")))

    compact_match = re.search(r"(?P<day>\d{1,2})\s+(?P<month>[a-záéíóú.]+)$", normalized)
    if compact_match:
        month = _MONTH_ALIASES.get(compact_match.group("month").rstrip("."))
        if month is None:
            return None
        return _safe_date(reference_date.year, month, int(compact_match.group("day")))

    return None


def parse_moodle_due_date(value: str, *, reference_date: date | None = None) -> str | None:
    """Convierte fechas Moodle frecuentes a `YYYY-MM-DD`.

    Soporta formatos ISO/numéricos y variantes en español como:
    - `16 de abr, 00:00`
    - `4 de may`
    - `3 de mayo de 2026`
    """
    text = (value or "").strip()
    if not text:
        return None

    ref_date = reference_date or date.today()
    normalized = _normalize_date_text(text)
    if not normalized:
        return None

    parsed = _parse_known_formats(normalized) or _parse_spanish_date(normalized, reference_date=ref_date)
    return parsed.isoformat() if parsed is not None else None


__all__ = ["parse_moodle_due_date"]
