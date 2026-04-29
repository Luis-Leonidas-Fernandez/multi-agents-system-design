"""Utilidades de texto puras para el dominio web.

Sin dependencias de capa de aplicación — solo stdlib.
"""
import re
from typing import Optional
from urllib.parse import urlparse

from features.web_scraping.domain.models import SourceDict, WebDigestContract, WebDigestSection


_TITLE_STOPWORDS = {
    "de", "la", "el", "en", "a", "los", "las", "del", "que", "un", "una",
    "por", "con", "se", "ha", "al", "es", "su", "y", "e", "o", "the", "of",
    "in", "to", "a", "and", "for", "on", "at", "by", "with", "from",
}

_MONTH_NAMES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}

_MONTH_NAMES_EN = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

_NO_INFO_RE = re.compile(
    # Pattern A: "no/sin + verb + noticias/información/news/contenido"
    r"\b(?:no|sin)\b.{0,50}\b(?:noticias?|informacion|news|contenido relevante)\b"
    # Pattern B: "subject + no + verb" — "la información proporcionada no incluye"
    r"|\b(?:informacion|pagina|sitio|texto|contenido)\b.{0,40}\bno\b.{0,40}"
    r"\b(?:incluye|proporciona|contiene|ofrece|tiene|encontr)\b"
    # Pattern C: explicit "doesn't address this topic" meta-commentary
    r"|sin abordar (?:directamente|este tema|el tema)"
    r"|no aborda (?:directamente|este tema)"
    r"|no trata (?:directamente|este tema)"
    r"|informacion proporcionada se centra en"
    # Pattern D: English equivalents
    r"|does not (?:contain|provide|include) (?:information|news|relevant)"
    r"|no relevant information|no results found|without relevant information",
    re.DOTALL,
)

_CITE_THIS_RE = re.compile(r"<<<CITE_THIS:[^>]+>>>")


def _extract_urls_from_text(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)\]]+", text or "")
    cleaned: list[str] = []
    seen: set[str] = set()
    for url in urls:
        normalized = url.rstrip(".,;:")
        if normalized and normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)
    return cleaned


def _clean_source_url(url: str) -> str:
    """Strip CITE_THIS artifacts (|domain=xxx>>>) from URLs."""
    return url.split("|")[0].rstrip(">").strip() if url else url


def _clean_digest_text(text: str) -> str:
    cleaned = _CITE_THIS_RE.sub("", text or "")
    cleaned = re.sub(r"^\s*#+\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _format_sources(sources: list[SourceDict]) -> str:
    if not sources:
        return ""
    lines = ["Sources:"]
    seen_urls: set[str] = set()
    for source in _unique_sources(sources):
        raw_url = source.get("url") or ""
        url = _clean_source_url(raw_url)
        if not url:
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        title = _display_source_title(source)
        lines.append(f"- [{title}]({url})")
    return "\n".join(lines)


def _unique_sources(sources: list[SourceDict]) -> list[SourceDict]:
    unique: list[SourceDict] = []
    seen: set[str] = set()
    for source in sources:
        raw_url = source.get("url") or ""
        url = _clean_source_url(raw_url)
        key = url or _strip_accents((source.get("title") or "").lower())
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def _display_source_title(source: SourceDict) -> str:
    raw_url = source.get("url") or ""
    url = _clean_source_url(raw_url)
    hostname = (urlparse(url).hostname or "").replace("www.", "").strip()
    title = _clean_digest_text(source.get("title") or "")
    if title:
        title_host = (urlparse(title).hostname or "").replace("www.", "").strip()
        if title_host and title_host == hostname:
            return hostname or title
        return title
    if hostname:
        return hostname
    return url or "source"


def _infer_section_label(text: str, source_title: str = "") -> str:
    haystack = _strip_accents(f"{source_title} {text}".lower())
    rules = [
        ("policiales", ("policia", "policial", "robo", "delito", "crimen", "motochor", "asesin", "apunal", "tiroteo", "detenid", "arrest", "banda", "narco")),
        ("operativos", ("operativo", "inciner", "droga", "cocaina", "narcot", "desarticul", "secuestr", "captura", "allan", "fiscal", "bandas")),
        ("seguridad", ("gobierno", "casa rosada", "prensa", "seguridad nacional", "ministerio", "policia federal", "smn", "institucional", "transparencia", "espionaje")),
        ("internacional", ("ee.uu", "estados unidos", "china", "geopolit", "internacional", "extranjero", "extradicion", "alianza")),
        ("sociedad", ("escuela", "amenaza", "evacu", "juvenil", "viral", "tiroteo", "patinete", "menor", "adolescente", "educativ", "colegio")),
    ]
    for label, signals in rules:
        if any(signal in haystack for signal in signals):
            return label
    return "Resumen"


def _split_sentences(text: str) -> list[str]:
    raw_parts = re.split(r"(?<=[.!?])\s+|\n+", (text or "").strip())
    parts: list[str] = []
    for part in raw_parts:
        cleaned = re.sub(r"\s+", " ", part).strip()
        if cleaned:
            parts.append(cleaned)
    return parts


def _line_signature(text: str) -> str:
    words = [
        word
        for word in re.findall(r"[\wáéíóúüñ]+", _strip_accents((text or "").lower()))
        if len(word) > 3 and word not in _TITLE_STOPWORDS
    ]
    return " ".join(words[:10])


def build_web_digest_contract(
    summary_lines: list[str],
    sources: list[SourceDict],
    *,
    intro: Optional[str] = None,
    conclusion: Optional[str] = None,
) -> WebDigestContract:
    sources = _unique_sources(sources)
    body = []
    seen_lines: set[str] = set()
    seen_signatures: set[str] = set()
    for line in summary_lines:
        cleaned = _clean_digest_text(line)
        if not cleaned:
            continue
        cleaned = re.sub(r"^[-•\u2022]\s+", "", cleaned)
        dedupe_key = re.sub(r"\s+", " ", cleaned).strip().lower()
        signature = _line_signature(cleaned)
        if dedupe_key in seen_lines or (signature and signature in seen_signatures):
            continue
        seen_lines.add(dedupe_key)
        if signature:
            seen_signatures.add(signature)
        body.append(cleaned)

    contract: WebDigestContract = {
        "version": "web_digest_v1",
        "intro": intro or "Te resumo lo más relevante y actualizado sobre seguridad en Argentina en los últimos días:",
        "sections": [],
        "conclusion": conclusion or "🧠 Conclusión: panorama mixto con actividad operativa, delito urbano y tensión institucional.",
        "sources": sources,
    }

    if not body:
        return contract

    if sources:
        usable_body = body[: max(len(sources) * 5, len(sources))]
        base_size, remainder = divmod(len(usable_body), len(sources))
        if base_size == 0:
            base_size = 1
        cursor = 0
        sections: list[WebDigestSection] = []
        for index, source in enumerate(sources):
            chunk_size = base_size + (1 if index < remainder else 0)
            chunk_size = min(5, max(1, chunk_size))
            chunk = usable_body[cursor:cursor + chunk_size]
            if not chunk:
                continue
            cursor += chunk_size
            paragraph = " ".join(_split_sentences(" ".join(chunk))[:5]).strip()
            raw_title = source.get("title") or ""
            title = _display_source_title(source)
            raw_url = source.get("url") or ""
            url = _clean_source_url(raw_url)
            section_label = _infer_section_label(paragraph, raw_title or title).lower()
            sections.append({
                "title": title or url or "source",
                "topic": section_label,
                "source": source,
                "bullets": [paragraph] if paragraph else [],
            })

        contract["sections"] = sections
        return contract

    contract["sections"] = [{
        "title": "Resumen",
        "topic": "resumen",
        "source": {},
        "bullets": body[:10],
    }]
    return contract


def format_web_digest_contract(contract: WebDigestContract) -> str:
    intro = (contract.get("intro") or "").strip()
    sections = contract.get("sections") or []
    conclusion = (contract.get("conclusion") or "").strip()
    sources = contract.get("sources") or []

    parts: list[str] = []
    if intro:
        parts.append(intro)
    for section in sections:
        title = _clean_digest_text(section.get("title") or "Resumen") or "Resumen"
        topic = _clean_digest_text(section.get("topic") or "resumen") or "resumen"
        bullets = section.get("bullets") or []
        source = section.get("source") or {}
        raw_url = source.get("url") or ""
        url = _clean_source_url(raw_url)
        source_title = _display_source_title(source)
        section_body = "\n".join(f"• {_clean_digest_text(bullet)}" for bullet in bullets if _clean_digest_text(bullet))
        if not section_body:
            continue
        source_line = f"Fuente: [{source_title or url or 'source'}]({url})" if url else f"Fuente: {source_title or 'source'}"
        parts.append(f"{title} — {topic}: {section_body}\n\n{source_line}")

    if conclusion:
        parts.append(conclusion)

    sources_block = _format_sources(sources)
    if sources_block:
        parts.append(sources_block)

    return "\n\n".join(part for part in parts if part.strip()).strip()


def _build_source_backed_response(summary_lines: list[str], sources: list[SourceDict]) -> str:
    return format_web_digest_contract(build_web_digest_contract(summary_lines, sources))


def _strip_accents(text: str) -> str:
    """Remove diacritics so 'japon' matches 'japón', 'ultima' matches 'última', etc."""
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _slugify_periodicos_label(value: str) -> str:
    normalized = _strip_accents((value or "").lower())
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return normalized


def _text_keywords(text: str) -> set[str]:
    return {w.lower() for w in text.split() if len(w) > 4 and w.lower() not in _TITLE_STOPWORDS}


def _candidate_url_has_date(url: str) -> bool:
    lowered = (url or "").lower()
    if re.search(r"[/\-](\d{4})[/\-](\d{2})[/\-](\d{2})", lowered):
        return True
    if re.search(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])", lowered):
        return True
    all_months = {**_MONTH_NAMES_ES, **_MONTH_NAMES_EN}
    month_pattern = "|".join(re.escape(m) for m in all_months)
    return bool(
        re.search(rf"(?:({month_pattern})[- ](\d{{4}})|(\d{{4}})[- ]({month_pattern}))", lowered)
    )


def _candidate_url_is_recent(url: str, days_threshold: int) -> bool:
    """Returns True if the date embedded in the URL is within `days_threshold` days, or no date found.

    Handles separated dates (/2026/04/02/), compact dates (yjj20260402...), and
    month-name slugs in Spanish or English (e.g. julio-2025, march-2025).
    """
    import datetime
    today = datetime.date.today()
    cutoff = today - datetime.timedelta(days=days_threshold)
    lowered = (url or "").lower()
    for match in re.finditer(r"[/\-](\d{4})[/\-](\d{2})[/\-](\d{2})", lowered):
        try:
            article_date = datetime.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            return article_date >= cutoff
        except ValueError:
            pass
    for match in re.finditer(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])", lowered):
        try:
            article_date = datetime.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            return article_date >= cutoff
        except ValueError:
            pass
    all_months = {**_MONTH_NAMES_ES, **_MONTH_NAMES_EN}
    month_pattern = "|".join(re.escape(m) for m in all_months)
    for match in re.finditer(
        rf"(?:({month_pattern})[- ](\d{{4}})|(\d{{4}})[- ]({month_pattern}))", lowered
    ):
        month_name = match.group(1) or match.group(4)
        year_str = match.group(2) or match.group(3)
        try:
            article_date = datetime.date(int(year_str), all_months[month_name], 1)
            return article_date >= cutoff
        except (ValueError, KeyError):
            pass
    return True  # No date in URL → don't filter (assume recent)


def _is_no_info_response(text: str) -> bool:
    lowered = _strip_accents((text or "").lower())
    return bool(_NO_INFO_RE.search(lowered))


def _enforce_synthesis_format(text: str) -> str:
    """Post-process LLM output to guarantee bullet spacing and strip header artifacts."""
    lines = text.splitlines()
    result: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^#{1,4}\s", stripped):
            continue
        if re.match(r"^[•\-\*]\s", stripped) and result and result[-1].strip():
            result.append("")
        result.append(stripped)
        if re.match(r"^[•\-\*]\s", stripped):
            pass
    final: list[str] = []
    for i, line in enumerate(result):
        final.append(line)
        if re.match(r"^[•\-\*]\s", line):
            if i + 1 < len(result) and result[i + 1].strip():
                final.append("")
    collapsed: list[str] = []
    blank_count = 0
    for line in final:
        if not line.strip():
            blank_count += 1
            if blank_count <= 2:
                collapsed.append(line)
        else:
            blank_count = 0
            collapsed.append(line)
    return "\n".join(collapsed).strip()


def _dedup_synthesis_bullets(text: str, query_terms: Optional[list[str]] = None) -> str:
    """Remove duplicate bullets from a synthesized response.

    Two bullets are considered duplicates when their non-query keyword overlap ≥ 3.
    Keeps the longer (more informative) bullet of each duplicate pair.
    """
    excluded = set(t.lower() for t in (query_terms or []))
    excluded.update(_TITLE_STOPWORDS)

    def kw(s: str) -> set[str]:
        words = set()
        for w in s.split():
            w = w.lower()
            if len(w) <= 4 or w in excluded:
                continue
            if w.endswith("s") and len(w) > 5:
                w = w[:-1]
            words.add(w)
        return words

    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if re.match(r"^[•\-\*]\s", line.strip()) and current:
            blocks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))

    bullet_blocks: list[str] = []
    prefix_lines: list[str] = []
    for block in blocks:
        if re.match(r"^[•\-\*]\s", block.strip()):
            bullet_blocks.append(block)
        else:
            prefix_lines.append(block)

    _DISASTER_KW = {"terremoto", "sismo", "maremoto", "tsunami", "earthquake", "seismic"}

    accepted: list[str] = []
    for block in bullet_blocks:
        block_kw = kw(block)
        duplicate = False
        for i, acc in enumerate(accepted):
            acc_kw = kw(acc)
            both_disaster = bool(block_kw & _DISASTER_KW) and bool(acc_kw & _DISASTER_KW)
            dedup_threshold = 2 if both_disaster else 4
            if len(block_kw & acc_kw) >= dedup_threshold:
                if len(block) > len(acc):
                    accepted[i] = block
                duplicate = True
                break
        if not duplicate:
            accepted.append(block)

    all_parts = prefix_lines + accepted
    return "\n\n".join(p.strip() for p in all_parts if p.strip())
