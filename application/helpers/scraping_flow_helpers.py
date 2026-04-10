"""Helpers puros de scraping reutilizables por tools y casos de uso."""
from typing import Any, Dict, Optional, Tuple

from application.helpers.text_truncation import truncate_suffix

_CACHE_TTL_SECONDS = 60
_SCRAPE_CACHE_MAX = 256
_SCRAPE_CACHE: Dict[str, Tuple[float, str]] = {}


def _cache_key(url: str, params: Dict[str, Any]) -> str:
    key_parts = [url] + [f"{k}={params[k]}" for k in sorted(params.keys())]
    return "|".join(key_parts)


def _get_cache(key: str) -> Optional[str]:
    entry = _SCRAPE_CACHE.get(key)
    if not entry:
        return None
    timestamp, value = entry
    import time
    if time.time() - timestamp > _CACHE_TTL_SECONDS:
        _SCRAPE_CACHE.pop(key, None)
        return None
    return value


def _set_cache(key: str, value: str) -> None:
    import time
    if len(_SCRAPE_CACHE) >= _SCRAPE_CACHE_MAX:
        oldest_key = next(iter(_SCRAPE_CACHE))
        _SCRAPE_CACHE.pop(oldest_key, None)
    _SCRAPE_CACHE[key] = (time.time(), value)


def _validate_url(url: str) -> Optional[str]:
    from urllib.parse import urlparse
    import ipaddress
    try:
        parsed = urlparse(url)
    except Exception:
        return "URL inválida"

    if parsed.scheme not in ("http", "https"):
        return f"Esquema no permitido: {parsed.scheme!r}. Solo se permiten http y https."

    hostname = parsed.hostname or ""
    blocked_hostnames = {"localhost", "0.0.0.0", "::1", "metadata.google.internal"}
    if hostname.lower() in blocked_hostnames:
        return f"Host no permitido: {hostname!r}"

    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved:
            return f"Dirección IP privada/reservada no permitida: {hostname!r}"
    except ValueError:
        pass

    return None


def _clean_text(text: str) -> str:
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return "\n".join(chunk for chunk in chunks if chunk)


def _truncate_text(text: str, max_chars: int) -> str:
    return truncate_suffix(text, max_chars=max_chars, suffix="... [texto truncado]")


def _extract_text(soup, max_chars: int, extract_selector: Optional[str] = None) -> str:
    if extract_selector:
        target = soup.select_one(extract_selector)
        if not target:
            return ""
        return _truncate_text(_clean_text(target.get_text()), max_chars)
    for script in soup(["script", "style", "noscript"]):
        script.decompose()
    return _truncate_text(_clean_text(soup.get_text()), max_chars)


def _extract_links(soup, base_url: str, max_links: int = 20) -> Tuple[int, str]:
    from urllib.parse import urljoin
    links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        link_text = link.get_text(strip=True)
        if href.startswith("/"):
            href = urljoin(base_url, href)
        links.append(f"- {link_text}: {href}")
    total = len(links)
    return total, "\n".join(links[:max_links])


def _build_result(url: str, text: Optional[str], links: Optional[str], total_links: int) -> str:
    parts = [f"URL: {url}\n"]
    if text:
        parts.append(f"\nTexto extraido:\n{text}")
    if links:
        parts.append(f"\n\nEnlaces encontrados ({total_links} total, mostrando primeros 20):\n")
        parts.append(links)
    return "\n".join(parts)


__all__ = [
    "_CACHE_TTL_SECONDS",
    "_SCRAPE_CACHE_MAX",
    "_SCRAPE_CACHE",
    "_cache_key",
    "_get_cache",
    "_set_cache",
    "_validate_url",
    "_clean_text",
    "_truncate_text",
    "_extract_text",
    "_extract_links",
    "_build_result",
]
