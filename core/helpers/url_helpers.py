"""URL utilities: article detection, redirect extraction, URL normalization."""
import ipaddress
import re
from typing import Optional
from urllib.parse import urlparse

from pydantic import AnyHttpUrl, BaseModel, ValidationError

# Matches URLs that look like specific news articles:
# - date segment in path (/2026/04/20/ or /20260420- or 2026-04-20)
# - long slug (≥15 chars covers IDs like yjj2026040500456)
_ARTICLE_URL_RE = re.compile(
    r"/\d{4}/\d{2}/\d{2}/|/\d{8}[-_]|\d{4}-\d{2}-\d{2}"
    r"|/[a-z0-9-]{15,}/?$"
)

# Redirect URL line emitted by fetch_web_page when the server redirects
_REDIRECT_URL_RE = re.compile(r"^Redirect URL:\s*(https?://\S+)$", re.MULTILINE)
_BLOCKED_HOSTNAMES = {"localhost", "0.0.0.0", "::1", "metadata.google.internal"}


class _ValidatedHttpUrl(BaseModel):
    url: AnyHttpUrl


def _is_article_url(url: str) -> bool:
    """Return True if the URL looks like a specific article page (not a hub/homepage)."""
    path = urlparse(url).path.rstrip("/")
    last_segment = path.rsplit("/", 1)[-1] if path else ""
    return bool(_ARTICLE_URL_RE.search(url)) or (
        path.count("/") >= 2 and len(last_segment) >= 5
    )


def _extract_web_fetch_redirect_url(result_text: str) -> Optional[str]:
    """Extract the redirect URL line that fetch_web_page injects into its output."""
    match = _REDIRECT_URL_RE.search(result_text or "")
    if match:
        return match.group(1).strip().rstrip(".,;:")
    return None


def _safe_hostname(url: str) -> str:
    """Return hostname or empty string without propagating malformed URL errors."""
    try:
        return urlparse(url).hostname or ""
    except ValueError:
        return ""


def _normalize_http_url(url: str) -> str:
    """Return a normalized HTTP(S) URL or empty string when invalid."""
    candidate = (url or "").strip()
    if not candidate:
        return ""
    try:
        return str(_ValidatedHttpUrl(url=candidate).url)
    except ValidationError:
        return ""


def _validate_public_http_url(url: str) -> tuple[str, Optional[str]]:
    """Validate and normalize a public HTTP(S) URL for scraping flows."""
    normalized = _normalize_http_url(url)
    if not normalized:
        return "", "URL inválida"

    hostname = _safe_hostname(normalized)
    if not hostname:
        return "", "URL inválida"
    if hostname.lower() in _BLOCKED_HOSTNAMES:
        return "", f"Host no permitido: {hostname!r}"

    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved:
            return "", f"Dirección IP privada/reservada no permitida: {hostname!r}"
    except ValueError:
        pass

    return normalized, None
