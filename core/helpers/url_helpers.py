"""URL utilities: article detection, redirect extraction, URL normalization."""
import re
from typing import Optional
from urllib.parse import urlparse

# Matches URLs that look like specific news articles:
# - date segment in path (/2026/04/20/ or /20260420- or 2026-04-20)
# - long slug (≥15 chars covers IDs like yjj2026040500456)
_ARTICLE_URL_RE = re.compile(
    r"/\d{4}/\d{2}/\d{2}/|/\d{8}[-_]|\d{4}-\d{2}-\d{2}"
    r"|/[a-z0-9-]{15,}/?$"
)

# Redirect URL line emitted by fetch_web_page when the server redirects
_REDIRECT_URL_RE = re.compile(r"^Redirect URL:\s*(https?://\S+)$", re.MULTILINE)


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
