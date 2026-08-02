"""Safe authenticated LinkedIn runtime diagnostics."""
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from features.web_scraping.infrastructure.linkedin_url_policy import (
    is_linkedin_auth_checkpoint,
)


def _safe_auth_diagnostic(page: object) -> dict[str, Any]:
    """Extrae señales booleanas y path; nunca cookies, query strings ni HTML."""

    raw_url = str(getattr(page, "url", "") or "")
    parsed = urlparse(raw_url)
    hostname = (parsed.hostname or "").lower()
    linkedin_host = hostname == "linkedin.com" or hostname.endswith(".linkedin.com")
    cookie_names: set[str] = set()
    try:
        cookie_names = {
            str(cookie.get("name") or "")
            for cookie in page.context.cookies()
            if isinstance(cookie, dict)
        }
    except Exception:
        pass

    def has_selector(selector: str) -> bool:
        try:
            return page.locator(selector).count() > 0
        except Exception:
            return False

    return {
        "final_path": (parsed.path or "/")[:160] if linkedin_host else "",
        "linkedin_host": linkedin_host,
        "is_auth_checkpoint": (
            is_linkedin_auth_checkpoint(raw_url) if linkedin_host else False
        ),
        "has_li_at": "li_at" in cookie_names,
        "has_global_nav": any(
            has_selector(selector)
            for selector in (
                "#global-nav",
                ".global-nav",
                "a[href*='/mynetwork/']",
                "a[href*='/messaging/']",
            )
        ),
        "has_login_form": any(
            has_selector(selector)
            for selector in (
                "form[action*='login']",
                "input[name='session_key']",
                "input[name='session_password']",
            )
        ),
    }
