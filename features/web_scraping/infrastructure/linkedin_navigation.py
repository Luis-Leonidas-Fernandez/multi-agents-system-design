"""Navigation and browser-state helpers for the LinkedIn jobs scraper."""
from __future__ import annotations

import re
import time

from features.web_scraping.infrastructure.linkedin_url_policy import (
    is_linkedin_auth_checkpoint,
    validate_linkedin_jobs_url,
)


_HYDRATION_POLL_INTERVALS_MS = (100, 150, 250, 400, 600, 800, 1000)


_DETAIL_HYDRATION_MAX_MS = 6000


_LOGIN_SIGNAL_SELECTORS = (
    "form[action*='login']",
    "input[name='session_key']",
    "input[name='session_password']",
)


_BLOCK_SIGNAL_SELECTORS = (
    "iframe[src*='captcha']",
    "[id*='captcha']",
    "[class*='captcha']",
    ".challenge-dialog",
    "text=Security verification",
    "text=Verificación de seguridad",
    "text=CAPTCHA",
)


class LinkedInAuthRequiredError(RuntimeError):
    pass


class LinkedInBlockedError(RuntimeError):
    pass


class LinkedInDetailPanelError(RuntimeError):
    def __init__(self, reason: str, *, safe_label: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.safe_label = safe_label


def _safe_error_label(exc: Exception) -> str:
    raw_message = str(exc or "")
    message = raw_message.casefold()
    exception_type = type(exc).__name__
    exception_type_folded = exception_type.casefold()
    if (
        "targetclosed" in exception_type_folded
        or "target closed" in message
        or "has been closed" in message
        or "persistent context closed" in message
        or "replacement page closed" in message
    ):
        category = "target_closed"
    elif "err_name_not_resolved" in message:
        category = "dns"
    elif "err_internet_disconnected" in message:
        category = "offline"
    elif any(
        token in message
        for token in (
            "err_connection_reset",
            "err_connection_refused",
            "err_connection_closed",
            "err_connection_timed_out",
        )
    ):
        category = "connection"
    elif "err_aborted" in message:
        category = "network_aborted"
    elif chromium_error := re.search(r"\bERR_[A-Z_]+\b", raw_message.upper()):
        category = f"chromium_{chromium_error.group(0).lower()}"
    elif "timeout" in message:
        category = "timeout"
    elif any(token in message for token in ("net::", "network", "connection")):
        category = "network"
    elif any(token in message for token in ("429", "rate limit", "too many requests")):
        category = "rate_limited"
    elif any(token in message for token in ("navigation", "goto")):
        category = "navigation"
    else:
        category = "runtime"
    return f"{exception_type}:{category}"


def _error_category(label: str) -> str:
    return label.rsplit(":", 1)[-1] if ":" in label else "runtime"


def _is_page_recoverable_error(label: str) -> bool:
    category = _error_category(label)
    return category.startswith("chromium_err_") or category in {
        "dns",
        "offline",
        "connection",
        "network_aborted",
        "network",
        "target_closed",
    }


def _is_http_response_code_failure(label: str) -> bool:
    return _error_category(label) == "chromium_err_http_response_code_failure"


def _safe_page_pause(page, milliseconds: int) -> None:
    try:
        page.wait_for_timeout(milliseconds)
    except Exception:
        time.sleep(milliseconds / 1000)


def _locator_has_signal(page, selector: str, *, require_text: bool = False) -> bool:
    try:
        locator = page.locator(selector).first
        if not locator.count():
            return False
        try:
            if not locator.is_visible(timeout=0):
                return False
        except Exception:
            pass
        if not require_text:
            return True
        return bool(
            re.sub(
                r"\s+",
                " ",
                locator.inner_text(timeout=500) or "",
            ).strip()
        )
    except Exception:
        return False


def _raise_for_terminal_page_signal(page) -> None:
    current_url = str(getattr(page, "url", "") or "")
    if is_linkedin_auth_checkpoint(current_url):
        raise LinkedInAuthRequiredError(
            "La sesión LinkedIn requiere login, 2FA o checkpoint manual."
        )
    try:
        validate_linkedin_jobs_url(current_url)
    except ValueError as exc:
        raise LinkedInBlockedError(
            "LinkedIn redirigió fuera del área de empleos permitida."
        ) from exc
    if any(_locator_has_signal(page, selector) for selector in _LOGIN_SIGNAL_SELECTORS):
        raise LinkedInAuthRequiredError(
            "La sesión LinkedIn requiere login manual."
        )
    if any(_locator_has_signal(page, selector) for selector in _BLOCK_SIGNAL_SELECTORS):
        raise LinkedInBlockedError(
            "LinkedIn solicitó una verificación manual."
        )


def _validate_authenticated_page(page) -> None:
    current_url = str(page.url or "")
    if is_linkedin_auth_checkpoint(current_url):
        raise LinkedInAuthRequiredError(
            "La sesión LinkedIn requiere login, 2FA o checkpoint manual."
        )
    try:
        validate_linkedin_jobs_url(current_url)
    except ValueError as exc:
        raise LinkedInBlockedError(
            "LinkedIn redirigió fuera del área de empleos permitida."
        ) from exc
    body_text = (page.locator("body").inner_text(timeout=5000) or "").lower()
    if any(token in body_text for token in ("security verification", "verificación de seguridad", "captcha")):
        raise LinkedInBlockedError(
            "LinkedIn solicitó una verificación manual. El scraper fue detenido."
        )
    cookie_names: set[str] = set()
    try:
        cookie_names = {
            str(cookie.get("name") or "")
            for cookie in page.context.cookies()
            if isinstance(cookie, dict)
        }
    except Exception:
        pass
    authenticated_marker = False
    for selector in (
        "#global-nav",
        ".global-nav",
        "a[href*='/mynetwork/']",
        "a[href*='/messaging/']",
    ):
        try:
            if page.locator(selector).count() > 0:
                authenticated_marker = True
                break
        except Exception:
            continue
    if "li_at" not in cookie_names and not authenticated_marker:
        raise LinkedInAuthRequiredError(
            "No se encontró una sesión autenticada válida de LinkedIn."
        )



__all__ = [
    "LinkedInAuthRequiredError",
    "LinkedInBlockedError",
    "LinkedInDetailPanelError",
    "_DETAIL_HYDRATION_MAX_MS",
    "_HYDRATION_POLL_INTERVALS_MS",
    "_error_category",
    "_is_http_response_code_failure",
    "_is_page_recoverable_error",
    "_locator_has_signal",
    "_raise_for_terminal_page_signal",
    "_safe_error_label",
    "_safe_page_pause",
    "_validate_authenticated_page",
]
