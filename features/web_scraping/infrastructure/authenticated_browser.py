"""Contexto Playwright autenticado para verticales read-only."""
from __future__ import annotations

from dataclasses import dataclass, field
import errno
import os
from pathlib import Path
import threading
from typing import Any

from features.web_scraping.infrastructure.scraping_infra import _get_playwright


_BROWSER_ALIASES = {
    "chromium": "chromium",
    "chrome": "chrome",
    "google-chrome": "chrome",
    "edge": "msedge",
    "msedge": "msedge",
    "brave": "brave",
    "executable": "executable",
}
_REUSABLE_SESSION_LOCK = threading.RLock()
_REUSABLE_SESSIONS: dict[
    tuple[int, str, str, str, bool],
    "AuthenticatedBrowserSession",
] = {}


@dataclass(frozen=True)
class AuthenticatedBrowserLaunchConfig:
    """Selección explícita del binario sin reutilizar perfiles personales."""

    browser: str = "chromium"
    executable_path: Path | None = None
    source: str = "default"

    @classmethod
    def from_env(
        cls,
        *,
        browser: str | None = None,
        executable_path: str | Path | None = None,
        persisted_browser: str | None = None,
        persisted_executable_path: str | Path | None = None,
    ) -> "AuthenticatedBrowserLaunchConfig":
        cli_configured = browser is not None or executable_path is not None
        env_browser = (os.getenv("LINKEDIN_BROWSER") or "").strip()
        env_executable = (
            os.getenv("LINKEDIN_BROWSER_EXECUTABLE_PATH") or ""
        ).strip()
        persisted_browser_value = (persisted_browser or "").strip()
        persisted_executable_value = str(persisted_executable_path or "").strip()

        if cli_configured:
            requested = browser or ("executable" if executable_path else "chromium")
            raw_path = executable_path or ""
            source = "cli"
        elif env_browser or env_executable:
            requested = env_browser or ("executable" if env_executable else "chromium")
            raw_path = env_executable
            source = "env"
        elif persisted_browser_value or persisted_executable_value:
            requested = persisted_browser_value or (
                "executable" if persisted_executable_value else "chromium"
            )
            raw_path = persisted_executable_value
            source = "session_metadata"
        else:
            requested = "chromium"
            raw_path = ""
            source = "default"

        normalized = _BROWSER_ALIASES.get(str(requested).strip().lower())
        if normalized is None:
            allowed = ", ".join(sorted(_BROWSER_ALIASES))
            raise ValueError(
                f"LINKEDIN_BROWSER inválido. Valores permitidos: {allowed}."
            )

        resolved_path: Path | None = None
        if str(raw_path).strip():
            resolved_path = Path(str(raw_path)).expanduser()
            if not resolved_path.is_absolute():
                raise ValueError(
                    "LINKEDIN_BROWSER_EXECUTABLE_PATH debe ser una ruta absoluta."
                )
            if not resolved_path.is_file():
                raise ValueError(
                    "LINKEDIN_BROWSER_EXECUTABLE_PATH no apunta a un ejecutable existente."
                )
            if not os.access(resolved_path, os.X_OK):
                raise ValueError(
                    "LINKEDIN_BROWSER_EXECUTABLE_PATH no tiene permiso de ejecución."
                )

        if normalized in {"brave", "executable"} and resolved_path is None:
            raise ValueError(
                f"LINKEDIN_BROWSER={normalized} requiere "
                "LINKEDIN_BROWSER_EXECUTABLE_PATH."
            )
        return cls(
            browser=normalized,
            executable_path=resolved_path,
            source=source,
        )

    @property
    def uses_installed_browser(self) -> bool:
        return self.browser != "chromium" or self.executable_path is not None

    def launch_kwargs(self, *, headless: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"headless": headless}
        if self.executable_path is not None:
            kwargs["executable_path"] = str(self.executable_path)
        elif self.browser in {"chrome", "msedge"}:
            kwargs["channel"] = self.browser
        return kwargs

    def display_name(self) -> str:
        if self.executable_path is not None:
            return f"{self.browser} ({self.executable_path})"
        if self.browser == "chromium":
            return "Chromium bundled de Playwright"
        return self.browser


def configured_linkedin_headless(*, default: bool = False) -> bool:
    """Parsea el modo de ejecución sin aceptar valores ambiguos."""

    raw = (os.getenv("LINKEDIN_HEADLESS") or "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "LINKEDIN_HEADLESS debe ser true/false, 1/0, yes/no u on/off."
    )


class BrowserProfileInUseError(RuntimeError):
    """El perfil persistente ya está abierto por otro proceso."""


class BrowserProfileLock:
    """Lock advisory mantenido durante toda la vida del BrowserContext."""

    def __init__(self, profile_path: str | Path) -> None:
        self.profile_path = Path(profile_path)
        self.path = self.profile_path.parent / f".{self.profile_path.name}.lock"
        self._handle: Any | None = None

    def acquire(self) -> None:
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            os.chmod(self.path, 0o600)
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            handle.close()
            if isinstance(exc, BlockingIOError) or exc.errno in {
                errno.EACCES,
                errno.EAGAIN,
            }:
                raise BrowserProfileInUseError(
                    "El perfil dedicado de LinkedIn ya está en uso. Cerrá el "
                    "bootstrap o la búsqueda anterior y volvé a intentar."
                ) from exc
            raise
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None


@dataclass
class AuthenticatedBrowserJobPages:
    """Ownership de Pages creadas durante un único job autenticado."""

    preexisting_pages: tuple[object, ...]
    created_pages: list[object] = field(default_factory=list)
    closed_pages: list[object] = field(default_factory=list)
    closed: bool = False

    def owns(self, page: object) -> bool:
        return any(candidate is page for candidate in self.created_pages)

    def register(self, page: object) -> None:
        if any(candidate is page for candidate in self.preexisting_pages):
            return
        if not self.owns(page):
            self.created_pages.append(page)

    def close_owned_page(self, page: object) -> None:
        if not self.owns(page):
            return
        if any(candidate is page for candidate in self.closed_pages):
            return
        try:
            is_closed = getattr(page, "is_closed", None)
            if not (callable(is_closed) and is_closed() is True):
                page.close()
        except Exception:
            pass
        finally:
            self.closed_pages.append(page)

    def close_created_pages(self) -> None:
        if self.closed:
            return
        self.closed = True
        for page in reversed(self.created_pages):
            # El cleanup debe ser best-effort y nunca ocultar el resultado
            # o la excepción original del job.
            self.close_owned_page(page)


@dataclass
class AuthenticatedBrowserSession:
    context: object
    page: object
    browser: object | None = None
    profile_lock: BrowserProfileLock | None = None
    _active_job_pages: AuthenticatedBrowserJobPages | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def context_is_alive(self) -> bool:
        try:
            browser = self.browser or getattr(self.context, "browser", None)
            if browser is not None:
                is_connected = getattr(browser, "is_connected", None)
                if callable(is_connected) and is_connected() is False:
                    return False
            getattr(self.context, "pages")
            return True
        except Exception:
            return False

    def page_is_alive(self) -> bool:
        if not self.context_is_alive():
            return False
        try:
            is_closed = getattr(self.page, "is_closed", None)
            if callable(is_closed) and is_closed() is True:
                return False
            getattr(self.page, "url")
            return True
        except Exception:
            return False

    def begin_job_pages(self) -> AuthenticatedBrowserJobPages:
        """Captura Pages ajenas al job para no cerrarlas durante el cleanup."""

        if self._active_job_pages is not None:
            raise RuntimeError("authenticated browser job page scope already active")
        scope = AuthenticatedBrowserJobPages(
            preexisting_pages=tuple(getattr(self.context, "pages", ())),
        )
        self._active_job_pages = scope
        return scope

    def close_job_pages(self, scope: AuthenticatedBrowserJobPages) -> None:
        """Cierra sólo Pages creadas por el job; preserva context y browser."""

        if self._active_job_pages is not scope:
            raise RuntimeError("authenticated browser job page scope mismatch")
        try:
            scope.close_created_pages()
        finally:
            self._active_job_pages = None

    def new_job_page(self) -> object:
        """Abre una Page cuya vida queda limitada al scope del job activo."""

        scope = self._active_job_pages
        if scope is None:
            raise RuntimeError("authenticated browser job page scope not active")
        page = self.context.new_page()
        scope.register(page)
        try:
            is_closed = getattr(page, "is_closed", None)
            if callable(is_closed) and is_closed() is True:
                raise RuntimeError("job page closed")
            getattr(page, "url")
        except Exception:
            scope.close_owned_page(page)
            raise
        self.page = page
        return page

    def replace_page(self) -> object:
        """Recrea sólo el Page dentro del mismo perfil/contexto persistente."""

        if not self.context_is_alive():
            raise RuntimeError("persistent context closed")
        previous_page = self.page
        replacement = self.context.new_page()
        scope = self._active_job_pages
        if scope is not None:
            scope.register(replacement)
        try:
            is_closed = getattr(replacement, "is_closed", None)
            if callable(is_closed) and is_closed() is True:
                raise RuntimeError("replacement page closed")
            getattr(replacement, "url")
        except Exception:
            if scope is not None:
                scope.close_owned_page(replacement)
            else:
                try:
                    replacement.close()
                except Exception:
                    pass
            raise
        self.page = replacement
        if scope is not None and scope.owns(previous_page):
            scope.close_owned_page(previous_page)
        elif scope is None:
            try:
                previous_page.close()
            except Exception:
                pass
        return replacement

    def close(self) -> None:
        with _REUSABLE_SESSION_LOCK:
            stale_keys = [
                key
                for key, candidate in _REUSABLE_SESSIONS.items()
                if candidate is self
            ]
            for key in stale_keys:
                _REUSABLE_SESSIONS.pop(key, None)
        try:
            self.context.close()
        finally:
            try:
                if self.browser is not None:
                    self.browser.close()
            finally:
                if self.profile_lock is not None:
                    self.profile_lock.release()


def _reusable_session_key(
    *,
    profile: Path,
    config: AuthenticatedBrowserLaunchConfig,
    headless: bool,
) -> tuple[int, str, str, str, bool]:
    return (
        threading.get_ident(),
        str(profile.resolve()),
        config.browser,
        str(config.executable_path or ""),
        headless,
    )


def _shutdown_reusable_authenticated_sessions() -> None:
    with _REUSABLE_SESSION_LOCK:
        sessions = list(_REUSABLE_SESSIONS.values())
        _REUSABLE_SESSIONS.clear()
    for session in sessions:
        try:
            session.close()
        except Exception:
            pass


def open_persistent_authenticated_context(
    *,
    profile_path: str | Path,
    headless: bool = False,
    launch_config: AuthenticatedBrowserLaunchConfig | None = None,
    playwright: object | None = None,
    reuse: bool = False,
) -> AuthenticatedBrowserSession:
    """Abre el perfil dedicado completo; no inyecta cookies ni storage_state."""

    config = launch_config or AuthenticatedBrowserLaunchConfig.from_env()
    profile = Path(profile_path)
    reusable_key = _reusable_session_key(
        profile=profile,
        config=config,
        headless=headless,
    )
    if reuse:
        with _REUSABLE_SESSION_LOCK:
            cached = _REUSABLE_SESSIONS.get(reusable_key)
            if cached is not None and cached.context_is_alive():
                return cached
            if cached is not None:
                _REUSABLE_SESSIONS.pop(reusable_key, None)
                try:
                    cached.close()
                except Exception:
                    pass

    profile_lock = BrowserProfileLock(profile)
    profile_lock.acquire()
    context = None
    try:
        runtime = playwright or _get_playwright()
        context = runtime.chromium.launch_persistent_context(
            str(profile),
            **config.launch_kwargs(headless=headless),
            viewport={"width": 1366, "height": 900},
            locale="es-AR",
        )
        pages = tuple(context.pages)
        page = pages[0] if pages else context.new_page()
        session = AuthenticatedBrowserSession(
            context=context,
            page=page,
            profile_lock=profile_lock,
        )
        if reuse:
            import atexit

            with _REUSABLE_SESSION_LOCK:
                _REUSABLE_SESSIONS[reusable_key] = session
            atexit.register(_shutdown_reusable_authenticated_sessions)
        return session
    except Exception:
        try:
            if context is not None:
                context.close()
        finally:
            profile_lock.release()
        raise


__all__ = [
    "AuthenticatedBrowserLaunchConfig",
    "AuthenticatedBrowserJobPages",
    "AuthenticatedBrowserSession",
    "BrowserProfileInUseError",
    "BrowserProfileLock",
    "configured_linkedin_headless",
    "open_persistent_authenticated_context",
]
