"""Bootstrap manual y visible de una sesión LinkedIn read-only."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright

from features.web_scraping.infrastructure.authenticated_browser import (
    AuthenticatedBrowserLaunchConfig,
    BrowserProfileInUseError,
    open_persistent_authenticated_context,
)
from features.web_scraping.infrastructure.linkedin_auth_diagnostics import (
    _safe_auth_diagnostic,
)
from features.web_scraping.infrastructure.linkedin_scraper import (
    LinkedInAuthRequiredError,
    LinkedInBlockedError,
    _validate_authenticated_page,
)
from features.web_scraping.infrastructure.linkedin_session_store import LinkedInSessionStore


_LINKEDIN_JOBS_URL = "https://www.linkedin.com/jobs/"
_EXIT_AUTH_NOT_READY = 1
_EXIT_INVALID_CONFIGURATION = 2
_EXIT_GOOGLE_OAUTH_BLOCKED = 3
_EXIT_PROFILE_IN_USE = 4
_EXIT_READINESS_TIMEOUT = 5
_READINESS_READY = "ready"
_READINESS_GOOGLE_BLOCKED = "google_blocked"
_READINESS_TIMEOUT = "timeout"
_READINESS_BROWSER_CLOSED = "browser_closed"
_GOOGLE_AUTOMATION_BLOCK_MARKERS = (
    "this browser or app may not be secure",
    "es posible que no sean seguros este navegador o la app",
    "es posible que este navegador o esta app no sean seguros",
)


def is_google_oauth_automation_block(url: str, body_text: str) -> bool:
    """Reconoce el bloqueo documentado sin intentar eludirlo."""

    parsed = urlparse(url or "")
    if parsed.scheme != "https" or (parsed.hostname or "").lower() != "accounts.google.com":
        return False
    normalized = " ".join((body_text or "").lower().split())
    return any(marker in normalized for marker in _GOOGLE_AUTOMATION_BLOCK_MARKERS)


def detect_google_oauth_automation_block(page: object) -> bool:
    try:
        body_text = page.locator("body").inner_text(timeout=3000)
    except Exception:
        body_text = ""
    return is_google_oauth_automation_block(str(page.url), body_text)


def detect_google_oauth_automation_block_in_context(context: object) -> bool:
    """Incluye popups abiertos por el botón de Google."""

    return any(
        detect_google_oauth_automation_block(candidate)
        for candidate in tuple(context.pages)
    )


def observe_linkedin_jobs_readiness(
    context: object,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float = 1.0,
    monotonic=time.monotonic,
    sleep=time.sleep,
) -> str:
    """Observa login manual hasta que LinkedIn Jobs sea usable o expire."""

    deadline = monotonic() + timeout_seconds
    jobs_navigation_requested: set[int] = set()
    while True:
        try:
            pages = tuple(context.pages)
        except Exception:
            return _READINESS_BROWSER_CLOSED
        if not pages:
            return _READINESS_BROWSER_CLOSED
        if detect_google_oauth_automation_block_in_context(context):
            return _READINESS_GOOGLE_BLOCKED

        for candidate in reversed(pages):
            try:
                _validate_authenticated_page(candidate)
            except (LinkedInAuthRequiredError, LinkedInBlockedError):
                pass
            except Exception:
                continue
            else:
                return _READINESS_READY

            diagnostic = _safe_auth_diagnostic(candidate)
            final_path = str(diagnostic.get("final_path") or "")
            authenticated = bool(
                diagnostic.get("has_li_at") or diagnostic.get("has_global_nav")
            )
            may_open_jobs = (
                diagnostic.get("linkedin_host") is True
                and authenticated
                and diagnostic.get("is_auth_checkpoint") is False
                and diagnostic.get("has_login_form") is False
                and not final_path.startswith("/jobs")
                and id(candidate) not in jobs_navigation_requested
            )
            if may_open_jobs:
                jobs_navigation_requested.add(id(candidate))
                try:
                    candidate.goto(
                        _LINKEDIN_JOBS_URL,
                        wait_until="domcontentloaded",
                        timeout=60000,
                    )
                    _validate_authenticated_page(candidate)
                except (LinkedInAuthRequiredError, LinkedInBlockedError):
                    pass
                except Exception:
                    continue
                else:
                    return _READINESS_READY

        if monotonic() >= deadline:
            return _READINESS_TIMEOUT
        sleep(max(0.0, poll_interval_seconds))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crea manualmente una sesión privada para LinkedIn Jobs."
    )
    parser.add_argument(
        "--browser",
        help=(
            "chromium (default), chrome, edge/msedge, brave o executable. "
            "También se configura con LINKEDIN_BROWSER."
        ),
    )
    parser.add_argument(
        "--executable-path",
        help=(
            "Ruta absoluta a Brave u otro Chromium instalado. "
            "También se configura con LINKEDIN_BROWSER_EXECUTABLE_PATH."
        ),
    )
    parser.add_argument(
        "--profile-dir",
        help=(
            "Subdirectorio persistente dedicado dentro del área privada de "
            "LinkedIn. También se configura con LINKEDIN_PROFILE_DIR."
        ),
    )
    parser.add_argument(
        "--observe-ready",
        action="store_true",
        help=(
            "Espera sin leer stdin hasta detectar que el login manual dejó "
            "LinkedIn Jobs autenticado y usable."
        ),
    )
    parser.add_argument(
        "--ready-timeout-seconds",
        type=float,
        default=300.0,
        help="Timeout explícito del observador de login manual (default: 300).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.ready_timeout_seconds <= 0:
        print("--ready-timeout-seconds debe ser mayor que cero.", file=sys.stderr)
        return _EXIT_INVALID_CONFIGURATION
    try:
        launch_config = AuthenticatedBrowserLaunchConfig.from_env(
            browser=args.browser,
            executable_path=args.executable_path,
        )
    except ValueError as exc:
        print(f"Configuración de navegador inválida: {exc}", file=sys.stderr)
        return _EXIT_INVALID_CONFIGURATION

    store = LinkedInSessionStore()
    try:
        profile_path = store.resolve_profile_path(
            profile_dir=args.profile_dir,
            create=True,
        )
    except (OSError, ValueError) as exc:
        print(f"Perfil persistente inválido: {exc}", file=sys.stderr)
        return _EXIT_INVALID_CONFIGURATION

    with sync_playwright() as playwright:
        try:
            session = open_persistent_authenticated_context(
                playwright=playwright,
                profile_path=profile_path,
                headless=False,
                launch_config=launch_config,
            )
        except BrowserProfileInUseError as exc:
            print(str(exc), file=sys.stderr)
            return _EXIT_PROFILE_IN_USE
        except Exception as exc:
            print(
                "No se pudo abrir el navegador configurado "
                f"({launch_config.display_name()}): {exc}",
                file=sys.stderr,
            )
            print(
                "Corregí LINKEDIN_BROWSER/LINKEDIN_BROWSER_EXECUTABLE_PATH. "
                "No se aplicó un fallback silencioso.",
                file=sys.stderr,
            )
            return _EXIT_INVALID_CONFIGURATION

        context = session.context
        page = session.page
        try:
            page.goto(
                _LINKEDIN_JOBS_URL if args.observe_ready else "https://www.linkedin.com/login",
                wait_until="domcontentloaded",
                timeout=60000,
            )
            print(
                f"Navegador: {launch_config.display_name()}.",
                flush=True,
            )
            print(f"Perfil dedicado: {profile_path}.", flush=True)
            if not launch_config.uses_installed_browser:
                print(
                    "Se usa el Chromium bundled como fallback explícito. "
                    "Podés seleccionar Chrome/Edge o una ruta a Brave con --browser.",
                    flush=True,
                )
            print(
                "Iniciá sesión MANUALMENTE con el email/teléfono y la contraseña de LinkedIn. "
                "No uses «Continuar con Google»: Google puede bloquear cualquier navegador "
                "controlado por automatización, incluso Chrome/Edge instalados.",
                flush=True,
            )
            if args.observe_ready:
                print(
                    "Completá manualmente login, 2FA o checkpoint. El proceso observará "
                    "la ventana y continuará sólo cuando LinkedIn Jobs esté autenticado "
                    "y usable; no ingresará credenciales ni evadirá verificaciones.",
                    flush=True,
                )
                readiness = observe_linkedin_jobs_readiness(
                    context,
                    timeout_seconds=args.ready_timeout_seconds,
                )
                if readiness == _READINESS_GOOGLE_BLOCKED:
                    print(
                        "Google bloqueó el OAuth porque detectó un navegador controlado por "
                        "automatización. Esto no se puede ni se debe evadir. Volvé a ejecutar "
                        "el bootstrap y usá el login directo de LinkedIn con contraseña.",
                        file=sys.stderr,
                    )
                    return _EXIT_GOOGLE_OAUTH_BLOCKED
                if readiness == _READINESS_TIMEOUT:
                    print(
                        "El tiempo de espera terminó antes de que LinkedIn Jobs quedara "
                        "autenticado y usable. Completá login/2FA/checkpoint manualmente "
                        "y reintentá.",
                        file=sys.stderr,
                    )
                    return _EXIT_READINESS_TIMEOUT
                if readiness != _READINESS_READY:
                    print(
                        "El navegador se cerró antes de validar LinkedIn Jobs.",
                        file=sys.stderr,
                    )
                    return _EXIT_AUTH_NOT_READY
            else:
                print(
                    "Si tu cuenta sólo usa Google, restablecé primero una contraseña de LinkedIn "
                    "desde tu navegador habitual y después repetí este bootstrap. "
                    "No cierres la ventana; al terminar, volvé acá y presioná Enter.",
                    flush=True,
                )
                input()
            if detect_google_oauth_automation_block_in_context(context):
                print(
                    "Google bloqueó el OAuth porque detectó un navegador controlado por "
                    "automatización. Esto no se puede ni se debe evadir. Volvé a ejecutar "
                    "el bootstrap y usá el login directo de LinkedIn con contraseña.",
                    file=sys.stderr,
                )
                return _EXIT_GOOGLE_OAUTH_BLOCKED
            if not args.observe_ready:
                page.goto(_LINKEDIN_JOBS_URL, wait_until="domcontentloaded", timeout=60000)
                try:
                    _validate_authenticated_page(page)
                except (LinkedInAuthRequiredError, LinkedInBlockedError):
                    print(
                        "No se pudo validar la sesión. Completá login/2FA/checkpoint "
                        "manualmente y reintentá.",
                        file=sys.stderr,
                    )
                    return _EXIT_AUTH_NOT_READY
            store.save_from_context(
                context,
                launch_config=launch_config,
                profile_path=profile_path,
            )
        finally:
            session.close()
    print("Sesión LinkedIn guardada de forma privada.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
