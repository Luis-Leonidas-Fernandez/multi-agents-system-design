"""Bootstrap manual y visible de una sesión LinkedIn read-only."""
from __future__ import annotations

import argparse
import sys
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
from features.web_scraping.infrastructure.linkedin_session_store import LinkedInSessionStore
from features.web_scraping.infrastructure.linkedin_scraper import (
    LinkedInAuthRequiredError,
    LinkedInBlockedError,
    _validate_authenticated_page,
)


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        launch_config = AuthenticatedBrowserLaunchConfig.from_env(
            browser=args.browser,
            executable_path=args.executable_path,
        )
    except ValueError as exc:
        print(f"Configuración de navegador inválida: {exc}", file=sys.stderr)
        return 2

    store = LinkedInSessionStore()
    try:
        profile_path = store.resolve_profile_path(
            profile_dir=args.profile_dir,
            create=True,
        )
    except (OSError, ValueError) as exc:
        print(f"Perfil persistente inválido: {exc}", file=sys.stderr)
        return 2

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
            return 4
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
            return 2

        context = session.context
        page = session.page
        try:
            page.goto("https://www.linkedin.com/login", wait_until="domcontentloaded", timeout=60000)
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
                return 3
            page.goto("https://www.linkedin.com/jobs/", wait_until="domcontentloaded", timeout=60000)
            try:
                _validate_authenticated_page(page)
            except (LinkedInAuthRequiredError, LinkedInBlockedError):
                print(
                    "No se pudo validar la sesión. Completá login/2FA/checkpoint "
                    "manualmente y reintentá.",
                    file=sys.stderr,
                )
                return 1
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
