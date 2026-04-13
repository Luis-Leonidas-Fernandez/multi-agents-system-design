"""
Infraestructura de scraping: cache, Playwright, helpers de extracción y guardado.

Exporta funciones usadas por los tools de scraping en agents.py.
Separado para reducir el tamaño de agents.py y facilitar el testing de infra.
"""
import asyncio
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from application.helpers.scraping_flow_helpers import (
    _CACHE_TTL_SECONDS,
    _SCRAPE_CACHE_MAX,
    _SCRAPE_CACHE,
    _cache_key,
    _get_cache,
    _set_cache,
    _validate_url,
    _clean_text,
    _truncate_text,
    _extract_text,
    _extract_links,
    _build_result,
)


# ==================== CONSTANTES ====================
_PLAYWRIGHT = None
_BROWSER = None
_PLAYWRIGHT_LOCK: threading.RLock = threading.RLock()

DATA_TRADING_DIR = Path(__file__).parent / "data_trading"


# ==================== DATA TRADING HELPERS ====================

def _ensure_data_trading_dir() -> Path:
    """Crea la carpeta data_trading si no existe."""
    DATA_TRADING_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_TRADING_DIR


def _safe_slug(s: str, max_len: int = 80) -> str:
    """Genera un slug seguro para nombres de archivo."""
    safe = "".join(c if c.isalnum() else "_" for c in s)[:max_len]
    return safe.strip("_") or "page"


def _save_json_bundle(page_url: str, captured: List[Dict[str, Any]]) -> str:
    """Guarda el bundle JSON de respuestas capturadas."""
    _ensure_data_trading_dir()
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    h = hashlib.sha256(page_url.encode("utf-8")).hexdigest()[:10]
    slug = _safe_slug(page_url)
    filename = f"{slug}_{h}_{int(time.time())}.json"
    out_path = DATA_TRADING_DIR / filename

    payload = {
        "page_url": page_url,
        "captured_at": now,
        "responses": captured,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def _is_json_content_type(ct: str) -> bool:
    """Verifica si el content-type es JSON (application/json o application/*+json)."""
    ct = (ct or "").lower()
    return ("application/json" in ct) or ct.endswith("+json") or ("+json;" in ct)


# ==================== HTTP HELPERS ====================

def _build_headers() -> Dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }


def _fetch_html(url: str, timeout_seconds: int = 10) -> bytes:
    import requests
    response = requests.get(url, headers=_build_headers(), timeout=timeout_seconds)
    response.raise_for_status()
    return response.content


# ==================== PLAYWRIGHT ====================

def _get_playwright():
    global _PLAYWRIGHT
    with _PLAYWRIGHT_LOCK:
        if _PLAYWRIGHT is None:
            import atexit
            from playwright.sync_api import sync_playwright  # pyright: ignore[reportMissingImports]
            _PLAYWRIGHT = sync_playwright().start()
            atexit.register(_shutdown_playwright)
    return _PLAYWRIGHT


def _get_browser():
    global _BROWSER
    with _PLAYWRIGHT_LOCK:
        if _BROWSER is None:
            _BROWSER = _get_playwright().chromium.launch(headless=True)
    return _BROWSER


def _shutdown_playwright() -> None:
    """Cierra browser y playwright al terminar el proceso (registrado via atexit)."""
    global _BROWSER, _PLAYWRIGHT
    try:
        if _BROWSER is not None:
            _BROWSER.close()
            _BROWSER = None
    except Exception:
        pass
    try:
        if _PLAYWRIGHT is not None:
            _PLAYWRIGHT.stop()
            _PLAYWRIGHT = None
    except Exception:
        pass


def _configure_page(page, block_resources: bool = True) -> None:
    if not block_resources:
        return

    def route_handler(route):
        request = route.request
        if request.resource_type in {"image", "media", "font"}:
            route.abort()
        else:
            route.continue_()

    page.route("**/*", route_handler)


# ==================== ASYNC SCRAPING CON CAPTURA JSON ====================

async def _scrape_dynamic_async(
    url: str,
    wait_for_selector: Optional[str] = None,
    extract_selector: Optional[str] = None,
    text_limit: int = 2000,
    timeout_ms: int = 20000,
    block_resources: bool = True,
    capture_json: bool = True,
    max_json_responses: int = 50,
    max_total_json_bytes: int = 2_000_000,  # 2MB total aprox
) -> Dict[str, Any]:
    """
    Scrapea una página con Playwright async y captura respuestas JSON.

    Guarda un bundle JSON en data_trading/ con todas las respuestas capturadas.
    """
    from playwright.async_api import async_playwright  # pyright: ignore[reportMissingImports]
    from bs4 import BeautifulSoup

    captured: List[Dict[str, Any]] = []
    tasks: List[asyncio.Task] = []
    total_bytes = 0
    json_bundle_path: Optional[str] = None

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()

        if block_resources:
            async def route_handler(route):
                rtype = route.request.resource_type
                if rtype in {"image", "font", "media"}:
                    await route.abort()
                else:
                    await route.continue_()
            await page.route("**/*", route_handler)

        # --- JSON capture handler ---
        async def _capture_response(resp):
            nonlocal total_bytes
            if len(captured) >= max_json_responses:
                return
            try:
                headers = await resp.all_headers()
                ct = headers.get("content-type", "")
                if not _is_json_content_type(ct):
                    return

                body = await resp.body()
                if not body:
                    data = None
                    body_len = 0
                else:
                    body_len = len(body)
                    if (total_bytes + body_len) > max_total_json_bytes:
                        return
                    total_bytes += body_len

                    try:
                        data = json.loads(body.decode("utf-8", errors="replace"))
                    except Exception:
                        data = body.decode("utf-8", errors="replace")[:5000]

                captured.append({
                    "url": resp.url,
                    "status": resp.status,
                    "headers": {"content-type": ct},
                    "data": data,
                })
            except Exception:
                return

        def on_response(resp):
            if not capture_json:
                return
            tasks.append(asyncio.create_task(_capture_response(resp)))

        if capture_json:
            page.on("response", on_response)

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

            if wait_for_selector:
                await page.wait_for_selector(wait_for_selector, timeout=timeout_ms)
            else:
                await page.wait_for_timeout(800)

            # Esperar para capturar calls tardías
            await page.wait_for_timeout(600)

            # Asegurar que terminaron tasks de captura
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Guardar bundle SIEMPRE (aunque esté vacío)
            json_bundle_path = _save_json_bundle(page.url, captured)

            # --- Extracción HTML normal ---
            html = await page.content()
            title = await page.title()

            soup = BeautifulSoup(html, "html.parser")
            node = soup.select_one(extract_selector) if extract_selector else soup
            main_text = node.get_text(" ", strip=True) if node else ""
            main_text = " ".join(main_text.split())
            if len(main_text) > text_limit:
                main_text = main_text[:text_limit] + "... [texto truncado]"

            links = []
            if node:
                for a in node.select("a[href]")[:30]:
                    t = " ".join(a.get_text(" ", strip=True).split())[:80]
                    href = a.get("href", "")
                    if href:
                        links.append({"text": t, "href": href})

            return {
                "requested_url": url,
                "url": page.url,
                "title": title,
                "rendered": True,
                "main_text": main_text,
                "links": links,
                "json_bundle_path": json_bundle_path,
                "json_captured_count": len(captured),
                "json_total_bytes": total_bytes,
            }

        finally:
            await context.close()
            await browser.close()


# ==================== SYNC SCRAPING SIN CAPTURA JSON ====================

def _scrape_page_sync(
    url: str,
    wait_for_selector: Optional[str] = None,
    extract_selector: Optional[str] = None,
    text_limit: int = 2000,
    timeout_ms: int = 20000,
    block_resources: bool = True,
) -> Dict[str, Any]:
    """
    Scrapea una página con Playwright sync (sin captura JSON).

    Usa el browser singleton sync (sync_playwright). Retorna el mismo
    contrato de dict que _scrape_dynamic_async para que los callers puedan
    intercambiar una implementación por la otra sin cambios adicionales.
    """
    from bs4 import BeautifulSoup

    browser = _get_browser()
    page = browser.new_page()
    _configure_page(page, block_resources=block_resources)

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        if wait_for_selector:
            page.wait_for_selector(wait_for_selector, timeout=timeout_ms)
        else:
            page.wait_for_timeout(800)

        final_url = page.url
        title = page.title()
        html = page.content()
    finally:
        page.close()

    soup = BeautifulSoup(html, "html.parser")
    node = soup.select_one(extract_selector) if extract_selector else soup
    main_text = node.get_text(" ", strip=True) if node else ""
    main_text = " ".join(main_text.split())
    if len(main_text) > text_limit:
        main_text = main_text[:text_limit] + "... [texto truncado]"

    links: List[Dict[str, Any]] = []
    target = node if node else soup
    for a in target.select("a[href]")[:30]:
        t = " ".join(a.get_text(" ", strip=True).split())[:80]
        href = a.get("href", "")
        if href:
            links.append({"text": t, "href": href})

    return {
        "requested_url": url,
        "url": final_url,
        "title": title,
        "rendered": True,
        "main_text": main_text,
        "links": links,
    }
