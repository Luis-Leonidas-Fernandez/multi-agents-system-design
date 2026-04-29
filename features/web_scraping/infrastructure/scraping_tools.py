"""Herramientas de scraping y fetch web para el sistema multi-agentes."""

from typing import Annotated, Optional
from urllib.parse import urlparse

from features.web_scraping.infrastructure.web_cache import (
    _get_web_fetch_cache,
    _set_web_fetch_cache,
    _web_fetch_cache_key,
)

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import Field

from core.helpers.scraping_flow_helpers import (
    _validate_url,
    _cache_key,
    _get_cache,
    _set_cache,
    _build_result,
    _extract_text,
    _extract_links,
)
from features.web_scraping.infrastructure.scraping_core import (
    _domain_allowed,
    _build_web_fetch_prompt,
)
from features.web_scraping.infrastructure.web_fetch_helpers import build_web_fetch_draft
from features.web_scraping.infrastructure import scraping_infra


@tool
def scrape_website_simple(
    url: Annotated[str, Field(description="URL completa incluyendo https://, para páginas estáticas (blogs, docs, noticias)")],
    extract_text: Annotated[bool, Field(description="Si True, extrae el texto principal de la página")] = True,
    extract_links: Annotated[bool, Field(description="Si True, extrae los enlaces encontrados en la página")] = False,
    max_chars: Annotated[int, Field(description="Límite de caracteres del texto extraído", ge=100, le=10000)] = 2000,
) -> str:
    """Extrae información de una página web estática usando requests + BeautifulSoup."""
    url_error = _validate_url(url)
    if url_error:
        return f"URL rechazada: {url_error}"
    if not _domain_allowed(url):
        return f"URL rechazada: dominio no permitido por la configuración actual ({urlparse(url).hostname or 'desconocido'})"
    try:
        from bs4 import BeautifulSoup
        import requests

        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        fetched_url = str(response.url or url)
        if fetched_url != url and not _domain_allowed(fetched_url):
            return f"URL rechazada: dominio no permitido por la configuración actual ({urlparse(fetched_url).hostname or 'desconocido'})"

        soup = BeautifulSoup(response.content, "html.parser")
        text = None
        links_text = None
        total_links = 0

        if extract_text:
            text = _extract_text(soup, max_chars)

        if extract_links:
            total_links, links_text = _extract_links(soup, url)

        return _build_result(fetched_url, text, links_text, total_links)
    except Exception as e:
        return f"Error al procesar la pagina web: {str(e)}"


@tool
def scrape_website_dynamic(
    url: Annotated[str, Field(description="URL completa incluyendo https://, para páginas con JavaScript (precios, dashboards, SPAs)")],
    wait_for_selector: Annotated[Optional[str], Field(description="Selector CSS a esperar antes de extraer, ej: '.price', '#content'")] = None,
    extract_selector: Annotated[Optional[str], Field(description="Selector CSS del bloque específico a extraer, ej: 'main', '.article-body'")] = None,
    extract_text: Annotated[bool, Field(description="Si True, extrae el texto principal de la página")] = True,
    extract_links: Annotated[bool, Field(description="Si True, extrae los enlaces encontrados")] = False,
    max_chars: Annotated[int, Field(description="Límite de caracteres del texto extraído", ge=100, le=10000)] = 2000,
    block_resources: Annotated[bool, Field(description="Si True, bloquea imágenes y fonts para mayor velocidad")] = True,
    use_cache: Annotated[bool, Field(description="Si True, usa caché de 60s por URL para evitar requests repetidos")] = True,
) -> str:
    """Extrae información de páginas web con JavaScript usando Playwright (sync)."""
    url_error = _validate_url(url)
    if url_error:
        return f"URL rechazada: {url_error}"
    if not _domain_allowed(url):
        return f"URL rechazada: dominio no permitido por la configuración actual ({urlparse(url).hostname or 'desconocido'})"
    cache_params = {
        "wait_for_selector": wait_for_selector,
        "extract_selector": extract_selector,
        "extract_text": extract_text,
        "extract_links": extract_links,
        "max_chars": max_chars,
        "block_resources": block_resources,
    }
    cache_key = _cache_key(url, cache_params)
    if use_cache:
        cached = _get_cache(cache_key)
        if cached:
            return cached

    try:
        from bs4 import BeautifulSoup

        browser = scraping_infra._get_browser()
        page = browser.new_page()
        scraping_infra._configure_page(page, block_resources=block_resources)

        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        fetched_url = str(page.url or url)
        if fetched_url != url and not _domain_allowed(fetched_url):
            page.close()
            return f"URL rechazada: dominio no permitido por la configuración actual ({urlparse(fetched_url).hostname or 'desconocido'})"
        if wait_for_selector:
            page.wait_for_selector(wait_for_selector, timeout=30000)

        html = page.content()
        page.close()

        soup = BeautifulSoup(html, "html.parser")
        text = None
        links_text = None
        total_links = 0

        if extract_text:
            text = _extract_text(soup, max_chars, extract_selector=extract_selector)

        if extract_links:
            total_links, links_text = _extract_links(soup, url)

        result = _build_result(fetched_url, text, links_text, total_links)
        if use_cache:
            _set_cache(cache_key, result)
        return result
    except Exception as e:
        return f"Error al procesar la pagina web: {str(e)}"


@tool
async def scrape_website_with_json_capture(
    url: Annotated[str, Field(description="URL completa incluyendo https://, ideal para páginas con APIs/endpoints JSON (trading, precios, datos en tiempo real)")],
    wait_for_selector: Annotated[Optional[str], Field(description="Selector CSS a esperar antes de extraer, ej: '.price', '#ticker'")] = None,
    extract_selector: Annotated[Optional[str], Field(description="Selector CSS del bloque específico a extraer")] = None,
    max_chars: Annotated[int, Field(description="Límite de caracteres del texto extraído", ge=100, le=10000)] = 2000,
    capture_json: Annotated[bool, Field(description="Si True, intercepta y guarda respuestas JSON de APIs en data/web_scraping/data_trading/")] = True,
) -> str:
    """Extrae información de páginas con JS y captura endpoints JSON automáticamente."""
    url_error = _validate_url(url)
    if url_error:
        return f"URL rechazada: {url_error}"
    if not _domain_allowed(url):
        return f"URL rechazada: dominio no permitido por la configuración actual ({urlparse(url).hostname or 'desconocido'})"
    try:
        result = await scraping_infra._scrape_dynamic_async(
            url=url,
            wait_for_selector=wait_for_selector,
            extract_selector=extract_selector,
            text_limit=max_chars,
            capture_json=capture_json,
        )

        parts = [f"URL: {result['url']}"]
        if result.get("title"):
            parts.append(f"Titulo: {result['title']}")
        parts.append(f"\nTexto extraido:\n{result['main_text']}")

        if result.get("links"):
            links_str = "\n".join([f"- {l['text']}: {l['href']}" for l in result["links"][:20]])
            parts.append(f"\n\nEnlaces encontrados ({len(result['links'])} total):\n{links_str}")

        if result.get("json_bundle_path"):
            parts.append(f"\n\n[JSON Capturado]")
            parts.append(f"Archivo: {result['json_bundle_path']}")
            parts.append(f"Respuestas capturadas: {result['json_captured_count']}")
            parts.append(f"Total bytes JSON: {result['json_total_bytes']}")

        return "\n".join(parts)
    except Exception as e:
        return f"Error al procesar la pagina web: {str(e)}"


@tool
async def web_fetch(
    url: Annotated[str, Field(description="URL completa incluyendo https://, para recuperar y sintetizar el contenido de una página web")],
    prompt: Annotated[str, Field(description="Prompt que define qué información querés extraer o sintetizar de la página")],
    use_dynamic: Annotated[bool, Field(description="Si True, usa Playwright para páginas con JavaScript")] = True,
    wait_for_selector: Annotated[Optional[str], Field(description="Selector CSS a esperar antes de extraer")] = None,
    extract_selector: Annotated[Optional[str], Field(description="Selector CSS del bloque específico a extraer")] = None,
    max_chars: Annotated[int, Field(description="Límite de caracteres del contenido base", ge=100, le=20000)] = 8000,
    block_resources: Annotated[bool, Field(description="Si True, bloquea imágenes y fonts para mayor velocidad")] = True,
    use_cache: Annotated[bool, Field(description="Si True, cachea resultados por URL+prompt por 15 minutos")] = True,
) -> str:
    """Recupera una página web, la convierte a texto estilo markdown y la sintetiza con un modelo chico."""
    url_error = _validate_url(url)
    if url_error:
        return f"URL rechazada: {url_error}"
    if not _domain_allowed(url):
        return f"URL rechazada: dominio no permitido por la configuración actual ({urlparse(url).hostname or 'desconocido'})"

    cache_key = _web_fetch_cache_key(url, prompt, use_dynamic, wait_for_selector, extract_selector, max_chars)
    if use_cache:
        cached = _get_web_fetch_cache(cache_key)
        if cached:
            return cached

    try:
        draft = await build_web_fetch_draft(
            url=url,
            prompt=prompt,
            use_dynamic=use_dynamic,
            wait_for_selector=wait_for_selector,
            extract_selector=extract_selector,
            max_chars=max_chars,
            block_resources=block_resources,
        )

        if isinstance(draft, str):
            return draft

        final_url = draft.final_url
        markdown_content = draft.markdown_content

        from core.helpers.config_flow_helpers import get_llm

        llm = get_llm()
        synthesized = await llm.ainvoke([
            HumanMessage(content=_build_web_fetch_prompt(markdown_content, prompt, draft.is_preapproved))
        ])
        summary = getattr(synthesized, "content", str(synthesized)).strip()
        domain = urlparse(final_url).hostname or final_url
        result_text = f"{summary}\n\n<<<CITE_THIS: title={draft.title}|url={final_url}|domain={domain}>>>"

        if use_cache:
            _set_web_fetch_cache(cache_key, result_text)
        return result_text
    except Exception as e:
        return f"Error al procesar la pagina web: {str(e)}"


async def fetch_web_page(**kwargs) -> str:
    """Helper async directo para reutilizar la lógica WebFetch sin invocar el wrapper LangChain."""
    return await web_fetch.coroutine(**kwargs)  # pyright: ignore[reportAttributeAccessIssue]


@tool
def scrape_moodle_assignments(
    base_url: Annotated[str, Field(description="URL base del Moodle, ej: https://virtual.instituto.edu/mld-1. Si se omite, usa MOODLE_URL del .env")] = "",
) -> str:
    """Inicia sesión en Moodle con Playwright y extrae tareas pendientes (incluyendo vencidas). Requiere MOODLE_USERNAME y MOODLE_PASSWORD en variables de entorno."""
    import os
    from bs4 import BeautifulSoup

    moodle_url = (base_url or os.getenv("MOODLE_URL", "")).rstrip("/")
    moodle_user = os.getenv("MOODLE_USERNAME", "")
    moodle_pass = os.getenv("MOODLE_PASSWORD", "")

    if not moodle_url:
        return "Error: falta la URL de Moodle. Configurá MOODLE_URL en .env o pasá base_url."
    if not moodle_user or not moodle_pass:
        return "Error: falta MOODLE_USERNAME o MOODLE_PASSWORD en variables de entorno."

    browser = scraping_infra._get_browser()
    context = browser.new_context()
    html_dashboard = ""
    html_calendar = ""
    try:
        page = context.new_page()

        # --- Login ---
        page.goto(f"{moodle_url}/login/index.php", wait_until="domcontentloaded", timeout=30000)
        page.fill("#username", moodle_user)
        page.fill("#password", moodle_pass)
        page.click("#loginbtn")
        # Esperar a que el DOM de la siguiente página esté listo
        page.wait_for_load_state("domcontentloaded", timeout=20000)

        # Moodle puede mostrar "sesión en uso" con un botón de continuar
        try:
            btn = page.locator("input[type='submit']").first
            if btn.is_visible(timeout=2000):
                btn.click()
                page.wait_for_load_state("domcontentloaded", timeout=10000)
        except Exception:
            pass

        if "/login/" in page.url:
            return "Error: login fallido — verificá MOODLE_USERNAME y MOODLE_PASSWORD."

        # --- Dashboard (contenido AJAX) ---
        page.goto(f"{moodle_url}/my/", wait_until="domcontentloaded", timeout=30000)
        # Esperar a que las llamadas AJAX del dashboard terminen antes de leer el DOM
        try:
            page.wait_for_load_state("networkidle", timeout=12000)
        except Exception:
            pass
        # Intentar esperar por cualquiera de los selectores del timeline
        _dashboard_selectors = [
            "li.event-list-item",
            "[data-region='event-list-item']",
            "[data-region='event-list-content']",
            ".timeline-event-list",
            ".event-name",
        ]
        for _sel in _dashboard_selectors:
            try:
                page.wait_for_selector(_sel, timeout=5000)
                break
            except Exception:
                continue
        else:
            page.wait_for_timeout(3000)
        html_dashboard = page.content()

        # --- Calendario (server-rendered, no AJAX) ---
        page.goto(
            f"{moodle_url}/calendar/view.php?view=upcoming&lookahead=365",
            wait_until="domcontentloaded",
            timeout=30000,
        )
        page.wait_for_timeout(1500)
        html_calendar = page.content()
    except Exception as e:
        return f"Error durante el scraping de Moodle: {str(e)}"
    finally:
        context.close()

    # Volcar HTML para diagnóstico
    import tempfile, pathlib
    _tmp = pathlib.Path(tempfile.gettempdir())
    (_tmp / "moodle_debug_dashboard.html").write_text(html_dashboard, encoding="utf-8")
    (_tmp / "moodle_debug_calendar.html").write_text(html_calendar, encoding="utf-8")

    assignments: list[dict] = []
    seen: set[str] = set()

    def _add(name: str, date_str: str, course: str, href: str, status: str = "") -> None:
        key = name.strip().lower()
        if key and key not in seen:
            seen.add(key)
            assignments.append({"name": name, "date": date_str, "course": course, "url": href, "status": status})

    # --- Dashboard: múltiples selectores para distintas versiones de Moodle ---
    soup = BeautifulSoup(html_dashboard, "html.parser")

    # v3.x / v4.x: li.event-list-item
    for item in soup.select("li.event-list-item, [data-region='event-list-item']"):
        name_el = item.select_one("a.event-name, a[data-eventid], .event-name a")
        course_el = item.select_one("p.small.text-muted, .event-item-details .text-muted, [data-region='event-course']")
        date_el = item.select_one(".col-lg-5.text-xs-right, .event-item-details .col-sm-6, .badge-info, time")
        name = name_el.get_text(strip=True) if name_el else ""
        course = course_el.get_text(strip=True) if course_el else ""
        date_str = date_el.get_text(strip=True) if date_el else ""
        href = name_el.get("href", "") if name_el else ""
        group = item.find_parent("[data-region='event-list-group-container'], [data-region='day-group-container']")
        title_el = group.select_one("h5, h4, .event-listitem-date") if group else None
        group_label = title_el.get_text(strip=True) if title_el else ""
        status = "VENCIDA" if "vencid" in group_label.lower() else ""
        _add(name, date_str, course, href, status)

    # v4.x timeline alternativo: [data-region="event-list-content"]
    if not assignments:
        for item in soup.select("[data-region='event-list-content'] .event-name, .timeline-event-list .event-name"):
            name = item.get_text(strip=True)
            href = item.get("href", "") if item.name == "a" else (item.find("a") or {}).get("href", "")
            _add(name, "", "", href)

    # --- Calendario: múltiples selectores ---
    soup_cal = BeautifulSoup(html_calendar, "html.parser")

    # v3.x: .event con h3.referer
    for event in soup_cal.select(".event"):
        name_el = event.select_one("h3.referer a, h3 a, .card-title a, a[data-eventid]")
        date_el = event.select_one(".date, .calendar-event-date, time")
        header = event.select_one(".card-header, .card-block, .card-body")
        course_el = None
        if header:
            for a in header.find_all("a"):
                if "course" in (a.get("href", "") or ""):
                    course_el = a
                    break
        name = name_el.get_text(strip=True) if name_el else ""
        date_str = date_el.get_text(strip=True) if date_el else ""
        course = course_el.get_text(strip=True) if course_el else ""
        href = name_el.get("href", "") if name_el else ""
        _add(name, date_str, course, href)

    # v4.x: .calendar-event-template o [data-eventid]
    if not assignments:
        for event in soup_cal.select("[data-eventid], .event-item, .calendar_event_assign, .calendar_event_due"):
            name_el = event.select_one("a[href], .eventname a, .event-title a")
            date_el = event.select_one("time, .date, .when")
            name = name_el.get_text(strip=True) if name_el else event.get_text(strip=True)[:80]
            date_str = date_el.get_text(strip=True) if date_el else ""
            href = name_el.get("href", "") if name_el else ""
            _add(name, date_str, "", href)

    if not assignments:
        dash_snippet = soup.get_text(separator=" ", strip=True)[:400]
        cal_snippet = soup_cal.get_text(separator=" ", strip=True)[:400]
        return (
            "No se encontraron tareas en el dashboard ni en el calendario.\n"
            "HTML volcado en /tmp/moodle_debug_dashboard.html y /tmp/moodle_debug_calendar.html para diagnóstico.\n\n"
            f"[Dashboard snippet]\n{dash_snippet}\n\n"
            f"[Calendario snippet]\n{cal_snippet}"
        )

    total = len(assignments)
    lines = [f"TAREAS EN MOODLE  ({total} pendiente{'s' if total != 1 else ''})", "─" * 44, ""]
    for i, a in enumerate(assignments, 1):
        status_tag = f"  ⚠ VENCIDA" if a["status"] == "VENCIDA" else ""
        lines.append(f"{i}. {a['name']}{status_tag}")
        if a["course"]:
            lines.append(f"   Curso  {a['course']}")
        if a["date"]:
            lines.append(f"   Fecha  {a['date']}")
        if a["url"]:
            lines.append(f"   URL    {a['url']}")
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "scrape_website_simple",
    "scrape_website_dynamic",
    "scrape_website_with_json_capture",
    "web_fetch",
    "fetch_web_page",
    "scrape_moodle_assignments",
]
