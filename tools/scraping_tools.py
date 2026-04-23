"""Herramientas de scraping y fetch web para el sistema multi-agentes."""

from typing import Annotated, Optional
from urllib.parse import urlparse

from infra.web_cache import (
    _get_web_fetch_cache,
    _set_web_fetch_cache,
    _web_fetch_cache_key,
)

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import Field

from application.helpers.scraping_flow_helpers import (
    _validate_url,
    _cache_key,
    _get_cache,
    _set_cache,
    _build_result,
    _extract_text,
    _extract_links,
)
from tools.scraping_core import (
    _domain_allowed,
    _build_web_fetch_prompt,
)
from tools.web_fetch_helpers import build_web_fetch_draft
from infra import scraping_infra


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
    capture_json: Annotated[bool, Field(description="Si True, intercepta y guarda respuestas JSON de APIs en data_trading/")] = True,
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

        from application.helpers.config_flow_helpers import get_llm

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


__all__ = [
    "scrape_website_simple",
    "scrape_website_dynamic",
    "scrape_website_with_json_capture",
    "web_fetch",
    "fetch_web_page",
]
