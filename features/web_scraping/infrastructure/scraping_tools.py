"""Herramientas de scraping y fetch web para el sistema multi-agentes."""

from typing import Annotated, Optional
import json
from pathlib import Path

from features.web_scraping.infrastructure.web_cache import (
    _get_web_fetch_cache,
    _set_web_fetch_cache,
    _web_fetch_cache_key,
)

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import Field

from core.helpers.scraping_flow_helpers import (
    _cache_key,
    _get_cache,
    _set_cache,
    _build_result,
    _extract_text,
    _extract_links,
)
from core.helpers.url_helpers import _safe_hostname, _validate_public_http_url
from features.web_scraping.infrastructure.scraping_core import (
    _domain_allowed,
    _build_web_fetch_prompt,
)
from features.web_scraping.infrastructure.web_fetch_helpers import build_web_fetch_draft
from features.web_scraping.infrastructure import scraping_infra
from features.web_scraping.application.moodle_artifacts import get_moodle_artifact_dir


@tool
def scrape_website_simple(
    url: Annotated[str, Field(description="URL completa incluyendo https://, para páginas estáticas (blogs, docs, noticias)")],
    extract_text: Annotated[bool, Field(description="Si True, extrae el texto principal de la página")] = True,
    extract_links: Annotated[bool, Field(description="Si True, extrae los enlaces encontrados en la página")] = False,
    max_chars: Annotated[int, Field(description="Límite de caracteres del texto extraído", ge=100, le=10000)] = 2000,
) -> str:
    """Extrae información de una página web estática usando requests + BeautifulSoup."""
    normalized_url, url_error = _validate_public_http_url(url)
    if url_error:
        return f"URL rechazada: {url_error}"
    url = normalized_url
    if not _domain_allowed(url):
        return f"URL rechazada: dominio no permitido por la configuración actual ({_safe_hostname(url) or 'desconocido'})"
    try:
        from bs4 import BeautifulSoup
        import requests

        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        fetched_url = str(response.url or url)
        if fetched_url != url and not _domain_allowed(fetched_url):
            return f"URL rechazada: dominio no permitido por la configuración actual ({_safe_hostname(fetched_url) or 'desconocido'})"

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
    normalized_url, url_error = _validate_public_http_url(url)
    if url_error:
        return f"URL rechazada: {url_error}"
    url = normalized_url
    if not _domain_allowed(url):
        return f"URL rechazada: dominio no permitido por la configuración actual ({_safe_hostname(url) or 'desconocido'})"
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
            return f"URL rechazada: dominio no permitido por la configuración actual ({_safe_hostname(fetched_url) or 'desconocido'})"
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
    normalized_url, url_error = _validate_public_http_url(url)
    if url_error:
        return f"URL rechazada: {url_error}"
    url = normalized_url
    if not _domain_allowed(url):
        return f"URL rechazada: dominio no permitido por la configuración actual ({_safe_hostname(url) or 'desconocido'})"
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
    normalized_url, url_error = _validate_public_http_url(url)
    if url_error:
        return f"URL rechazada: {url_error}"
    url = normalized_url
    if not _domain_allowed(url):
        return f"URL rechazada: dominio no permitido por la configuración actual ({_safe_hostname(url) or 'desconocido'})"

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
        domain = _safe_hostname(final_url) or final_url
        result_text = f"{summary}\n\n<<<CITE_THIS: title={draft.title}|url={final_url}|domain={domain}>>>"

        if use_cache:
            _set_web_fetch_cache(cache_key, result_text)
        return result_text
    except Exception as e:
        return f"Error al procesar la pagina web: {str(e)}"


async def fetch_web_page(**kwargs) -> str:
    """Helper async directo para reutilizar la lógica WebFetch sin invocar el wrapper LangChain."""
    return await web_fetch.coroutine(**kwargs)  # pyright: ignore[reportAttributeAccessIssue]


def _moodle_absolute_url(base_url: str, href: str) -> str:
    from urllib.parse import urljoin

    return urljoin(f"{base_url.rstrip('/')}/", (href or "").strip())


def _basename_from_url(url: str) -> str:
    from urllib.parse import urlparse, unquote

    path = unquote(urlparse(url).path or "")
    return path.rsplit("/", 1)[-1] if path else ""


def _moodle_debug_dir() -> Path:
    debug_dir = get_moodle_artifact_dir() / "debug" / "login"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def _write_moodle_debug_log(name: str, payload: object) -> None:
    debug_dir = _moodle_debug_dir()
    path = debug_dir / name
    if isinstance(payload, (dict, list)):
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    path.write_text(str(payload), encoding="utf-8")


def _write_moodle_debug_screenshot(name: str, page) -> None:
    debug_dir = _moodle_debug_dir()
    path = debug_dir / name
    try:
        page.screenshot(path=str(path), full_page=True)
    except Exception as exc:
        _write_moodle_debug_log(
            f"{Path(name).stem}_screenshot_error.json",
            {"name": name, "error": str(exc)},
        )


def _safe_page_content(page) -> str:
    try:
        return str(page.content() or "")
    except Exception:
        return ""


def _safe_page_text(page, *, limit: int = 4000) -> str:
    try:
        return str(page.locator("body").inner_text(timeout=5000) or "")[:limit]
    except Exception:
        return ""


def _extract_login_error_message(*, html: str = "", text: str = "") -> str:
    import re

    combined = "\n".join(part for part in (text, html) if part).lower()
    patterns = (
        r"(datos de acceso incorrectos[^<\n]*)",
        r"(nombre de usuario o contraseña[^<\n]*)",
        r"(invalid login[^<\n]*)",
        r"(invalid username or password[^<\n]*)",
        r"(login failed[^<\n]*)",
        r"(acceso inválido[^<\n]*)",
        r"(credenciales inválidas[^<\n]*)",
    )
    for pattern in patterns:
        match = re.search(pattern, combined, re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    return ""


def _extract_login_form_debug(*, html: str, current_url: str) -> dict[str, object]:
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin

    soup = BeautifulSoup(html or "", "html.parser")
    forms_payload: list[dict[str, object]] = []
    sso_providers: list[str] = []
    login_form_found = False
    token_fields: list[str] = []

    for idx, form in enumerate(soup.find_all("form"), start=1):
        action = str(form.get("action") or "").strip()
        resolved_action = urljoin(current_url, action) if action else current_url
        method = str(form.get("method") or "get").strip().lower()
        inputs = form.find_all(["input", "button", "select"])
        field_names: list[str] = []
        field_types: list[str] = []
        for tag in inputs:
            field_name = str(tag.get("name") or tag.get("id") or "").strip()
            field_type = str(tag.get("type") or tag.name or "").strip().lower()
            if field_name:
                field_names.append(field_name)
            if field_type:
                field_types.append(field_type)
            if field_name and "token" in field_name.lower():
                token_fields.append(field_name)
        has_password = "password" in field_types or any("password" in name.lower() for name in field_names)
        has_username = any(
            token in name.lower()
            for name in field_names
            for token in ("username", "user", "email", "login")
        )
        is_login_form = has_password or has_username or "/login/" in action or "login" in action.lower()
        login_form_found = login_form_found or is_login_form
        forms_payload.append(
            {
                "index": idx,
                "action": action,
                "resolved_action": resolved_action,
                "method": method,
                "field_names": field_names[:20],
                "field_types": field_types[:20],
                "has_password": has_password,
                "has_username": has_username,
                "is_login_form": is_login_form,
            }
        )

    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href") or "").strip().lower()
        text = " ".join(anchor.stripped_strings).lower()
        if any(token in href or token in text for token in ("sso", "oauth", "microsoft", "google", "office365", "saml", "openid", "cas", "auth/")):
            provider = text or href
            if provider and provider not in sso_providers:
                sso_providers.append(provider[:160])

    second_form_detected = len(forms_payload) > 1
    return {
        "current_url": current_url,
        "form_count": len(forms_payload),
        "second_form_detected": second_form_detected,
        "login_form_found": login_form_found,
        "forms": forms_payload,
        "sso_detected": bool(sso_providers),
        "sso_providers": sso_providers[:10],
        "token_fields": sorted(set(token_fields)),
    }


def _context_cookie_debug(context, *, expected_root_host: str) -> dict[str, object]:
    try:
        cookies = context.cookies()
    except Exception as exc:
        return {"cookie_error": str(exc), "cookie_count": 0, "cookies": []}

    filtered: list[dict[str, object]] = []
    for cookie in cookies:
        domain = str(cookie.get("domain") or "").lower().lstrip(".")
        if expected_root_host and expected_root_host not in domain:
            continue
        filtered.append(
            {
                "name": str(cookie.get("name") or ""),
                "domain": domain,
                "path": str(cookie.get("path") or ""),
                "expires": cookie.get("expires"),
                "httpOnly": bool(cookie.get("httpOnly")),
                "secure": bool(cookie.get("secure")),
                "sameSite": str(cookie.get("sameSite") or ""),
                "value_length": len(str(cookie.get("value") or "")),
            }
        )
    return {"cookie_count": len(filtered), "cookies": filtered}


def _normalized_host(url: str) -> str:
    return (_safe_hostname(url) or "").lower().removeprefix("www.")


def _canonicalize_moodle_crawl_url(url: str, *, root_host: str) -> str:
    from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

    raw = (url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    normalized_host = _normalized_host(raw)
    filtered_items = query_items
    if normalized_host and normalized_host == (root_host or ""):
        filtered_items = [(key, value) for key, value in query_items if key.lower() != "lang"]
    normalized_query = urlencode(filtered_items, doseq=True)
    normalized_path = parsed.path or "/"
    if normalized_path != "/":
        normalized_path = normalized_path.rstrip("/") or "/"
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            normalized_path,
            parsed.params,
            normalized_query,
            "",
        )
    )


def _warning_bucket(warning: str) -> str:
    lowered = (warning or "").strip().lower()
    if not lowered:
        return "unknown"
    return lowered.split(":", 1)[0]


def _summarize_moodle_audit_stats(
    pages: list[dict[str, object]],
    warnings: list[str],
    *,
    assignment_count: int,
    visited_count_raw: int,
) -> dict[str, object]:
    resource_type_counts: dict[str, int] = {}
    warning_types: dict[str, int] = {}
    external_redirect_count = 0
    download_document_count = 0
    for page in pages:
        if not isinstance(page, dict):
            continue
        resource_type = str(page.get("resource_type") or "unknown").strip() or "unknown"
        resource_type_counts[resource_type] = resource_type_counts.get(resource_type, 0) + 1
        final_url = str(page.get("final_url") or page.get("url") or "").strip()
        if final_url and _normalized_host(final_url) and _normalized_host(final_url) != "":
            if _normalized_host(final_url) != _normalized_host(str(page.get("url") or "")) and page.get("page_kind") == "linked_resource":
                external_redirect_count += 1
        attachments = page.get("attachments")
        if resource_type == "document" and isinstance(attachments, list) and attachments:
            download_document_count += 1
    for warning in warnings:
        bucket = _warning_bucket(warning)
        warning_types[bucket] = warning_types.get(bucket, 0) + 1
    return {
        "stats": {
            "visited_count_raw": visited_count_raw,
            "retained_page_count": len(pages),
            "external_redirect_count": external_redirect_count,
            "download_document_count": download_document_count,
            "assignment_like_count": assignment_count,
        },
        "resource_type_counts": resource_type_counts,
        "warning_types": warning_types,
    }


def _mime_hint_from_name(name: str) -> str:
    filename = (name or "").strip().lower()
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1]


def _kind_from_file_metadata(filename: str, content_type: str = "") -> str:
    lowered = f"{filename} {content_type}".lower()
    if any(token in lowered for token in ("pdf", "msword", "wordprocessingml", "presentation", "spreadsheet", "excel", "powerpoint", "text/csv")):
        return "document"
    if any(token in lowered for token in ("zip", "rar", "7z", "tar", "gzip")):
        return "archive"
    if any(token in lowered for token in ("image/", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")):
        return "image"
    if any(token in lowered for token in ("video/", ".mp4", ".webm", ".mov", ".m3u8")):
        return "video"
    if any(token in lowered for token in ("audio/", ".mp3", ".wav", ".ogg", ".m4a")):
        return "audio"
    return "file"


def _filename_from_content_disposition(content_disposition: str) -> str:
    import re
    from urllib.parse import unquote

    header = (content_disposition or "").strip()
    if not header:
        return ""
    match = re.search(r"filename\\*=UTF-8''([^;]+)", header, flags=re.IGNORECASE)
    if match:
        return unquote(match.group(1).strip().strip('"'))
    match = re.search(r'filename="?([^";]+)"?', header, flags=re.IGNORECASE)
    if match:
        return unquote(match.group(1).strip())
    return ""


def _int_from_header(value: str) -> Optional[int]:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _build_file_audit_record(
    *,
    label: str,
    url: str,
    final_url: str = "",
    mime_hint: str = "",
    kind: str = "file",
    content_type: str = "",
    content_length: Optional[int] = None,
    content_disposition: str = "",
    status_code: Optional[int] = None,
    redirect_chain: Optional[list[str]] = None,
) -> dict[str, object]:
    resolved_final_url = final_url or url
    disposition_filename = _filename_from_content_disposition(content_disposition)
    filename = disposition_filename or _basename_from_url(resolved_final_url) or _basename_from_url(url)
    resolved_mime_hint = mime_hint or _mime_hint_from_name(filename)
    chain = [item for item in (redirect_chain or [url]) if item]
    if resolved_final_url and resolved_final_url not in chain:
        chain.append(resolved_final_url)
    resolved_kind = kind if kind != "file" else _kind_from_file_metadata(filename, content_type)
    return {
        "label": label,
        "filename": filename,
        "url": url,
        "final_url": resolved_final_url,
        "redirect_target": resolved_final_url if resolved_final_url != url else "",
        "redirect_chain": chain,
        "mime_hint": resolved_mime_hint,
        "kind": resolved_kind,
        "content_type": (content_type or "").split(";", 1)[0].strip(),
        "content_length": content_length,
        "content_disposition": content_disposition or "",
        "status_code": status_code,
        "is_download": "attachment" in (content_disposition or "").lower() or resolved_kind in {"document", "archive", "audio", "video"},
    }


def _requests_session_from_playwright_context(context):
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
    try:
        for cookie in context.cookies():
            session.cookies.set(
                cookie.get("name", ""),
                cookie.get("value", ""),
                domain=cookie.get("domain"),
                path=cookie.get("path", "/"),
            )
    except Exception:
        pass
    return session


def _fetch_file_http_metadata(session, source_url: str) -> dict[str, object]:
    response = None
    try:
        response = session.head(source_url, allow_redirects=True, timeout=10)
        if response.status_code >= 400 or not (response.headers.get("content-type") or response.headers.get("content-disposition")):
            response.close()
            response = session.get(source_url, allow_redirects=True, timeout=10, stream=True)
        final_url = str(getattr(response, "url", "") or source_url)
        headers = response.headers
        redirect_chain = [item.url for item in getattr(response, "history", []) if getattr(item, "url", "")]
        redirect_chain.append(final_url)
        return _build_file_audit_record(
            label="",
            url=source_url,
            final_url=final_url,
            content_type=headers.get("content-type", ""),
            content_length=_int_from_header(headers.get("content-length", "")),
            content_disposition=headers.get("content-disposition", ""),
            status_code=getattr(response, "status_code", None),
            redirect_chain=redirect_chain,
        )
    finally:
        try:
            if response is not None:
                response.close()
        except Exception:
            pass


def _youtube_watch_and_preview(url: str) -> tuple[str, str]:
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    video_id = ""
    if "youtu.be" in host:
        video_id = (parsed.path or "").strip("/")
    elif "youtube.com" in host:
        if parsed.path.startswith("/embed/"):
            video_id = parsed.path.split("/embed/", 1)[1].split("/", 1)[0]
        else:
            video_id = parse_qs(parsed.query).get("v", [""])[0]
    if not video_id:
        return "", ""
    return (
        f"https://www.youtube.com/watch?v={video_id}",
        f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
    )


def _extract_google_slides_id(url: str) -> str:
    import re

    match = re.search(r"/presentation/d/([^/]+)", url or "", flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _extract_google_drive_id(url: str) -> str:
    import re

    match = re.search(r"/file/d/([^/]+)", url or "", flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _infer_google_slides_slide_count(text: str) -> Optional[int]:
    import re

    numbers = [int(value) for value in re.findall(r"\b\d+\b", text or "")]
    if not numbers:
        return None
    best = 0
    current = 0
    expected = 1
    for number in numbers:
        if number == expected:
            current += 1
            expected += 1
            best = max(best, current)
        elif number == 1:
            current = 1
            expected = 2
            best = max(best, current)
    return best or None


def _extract_google_slides_resource_metadata(soup, final_url: str, text_excerpt: str) -> dict[str, object]:
    canonical_url = (final_url or "").split("#", 1)[0]
    htmlpresent_url = ""
    access_url = ""
    for anchor in soup.select("a[href]"):
        href = _moodle_absolute_url(final_url, anchor.get("href", ""))
        label = anchor.get_text(" ", strip=True)
        if not htmlpresent_url and "/htmlpresent" in href:
            htmlpresent_url = href
        if not access_url and "accounts.google.com/ServiceLogin" in href:
            access_url = href
        if htmlpresent_url and access_url:
            break
    preview_url = ""
    meta_image = soup.select_one('meta[property="og:image"][content]')
    if meta_image:
        preview_url = (meta_image.get("content") or "").strip()
    content_blocks: list[str] = []
    title = ""
    meta_title = soup.select_one('meta[property="og:title"][content]')
    if meta_title:
        title = (meta_title.get("content") or "").strip()
    if title:
        content_blocks.append(title)
    for marker in ("Vista HTML de la presentación", "Presentación de diapositivas", "Solo ver"):
        if marker in (text_excerpt or "") and marker not in content_blocks:
            content_blocks.append(marker)
    return {
        "provider": "google_slides",
        "resource_id": _extract_google_slides_id(final_url),
        "resource_type": "presentation",
        "canonical_url": canonical_url,
        "htmlpresent_url": htmlpresent_url,
        "preview_url": preview_url,
        "access_url": access_url,
        "download_url": "",
        "requires_login": bool(access_url or "Acceder" in (text_excerpt or "")),
        "slide_count": _infer_google_slides_slide_count(text_excerpt),
        "content_blocks": content_blocks[:12],
    }


def _extract_google_drive_resource_metadata(soup, final_url: str, text_excerpt: str) -> dict[str, object]:
    canonical_url = (final_url or "").split("#", 1)[0]
    access_url = ""
    preview_url = ""
    file_kind = "file"
    content_blocks: list[str] = []
    for anchor in soup.select("a[href]"):
        href = _moodle_absolute_url(final_url, anchor.get("href", ""))
        label = anchor.get_text(" ", strip=True)
        if not access_url and "accounts.google.com/ServiceLogin" in href:
            access_url = href
        if label and label not in content_blocks and label in {"Transcripción", "Descargar", "Abrir", "Detalles"}:
            content_blocks.append(label)
    meta_title = soup.select_one('meta[property="og:title"][content]')
    if meta_title:
        title = (meta_title.get("content") or "").strip()
        if title and title not in content_blocks:
            content_blocks.insert(0, title)
        lowered_title = title.lower()
        if lowered_title.endswith(".mp4") or lowered_title.endswith(".mov") or lowered_title.endswith(".webm"):
            file_kind = "video"
    viewer_image = soup.select_one('meta[property="og:image"][content]')
    if viewer_image:
        preview_url = (viewer_image.get("content") or "").strip()
    if not preview_url:
        for image in soup.select("img[src]"):
            src = _moodle_absolute_url(final_url, image.get("src", ""))
            if "drive-viewer/" in src:
                preview_url = src
                break
    lowered_text = (text_excerpt or "").lower()
    if "video/mp4" in lowered_text or "transcripción" in lowered_text:
        file_kind = "video"
    return {
        "provider": "google_drive",
        "resource_id": _extract_google_drive_id(final_url),
        "resource_type": file_kind,
        "canonical_url": canonical_url,
        "htmlpresent_url": "",
        "preview_url": preview_url,
        "access_url": access_url,
        "download_url": f"https://drive.google.com/uc?export=download&id={_extract_google_drive_id(final_url)}" if _extract_google_drive_id(final_url) else "",
        "requires_login": bool(access_url or "Acceder" in (text_excerpt or "")),
        "slide_count": None,
        "content_blocks": content_blocks[:12],
    }


def _extract_external_resource_metadata(soup, final_url: str, text_excerpt: str) -> Optional[dict[str, object]]:
    lowered = (final_url or "").lower()
    if "docs.google.com/presentation/" in lowered:
        return _extract_google_slides_resource_metadata(soup, final_url, text_excerpt)
    if "drive.google.com/file/" in lowered:
        return _extract_google_drive_resource_metadata(soup, final_url, text_excerpt)
    return None


def _guess_moodle_resource_type(url: str, label: str = "") -> str:
    lowered = f"{url} {label}".lower()
    if "docs.google.com/presentation/" in lowered:
        return "google_slides"
    if "drive.google.com/file/" in lowered:
        return "google_drive"
    if "/course/view.php" in lowered and "section=" in lowered:
        return "course_section"
    if "/mod/assign/" in lowered or "entrega" in lowered or "submit" in lowered:
        return "assignment"
    if "/mod/url/" in lowered:
        return "redirect_link"
    if "/mod/lti/" in lowered or "turnitin" in lowered or "external tool" in lowered or "herramienta externa" in lowered:
        return "external_tool"
    if "/mod/folder/" in lowered:
        return "folder"
    if "/mod/resource/" in lowered or "pluginfile.php" in lowered:
        return "document"
    if "/mod/page/" in lowered:
        return "page"
    if "/mod/forum/" in lowered:
        return "forum"
    if "/mod/quiz/" in lowered:
        return "quiz"
    if "/mod/workshop/" in lowered:
        return "workshop"
    if "youtube" in lowered or "youtu.be" in lowered or "/embed/" in lowered:
        return "video"
    return "link"


def _looks_like_submission_target(url: str, label: str = "") -> bool:
    lowered = f"{url} {label}".lower()
    tokens = (
        "add submission",
        "edit submission",
        "submit assignment",
        "submission",
        "deliver",
        "entrega",
        "enviar tarea",
        "realizar entrega",
        "assign/view.php",
        "assign/submission",
    )
    return any(token in lowered for token in tokens)


def _extract_moodle_course_id(url: str) -> str:
    from urllib.parse import parse_qs, urlparse

    try:
        parsed = urlparse(url)
        return (parse_qs(parsed.query).get("id") or [""])[0].strip()
    except Exception:
        return ""


def _is_same_course_section_link(url: str, current_page_url: str) -> bool:
    from urllib.parse import parse_qs, urlparse

    try:
        parsed = urlparse(url)
        current_parsed = urlparse(current_page_url)
        if "/course/view.php" not in (parsed.path or ""):
            return False
        query = parse_qs(parsed.query)
        current_query = parse_qs(current_parsed.query)
        course_id = (query.get("id") or [""])[0].strip()
        current_course_id = (current_query.get("id") or [""])[0].strip()
        has_section = "section" in query
        return bool(course_id and current_course_id and has_section and course_id == current_course_id)
    except Exception:
        return False


def _should_follow_moodle_child_link(url: str, label: str = "", *, root_host: str = "", current_page_url: str = "") -> bool:
    resource_type = _guess_moodle_resource_type(url, label)
    lowered = f"{url} {label}".lower()
    same_site = bool(root_host) and _normalized_host(url) == root_host
    if _looks_like_submission_target(url, label):
        return True
    if same_site and _is_same_course_section_link(url, current_page_url):
        return True
    if same_site and resource_type in {"assignment", "redirect_link", "external_tool", "page", "document", "forum", "quiz", "workshop", "video", "folder"}:
        return True
    external_tokens = (
        "turnitin",
        "lti",
        "external tool",
        "herramienta externa",
        "drive.google",
        "docs.google",
        "youtube",
        "youtu.be",
        "vimeo",
        "forms",
        "teams",
        "zoom",
        "meet.google",
    )
    return resource_type in {"redirect_link", "external_tool", "video"} and any(token in lowered for token in external_tokens)


_MOODLE_CRAWL_MAX_DEPTH = 3
_MOODLE_CRAWL_MAX_PAGES = 24
_MOODLE_CRAWL_MAX_CHILDREN_PER_PAGE = 10
_MOODLE_FILE_METADATA_LIMIT = 32


def _resolve_moodle_settings(base_url: str = "") -> tuple[str, str, str]:
    import os

    moodle_url = (base_url or os.getenv("MOODLE_URL", "")).rstrip("/")
    moodle_user = os.getenv("MOODLE_USERNAME", "")
    moodle_pass = os.getenv("MOODLE_PASSWORD", "")

    if not moodle_url:
        raise ValueError("falta la URL de Moodle. Configurá MOODLE_URL en .env o pasá base_url.")
    if not moodle_user or not moodle_pass:
        raise ValueError("falta MOODLE_USERNAME o MOODLE_PASSWORD en variables de entorno.")
    return moodle_url, moodle_user, moodle_pass


def _session_invalid(
    page,
    *,
    expected_root_host: str,
    allow_external: bool = False,
) -> tuple[bool, str]:
    current_url = str(page.url or "")
    current_host = _normalized_host(current_url)
    same_site = bool(expected_root_host) and current_host == expected_root_host
    if allow_external and not same_site:
        return False, ""
    if not same_site:
        return False, ""
    html = _safe_page_content(page)
    text = _safe_page_text(page, limit=2000)
    lowered = text.lower()
    signals = (
        "usted no se ha identificado",
        "su sesión ha excedido el tiempo límite",
        "su sesión ha excedido el tiempo limite",
        "nombre de usuario",
        "contraseña",
        "acceder",
        "iniciar sesión en el sitio",
    )
    explicit_error = _extract_login_error_message(html=html, text=text)
    if "/login/" in current_url:
        if explicit_error:
            return True, f"credential_error_detected url={current_url} message={explicit_error}"
        return True, f"redirected_to_login url={current_url}"
    if any(signal in lowered for signal in signals):
        if explicit_error:
            return True, f"credential_error_detected url={current_url} message={explicit_error}"
        return True, f"login_screen_detected url={current_url}"
    return False, ""


def _authenticate_moodle_session(base_url: str = "") -> tuple[str, object, object]:
    moodle_url, moodle_user, moodle_pass = _resolve_moodle_settings(base_url)
    browser = scraping_infra._get_browser()
    context = browser.new_context()
    page = context.new_page()
    redirect_chain: list[str] = []

    def _track_navigation(frame) -> None:
        try:
            main_frame = getattr(page, "main_frame", None)
            if callable(main_frame):
                main_frame = main_frame()
            if main_frame is not None and frame != main_frame:
                return
            current = str(page.url or "")
            if current and current not in redirect_chain:
                redirect_chain.append(current)
        except Exception:
            return

    try:
        page.on("framenavigated", _track_navigation)
    except Exception:
        pass

    _write_moodle_debug_log(
        "login_context.json",
        {
            "moodle_url": moodle_url,
            "username_present": bool(moodle_user),
            "password_present": bool(moodle_pass),
        },
    )

    page.goto(f"{moodle_url}/login/index.php", wait_until="domcontentloaded", timeout=30000)
    before_submit_url = str(page.url or "")
    before_submit_html = _safe_page_content(page)
    _write_moodle_debug_log("01_login_page.html", before_submit_html)
    _write_moodle_debug_screenshot("01_login_page.png", page)
    _write_moodle_debug_log(
        "01_login_page_meta.json",
        {
            "url": before_submit_url,
            **_extract_login_form_debug(html=before_submit_html, current_url=before_submit_url),
        },
    )
    page.fill("#username", moodle_user)
    page.fill("#password", moodle_pass)
    page.click("#loginbtn")
    page.wait_for_load_state("domcontentloaded", timeout=20000)
    after_submit_url = str(page.url or "")

    try:
        btn = page.locator("input[type='submit']").first
        if btn.is_visible(timeout=2000):
            btn.click()
            page.wait_for_load_state("domcontentloaded", timeout=10000)
    except Exception:
        pass

    final_url = str(page.url or "")
    final_html = _safe_page_content(page)
    final_text = _safe_page_text(page, limit=4000)
    if before_submit_url and before_submit_url not in redirect_chain:
        redirect_chain.insert(0, before_submit_url)
    if after_submit_url and after_submit_url not in redirect_chain:
        redirect_chain.append(after_submit_url)
    if final_url and final_url not in redirect_chain:
        redirect_chain.append(final_url)
    login_form_debug = _extract_login_form_debug(html=final_html, current_url=final_url)
    cookie_debug = _context_cookie_debug(context, expected_root_host=_normalized_host(moodle_url))
    explicit_error = _extract_login_error_message(html=final_html, text=final_text)
    login_debug_payload = {
        "login_url": f"{moodle_url}/login/index.php",
        "before_submit_url": before_submit_url,
        "after_submit_url": after_submit_url,
        "final_url": final_url,
        "redirect_chain": redirect_chain,
        "credential_error_message": explicit_error,
        "body_preview": final_text[:1500],
        **login_form_debug,
        **cookie_debug,
    }
    _write_moodle_debug_log("login_attempt.json", login_debug_payload)

    invalid_session, reason = _session_invalid(page, expected_root_host=_normalized_host(moodle_url))
    if invalid_session:
        _write_moodle_debug_log(
            "login_failure.json",
            {
                "reason": reason,
                "final_url": final_url,
                "before_submit_url": before_submit_url,
                "after_submit_url": after_submit_url,
                "redirect_chain": redirect_chain,
                "credential_error_message": explicit_error,
                "login_form_found": login_form_debug.get("login_form_found", False),
                "second_form_detected": login_form_debug.get("second_form_detected", False),
                "sso_detected": login_form_debug.get("sso_detected", False),
                "token_fields": login_form_debug.get("token_fields", []),
                "cookie_count": cookie_debug.get("cookie_count", 0),
            },
        )
        _write_moodle_debug_log("02_login_failure.html", final_html)
        _write_moodle_debug_screenshot("02_login_failure.png", page)
        print(f"[MOODLE_DEBUG] login_invalid reason={reason}", flush=True)
        context.close()
        raise RuntimeError("login fallido — verificá MOODLE_USERNAME y MOODLE_PASSWORD.")
    _write_moodle_debug_log(
        "login_success.json",
        {
            "before_submit_url": before_submit_url,
            "after_submit_url": after_submit_url,
            "final_url": final_url,
            "redirect_chain": redirect_chain,
            "title": page.title() if hasattr(page, "title") else "",
            "cookie_count": cookie_debug.get("cookie_count", 0),
            "token_fields": login_form_debug.get("token_fields", []),
            "sso_detected": login_form_debug.get("sso_detected", False),
        },
    )
    _write_moodle_debug_log("02_login_success.html", final_html)
    _write_moodle_debug_screenshot("02_login_success.png", page)
    return moodle_url, context, page


def _extract_submission_status_from_text(page_text: str) -> str:
    lowered = (page_text or "").lower()
    submitted_tokens = (
        "entrega realizada",
        "estado de la entrega entregado",
        "estado de la entrega enviado para calificar",
        "enviado para calificar",
        "submission status submitted for grading",
        "submitted for grading",
        "calificado",
        "graded",
    )
    pending_tokens = (
        "estado de la entrega no entregado",
        "no entregado",
        "nothing has been submitted",
        "no se ha realizado ningún envío",
        "no se ha realizado ningun envio",
        "pendiente",
        "por entregar",
    )
    overdue_tokens = (
        "vencida",
        "vencido",
        "atrasada",
        "overdue",
    )
    print(
        "[MOODLE_DEBUG] detail_status_probe "
        f"submitted_tokens_found={[token for token in submitted_tokens if token in lowered]} "
        f"pending_tokens_found={[token for token in pending_tokens if token in lowered]} "
        f"overdue_tokens_found={[token for token in overdue_tokens if token in lowered]}",
        flush=True,
    )
    if any(token in lowered for token in submitted_tokens):
        return "entregado"
    if any(token in lowered for token in pending_tokens):
        return "no entregado"
    if any(token in lowered for token in overdue_tokens):
        return "vencida"
    return ""


def fetch_moodle_submission_statuses(
    assignments: list[dict[str, str]],
    base_url: str = "",
) -> dict[str, str]:
    """Consulta el detalle de cada tarea Moodle para enriquecer el estado real de entrega."""
    if not assignments:
        return {}
    moodle_url, context, page = _authenticate_moodle_session(base_url)
    statuses: dict[str, str] = {}
    try:
        for assignment in assignments:
            href = str(assignment.get("url") or "").strip()
            if not href:
                continue
            target_url = href if href.startswith("http") else f"{moodle_url}/{href.lstrip('/')}"
            try:
                page.goto(target_url, wait_until="domcontentloaded", timeout=30000)
                invalid_session, reason = _session_invalid(page, expected_root_host=_normalized_host(moodle_url))
                if invalid_session:
                    print(f"[MOODLE_DEBUG] detail_invalid reason={reason} url={target_url}", flush=True)
                    raise RuntimeError(
                        "sesión expirada o redirigida al login al abrir el detalle de una tarea de Moodle."
                    )
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass
                detail_text = page.locator("body").inner_text(timeout=5000)
                print(
                    f"[MOODLE_DEBUG] detail_status_scan url={target_url} text_preview={detail_text[:400]!r}",
                    flush=True,
                )
                status = _extract_submission_status_from_text(detail_text)
                if status:
                    statuses[target_url] = status
                    print(
                        f"[MOODLE_DEBUG] detail_status_detected url={target_url} status={status!r}",
                        flush=True,
                    )
                else:
                    print(
                        f"[MOODLE_DEBUG] detail_status_not_detected url={target_url}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[MOODLE_DEBUG] detail_status_error url={target_url} error={exc}", flush=True)
                continue
    finally:
        context.close()
    return statuses


def _extract_moodle_page_links(soup, current_url: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    links: list[dict[str, object]] = []
    attachments: list[dict[str, str]] = []
    seen_links: set[tuple[str, str]] = set()
    seen_attachments: set[tuple[str, str]] = set()
    file_tokens = (
        ".pdf",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".xls",
        ".xlsx",
        ".zip",
        ".rar",
        ".txt",
        ".csv",
        "pluginfile.php",
        "/mod/resource/",
    )
    for anchor in soup.select("a[href]"):
        href = (anchor.get("href") or "").strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        absolute_url = _moodle_absolute_url(current_url, href)
        label = anchor.get_text(" ", strip=True)
        key = (label, absolute_url)
        if key not in seen_links:
            seen_links.add(key)
            resource_type = _guess_moodle_resource_type(absolute_url, label)
            links.append(
                {
                    "label": label,
                    "url": absolute_url,
                    "final_url": absolute_url,
                    "redirect_target": absolute_url if absolute_url != href else "",
                    "redirect_chain": [absolute_url],
                    "resource_type": resource_type,
                    "is_submission_target": _looks_like_submission_target(absolute_url, label),
                }
            )
        lowered = absolute_url.lower()
        if any(token in lowered for token in file_tokens):
            attachment_key = (label, absolute_url)
            if attachment_key in seen_attachments:
                continue
            seen_attachments.add(attachment_key)
            mime_hint = ""
            if "." in lowered.rsplit("/", 1)[-1]:
                mime_hint = lowered.rsplit(".", 1)[-1]
            attachments.append(
                _build_file_audit_record(
                    label=label,
                    url=absolute_url,
                    final_url=absolute_url,
                    mime_hint=mime_hint,
                    kind="document" if mime_hint in {"pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx"} else "file",
                    redirect_chain=[absolute_url],
                )
            )
    return links, attachments


def _extract_moodle_page_videos(soup, current_url: str) -> list[dict[str, str]]:
    videos: list[dict[str, str]] = []
    seen: set[str] = set()
    for iframe in soup.select("iframe[src]"):
        src = _moodle_absolute_url(current_url, iframe.get("src", ""))
        if not src or src in seen:
            continue
        seen.add(src)
        provider = "youtube" if "youtube" in src or "youtu.be" in src else "vimeo" if "vimeo" in src else "iframe"
        watch_url, preview_url = _youtube_watch_and_preview(src)
        videos.append(
            {
                "label": iframe.get("title", "") or iframe.get("aria-label", ""),
                "embed_url": src,
                "watch_url": watch_url,
                "provider": provider,
                "preview_url": preview_url,
            }
        )
    for video in soup.select("video"):
        src = ""
        source = video.get("src")
        if source:
            src = _moodle_absolute_url(current_url, source)
        else:
            source_el = video.select_one("source[src]")
            if source_el:
                src = _moodle_absolute_url(current_url, source_el.get("src", ""))
        if not src or src in seen:
            continue
        seen.add(src)
        videos.append(
            {
                "label": video.get("title", "") or video.get("aria-label", ""),
                "embed_url": src,
                "watch_url": "",
                "provider": "html5",
                "preview_url": _moodle_absolute_url(current_url, video.get("poster", "")) if video.get("poster") else "",
            }
        )
    return videos


def _extract_moodle_page_images(soup, current_url: str) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    seen: set[str] = set()
    for image in soup.select("img[src]"):
        src = _moodle_absolute_url(current_url, image.get("src", ""))
        if not src or src in seen:
            continue
        seen.add(src)
        images.append(
            {
                "label": image.get("alt", "") or image.get("title", ""),
                "url": src,
                "kind": "image",
            }
        )
    return images


def _extract_table_pairs(soup) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for row in soup.select("table tr"):
        header = row.select_one("th, .cell.c0")
        value = row.select_one("td, .cell.c1")
        if not header or not value:
            cells = row.select("td")
            if len(cells) >= 2:
                header, value = cells[0], cells[1]
        if not header or not value:
            continue
        key = header.get_text(" ", strip=True)
        val = value.get_text(" ", strip=True)
        if key and val and key not in pairs:
            pairs[key] = val
    for item in soup.select(".submissionstatustable .row, .boxaligncenter table tr"):
        cells = item.select(".cell")
        if len(cells) >= 2:
            key = cells[0].get_text(" ", strip=True)
            val = cells[1].get_text(" ", strip=True)
            if key and val and key not in pairs:
                pairs[key] = val
    return pairs


def _extract_submission_files(soup, current_url: str) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    seen: set[str] = set()
    candidate_selectors = [
        ".submissionstatustable a[href]",
        ".fileuploadsubmission a[href]",
        ".submissionsummarytable a[href]",
        "a[href*='pluginfile.php']",
    ]
    for selector in candidate_selectors:
        for anchor in soup.select(selector):
            href = (anchor.get("href") or "").strip()
            if not href:
                continue
            absolute_url = _moodle_absolute_url(current_url, href)
            if absolute_url in seen:
                continue
            seen.add(absolute_url)
            filename = _basename_from_url(absolute_url)
            mime_hint = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            files.append(
                _build_file_audit_record(
                    label=anchor.get_text(" ", strip=True),
                    url=absolute_url,
                    final_url=absolute_url,
                    mime_hint=mime_hint,
                    redirect_chain=[absolute_url],
                )
            )
    return files[:20]


def _extract_submission_actions(soup) -> list[str]:
    actions: list[str] = []
    seen: set[str] = set()
    for node in soup.select("a, button, input[type='submit']"):
        label = node.get_text(" ", strip=True) or (node.get("value") or "").strip()
        lowered = label.lower()
        if not label:
            continue
        if not any(
            token in lowered
            for token in (
                "add submission",
                "edit submission",
                "submit assignment",
                "realizar entrega",
                "enviar tarea",
                "agregar entrega",
                "editar entrega",
                "save changes",
            )
        ):
            continue
        if label in seen:
            continue
        seen.add(label)
        actions.append(label)
    return actions[:10]


def _extract_submission_instructions(soup) -> list[str]:
    instructions: list[str] = []
    seen: set[str] = set()
    for selector in ("#intro", ".activity-description", ".submissioninstructions", ".no-overflow"):
        for node in soup.select(selector):
            text = node.get_text(" ", strip=True)
            if not text or text in seen:
                continue
            seen.add(text)
            instructions.append(text[:1000])
    return instructions[:5]


def _extract_submission_state(soup, current_url: str) -> Optional[dict[str, object]]:
    raw_fields = _extract_table_pairs(soup)
    normalized_fields = {key.lower(): value for key, value in raw_fields.items()}

    def _match_field(*needles: str) -> str:
        for key, value in normalized_fields.items():
            if any(needle in key for needle in needles):
                return value
        return ""

    submission_status = _match_field("submission status", "estado de la entrega", "estado del envío", "estado del envio")
    grading_status = _match_field("grading status", "estado de calificación", "estado de calificacion")
    due_date_text = _match_field("due date", "fecha de entrega", "cut-off date", "fecha límite", "fecha limite")
    time_remaining_text = _match_field("time remaining", "tiempo restante", "remaining time")
    last_modified_text = _match_field("last modified", "última modificación", "ultima modificacion", "modified")
    attempt_text = _match_field("attempt number", "número de intento", "numero de intento", "attempt")

    instructions = _extract_submission_instructions(soup)
    available_actions = _extract_submission_actions(soup)
    submitted_files = _extract_submission_files(soup, current_url)

    lowered_submission = submission_status.lower()
    lowered_grading = grading_status.lower()
    lowered_actions = " ".join(available_actions).lower()

    can_submit = any(
        token in lowered_actions
        for token in ("realizar entrega", "enviar tarea", "add submission", "edit submission", "submit assignment")
    )
    is_submitted = any(
        token in lowered_submission
        for token in ("entregado", "enviado", "submitted", "submission submitted")
    ) or bool(submitted_files)
    is_graded = (
        any(token in lowered_grading for token in ("calificado", "graded"))
        and not any(token in lowered_grading for token in ("no calificado", "not graded", "sin calificar"))
    )
    is_locked = any(
        token in f"{lowered_submission} {lowered_actions}"
        for token in ("cerrado", "closed", "bloqueado", "locked", "no disponible")
    ) and not can_submit

    field_confidence = {
        "submission_status": 0.95 if submission_status else 0.0,
        "grading_status": 0.95 if grading_status else 0.0,
        "due_date_text": 0.9 if due_date_text else 0.0,
        "time_remaining_text": 0.8 if time_remaining_text else 0.0,
        "last_modified_text": 0.75 if last_modified_text else 0.0,
        "attempt_text": 0.75 if attempt_text else 0.0,
        "instructions": 0.8 if instructions else 0.0,
        "available_actions": 0.9 if available_actions else 0.0,
        "submitted_files": 0.95 if submitted_files else 0.0,
    }

    if not any([submission_status, grading_status, due_date_text, time_remaining_text, last_modified_text, attempt_text, instructions, available_actions, submitted_files, raw_fields]):
        return None

    return {
        "submission_status": submission_status,
        "grading_status": grading_status,
        "due_date_text": due_date_text,
        "time_remaining_text": time_remaining_text,
        "last_modified_text": last_modified_text,
        "attempt_text": attempt_text,
        "instructions": instructions,
        "available_actions": available_actions,
        "submitted_files": submitted_files,
        "raw_fields": raw_fields,
        "field_confidence": field_confidence,
        "can_submit": can_submit,
        "is_submitted": is_submitted,
        "is_graded": is_graded,
        "is_locked": is_locked,
    }


def _extract_moodle_page_metadata(
    page_kind: str,
    page_url: str,
    final_url: str,
    html: str,
    item_count: int = 0,
    *,
    parent_url: str = "",
    source_link_label: str = "",
    crawl_depth: int = 0,
    visit_order: int = 0,
) -> dict[str, object]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    title = ""
    for selector in (".page-header-headings h1", "h1", "title"):
        el = soup.select_one(selector)
        if el:
            title = el.get_text(" ", strip=True)
            if title:
                break
    subtitle = ""
    for selector in (".page-header-headings h2", "h2", ".lead", ".small.text-muted"):
        el = soup.select_one(selector)
        if el:
            subtitle = el.get_text(" ", strip=True)
            if subtitle:
                break
    description = ""
    for selector in ("#intro", ".description", ".no-overflow", "[role='main'] p", ".activity-description"):
        el = soup.select_one(selector)
        if el:
            description = el.get_text(" ", strip=True)
            if description:
                break
    breadcrumbs = [
        crumb.get_text(" ", strip=True)
        for crumb in soup.select(".breadcrumb li, nav[aria-label='breadcrumb'] li, .breadcrumbs li")
        if crumb.get_text(" ", strip=True)
    ]
    links, attachments = _extract_moodle_page_links(soup, final_url or page_url)
    videos = _extract_moodle_page_videos(soup, final_url or page_url)
    images = _extract_moodle_page_images(soup, final_url or page_url)
    submission_state = _extract_submission_state(soup, final_url or page_url)
    text_excerpt = soup.get_text(" ", strip=True)[:500]
    external_resource = _extract_external_resource_metadata(soup, final_url or page_url, text_excerpt)
    return {
        "page_kind": page_kind,
        "url": page_url,
        "final_url": final_url or page_url,
        "title": title,
        "subtitle": subtitle,
        "description": description[:2000],
        "resource_type": _guess_moodle_resource_type(final_url or page_url, title or source_link_label),
        "parent_url": parent_url,
        "source_link_label": source_link_label,
        "crawl_depth": crawl_depth,
        "visit_order": visit_order,
        "breadcrumbs": breadcrumbs[:12],
        "text_excerpt": text_excerpt,
        "links": links[:200],
        "attachments": attachments[:100],
        "videos": videos[:50],
        "images": images[:100],
        "external_resource": external_resource,
        "submission_state": submission_state,
        "extracted_items_count": item_count,
        "raw_html": html,
    }


def _enrich_page_file_metadata(
    page_payload: dict[str, object],
    *,
    http_session,
    warnings: list[str],
    budget: dict[str, int],
) -> None:
    if budget["remaining"] <= 0:
        return

    def _enrich_collection(items: list[dict[str, object]], field_name: str) -> None:
        for item in items:
            if budget["remaining"] <= 0:
                return
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            try:
                metadata = _fetch_file_http_metadata(http_session, url)
            except Exception as exc:
                warnings.append(f"file_metadata_error:{field_name}:{url}:{exc}")
                budget["remaining"] -= 1
                continue
            item.update(
                {
                    "filename": metadata.get("filename", item.get("filename", "")),
                    "final_url": metadata.get("final_url", item.get("final_url", url)),
                    "redirect_target": metadata.get("redirect_target", item.get("redirect_target", "")),
                    "redirect_chain": metadata.get("redirect_chain", item.get("redirect_chain", [url])),
                    "mime_hint": metadata.get("mime_hint", item.get("mime_hint", "")),
                    "kind": metadata.get("kind", item.get("kind", "file")),
                    "content_type": metadata.get("content_type", ""),
                    "content_length": metadata.get("content_length"),
                    "content_disposition": metadata.get("content_disposition", ""),
                    "status_code": metadata.get("status_code"),
                    "is_download": metadata.get("is_download", False),
                }
            )
            budget["remaining"] -= 1

    attachments = page_payload.get("attachments")
    if isinstance(attachments, list):
        _enrich_collection([item for item in attachments if isinstance(item, dict)], "attachments")
    submission_state = page_payload.get("submission_state")
    if isinstance(submission_state, dict):
        submitted_files = submission_state.get("submitted_files")
        if isinstance(submitted_files, list):
            _enrich_collection([item for item in submitted_files if isinstance(item, dict)], "submitted_files")


def _build_download_backed_audit_page(
    *,
    target_url: str,
    page_kind: str,
    source_link_label: str,
    parent_url: str,
    crawl_depth: int,
    visit_order: int,
    metadata: dict[str, object],
) -> dict[str, object]:
    title = str(source_link_label or metadata.get("filename") or metadata.get("label") or "Documento Moodle").strip()
    final_url = str(metadata.get("final_url") or target_url)
    attachment = _build_file_audit_record(
        label=str(metadata.get("label") or source_link_label or title),
        url=target_url,
        final_url=final_url,
        mime_hint=str(metadata.get("mime_hint") or ""),
        kind=str(metadata.get("kind") or "file"),
        content_type=str(metadata.get("content_type") or ""),
        content_length=metadata.get("content_length"),
        content_disposition=str(metadata.get("content_disposition") or ""),
        status_code=metadata.get("status_code"),
        redirect_chain=list(metadata.get("redirect_chain") or [target_url]),
    )
    return {
        "page_kind": page_kind,
        "url": target_url,
        "final_url": final_url,
        "title": title,
        "subtitle": "",
        "description": "download resource",
        "resource_type": "document",
        "parent_url": parent_url,
        "source_link_label": source_link_label,
        "crawl_depth": crawl_depth,
        "visit_order": visit_order,
        "breadcrumbs": [],
        "text_excerpt": "",
        "links": [],
        "attachments": [attachment],
        "videos": [],
        "images": [],
        "submission_state": None,
        "extracted_items_count": 1,
        "raw_html": "",
        "notes": ["download_backed_resource"],
    }


def _infer_child_page_kind(parent_kind: str, url: str, label: str = "") -> str:
    if _looks_like_submission_target(url, label):
        return "submission_action_page" if parent_kind in {"submission_page", "submission_action_page"} else "submission_page"
    resource_type = _guess_moodle_resource_type(url, label)
    if resource_type == "course_section":
        return "course_section"
    if resource_type == "assignment":
        return "assignment_detail"
    if resource_type in {"page", "forum", "quiz", "workshop"}:
        return "linked_resource"
    if resource_type in {"redirect_link", "external_tool", "document", "video", "folder"}:
        return "linked_resource"
    return "linked_resource"


def _build_child_crawl_targets(
    page_payload: dict[str, object],
    *,
    root_host: str,
    max_depth: int,
    max_children: int,
) -> list[dict[str, object]]:
    from bs4 import BeautifulSoup

    current_depth = int(page_payload.get("crawl_depth") or 0)
    if current_depth >= max_depth:
        return []

    html = str(page_payload.get("raw_html") or "")
    final_url = str(page_payload.get("final_url") or page_payload.get("url") or "")
    if not html or not final_url:
        return []

    soup = BeautifulSoup(html, "html.parser")
    prioritized_targets = _extract_submission_targets(soup, final_url)
    generic_targets = _collect_followable_child_links(page_payload, root_host=root_host, limit=max_children)
    prioritized_keys = {
        _canonicalize_moodle_crawl_url(str(item.get("url") or ""), root_host=root_host)
        for item in prioritized_targets
        if str(item.get("url") or "").strip()
    }
    ordered_targets = prioritized_targets + [
        target
        for target in generic_targets
        if _canonicalize_moodle_crawl_url(str(target.get("url") or ""), root_host=root_host) not in prioritized_keys
    ]

    parent_kind = str(page_payload.get("page_kind") or "")
    parent_final_url = str(page_payload.get("final_url") or page_payload.get("url") or "")
    next_depth = current_depth + 1
    crawl_targets: list[dict[str, object]] = []
    seen: set[str] = set()
    for child_link in ordered_targets[:max_children]:
        child_url = str(child_link.get("url") or "").strip()
        label = str(child_link.get("label") or "").strip()
        canonical_child_url = _canonicalize_moodle_crawl_url(child_url, root_host=root_host)
        if not child_url or canonical_child_url in seen:
            continue
        seen.add(canonical_child_url)
        crawl_targets.append(
            {
                "url": canonical_child_url or child_url,
                "label": label,
                "parent_url": parent_final_url,
                "page_kind": _infer_child_page_kind(parent_kind, child_url, label),
                "crawl_depth": next_depth,
            }
        )
    return crawl_targets


def _extract_submission_targets(soup, current_url: str) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    seen: set[str] = set()
    root_host = _normalized_host(current_url)
    selectors = [
        "a[href]",
        "button[onclick]",
        "form[action]",
    ]
    for selector in selectors:
        for node in soup.select(selector):
            label = node.get_text(" ", strip=True)
            href = ""
            if node.name == "a":
                href = (node.get("href") or "").strip()
            elif node.name == "form":
                href = (node.get("action") or "").strip()
            else:
                onclick = (node.get("onclick") or "").strip()
                if "location.href" in onclick:
                    href = onclick.split("location.href", 1)[1].split("=", 1)[-1].strip(" '\" );")
            if not href:
                continue
            absolute_url = _moodle_absolute_url(current_url, href)
            canonical_url = _canonicalize_moodle_crawl_url(absolute_url, root_host=root_host)
            if canonical_url in seen:
                continue
            if not _looks_like_submission_target(absolute_url, label):
                continue
            seen.add(canonical_url)
            targets.append({"url": canonical_url or absolute_url, "label": label or "submission_target"})
    return targets[:6]


def _collect_followable_child_links(
    page_payload: dict[str, object],
    *,
    root_host: str = "",
    limit: int = 8,
) -> list[dict[str, str]]:
    links = page_payload.get("links")
    if not isinstance(links, list):
        return []
    followable: list[dict[str, str]] = []
    seen: set[str] = set()
    current_page_url = str(page_payload.get("final_url") or page_payload.get("url") or "").strip()
    for link in links:
        if not isinstance(link, dict):
            continue
        url = str(link.get("url") or "").strip()
        label = str(link.get("label") or "").strip()
        canonical_url = _canonicalize_moodle_crawl_url(url, root_host=root_host)
        if not url or canonical_url in seen:
            continue
        if not _should_follow_moodle_child_link(url, label, root_host=root_host, current_page_url=current_page_url):
            continue
        seen.add(canonical_url)
        followable.append({"url": canonical_url or url, "label": label})
        if len(followable) >= limit:
            break
    return followable


def _update_page_link_resolution(page_payload: dict[str, object], source_url: str, final_url: str, *, root_host: str = "") -> None:
    links = page_payload.get("links")
    if not isinstance(links, list):
        return
    source_key = _canonicalize_moodle_crawl_url(source_url, root_host=root_host)
    final_key = _canonicalize_moodle_crawl_url(final_url, root_host=root_host)
    for link in links:
        if not isinstance(link, dict):
            continue
        link_key = _canonicalize_moodle_crawl_url(str(link.get("url") or ""), root_host=root_host)
        if link_key != source_key:
            continue
        redirect_chain = link.get("redirect_chain")
        if not isinstance(redirect_chain, list):
            redirect_chain = [source_url]
        if final_url and final_key != source_key and final_url not in redirect_chain:
            redirect_chain.append(final_url)
        link["final_url"] = final_url or source_url
        link["redirect_target"] = final_url if final_url and final_url != source_url else ""
        link["redirect_chain"] = redirect_chain
        break


def _assignment_status_from_page(page_payload: dict[str, object]) -> str:
    submission_state = page_payload.get("submission_state")
    if not isinstance(submission_state, dict):
        return ""
    if bool(submission_state.get("is_graded")):
        return "calificado"
    if bool(submission_state.get("is_submitted")):
        return "entregado"
    if bool(submission_state.get("can_submit")):
        return "pendiente"
    if bool(submission_state.get("is_locked")):
        return "bloqueado"
    return str(submission_state.get("submission_status") or "")


def _normalize_course_name(value: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", value or "")
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(without_accents.lower().split())


def _extract_visible_moodle_courses_from_html(html: str, current_url: str) -> list[dict[str, str]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    selectors = (
        "a[href*='/course/view.php']",
        "a.course-title[href]",
        ".card a[href*='/course/view.php']",
        "[data-course-id] a[href]",
        ".coursebox a[href*='/course/view.php']",
        ".block_myoverview a[href*='/course/view.php']",
    )
    courses: list[dict[str, str]] = []
    seen: set[str] = set()
    for selector in selectors:
        for anchor in soup.select(selector):
            href = (anchor.get("href") or "").strip()
            if not href:
                continue
            absolute_url = _moodle_absolute_url(current_url, href)
            if "/course/view.php" not in absolute_url and "/course/" not in absolute_url:
                continue
            if absolute_url in seen:
                continue
            course_name = anchor.get_text(" ", strip=True)
            if not course_name:
                title_attr = (anchor.get("title") or "").strip()
                aria_label = (anchor.get("aria-label") or "").strip()
                course_name = title_attr or aria_label
            if not course_name:
                continue
            seen.add(absolute_url)
            courses.append({"course_name": course_name, "course_url": absolute_url})
    return courses


def _resolve_course_match(
    courses: list[dict[str, str]],
    course_query: str,
) -> tuple[Optional[dict[str, str]], list[dict[str, str]], str]:
    query = (course_query or "").strip()
    if not query:
        return None, [], "empty_query"

    if query.isdigit():
        idx = int(query)
        if 1 <= idx <= len(courses):
            return courses[idx - 1], [], "index"
        return None, [], "index_out_of_range"

    normalized_query = _normalize_course_name(query)
    exact = [course for course in courses if _normalize_course_name(course.get("course_name", "")) == normalized_query]
    if len(exact) == 1:
        return exact[0], exact, "exact"
    if len(exact) > 1:
        return None, exact, "ambiguous_exact"

    contains = [course for course in courses if normalized_query in _normalize_course_name(course.get("course_name", ""))]
    if len(contains) == 1:
        return contains[0], contains, "contains"
    if len(contains) > 1:
        return None, contains, "ambiguous_contains"

    tokens = [token for token in normalized_query.split() if token]
    fuzzy = [
        course for course in courses
        if tokens and all(token in _normalize_course_name(course.get("course_name", "")) for token in tokens)
    ]
    if len(fuzzy) == 1:
        return fuzzy[0], fuzzy, "fuzzy"
    if len(fuzzy) > 1:
        return None, fuzzy, "ambiguous_fuzzy"

    return None, [], "not_found"


def _extract_assignment_records_from_pages(
    pages: list[dict[str, object]],
    *,
    course_name: str = "",
) -> list[dict[str, str]]:
    assignments: list[dict[str, str]] = []
    seen: set[str] = set()
    for page_payload in pages:
        page_kind = str(page_payload.get("page_kind") or "")
        resource_type = str(page_payload.get("resource_type") or "")
        if page_kind not in {"assignment_detail", "submission_page", "submission_action_page"} and resource_type != "assignment":
            continue
        final_url = str(page_payload.get("final_url") or page_payload.get("url") or "").strip()
        if not final_url or final_url in seen:
            continue
        seen.add(final_url)
        submission_state = page_payload.get("submission_state")
        due_date_text = ""
        if isinstance(submission_state, dict):
            due_date_text = str(submission_state.get("due_date_text") or "")
        assignments.append(
            {
                "name": str(page_payload.get("title") or page_payload.get("source_link_label") or "Actividad Moodle"),
                "date": due_date_text,
                "course": course_name,
                "url": final_url,
                "status": _assignment_status_from_page(page_payload),
                "source_stage": "course_audit",
            }
        )
    return assignments


def extract_moodle_audit_bundle(base_url: str = "") -> dict[str, object]:
    """Retorna tareas y páginas auditables capturadas desde Moodle."""
    from bs4 import BeautifulSoup
    from collections import deque

    moodle_url, context, page = _authenticate_moodle_session(base_url)
    http_session = _requests_session_from_playwright_context(context)
    html_dashboard = ""
    html_calendar = ""
    dashboard_url = f"{moodle_url}/my/"
    calendar_url = f"{moodle_url}/calendar/view.php?view=upcoming&lookahead=365"
    crawled_pages: list[dict[str, object]] = []
    warnings: list[str] = []
    root_host = _normalized_host(moodle_url)
    try:
        page.goto(dashboard_url, wait_until="domcontentloaded", timeout=30000)
        invalid_session, reason = _session_invalid(page, expected_root_host=root_host)
        if invalid_session:
            print(f"[MOODLE_DEBUG] dashboard_invalid reason={reason}", flush=True)
            raise RuntimeError("sesión expirada o redirigida al login al abrir el dashboard de Moodle.")
        try:
            page.wait_for_load_state("networkidle", timeout=12000)
        except Exception:
            pass
        for _sel in (
            "li.event-list-item",
            "[data-region='event-list-item']",
            "[data-region='event-list-content']",
            ".timeline-event-list",
            ".event-name",
        ):
            try:
                page.wait_for_selector(_sel, timeout=5000)
                break
            except Exception:
                continue
        else:
            page.wait_for_timeout(3000)
        html_dashboard = page.content()
        dashboard_final_url = str(page.url or dashboard_url)

        page.goto(calendar_url, wait_until="domcontentloaded", timeout=30000)
        invalid_session, reason = _session_invalid(page, expected_root_host=root_host)
        if invalid_session:
            print(f"[MOODLE_DEBUG] calendar_invalid reason={reason}", flush=True)
            raise RuntimeError("sesión expirada o redirigida al login al abrir el calendario de Moodle.")
        page.wait_for_timeout(1500)
        html_calendar = page.content()
        calendar_final_url = str(page.url or calendar_url)
    except Exception as e:
        raise RuntimeError(f"Error durante el scraping de Moodle: {str(e)}") from e

    assignments: list[dict] = []
    seen: set[str] = set()

    def _add(name: str, date_str: str, course: str, href: str, status: str = "") -> None:
        key = name.strip().lower()
        if key and key not in seen:
            seen.add(key)
            assignments.append({"name": name, "date": date_str, "course": course, "url": href, "status": status})

    soup = BeautifulSoup(html_dashboard, "html.parser")
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

    if not assignments:
        for item in soup.select("[data-region='event-list-content'] .event-name, .timeline-event-list .event-name"):
            name = item.get_text(strip=True)
            href = item.get("href", "") if item.name == "a" else (item.find("a") or {}).get("href", "")
            _add(name, "", "", href)

    soup_cal = BeautifulSoup(html_calendar, "html.parser")
    for event in soup_cal.select(".event"):
        name_el = event.select_one("h3.referer a, h3 a, .card-title a, a[data-eventid]")
        date_el = event.select_one(".date, .calendar-event-date, time")
        header = event.select_one(".card-header, .card-block, .card-body")
        course_el = None
        if header:
            for anchor in header.find_all("a"):
                if "course" in (anchor.get("href", "") or ""):
                    course_el = anchor
                    break
        name = name_el.get_text(strip=True) if name_el else ""
        date_str = date_el.get_text(strip=True) if date_el else ""
        course = course_el.get_text(strip=True) if course_el else ""
        href = name_el.get("href", "") if name_el else ""
        _add(name, date_str, course, href)

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
        raise RuntimeError(
            "No se encontraron tareas en el dashboard ni en el calendario.\n"
            "HTML volcado en /tmp/moodle_debug_dashboard.html y /tmp/moodle_debug_calendar.html para diagnóstico.\n\n"
            f"[Dashboard snippet]\n{dash_snippet}\n\n"
            f"[Calendario snippet]\n{cal_snippet}"
        )

    dashboard_page = _extract_moodle_page_metadata(
        "dashboard",
        dashboard_url,
        dashboard_final_url,
        html_dashboard,
        len(assignments),
        visit_order=1,
    )
    calendar_page = _extract_moodle_page_metadata(
        "calendar_upcoming",
        calendar_url,
        calendar_final_url,
        html_calendar,
        len(assignments),
        visit_order=2,
    )

    visited_urls: set[str] = set()
    file_metadata_budget = {"remaining": _MOODLE_FILE_METADATA_LIMIT}

    def _capture_page(
        target_url: str,
        *,
        page_kind: str,
        parent_url: str = "",
        source_link_label: str = "",
        crawl_depth: int = 0,
        visit_order: int = 0,
    ) -> dict[str, object] | None:
        target_resource_type = _guess_moodle_resource_type(target_url, source_link_label)
        try:
            page.goto(target_url, wait_until="domcontentloaded", timeout=30000)
            invalid_session, reason = _session_invalid(page, expected_root_host=root_host, allow_external=True)
            if invalid_session:
                warnings.append(f"{page_kind}_invalid:{target_url}:{reason}")
                return None
            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            captured_html = page.content()
            payload = _extract_moodle_page_metadata(
                page_kind,
                target_url,
                str(page.url or target_url),
                captured_html,
                0,
                parent_url=parent_url,
                source_link_label=source_link_label,
                crawl_depth=crawl_depth,
                visit_order=visit_order,
            )
            _enrich_page_file_metadata(
                payload,
                http_session=http_session,
                warnings=warnings,
                budget=file_metadata_budget,
            )
            return payload
        except Exception as exc:
            if target_resource_type == "document" or "Download is starting" in str(exc):
                try:
                    metadata = _fetch_file_http_metadata(http_session, target_url)
                    return _build_download_backed_audit_page(
                        target_url=target_url,
                        page_kind=page_kind,
                        source_link_label=source_link_label,
                        parent_url=parent_url,
                        crawl_depth=crawl_depth,
                        visit_order=visit_order,
                        metadata=metadata,
                    )
                except Exception as metadata_exc:
                    warnings.append(f"document_metadata_error:{target_url}:{metadata_exc}")
                    return None
            warnings.append(f"{page_kind}_error:{target_url}:{exc}")
            return None

    try:
        crawl_queue: deque[dict[str, object]] = deque()
        for assignment in assignments:
            href = str(assignment.get("url") or "").strip()
            if not href:
                continue
            target_url = href if href.startswith("http") else _moodle_absolute_url(moodle_url, href)
            crawl_queue.append(
                {
                    "url": target_url,
                    "page_kind": "assignment_detail",
                    "parent_url": "",
                    "source_link_label": "",
                    "crawl_depth": 0,
                }
            )

        visit_order = 3
        while crawl_queue and len(crawled_pages) < _MOODLE_CRAWL_MAX_PAGES:
            target = crawl_queue.popleft()
            target_url = str(target.get("url") or "").strip()
            canonical_target_url = _canonicalize_moodle_crawl_url(target_url, root_host=root_host)
            if not target_url or canonical_target_url in visited_urls:
                continue
            visited_urls.add(canonical_target_url)
            captured_page = _capture_page(
                target_url,
                page_kind=str(target.get("page_kind") or "linked_resource"),
                parent_url=str(target.get("parent_url") or ""),
                source_link_label=str(target.get("source_link_label") or ""),
                crawl_depth=int(target.get("crawl_depth") or 0),
                visit_order=visit_order,
            )
            visit_order += 1
            if captured_page is None:
                continue
            crawled_pages.append(captured_page)

            final_url = str(captured_page.get("final_url") or target_url)
            canonical_final_url = _canonicalize_moodle_crawl_url(final_url, root_host=root_host)
            visited_urls.add(canonical_final_url)
            parent_url = str(target.get("parent_url") or "")
            if parent_url:
                for page_payload in reversed(crawled_pages):
                    page_final_url = _canonicalize_moodle_crawl_url(
                        str(page_payload.get("final_url") or page_payload.get("url") or ""),
                        root_host=root_host,
                    )
                    if page_final_url == _canonicalize_moodle_crawl_url(parent_url, root_host=root_host):
                        _update_page_link_resolution(page_payload, target_url, final_url, root_host=root_host)
                        break

            child_targets = _build_child_crawl_targets(
                captured_page,
                root_host=root_host,
                max_depth=_MOODLE_CRAWL_MAX_DEPTH,
                max_children=_MOODLE_CRAWL_MAX_CHILDREN_PER_PAGE,
            )
            for child_target in child_targets:
                child_url = str(child_target.get("url") or "").strip()
                canonical_child_url = _canonicalize_moodle_crawl_url(child_url, root_host=root_host)
                if not child_url or canonical_child_url in visited_urls:
                    continue
                crawl_queue.append(child_target)
    finally:
        context.close()
        try:
            http_session.close()
        except Exception:
            pass

    import tempfile, pathlib

    _tmp = pathlib.Path(tempfile.gettempdir())
    (_tmp / "moodle_debug_dashboard.html").write_text(html_dashboard, encoding="utf-8")
    (_tmp / "moodle_debug_calendar.html").write_text(html_calendar, encoding="utf-8")

    return {
        "assignments": assignments,
        "pages": [dashboard_page, calendar_page, *crawled_pages],
        "warnings": warnings,
    }


def list_moodle_courses(base_url: str = "") -> list[dict[str, str]]:
    """Descubre materias visibles del usuario autenticado en Moodle."""
    moodle_url, context, page = _authenticate_moodle_session(base_url)
    dashboard_url = f"{moodle_url}/my/"
    courses_url = f"{moodle_url}/my/courses.php"
    html_fragments: list[tuple[str, str]] = []
    try:
        for target_url in (dashboard_url, courses_url):
            try:
                page.goto(target_url, wait_until="domcontentloaded", timeout=30000)
                invalid_session, reason = _session_invalid(page, expected_root_host=_normalized_host(moodle_url))
                if invalid_session:
                    raise RuntimeError(f"sesión inválida al abrir {target_url}: {reason}")
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass
                current_url = str(page.url or target_url)
                current_html = page.content()
                html_fragments.append((current_url, current_html))
                suffix = "dashboard" if "my/" in target_url else "courses"
                _write_moodle_debug_log(f"courses_{suffix}.html", current_html)
                _write_moodle_debug_log(
                    f"courses_{suffix}.json",
                    {
                        "requested_url": target_url,
                        "final_url": current_url,
                    },
                )
            except Exception:
                continue
    finally:
        context.close()

    courses: list[dict[str, str]] = []
    seen: set[str] = set()
    for current_url, html in html_fragments:
        for course in _extract_visible_moodle_courses_from_html(html, current_url):
            course_url = course["course_url"]
            if course_url in seen:
                continue
            seen.add(course_url)
            courses.append(course)
    _write_moodle_debug_log("courses_discovered.json", {"courses": courses})
    return courses


def resolve_moodle_course_by_name(course_query: str, base_url: str = "") -> dict[str, object]:
    """Resuelve una materia Moodle visible por nombre o índice 1-based."""
    courses = list_moodle_courses(base_url)
    match, candidates, strategy = _resolve_course_match(courses, course_query)
    return {
        "query": course_query,
        "strategy": strategy,
        "matched_course": match,
        "candidates": candidates if candidates else ([] if match else courses[:10] if strategy in {"not_found", "index_out_of_range"} else []),
        "course_count": len(courses),
    }


def extract_moodle_course_audit_bundle(course_url: str, base_url: str = "") -> dict[str, object]:
    """Audita una materia/curso Moodle específico recorriendo su grafo interno."""
    moodle_url, context, page = _authenticate_moodle_session(base_url)
    http_session = _requests_session_from_playwright_context(context)
    warnings: list[str] = []
    root_host = _normalized_host(moodle_url)
    file_metadata_budget = {"remaining": _MOODLE_FILE_METADATA_LIMIT}
    visited_urls: set[str] = set()
    crawled_pages: list[dict[str, object]] = []

    target_course_url = course_url if str(course_url).startswith("http") else _moodle_absolute_url(moodle_url, course_url)
    _write_moodle_debug_log(
        "course_audit_request.json",
        {
            "input_course_url": course_url,
            "resolved_course_url": target_course_url,
            "base_url": moodle_url,
        },
    )

    def _capture_page(
        target_url: str,
        *,
        page_kind: str,
        parent_url: str = "",
        source_link_label: str = "",
        crawl_depth: int = 0,
        visit_order: int = 0,
    ) -> dict[str, object] | None:
        target_resource_type = _guess_moodle_resource_type(target_url, source_link_label)
        try:
            page.goto(target_url, wait_until="domcontentloaded", timeout=30000)
            invalid_session, reason = _session_invalid(page, expected_root_host=root_host, allow_external=True)
            if invalid_session:
                warnings.append(f"{page_kind}_invalid:{target_url}:{reason}")
                return None
            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            captured_html = page.content()
            payload = _extract_moodle_page_metadata(
                page_kind,
                target_url,
                str(page.url or target_url),
                captured_html,
                0,
                parent_url=parent_url,
                source_link_label=source_link_label,
                crawl_depth=crawl_depth,
                visit_order=visit_order,
            )
            _enrich_page_file_metadata(
                payload,
                http_session=http_session,
                warnings=warnings,
                budget=file_metadata_budget,
            )
            debug_name = f"course_visit_{visit_order:02d}_{payload.get('page_kind','page')}.html"
            _write_moodle_debug_log(debug_name, captured_html)
            return payload
        except Exception as exc:
            if target_resource_type == "document" or "Download is starting" in str(exc):
                try:
                    metadata = _fetch_file_http_metadata(http_session, target_url)
                    return _build_download_backed_audit_page(
                        target_url=target_url,
                        page_kind=page_kind,
                        source_link_label=source_link_label,
                        parent_url=parent_url,
                        crawl_depth=crawl_depth,
                        visit_order=visit_order,
                        metadata=metadata,
                    )
                except Exception as metadata_exc:
                    warnings.append(f"document_metadata_error:{target_url}:{metadata_exc}")
                    return None
            warnings.append(f"{page_kind}_error:{target_url}:{exc}")
            return None

    try:
        from collections import deque

        crawl_queue: deque[dict[str, object]] = deque(
            [
                {
                    "url": target_course_url,
                    "page_kind": "course_home",
                    "parent_url": "",
                    "source_link_label": "",
                    "crawl_depth": 0,
                }
            ]
        )
        visit_order = 1
        while crawl_queue and len(crawled_pages) < _MOODLE_CRAWL_MAX_PAGES:
            target = crawl_queue.popleft()
            target_url = str(target.get("url") or "").strip()
            canonical_target_url = _canonicalize_moodle_crawl_url(target_url, root_host=root_host)
            if not target_url or canonical_target_url in visited_urls:
                continue
            visited_urls.add(canonical_target_url)
            captured_page = _capture_page(
                target_url,
                page_kind=str(target.get("page_kind") or "linked_resource"),
                parent_url=str(target.get("parent_url") or ""),
                source_link_label=str(target.get("source_link_label") or ""),
                crawl_depth=int(target.get("crawl_depth") or 0),
                visit_order=visit_order,
            )
            visit_order += 1
            if captured_page is None:
                continue
            crawled_pages.append(captured_page)

            final_url = str(captured_page.get("final_url") or target_url)
            canonical_final_url = _canonicalize_moodle_crawl_url(final_url, root_host=root_host)
            visited_urls.add(canonical_final_url)
            parent_url = str(target.get("parent_url") or "")
            if parent_url:
                for page_payload in reversed(crawled_pages):
                    page_final_url = _canonicalize_moodle_crawl_url(
                        str(page_payload.get("final_url") or page_payload.get("url") or ""),
                        root_host=root_host,
                    )
                    if page_final_url == _canonicalize_moodle_crawl_url(parent_url, root_host=root_host):
                        _update_page_link_resolution(page_payload, target_url, final_url, root_host=root_host)
                        break

            child_targets = _build_child_crawl_targets(
                captured_page,
                root_host=root_host,
                max_depth=_MOODLE_CRAWL_MAX_DEPTH,
                max_children=_MOODLE_CRAWL_MAX_CHILDREN_PER_PAGE,
            )
            for child_target in child_targets:
                child_url = str(child_target.get("url") or "").strip()
                canonical_child_url = _canonicalize_moodle_crawl_url(child_url, root_host=root_host)
                if not child_url or canonical_child_url in visited_urls:
                    continue
                crawl_queue.append(child_target)
    finally:
        context.close()
        try:
            http_session.close()
        except Exception:
            pass

    course_root = crawled_pages[0] if crawled_pages else None
    course_name = str(course_root.get("title") or "") if isinstance(course_root, dict) else ""
    assignments = _extract_assignment_records_from_pages(crawled_pages, course_name=course_name)
    audit_stats = _summarize_moodle_audit_stats(
        crawled_pages,
        warnings,
        assignment_count=len(assignments),
        visited_count_raw=len(visited_urls),
    )
    summary_payload = {
        "course_url": target_course_url,
        "course_name": course_name,
        "page_count": len(crawled_pages),
        "assignment_count": len(assignments),
        "warnings": warnings,
        **audit_stats["stats"],
        "resource_type_counts": audit_stats["resource_type_counts"],
        "warning_types": audit_stats["warning_types"],
    }
    _write_moodle_debug_log(
        "course_audit_summary.json",
        summary_payload,
    )
    return {
        "course_url": target_course_url,
        "course_name": course_name,
        "assignments": assignments,
        "pages": crawled_pages,
        "warnings": warnings,
        **audit_stats["stats"],
        "resource_type_counts": audit_stats["resource_type_counts"],
        "warning_types": audit_stats["warning_types"],
    }


def extract_moodle_assignments(
    base_url: str = "",
) -> list[dict[str, str]]:
    """Retorna tareas Moodle estructuradas desde el dashboard/calendario."""
    bundle = extract_moodle_audit_bundle(base_url)
    assignments = bundle.get("assignments")
    return assignments if isinstance(assignments, list) else []


@tool
def scrape_moodle_assignments(
    base_url: Annotated[str, Field(description="URL base del Moodle, ej: https://virtual.instituto.edu/mld-1. Si se omite, usa MOODLE_URL del .env")] = "",
) -> str:
    """Inicia sesión en Moodle con Playwright y extrae tareas pendientes (incluyendo vencidas). Requiere MOODLE_USERNAME y MOODLE_PASSWORD en variables de entorno."""
    try:
        assignments = extract_moodle_assignments(base_url)
    except Exception as exc:
        return f"Error durante el scraping de Moodle: {exc}"

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


@tool
def scrape_moodle_courses(
    base_url: Annotated[str, Field(description="URL base del Moodle. Si se omite, usa MOODLE_URL del .env")] = "",
) -> str:
    """Lista las materias/cursos visibles del usuario autenticado en Moodle."""
    try:
        courses = list_moodle_courses(base_url)
    except Exception as exc:
        return f"Error al listar materias de Moodle: {exc}"

    if not courses:
        return "No encontré materias visibles en Moodle."

    lines = ["MATERIAS EN MOODLE", "─" * 30, ""]
    for idx, course in enumerate(courses, start=1):
        lines.append(f"{idx}. {course['course_name']}")
        lines.append(f"   URL    {course['course_url']}")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "scrape_website_simple",
    "scrape_website_dynamic",
    "scrape_website_with_json_capture",
    "web_fetch",
    "fetch_web_page",
    "extract_moodle_audit_bundle",
    "extract_moodle_course_audit_bundle",
    "extract_moodle_assignments",
    "list_moodle_courses",
    "resolve_moodle_course_by_name",
    "fetch_moodle_submission_statuses",
    "scrape_moodle_assignments",
    "scrape_moodle_courses",
]
