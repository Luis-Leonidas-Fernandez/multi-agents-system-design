"""Cliente interno mínimo para sincronizar tareas con una database de Notion."""
from __future__ import annotations

import os
import re
from urllib.parse import urlparse
from typing import Any

from integrations.notion_task_models import NotionTaskRecord


_NOTION_API_BASE_URL = "https://api.notion.com/v1"
_NOTION_API_VERSION = os.getenv("NOTION_API_VERSION", "2022-06-28")
_TITLE_PROPERTY = "Name"
_EXTERNAL_ID_PROPERTY = "External ID"
_COURSE_PROPERTY = "Course"
_DUE_DATE_PROPERTY = "Due Date"
_STATUS_PROPERTY = "Status"
_SOURCE_URL_PROPERTY = "Source URL"
_SOURCE_PROPERTY = "Source"
_RAW_DATE_PROPERTY = "Raw Date Text"
_LAST_SYNCED_AT_PROPERTY = "Last Synced At"


class NotionIntegrationError(RuntimeError):
    """Error base de la integración con Notion."""


class NotionConfigError(NotionIntegrationError):
    """Falta configuración local requerida."""


class NotionAuthError(NotionIntegrationError):
    """Credenciales inválidas o revocadas."""


class NotionPermissionError(NotionIntegrationError):
    """La integration no tiene acceso al recurso solicitado."""


class NotionDatabaseNotFoundError(NotionIntegrationError):
    """El database id no existe o no es accesible."""


class NotionSchemaError(NotionIntegrationError):
    """El schema de la database no coincide con lo esperado."""


class NotionRateLimitError(NotionIntegrationError):
    """Rate limit remoto de Notion."""


class NotionApiTransientError(NotionIntegrationError):
    """Error temporal remoto de Notion."""


def _notion_api_key() -> str:
    value = os.getenv("NOTION_API_KEY", "").strip()
    if not value:
        raise NotionConfigError("Falta NOTION_API_KEY en variables de entorno.")
    return value


def _notion_database_id() -> str:
    raw_value = os.getenv("NOTION_DATABASE_ID", "").strip()
    if not raw_value:
        raise NotionConfigError("Falta NOTION_DATABASE_ID en variables de entorno.")
    value = raw_value
    if "notion.so/" in value:
        parsed = urlparse(value)
        value = parsed.path.rsplit("-", 1)[-1].strip("/")
    value = value.split("?", 1)[0].strip().strip("/")
    match = re.search(r"([0-9a-fA-F]{32})", value.replace("-", ""))
    if match:
        return match.group(1)
    raise NotionConfigError(
        "NOTION_DATABASE_ID debe ser solo el id de la database de Notion (sin `?v=` ni URL completa inválida)."
    )


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_notion_api_key()}",
        "Content-Type": "application/json",
        "Notion-Version": _NOTION_API_VERSION,
    }


def _api_request(method: str, path: str, *, json_body: dict[str, Any] | None = None) -> dict[str, Any]:
    import requests

    url = f"{_NOTION_API_BASE_URL}{path}"
    print(f"[NOTION_DEBUG] request method={method.upper()} path={path}", flush=True)
    try:
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=_headers(),
            json=json_body,
            timeout=30,
        )
    except requests.Timeout as exc:
        print(f"[NOTION_DEBUG] timeout path={path}", flush=True)
        raise NotionApiTransientError("Timeout al hablar con Notion.") from exc
    except requests.RequestException as exc:
        print(f"[NOTION_DEBUG] request_exception path={path} error={exc}", flush=True)
        raise NotionApiTransientError(f"Error de red al hablar con Notion: {exc}") from exc

    if response.status_code >= 400:
        message = ""
        try:
            error_payload = response.json()
            if isinstance(error_payload, dict):
                message = str(error_payload.get("message") or error_payload.get("error") or "")
        except Exception:
            message = response.text[:400]
        print(
            f"[NOTION_DEBUG] http_error status={response.status_code} path={path} message={message!r}",
            flush=True,
        )
        lowered = message.lower()
        if response.status_code == 400:
            raise NotionSchemaError(
                f"Notion rechazó el payload o el schema de la database no coincide. Detalle: {message or 'sin detalle'}"
            )
        if response.status_code in {401, 403}:
            if "database" in lowered and ("not found" in lowered or "could not be found" in lowered):
                raise NotionDatabaseNotFoundError(
                    "NOTION_DATABASE_ID no corresponde a una database accesible para la integration."
                )
            raise NotionPermissionError(
                f"La integration de Notion no tiene acceso suficiente o el token es inválido. Detalle: {message or 'sin detalle'}"
            )
        if response.status_code == 404:
            raise NotionDatabaseNotFoundError(
                "La database de Notion no existe o no es accesible con el NOTION_DATABASE_ID configurado."
            )
        if response.status_code == 429:
            raise NotionRateLimitError("Notion devolvió rate limit. Reintentá en unos segundos.")
        if response.status_code >= 500:
            raise NotionApiTransientError(
                f"Notion devolvió un error temporal ({response.status_code})."
            )
        raise NotionIntegrationError(
            f"Error inesperado de Notion ({response.status_code}): {message or 'sin detalle'}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        print(f"[NOTION_DEBUG] invalid_json path={path}", flush=True)
        raise NotionIntegrationError("Notion devolvió una respuesta no JSON.") from exc
    if not isinstance(payload, dict):
        raise NotionIntegrationError("Respuesta inesperada de Notion API.")
    return payload


def _title_property(value: str) -> dict[str, Any]:
    return {
        _TITLE_PROPERTY: {
            "title": [{"type": "text", "text": {"content": value}}],
        }
    }


def _rich_text_property(name: str, value: str) -> dict[str, Any]:
    return {
        name: {
            "rich_text": [{"type": "text", "text": {"content": value}}] if value else [],
        }
    }


def _date_property(name: str, value: str) -> dict[str, Any]:
    return {name: {"date": {"start": value} if value else None}}


def _select_property(name: str, value: str) -> dict[str, Any]:
    return {name: {"select": {"name": value} if value else None}}


def _url_property(name: str, value: str) -> dict[str, Any]:
    return {name: {"url": value or None}}


def build_notion_task_properties(record: NotionTaskRecord) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    properties.update(_title_property(record.title))
    properties.update(_rich_text_property(_EXTERNAL_ID_PROPERTY, record.external_id))
    properties.update(_rich_text_property(_COURSE_PROPERTY, record.course))
    properties.update(_date_property(_DUE_DATE_PROPERTY, record.due_date))
    properties.update(_select_property(_STATUS_PROPERTY, record.status))
    properties.update(_url_property(_SOURCE_URL_PROPERTY, record.source_url))
    properties.update(_rich_text_property(_SOURCE_PROPERTY, record.source))
    properties.update(_rich_text_property(_RAW_DATE_PROPERTY, record.raw_date_text))
    properties.update(_date_property(_LAST_SYNCED_AT_PROPERTY, record.last_synced_at))
    return properties


def _read_rich_text_value(prop: dict[str, Any]) -> str:
    rich_text = prop.get("rich_text")
    if isinstance(rich_text, list):
        return "".join(str(part.get("plain_text") or "") for part in rich_text if isinstance(part, dict)).strip()
    title = prop.get("title")
    if isinstance(title, list):
        return "".join(str(part.get("plain_text") or "") for part in title if isinstance(part, dict)).strip()
    return ""


def _read_date_value(prop: dict[str, Any]) -> str:
    date_value = prop.get("date")
    if isinstance(date_value, dict):
        return str(date_value.get("start") or "")
    return ""


def _read_select_value(prop: dict[str, Any]) -> str:
    select_value = prop.get("select")
    if isinstance(select_value, dict):
        return str(select_value.get("name") or "")
    return ""


def _read_url_value(prop: dict[str, Any]) -> str:
    return str(prop.get("url") or "")


def extract_notion_task_snapshot(page_payload: dict[str, Any]) -> dict[str, str]:
    properties = page_payload.get("properties")
    if not isinstance(properties, dict):
        return {}
    return {
        "external_id": _read_rich_text_value(properties.get(_EXTERNAL_ID_PROPERTY, {})),
        "title": _read_rich_text_value(properties.get(_TITLE_PROPERTY, {})),
        "course": _read_rich_text_value(properties.get(_COURSE_PROPERTY, {})),
        "due_date": _read_date_value(properties.get(_DUE_DATE_PROPERTY, {})),
        "status": _read_select_value(properties.get(_STATUS_PROPERTY, {})),
        "source_url": _read_url_value(properties.get(_SOURCE_URL_PROPERTY, {})),
        "source": _read_rich_text_value(properties.get(_SOURCE_PROPERTY, {})),
        "raw_date_text": _read_rich_text_value(properties.get(_RAW_DATE_PROPERTY, {})),
    }


def notion_task_matches_record(page_payload: dict[str, Any], record: NotionTaskRecord) -> bool:
    snapshot = extract_notion_task_snapshot(page_payload)
    expected = {
        "external_id": record.external_id,
        "title": record.title,
        "course": record.course,
        "due_date": record.due_date,
        "status": record.status,
        "source_url": record.source_url,
        "source": record.source,
        "raw_date_text": record.raw_date_text,
    }
    return snapshot == expected


def query_task_by_external_id(external_id: str) -> dict[str, Any] | None:
    payload = _api_request(
        "POST",
        f"/databases/{_notion_database_id()}/query",
        json_body={
            "page_size": 1,
            "filter": {
                "property": _EXTERNAL_ID_PROPERTY,
                "rich_text": {
                    "equals": external_id,
                },
            },
        },
    )
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return None
    first = results[0]
    return first if isinstance(first, dict) else None


def create_task_page(record: NotionTaskRecord) -> dict[str, Any]:
    return _api_request(
        "POST",
        "/pages",
        json_body={
            "parent": {"database_id": _notion_database_id()},
            "properties": build_notion_task_properties(record),
        },
    )


def update_task_page(page_id: str, record: NotionTaskRecord) -> dict[str, Any]:
    return _api_request(
        "PATCH",
        f"/pages/{page_id}",
        json_body={
            "properties": build_notion_task_properties(record),
        },
    )


__all__ = [
    "build_notion_task_properties",
    "create_task_page",
    "extract_notion_task_snapshot",
    "NotionApiTransientError",
    "NotionAuthError",
    "NotionConfigError",
    "NotionDatabaseNotFoundError",
    "NotionIntegrationError",
    "NotionPermissionError",
    "NotionRateLimitError",
    "NotionSchemaError",
    "notion_task_matches_record",
    "query_task_by_external_id",
    "update_task_page",
]
