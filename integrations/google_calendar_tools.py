"""LangChain tool wrappers sobre Google Calendar MCP."""
from __future__ import annotations

import json
from typing import Annotated, Optional

from langchain_core.tools import tool
from pydantic import Field

from application.services.request_runtime import is_mcp_enabled
from features.web_scraping.application.moodle_audit import (
    load_moodle_audit_snapshot,
    persist_moodle_audit_snapshot,
)
from features.web_scraping.application.moodle_artifacts import (
    load_valid_moodle_assignments_from_artifact,
    load_validated_moodle_artifact,
    mark_moodle_artifact_calendar_sync,
    persist_moodle_artifacts,
)
from features.web_scraping.application.moodle_normalization import normalize_moodle_assignments
from features.web_scraping.application.moodle_review_render import (
    render_moodle_assignments_chat,
    render_moodle_assignments_review,
)
from features.web_scraping.application.moodle_submission_status_enrichment import (
    enrich_moodle_submission_statuses,
)
from features.web_scraping.application.moodle_validation import validate_moodle_assignments
from features.web_scraping.infrastructure.scraping_tools import (
    extract_moodle_audit_bundle,
    extract_moodle_course_audit_bundle,
    list_moodle_courses,
    resolve_moodle_course_by_name,
)
from integrations.google_calendar_dedupe import partition_calendar_drafts
from integrations.google_calendar_mcp import create_event, delete_event, list_events, update_event
from integrations.google_calendar_sync import build_calendar_draft_events


_GOOGLE_CALENDAR_MCP_KEY = "google_calendar"


def _guard_google_calendar() -> str | None:
    if is_mcp_enabled(_GOOGLE_CALENDAR_MCP_KEY):
        return None
    return (
        "Google Calendar MCP está desactivado para este turno. "
        "Activá Google Calendar en el chat antes de usar estas herramientas."
    )


def _format_payload(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def prepare_moodle_assignments_payload(base_url: str = "") -> dict[str, object]:
    print(
        f"[WEB_FLOW] tool=scrape_moodle_assignments_for_review base_url={base_url or '(env)'}",
        flush=True,
    )
    audit_bundle = extract_moodle_audit_bundle(base_url)
    raw_assignments = audit_bundle.get("assignments") if isinstance(audit_bundle.get("assignments"), list) else []
    audit_pages = audit_bundle.get("pages") if isinstance(audit_bundle.get("pages"), list) else []
    audit_warnings = audit_bundle.get("warnings") if isinstance(audit_bundle.get("warnings"), list) else []
    audit_paths = persist_moodle_audit_snapshot(
        raw_assignments,
        base_url=base_url,
        pages=audit_pages,
        warnings=[str(item) for item in audit_warnings],
        stats={
            "visited_count_raw": int(audit_bundle.get("visited_count_raw") or len(audit_pages)),
            "retained_page_count": int(audit_bundle.get("retained_page_count") or len(audit_pages)),
            "external_redirect_count": int(audit_bundle.get("external_redirect_count") or 0),
            "download_document_count": int(audit_bundle.get("download_document_count") or 0),
            "assignment_like_count": int(audit_bundle.get("assignment_like_count") or len(raw_assignments)),
        },
        resource_type_counts=dict(audit_bundle.get("resource_type_counts") or {}),
    )
    enriched_assignments = enrich_moodle_submission_statuses(raw_assignments, base_url=base_url)
    normalized = normalize_moodle_assignments(enriched_assignments)
    validated = validate_moodle_assignments(normalized)
    review_markdown = render_moodle_assignments_review(validated)
    paths = persist_moodle_artifacts(validated, review_markdown)
    return {
        "audit_json_path": str(audit_paths.json_path),
        "audit_schema_path": str(audit_paths.schema_path),
        "audit_summary_path": str(audit_paths.summary_path),
        "audit_job_uid": load_moodle_audit_snapshot(audit_paths.json_path).meta.job_uid,
        "json_path": str(paths.json_path),
        "markdown_path": str(paths.markdown_path),
        "valid_count": len(validated.valid),
        "invalid_count": len(validated.invalid),
        "issue_count": len(validated.issues),
        "issues": [issue.to_dict() for issue in validated.issues],
        "markdown_preview": review_markdown[:1600],
        "chat_response": render_moodle_assignments_chat(validated),
    }


def prepare_moodle_course_audit_payload(course_url: str, base_url: str = "") -> dict[str, object]:
    print(
        f"[WEB_FLOW] tool=scrape_moodle_course_for_audit course_url={course_url!r} base_url={base_url or '(env)'}",
        flush=True,
    )
    audit_bundle = extract_moodle_course_audit_bundle(course_url, base_url=base_url)
    raw_assignments = audit_bundle.get("assignments") if isinstance(audit_bundle.get("assignments"), list) else []
    audit_pages = audit_bundle.get("pages") if isinstance(audit_bundle.get("pages"), list) else []
    audit_warnings = audit_bundle.get("warnings") if isinstance(audit_bundle.get("warnings"), list) else []
    resolved_course_url = str(audit_bundle.get("course_url") or course_url)
    audit_paths = persist_moodle_audit_snapshot(
        raw_assignments,
        base_url=base_url,
        pages=audit_pages,
        warnings=[str(item) for item in audit_warnings],
        stats={
            "visited_count_raw": int(audit_bundle.get("visited_count_raw") or len(audit_pages)),
            "retained_page_count": int(audit_bundle.get("retained_page_count") or len(audit_pages)),
            "external_redirect_count": int(audit_bundle.get("external_redirect_count") or 0),
            "download_document_count": int(audit_bundle.get("download_document_count") or 0),
            "assignment_like_count": int(audit_bundle.get("assignment_like_count") or len(raw_assignments)),
        },
        resource_type_counts=dict(audit_bundle.get("resource_type_counts") or {}),
    )
    snapshot = load_moodle_audit_snapshot(audit_paths.json_path)
    meta_stats = getattr(snapshot.meta, "stats", {}) or {}
    meta_resource_type_counts = getattr(snapshot.meta, "resource_type_counts", {}) or {}
    return {
        "course_url": resolved_course_url,
        "course_name": str(audit_bundle.get("course_name") or ""),
        "audit_json_path": str(audit_paths.json_path),
        "audit_schema_path": str(audit_paths.schema_path),
        "audit_summary_path": str(audit_paths.summary_path),
        "audit_job_uid": snapshot.meta.job_uid,
        "page_count": len(snapshot.pages),
        "assignment_count": len(snapshot.assignments),
        "warning_count": len(snapshot.warnings),
        "warnings": snapshot.warnings,
        "root_page_title": snapshot.pages[0].title if snapshot.pages else "",
        "resource_type_counts": dict(audit_bundle.get("resource_type_counts") or meta_resource_type_counts),
        "warning_types": dict(audit_bundle.get("warning_types") or {}),
        "visited_count_raw": int(audit_bundle.get("visited_count_raw") or meta_stats.get("visited_count_raw", 0)),
        "retained_page_count": int(audit_bundle.get("retained_page_count") or meta_stats.get("retained_page_count", len(snapshot.pages))),
        "external_redirect_count": int(audit_bundle.get("external_redirect_count") or 0),
        "download_document_count": int(audit_bundle.get("download_document_count") or 0),
        "assignment_like_count": int(audit_bundle.get("assignment_like_count") or len(snapshot.assignments)),
    }


def prepare_moodle_courses_payload(base_url: str = "") -> dict[str, object]:
    print(
        f"[WEB_FLOW] tool=list_moodle_courses base_url={base_url or '(env)'}",
        flush=True,
    )
    courses = list_moodle_courses(base_url)
    return {
        "course_count": len(courses),
        "courses": [
            {
                "index": idx,
                "course_name": course["course_name"],
                "course_url": course["course_url"],
            }
            for idx, course in enumerate(courses, start=1)
        ],
    }


def prepare_moodle_course_audit_by_name_payload(course_query: str, base_url: str = "") -> dict[str, object]:
    print(
        f"[WEB_FLOW] tool=scrape_moodle_course_by_name course_query={course_query!r} base_url={base_url or '(env)'}",
        flush=True,
    )
    resolution = resolve_moodle_course_by_name(course_query, base_url=base_url)
    matched_course = resolution.get("matched_course")
    if not isinstance(matched_course, dict):
        return {
            "resolved": False,
            "query": course_query,
            "strategy": resolution.get("strategy", ""),
            "course_count": resolution.get("course_count", 0),
            "candidates": resolution.get("candidates", []),
            "message": (
                "No pude resolver la materia de forma unívoca. "
                "Probá con el nombre exacto o con el número de la lista de materias."
            ),
        }

    payload = prepare_moodle_course_audit_payload(
        str(matched_course.get("course_url") or ""),
        base_url=base_url,
    )
    payload["resolved"] = True
    payload["resolved_from_query"] = course_query
    payload["resolution_strategy"] = resolution.get("strategy", "")
    return payload


def create_calendar_events_from_validated_tasks_payload(
    artifact_path: str,
    calendar_id: str | None = None,
) -> dict[str, object] | str:
    print(
        f"[WEB_FLOW] tool=create_calendar_events_from_validated_tasks artifact_path={artifact_path!r}",
        flush=True,
    )
    blocked = _guard_google_calendar()
    if blocked:
        return blocked

    artifact_payload = load_validated_moodle_artifact(artifact_path)
    meta = artifact_payload.get("meta") if isinstance(artifact_payload.get("meta"), dict) else {}
    if not bool(meta.get("approved")):
        return "El JSON todavía no fue aprobado en el chat. Revisalo y aprobalo antes de crear eventos."

    assignments = load_valid_moodle_assignments_from_artifact(artifact_path)
    drafts = build_calendar_draft_events(assignments)
    if not drafts:
        mark_moodle_artifact_calendar_sync(artifact_path, 0)
        return {
            "artifact_path": artifact_path,
            "created_count": 0,
            "events": [],
            "message": "No hay tareas válidas con fecha para crear eventos.",
        }

    dedupe = partition_calendar_drafts(drafts, calendar_id=calendar_id)
    if not dedupe.to_create:
        mark_moodle_artifact_calendar_sync(artifact_path, 0)
        return {
            "artifact_path": artifact_path,
            "created_count": 0,
            "duplicate_count": len(dedupe.duplicates),
            "events": [],
            "duplicates": dedupe.duplicates,
            "message": "No se crearon eventos porque todas las tareas ya existen en Google Calendar.",
        }

    created: list[dict[str, object]] = []
    for draft in dedupe.to_create:
        result = create_event(
            {
                "summary": draft.summary,
                "start": draft.start,
                "end": draft.end,
                "description": draft.description,
                "location": draft.location or None,
                "calendar_id": calendar_id,
            }
        )
        created.append(
            {
                "source_title": draft.source_title,
                "summary": draft.summary,
                "start": draft.start,
                "end": draft.end,
                "calendar_event": result,
            }
        )

    mark_moodle_artifact_calendar_sync(artifact_path, len(created))
    return {
        "artifact_path": artifact_path,
        "created_count": len(created),
        "duplicate_count": len(dedupe.duplicates),
        "events": created,
        "duplicates": dedupe.duplicates,
    }


@tool
def list_calendar_events(
    time_min: Annotated[Optional[str], Field(description="Fecha/hora mínima RFC3339. Si se omite, usa ahora.")] = None,
    time_max: Annotated[Optional[str], Field(description="Fecha/hora máxima RFC3339.")] = None,
    max_results: Annotated[int, Field(description="Cantidad máxima de eventos", ge=1, le=50)] = 10,
    query: Annotated[Optional[str], Field(description="Texto libre para filtrar eventos.")] = None,
    calendar_id: Annotated[Optional[str], Field(description="Calendar ID; por defecto usa primary.")] = None,
) -> str:
    """Lista eventos de Google Calendar dentro de una ventana temporal."""
    blocked = _guard_google_calendar()
    if blocked:
        return blocked
    try:
        return _format_payload(
            list_events(
                {
                    "time_min": time_min,
                    "time_max": time_max,
                    "max_results": max_results,
                    "query": query,
                    "calendar_id": calendar_id,
                }
            )
        )
    except Exception as exc:  # noqa: BLE001 - tools deben responder texto legible
        return f"Error al listar eventos de Google Calendar: {exc}"


@tool
def create_calendar_event(
    summary: Annotated[str, Field(description="Título del evento.")],
    start: Annotated[Optional[str], Field(description="Inicio RFC3339 o YYYY-MM-DD.")] = None,
    end: Annotated[Optional[str], Field(description="Fin RFC3339 o YYYY-MM-DD.")] = None,
    timezone_name: Annotated[Optional[str], Field(description="Zona horaria IANA, ej. America/Argentina/Buenos_Aires.")] = None,
    description: Annotated[Optional[str], Field(description="Descripción del evento.")] = None,
    location: Annotated[Optional[str], Field(description="Ubicación del evento.")] = None,
    calendar_id: Annotated[Optional[str], Field(description="Calendar ID; por defecto usa primary.")] = None,
) -> str:
    """Crea un evento en Google Calendar."""
    blocked = _guard_google_calendar()
    if blocked:
        return blocked
    try:
        return _format_payload(
            create_event(
                {
                    "summary": summary,
                    "start": start,
                    "end": end,
                    "timezone_name": timezone_name,
                    "description": description,
                    "location": location,
                    "calendar_id": calendar_id,
                }
            )
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error al crear evento en Google Calendar: {exc}"


@tool
def update_calendar_event(
    event_id: Annotated[str, Field(description="ID del evento a actualizar.")],
    summary: Annotated[Optional[str], Field(description="Nuevo título del evento.")] = None,
    start: Annotated[Optional[str], Field(description="Nuevo inicio RFC3339 o YYYY-MM-DD.")] = None,
    end: Annotated[Optional[str], Field(description="Nuevo fin RFC3339 o YYYY-MM-DD.")] = None,
    timezone_name: Annotated[Optional[str], Field(description="Zona horaria IANA.")] = None,
    description: Annotated[Optional[str], Field(description="Nueva descripción.")] = None,
    location: Annotated[Optional[str], Field(description="Nueva ubicación.")] = None,
    calendar_id: Annotated[Optional[str], Field(description="Calendar ID; por defecto usa primary.")] = None,
) -> str:
    """Actualiza un evento existente en Google Calendar."""
    blocked = _guard_google_calendar()
    if blocked:
        return blocked
    try:
        return _format_payload(
            update_event(
                {
                    "event_id": event_id,
                    "summary": summary,
                    "start": start,
                    "end": end,
                    "timezone_name": timezone_name,
                    "description": description,
                    "location": location,
                    "calendar_id": calendar_id,
                }
            )
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error al actualizar evento de Google Calendar: {exc}"


@tool
def delete_calendar_event(
    event_id: Annotated[str, Field(description="ID del evento a borrar.")],
    calendar_id: Annotated[Optional[str], Field(description="Calendar ID; por defecto usa primary.")] = None,
) -> str:
    """Borra un evento de Google Calendar."""
    blocked = _guard_google_calendar()
    if blocked:
        return blocked
    try:
        return _format_payload(delete_event({"event_id": event_id, "calendar_id": calendar_id}))
    except Exception as exc:  # noqa: BLE001
        return f"Error al borrar evento de Google Calendar: {exc}"


@tool
def scrape_moodle_assignments_for_review(
    base_url: Annotated[str, Field(description="URL base de Moodle. Si se omite, usa MOODLE_URL.")] = "",
) -> str:
    """Extrae tareas de Moodle, las normaliza, valida y genera artefactos JSON/Markdown para revisión humana antes de calendarizar."""
    try:
        return _format_payload(prepare_moodle_assignments_payload(base_url))
    except Exception as exc:  # noqa: BLE001
        return f"Error al preparar tareas de Moodle para calendario: {exc}"


@tool
def prepare_moodle_assignments_for_calendar(
    base_url: Annotated[str, Field(description="URL base de Moodle. Si se omite, usa MOODLE_URL.")] = "",
) -> str:
    """Alias retrocompatible de scrape_moodle_assignments_for_review."""
    return scrape_moodle_assignments_for_review.invoke({"base_url": base_url})


@tool
def scrape_moodle_course_for_audit(
    course_url: Annotated[str, Field(description="URL absoluta o relativa de la materia Moodle, por ejemplo /course/view.php?id=123.")],
    base_url: Annotated[str, Field(description="URL base de Moodle. Si se omite, usa MOODLE_URL.")] = "",
) -> str:
    """Audita una materia Moodle específica y persiste un JSON auditable con páginas, URLs, redirecciones y metadata de archivos."""
    try:
        return _format_payload(prepare_moodle_course_audit_payload(course_url, base_url=base_url))
    except Exception as exc:  # noqa: BLE001
        return f"Error al auditar la materia específica de Moodle: {exc}"


@tool
def list_moodle_courses_for_user(
    base_url: Annotated[str, Field(description="URL base de Moodle. Si se omite, usa MOODLE_URL.")] = "",
) -> str:
    """Lista las materias visibles del usuario autenticado de forma amigable para UI."""
    try:
        return _format_payload(prepare_moodle_courses_payload(base_url=base_url))
    except Exception as exc:  # noqa: BLE001
        return f"Error al listar materias de Moodle: {exc}"


@tool
def scrape_moodle_course_by_name(
    course_query: Annotated[str, Field(description="Nombre de la materia o índice 1-based de la lista, por ejemplo 'Historia Argentina' o '2'.")],
    base_url: Annotated[str, Field(description="URL base de Moodle. Si se omite, usa MOODLE_URL.")] = "",
) -> str:
    """Audita una materia Moodle por nombre o número de lista, resolviendo la URL internamente."""
    try:
        return _format_payload(prepare_moodle_course_audit_by_name_payload(course_query, base_url=base_url))
    except Exception as exc:  # noqa: BLE001
        return f"Error al auditar materia de Moodle por nombre: {exc}"


@tool
def create_calendar_events_from_validated_tasks(
    artifact_path: Annotated[str, Field(description="Path al JSON validado generado por prepare_moodle_assignments_for_calendar.")],
    calendar_id: Annotated[Optional[str], Field(description="Calendar ID; por defecto usa primary.")] = None,
) -> str:
    """Crea eventos de Google Calendar a partir de un artifact JSON previamente validado."""
    try:
        payload = create_calendar_events_from_validated_tasks_payload(artifact_path, calendar_id=calendar_id)
        if isinstance(payload, str):
            return payload
        return _format_payload(payload)
    except Exception as exc:  # noqa: BLE001
        return f"Error al crear eventos desde tareas validadas: {exc}"


__all__ = [
    "prepare_moodle_assignments_payload",
    "prepare_moodle_courses_payload",
    "prepare_moodle_course_audit_payload",
    "prepare_moodle_course_audit_by_name_payload",
    "create_calendar_events_from_validated_tasks_payload",
    "scrape_moodle_assignments_for_review",
    "list_moodle_courses_for_user",
    "scrape_moodle_course_by_name",
    "scrape_moodle_course_for_audit",
    "prepare_moodle_assignments_for_calendar",
    "create_calendar_events_from_validated_tasks",
    "create_calendar_event",
    "delete_calendar_event",
    "list_calendar_events",
    "update_calendar_event",
]
