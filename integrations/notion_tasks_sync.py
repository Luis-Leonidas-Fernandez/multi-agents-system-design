"""Orquestación de upsert de tareas Moodle validadas hacia Notion."""
from __future__ import annotations

from application.services.request_runtime import get_request_runtime_config
from features.web_scraping.application.moodle_artifacts import (
    list_session_moodle_artifacts,
    load_valid_moodle_assignments_from_artifact,
    load_validated_moodle_artifact,
    mark_moodle_artifact_notion_sync,
)
from features.web_scraping.application.moodle_notion_mapper import map_moodle_assignments_to_notion_records
from integrations.notion_client import (
    NotionConfigError,
    NotionDatabaseNotFoundError,
    NotionIntegrationError,
    NotionPermissionError,
    NotionRateLimitError,
    NotionSchemaError,
    create_task_page,
    notion_task_matches_record,
    query_task_by_external_id,
    update_task_page,
)
from integrations.notion_task_models import NotionSyncSummary, NotionUpsertResult


def _resolve_artifact_path(artifact_path: str) -> str:
    normalized = str(artifact_path or "").strip()
    if normalized:
        return normalized
    session_id = get_request_runtime_config().session_id
    if not session_id:
        raise ValueError("Falta artifact_path y no hay session_id activo para resolver el último artifact Moodle.")
    artifacts = list_session_moodle_artifacts(session_id)
    if not artifacts:
        raise ValueError("No encontré artifacts Moodle en la sesión actual.")
    return artifacts[0].json_path


def _build_result(
    *,
    action: str,
    external_id: str,
    title: str,
    payload: dict | None = None,
    message: str = "",
) -> NotionUpsertResult:
    page_payload = payload if isinstance(payload, dict) else {}
    return NotionUpsertResult(
        action=action,  # type: ignore[arg-type]
        external_id=external_id,
        title=title,
        page_id=str(page_payload.get("id") or ""),
        page_url=str(page_payload.get("url") or ""),
        message=message,
    )


def sync_validated_moodle_artifact_to_notion_payload(artifact_path: str = "") -> dict[str, object]:
    resolved_artifact_path = _resolve_artifact_path(artifact_path)
    print(f"[NOTION_DEBUG] sync_start artifact_path={resolved_artifact_path}", flush=True)
    artifact_payload = load_validated_moodle_artifact(resolved_artifact_path)
    meta = artifact_payload.get("meta") if isinstance(artifact_payload.get("meta"), dict) else {}
    if not bool(meta.get("approved")):
        raise ValueError("El JSON todavía no fue aprobado en el chat. Revisalo y aprobalo antes de sincronizar a Notion.")

    assignments = load_valid_moodle_assignments_from_artifact(resolved_artifact_path)
    records = map_moodle_assignments_to_notion_records(assignments)
    print(f"[NOTION_DEBUG] sync_records count={len(records)}", flush=True)
    summary = NotionSyncSummary()
    created: list[NotionUpsertResult] = []
    updated: list[NotionUpsertResult] = []
    skipped: list[NotionUpsertResult] = []
    errors: list[NotionUpsertResult] = []

    for record in records:
        try:
            print(
                f"[NOTION_DEBUG] upsert_start external_id={record.external_id!r} title={record.title!r}",
                flush=True,
            )
            existing = query_task_by_external_id(record.external_id)
            if existing is None:
                created_payload = create_task_page(record)
                print(
                    f"[NOTION_DEBUG] upsert_created external_id={record.external_id!r} page_id={created_payload.get('id')!r}",
                    flush=True,
                )
                created.append(
                    _build_result(
                        action="created",
                        external_id=record.external_id,
                        title=record.title,
                        payload=created_payload,
                        message="Página creada en Notion.",
                    )
                )
                continue

            if notion_task_matches_record(existing, record):
                print(
                    f"[NOTION_DEBUG] upsert_skipped external_id={record.external_id!r}",
                    flush=True,
                )
                skipped.append(
                    _build_result(
                        action="skipped",
                        external_id=record.external_id,
                        title=record.title,
                        payload=existing,
                        message="La tarea ya estaba sincronizada sin cambios.",
                    )
                )
                continue

            page_id = str(existing.get("id") or "").strip()
            if not page_id:
                raise ValueError("La página existente de Notion no tiene id.")
            updated_payload = update_task_page(page_id, record)
            print(
                f"[NOTION_DEBUG] upsert_updated external_id={record.external_id!r} page_id={page_id!r}",
                flush=True,
            )
            updated.append(
                _build_result(
                    action="updated",
                    external_id=record.external_id,
                    title=record.title,
                    payload=updated_payload,
                    message="Página actualizada en Notion.",
                )
            )
        except (
            NotionConfigError,
            NotionPermissionError,
            NotionDatabaseNotFoundError,
            NotionSchemaError,
            NotionRateLimitError,
            NotionIntegrationError,
        ) as exc:
            print(
                f"[NOTION_DEBUG] upsert_error external_id={record.external_id!r} kind={exc.__class__.__name__} message={exc}",
                flush=True,
            )
            errors.append(
                _build_result(
                    action="error",
                    external_id=record.external_id,
                    title=record.title,
                    message=str(exc),
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[NOTION_DEBUG] upsert_error external_id={record.external_id!r} kind=UnexpectedError message={exc}",
                flush=True,
            )
            errors.append(
                _build_result(
                    action="error",
                    external_id=record.external_id,
                    title=record.title,
                    message=f"Error inesperado al sincronizar con Notion: {exc}",
                )
            )

    summary = NotionSyncSummary(
        created=created,
        updated=updated,
        skipped=skipped,
        errors=errors,
    )
    mark_moodle_artifact_notion_sync(
        resolved_artifact_path,
        created_count=len(created),
        updated_count=len(updated),
        skipped_count=len(skipped),
        error_count=len(errors),
    )
    print(
        "[NOTION_DEBUG] sync_done "
        f"created={len(created)} updated={len(updated)} skipped={len(skipped)} errors={len(errors)}",
        flush=True,
    )
    payload = summary.to_dict()
    payload["artifact_path"] = resolved_artifact_path
    return payload


__all__ = ["sync_validated_moodle_artifact_to_notion_payload"]
