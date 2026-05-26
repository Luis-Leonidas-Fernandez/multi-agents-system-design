"""Tools internas para sincronizar tareas validadas con Notion."""
from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

from integrations.notion_tasks_sync import sync_validated_moodle_artifact_to_notion_payload


def _format_payload(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


@tool
def sync_validated_moodle_artifact_to_notion(
    artifact_path: Annotated[
        str,
        Field(
            description=(
                "Ruta del artifact JSON validado de Moodle. Si se omite, usa el "
                "artifact más reciente de la sesión actual."
            )
        ),
    ] = "",
) -> str:
    """Sincroniza a Notion las tareas válidas del artifact Moodle ya revisado y aprobado."""
    try:
        return _format_payload(sync_validated_moodle_artifact_to_notion_payload(artifact_path))
    except Exception as exc:  # noqa: BLE001 - las tools deben responder texto legible
        return f"Error al sincronizar tareas de Moodle con Notion: {exc}"


__all__ = ["sync_validated_moodle_artifact_to_notion"]
