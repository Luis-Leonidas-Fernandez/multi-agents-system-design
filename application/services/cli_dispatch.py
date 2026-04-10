"""Despacho de comandos CLI compartido entre shell y TUI.

Este módulo mantiene la lógica de comandos en un solo lugar para que la
UI (legacy o Textual) pueda reutilizarla sin duplicar ramas.
"""
from __future__ import annotations

import asyncio
import json
import shlex
from dataclasses import dataclass
from typing import Any

from application.services.command_registry import COMMAND_REGISTRY
from application.services.prompt_versioning import prompt_version_service
from application.services.session_inspection import (
    format_background_task_state,
    format_background_task_summary,
    format_bookmark_detail,
    format_bookmark_list,
    format_command_detail,
    format_command_registry,
    format_context_budget,
    format_inspection_help,
    format_memory_search_results,
    format_prompt_snapshot,
    format_prompt_snapshot_list,
    format_replay_timeline,
    format_session_artifact,
    format_tool_approval_preview,
    format_tool_catalog,
    format_tool_impact_preview,
    format_worker_detail,
    format_worker_list,
    format_worker_messages,
)


@dataclass(frozen=True)
class CLICommandResult:
    handled: bool
    lines: list[str]


def _coerce_tool_argument(value: str):
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_tool_arguments(raw_args: str):
    if not raw_args:
        return {}, None
    if raw_args.startswith("{"):
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return {}, "[tool] JSON inválido en args"
        if not isinstance(parsed, dict):
            return {}, "[tool] los args deben ser un objeto JSON"
        return parsed, None

    arguments = {}
    try:
        tokens = shlex.split(raw_args)
    except ValueError:
        return {}, "[tool] args inválidos"

    for token in tokens:
        if "=" not in token:
            return {}, f"[tool] formato inválido: {token}. Usá key=value o JSON"
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            return {}, "[tool] key vacío en args"
        arguments[key] = _coerce_tool_argument(value)
    return arguments, None


def dispatch_inspection_command(user_input: str, lifecycle, runtime) -> CLICommandResult:
    command = user_input.strip()
    if not command.startswith("/"):
        return CLICommandResult(False, [])

    parts = command.split()
    name = parts[0].lstrip("/").lower()
    resolved = COMMAND_REGISTRY.resolve_name(name)
    if resolved:
        name = resolved

    if name in {"help", "?"}:
        return CLICommandResult(True, format_inspection_help())

    if name == "commands":
        return CLICommandResult(True, format_command_registry())

    if name == "command":
        if len(parts) < 2:
            return CLICommandResult(True, ["Usá /command <nombre>"])
        spec = COMMAND_REGISTRY.get(parts[1].lstrip("/"))
        if spec is None:
            return CLICommandResult(True, [f"[command] no encontrado: {parts[1]}"])
        return CLICommandResult(True, format_command_detail(spec))

    if name in {"context", "state"}:
        agent_name = parts[1] if len(parts) > 1 else None
        report = lifecycle.context_budget(agent_name)
        return CLICommandResult(True, format_context_budget(report))

    if name in {"inspect", "status"}:
        context_report = lifecycle.context_budget()
        summary = lifecycle.background_task_summary()
        artifact = lifecycle.export_artifact()
        lines = []
        lines.extend(format_context_budget(context_report))
        lines.extend(format_background_task_summary(summary))
        lines.extend(format_session_artifact(artifact))
        tasks = lifecycle.list_background_tasks()
        if tasks:
            lines.append("[tasks] detalle")
            for task in tasks[-5:]:
                lines.append(f"  - {task.get('task_id', '?')} [{task.get('status', '?')}] {task.get('title', '')}")
        return CLICommandResult(True, lines)

    if name == "tasks":
        summary = lifecycle.background_task_summary()
        lines = format_background_task_summary(summary)
        tasks = lifecycle.list_background_tasks()
        if tasks:
            for task in tasks:
                lines.extend(format_background_task_state(task))
        else:
            lines.append("[tasks] no hay tareas delegadas en esta sesión")
        return CLICommandResult(True, lines)

    if name == "task":
        if len(parts) < 2:
            return CLICommandResult(True, ["Usá /task <id>"])
        state = lifecycle.describe_background_task(parts[1])
        if state is None:
            return CLICommandResult(True, [f"[task] no encontrada: {parts[1]}"])
        return CLICommandResult(True, format_background_task_state(state))

    if name == "cancel":
        if len(parts) < 2:
            return CLICommandResult(True, ["Usá /cancel <id>"])
        result = lifecycle.cancel_background_task(parts[1], reason="cancelado desde CLI")
        if asyncio.iscoroutine(result):
            asyncio.create_task(result)
        return CLICommandResult(True, [f"[task] cancel requested: {parts[1]}"])

    if name == "retryable":
        tasks = lifecycle.list_retryable_background_tasks()
        if not tasks:
            return CLICommandResult(True, ["[tasks] no hay tareas reintentables"])
        lines: list[str] = []
        for task in tasks:
            lines.extend(format_background_task_state(task))
        return CLICommandResult(True, lines)

    if name == "artifact":
        artifact = lifecycle.export_artifact()
        lines = format_session_artifact(artifact)
        lines.append(f"[artifact] path={lifecycle.artifact_path()}")
        lines.append("[artifact] exportado y guardado")
        return CLICommandResult(True, lines)

    if name in {"bookmarks", "checkpoints"}:
        bookmarks = lifecycle.list_bookmarks()
        return CLICommandResult(True, format_bookmark_list(bookmarks))

    if name == "bookmark":
        raw = command[len(parts[0]):].strip()
        if raw:
            try:
                tokens = shlex.split(raw)
            except ValueError:
                return CLICommandResult(True, ["[bookmark] argumento inválido"])
            label = tokens[0] if tokens else None
            note = " ".join(tokens[1:]) if len(tokens) > 1 else ""
        else:
            label = None
            note = ""
        checkpoint = lifecycle.create_bookmark(label=label, note=note)
        lines = format_bookmark_detail(checkpoint)
        lines.append("[bookmark] checkpoint guardado")
        return CLICommandResult(True, lines)

    if name == "checkpoint":
        if len(parts) < 2:
            return CLICommandResult(True, ["Usá /checkpoint <id>"])
        bookmark = lifecycle.describe_bookmark(parts[1])
        if bookmark is None:
            return CLICommandResult(True, [f"[checkpoint] no encontrado: {parts[1]}"])
        return CLICommandResult(True, format_bookmark_detail(bookmark))

    if name == "prompts":
        agents = prompt_version_service.list_agents()
        return CLICommandResult(True, format_prompt_snapshot_list(agents))

    if name == "prompt":
        if len(parts) < 2:
            return CLICommandResult(True, ["Usá /prompt <agente>"])
        snapshot = prompt_version_service.load_snapshot(parts[1])
        if snapshot is None:
            return CLICommandResult(True, [f"[prompt] no encontrado: {parts[1]}"])
        snapshot["snapshot_path"] = str(prompt_version_service.snapshot_path(parts[1]))
        snapshot["history_path"] = str(prompt_version_service.history_path(parts[1]))
        lines = format_prompt_snapshot(snapshot)
        history = prompt_version_service.load_history(parts[1])
        lines.append(f"[prompt] historial={len(history)} versiones")
        return CLICommandResult(True, lines)

    if name == "replay":
        target_session_id = parts[1] if len(parts) > 1 else lifecycle.session_id
        replay = runtime.build_session_replay(target_session_id)
        return CLICommandResult(True, format_replay_timeline(replay))

    if name == "memory":
        if len(parts) < 2:
            sessions = runtime.list_memory_sessions()
            if sessions:
                lines = ["[memory] sesiones con MEMORY.md:"]
                lines.extend(f"  - {session_id}" for session_id in sessions)
                return CLICommandResult(True, lines)
            return CLICommandResult(True, ["[memory] no hay sesiones con memoria persistida"])
        query = " ".join(parts[1:])
        results = runtime.search_memory(query)
        return CLICommandResult(True, format_memory_search_results(query, results))

    if name == "tools":
        return CLICommandResult(True, format_tool_catalog(runtime.tool_catalog()))

    if name == "tool":
        if len(parts) < 2:
            return CLICommandResult(True, ["Usá /tool <nombre> [json_args|key=value ...]"])
        tool_name = parts[1]
        raw_args = " ".join(parts[2:]).strip()
        arguments, error = _parse_tool_arguments(raw_args)
        if error:
            return CLICommandResult(True, [error])
        preview = runtime.preview_tool(tool_name, arguments=arguments)
        return CLICommandResult(True, format_tool_approval_preview(preview))

    if name == "impact":
        if len(parts) < 2:
            return CLICommandResult(True, ["Usá /impact <nombre> [json_args|key=value ...]"])
        tool_name = parts[1]
        raw_args = " ".join(parts[2:]).strip()
        arguments, error = _parse_tool_arguments(raw_args)
        if error:
            return CLICommandResult(True, [error])
        preview = runtime.preview_tool(tool_name, arguments=arguments)
        if isinstance(preview, dict):
            impact_preview = preview.get("impact_preview", preview)
        else:
            impact_preview = getattr(preview, "impact_preview", preview)
        return CLICommandResult(True, format_tool_impact_preview(impact_preview))

    if name == "workers":
        workers = runtime.list_workers(lifecycle.session_id)
        return CLICommandResult(True, format_worker_list(workers))

    if name == "worker":
        if len(parts) < 2:
            return CLICommandResult(True, ["Usá /worker <id>"])
        worker = next((w for w in runtime.list_workers(lifecycle.session_id) if getattr(w, "worker_id", None) == parts[1] or (isinstance(w, dict) and w.get("worker_id") == parts[1])), None)
        if worker is None:
            return CLICommandResult(True, [f"[worker] no encontrado: {parts[1]}"])
        return CLICommandResult(True, format_worker_detail(worker))

    if name == "mailbox":
        messages = runtime.list_worker_messages(lifecycle.session_id)
        return CLICommandResult(True, format_worker_messages(messages))

    return CLICommandResult(True, [f"Comando no reconocido: {command}", *format_inspection_help()])


__all__ = ["CLICommandResult", "dispatch_inspection_command"]
