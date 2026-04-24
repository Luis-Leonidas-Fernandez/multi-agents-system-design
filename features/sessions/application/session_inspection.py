"""Helpers de presentación para inspección de sesión en CLI.

Mantiene el formateo fuera de `main.py` para que el entrypoint siga fino.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any, Mapping

from application.services.command_registry import COMMAND_REGISTRY, SlashCommandSpec
from features.sessions.application.context_budget import SessionContextBudget
from features.sessions.application.background_tasks import BackgroundTaskState, BackgroundTaskSummary
from features.sessions.application.memory_retrieval import MemorySearchHit
from features.sessions.application.coordinator_workers import CoordinatorMessage, CoordinatorWorker
from application.services.tool_impact import tool_impact_service
from application.services.tool_approval import ToolApprovalPreview
from features.sessions.application.session_replay import SessionReplay


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        data = dict(value.__dict__)
        if data:
            return data
    cls = getattr(value, "__class__", None)
    if cls is not None:
        data = {
            key: getattr(value, key)
            for key in dir(cls)
            if not key.startswith("_") and not callable(getattr(value, key, None)) and hasattr(value, key)
        }
        if data:
            return data
    return {}


def _supports_color() -> bool:
    return os.getenv("NO_COLOR") is None and sys.stdout.isatty()


def _style(text: str, code: str) -> str:
    if not _supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def _plain_length(text: str) -> int:
    return len(re.sub(r"\x1b\[[0-9;]*m", "", text))


def _wrap_box(title: str, body_lines: list[str]) -> list[str]:
    width = max([_plain_length(title), *(_plain_length(line) for line in body_lines)] + [0])
    top = f"┌─ {title} " + "─" * max(0, width - _plain_length(title)) + "┐"
    bottom = "└" + "─" * (width + 4) + "┘"
    lines = [top]
    for line in body_lines:
        pad = width - _plain_length(line)
        lines.append(f"│ {line}{' ' * pad} │")
    lines.append(bottom)
    return lines


def format_session_banner(snapshot: Any, context_budget: SessionContextBudget | Mapping[str, Any] | None = None) -> list[str]:
    data = _as_mapping(snapshot)
    lines = [
        f"[sesión] {data.get('session_id', '?')} · {data.get('message_count', 0)} mensajes previos · {'memoria cargada' if data.get('has_memory') else 'sin memoria persistida'}",
    ]
    if context_budget:
        report = _as_mapping(context_budget)
        lines.append(
            f"[contexto] scope={report.get('scope', 'session')} status={report.get('status', '?')} usado={report.get('estimated_context_chars', 0)} chars · restantes={report.get('estimated_remaining_chars', 0)} · tokens≈{report.get('estimated_tokens', 0)}"
        )
    lines.append("[atajos] /context · /commands · /bookmarks · /replay · /tool")
    return lines


def format_cli_status(snapshot: Any, context_budget: SessionContextBudget | Mapping[str, Any] | None = None) -> str:
    data = _as_mapping(snapshot)
    parts = [
        f"sesión={data.get('session_id', '?')}",
        f"mensajes={data.get('message_count', 0)}",
        f"memoria={'sí' if data.get('has_memory') else 'no'}",
    ]
    if context_budget:
        report = _as_mapping(context_budget)
        parts.append(f"contexto={report.get('estimated_remaining_chars', 0)} chars libres")
        parts.append(f"tokens≈{report.get('estimated_tokens', 0)}")
    return "[status] " + " · ".join(parts)


def format_agent_roster(agents: list[Any]) -> list[str]:
    if not agents:
        return ["[agentes] no hay agentes registrados"]
    roster = []
    for agent in agents:
        data = _as_mapping(agent)
        tag = data.get("risk_level", "?")
        roster.append(f"{data.get('name', '?')}[{tag}]")
    return ["[agentes] " + " · ".join(roster)]


def format_cli_chrome(snapshot: Any, context_budget: SessionContextBudget | Mapping[str, Any] | None, agents: list[Any]) -> list[str]:
    lines = _wrap_box(
        _style("Multi-Agentes", "1;36"),
        [
            _style(format_cli_status(snapshot, context_budget), "0;37"),
            _style("atajos: /help /context /commands /bookmarks /replay /tool", "2"),
        ],
    )
    lines.extend(format_agent_roster(agents))
    return lines


def format_cli_prompt(snapshot: Any) -> str:
    data = _as_mapping(snapshot)
    session_id = data.get("session_id", "?")
    return _style(f"[{session_id}] › ", "1;36")


def format_session_selector(sessions: list[str]) -> list[str]:
    if not sessions:
        return ["[sessions] no hay sesiones guardadas"]
    lines = ["[sessions] elegí una sesión por número o ID:"]
    for idx, session_id in enumerate(sessions, start=1):
        lines.append(f"  {idx}. {session_id}")
    lines.append("  Enter = nueva sesión")
    return lines


def format_chat_block(role: str, content: str) -> list[str]:
    title = _style(role.upper(), "1;35" if role == "user" else "1;32")
    body = [line if line else "" for line in content.splitlines() or [content]]
    return _wrap_box(title, body or [""])


def format_session_transcript(transcript: list[Any], limit: int = 4) -> list[str]:
    if not transcript:
        return ["[chat] todavía no hay mensajes en esta sesión"]
    items = transcript[-limit:]
    lines = [f"[chat] últimos {len(items)} mensajes"]
    for item in items:
        data = _as_mapping(item)
        role = str(data.get("role", "message")).upper()
        content = str(data.get("content", ""))
        preview = content.replace("\n", " ⏎ ")
        if len(preview) > 120:
            preview = preview[:117] + "…"
        lines.append(f"  {role}: {preview}")
    return lines


def format_transcript_blocks(transcript: list[Any], limit: int = 4) -> list[str]:
    if not transcript:
        return ["[chat] todavía no hay mensajes en esta sesión"]
    items = transcript[-limit:]
    lines: list[str] = []
    for item in items:
        data = _as_mapping(item)
        role = str(data.get("role", "message")).lower()
        label = {
            "human": "you",
            "user": "you",
            "ai": "claude",
            "assistant": "claude",
        }.get(role, role)
        lines.extend(format_chat_block(label, str(data.get("content", ""))))
        lines.append("")
    return lines[:-1] if lines else lines


def format_shell_frame(
    snapshot: Any,
    context_budget: SessionContextBudget | Mapping[str, Any] | None,
    agents: list[Any],
    transcript: list[Any],
    prompt_hint: str,
    transcript_limit: int = 4,
) -> list[str]:
    lines = format_cli_chrome(snapshot, context_budget, agents)
    lines.append("")
    lines.append(_style("conversation", "1;34"))
    lines.extend(format_transcript_blocks(transcript, limit=transcript_limit))
    lines.append("")
    lines.append(_style(prompt_hint, "2"))
    return lines


def format_background_task_summary(summary: BackgroundTaskSummary | Mapping[str, Any]) -> list[str]:
    data = _as_mapping(summary)
    session_id = data.get("session_id", "?")
    lines = [
        f"[tasks] sesión={session_id} total={data.get('total', 0)} active={data.get('active', 0)} terminal={data.get('terminal', 0)}",
        f"  queued={data.get('queued', 0)} running={data.get('running', 0)} completed={data.get('completed', 0)} failed={data.get('failed', 0)} cancelled={data.get('cancelled', 0)} retryable={data.get('retryable', 0)}",
    ]
    latest = data.get("latest_updated_at_ms")
    if latest:
        lines.append(f"  latest_update_ms={latest}")
    return lines


def format_background_task_state(state: BackgroundTaskState | Mapping[str, Any]) -> list[str]:
    data = _as_mapping(state)
    lines = [
        f"[task] {data.get('task_id', '?')} status={data.get('status', '?')} kind={data.get('state_kind', '?')} attempt={data.get('attempt_number', 1)} title={data.get('title', '')}",
        f"  session={data.get('session_id', '?')} parent={data.get('parent_task_id', '-') or '-'} request_id={data.get('request_id', '-') or '-'} trace_id={data.get('trace_id', '-') or '-'}",
    ]
    if data.get("result") is not None:
        lines.append(f"  result={data.get('result')}")
    if data.get("error"):
        lines.append(f"  error={data.get('error')}")
    if data.get("cancel_reason"):
        lines.append(f"  cancel_reason={data.get('cancel_reason')}")
    if data.get("metadata"):
        lines.append(f"  metadata={data.get('metadata')}")
    return lines


def format_session_artifact(artifact: Any) -> list[str]:
    data = _as_mapping(artifact)
    lines = [
        f"[artifact] sesión={data.get('session_id', '?')} messages={data.get('message_count', 0)} memory={'yes' if data.get('has_memory') else 'no'}",
        f"  audit_events={len(data.get('audit_events', []))} background_tasks={len(data.get('background_tasks', []))} prompt_snapshots={len(data.get('prompt_snapshots', []))} bookmarks={len(data.get('bookmarks', []))} traces={len(data.get('trace_ids', []))}",
    ]
    context_budget = data.get("context_budget") or {}
    if context_budget:
        report = _as_mapping(context_budget.get("report", context_budget))
        lines.append(
            f"  context scope={report.get('scope', '?')} status={report.get('status', '?')} chars={report.get('estimated_context_chars', 0)} remaining={report.get('estimated_remaining_chars', 0)}"
        )
    summary = data.get("background_task_summary") or {}
    if summary:
        lines.append(
            f"  task_summary total={summary.get('total', 0)} active={summary.get('active', 0)} completed={summary.get('completed', 0)} failed={summary.get('failed', 0)} cancelled={summary.get('cancelled', 0)}"
        )
    return lines


def format_prompt_snapshot(snapshot: Any) -> list[str]:
    data = _as_mapping(snapshot)
    lines = [
        f"[prompt] {data.get('agent_name', '?')} version={data.get('prompt_version', '?')} hash={str(data.get('prompt_hash', '?'))[:12]}",
        f"  created_at_ms={data.get('created_at_ms', '?')} extra_context_chars={len(str(data.get('extra_context', '')))} system_prompt_chars={len(str(data.get('system_prompt', '')))}",
    ]
    if data.get("snapshot_path"):
        lines.append(f"  snapshot_path={data.get('snapshot_path')}")
    if data.get("history_path"):
        lines.append(f"  history_path={data.get('history_path')}")
    return lines


def format_prompt_snapshot_list(agents: list[str]) -> list[str]:
    if not agents:
        return ["[prompt] no hay snapshots de prompts persistidos"]
    return ["[prompt] agentes con snapshots: " + ", ".join(agents)]


def format_inspection_help() -> list[str]:
    lines = ["Comandos de inspección y atajos:"]
    for group, commands in COMMAND_REGISTRY.grouped().items():
        lines.append(f"[{group}]")
        for command in commands:
            aliases = f" (alias: {', '.join(command.aliases)})" if command.aliases else ""
            lines.append(f"  /{command.name:<10} - {command.summary}{aliases}")
    lines.append("  /command <nombre> - muestra ayuda detallada de un comando")
    lines.append("  /commands         - lista comandos agrupados por categoría")
    return lines


def format_command_registry(groups: Mapping[str, list[SlashCommandSpec]] | None = None) -> list[str]:
    groups = groups or COMMAND_REGISTRY.grouped()
    lines = [f"[commands] total={len(COMMAND_REGISTRY.list_commands())}"]
    for group, commands in groups.items():
        lines.append(f"  [{group}] {len(commands)} comandos")
        for command in commands:
            alias_part = f" aliases={', '.join(command.aliases)}" if command.aliases else ""
            lines.append(f"    /{command.name} - {command.summary}{alias_part}")
    return lines


def format_command_detail(command: SlashCommandSpec | Mapping[str, Any]) -> list[str]:
    data = _as_mapping(command)
    lines = [
        f"[command] /{data.get('name', '?')} group={data.get('group', '?')}",
        f"  usage={data.get('usage', '?')}",
        f"  summary={data.get('summary', '?')}",
    ]
    if data.get("aliases"):
        lines.append(f"  aliases={', '.join(data.get('aliases', []))}")
    return lines


def format_context_budget(report: SessionContextBudget | Mapping[str, Any]) -> list[str]:
    data = _as_mapping(report)
    lines = [
        f"[context] sesión={data.get('session_id', '?')} scope={data.get('scope', '?')} status={data.get('status', '?')} budget={data.get('budget_chars', 0)} chars · usado={data.get('estimated_context_chars', 0)} · restantes={data.get('estimated_remaining_chars', 0)} · tokens≈{data.get('estimated_tokens', 0)}",
        f"  transcript_messages={data.get('transcript_message_count', 0)} memory={'yes' if data.get('memory_present') else 'no'}",
    ]
    items = data.get("items", []) or []
    included = [item for item in items if _as_mapping(item).get("role") == "included"]
    summarized = [item for item in items if _as_mapping(item).get("role") == "summarized"]
    excluded = [item for item in items if _as_mapping(item).get("role") == "excluded"]

    def _format_group(title: str, group: list[Any]) -> str:
        parts = []
        for item in group:
            item_data = _as_mapping(item)
            parts.append(f"{item_data.get('section', '?')}={item_data.get('chars', 0)}")
        return f"  {title}: " + (" ".join(parts) if parts else "-")

    lines.append(_format_group("incluido", included))
    lines.append(_format_group("resumido", summarized))
    lines.append(_format_group("afuera", excluded))
    for item in items:
        item_data = _as_mapping(item)
        lines.append(f"    - {item_data.get('section', '?')} [{item_data.get('role', '?')}] {item_data.get('detail', '')}")
    return lines


def format_bookmark_list(bookmarks: list[Mapping[str, Any]]) -> list[str]:
    if not bookmarks:
        return ["[bookmarks] no hay checkpoints guardados"]
    lines = [f"[bookmarks] checkpoints={len(bookmarks)}"]
    for bookmark in bookmarks:
        data = _as_mapping(bookmark)
        lines.append(
            f"  - {data.get('checkpoint_id', '?')} label={data.get('label', '?')} messages={data.get('message_count', 0)} memory={'yes' if data.get('has_memory') else 'no'}"
        )
    return lines


def format_bookmark_detail(bookmark: Mapping[str, Any]) -> list[str]:
    data = _as_mapping(bookmark)
    lines = [
        f"[checkpoint] {data.get('checkpoint_id', '?')} label={data.get('label', '?')} session={data.get('session_id', '?')}",
        f"  created_at_ms={data.get('created_at_ms', '?')} messages={data.get('message_count', 0)} memory={'yes' if data.get('has_memory') else 'no'} replay_items={data.get('replay_item_count', 0)}",
        f"  artifact_path={data.get('artifact_path', '?')}",
    ]
    if data.get("note"):
        lines.append(f"  note={data.get('note')}")
    context_budget = data.get("context_budget") or {}
    if context_budget:
        report = _as_mapping(context_budget.get("report", context_budget))
        lines.append(
            f"  context scope={report.get('scope', '?')} status={report.get('status', '?')} chars={report.get('estimated_context_chars', 0)} remaining={report.get('estimated_remaining_chars', 0)}"
        )
    if data.get("prompt_agents"):
        lines.append(f"  prompt_agents={', '.join(data.get('prompt_agents', []))}")
    return lines


def format_replay_timeline(replay: SessionReplay | Mapping[str, Any]) -> list[str]:
    data = _as_mapping(replay)
    items = data.get("items", [])
    lines = [f"[replay] sesión={data.get('session_id', '?')} items={len(items)}"]
    for item in items:
        item_data = _as_mapping(item)
        lines.append(f"  [{item_data.get('section', '?')}] {item_data.get('title', '?')} — {item_data.get('detail', '')}")
    return lines


def format_memory_search_results(query: str, results: list[MemorySearchHit] | list[Mapping[str, Any]]) -> list[str]:
    lines = [f"[memory] query={query!r} hits={len(results)}"]
    if not results:
        lines.append("  sin coincidencias")
        return lines

    for result in results:
        data = _as_mapping(result)
        lines.append(
            f"  [session={data.get('session_id', '?')}] score={data.get('score', 0):.1f} terms={','.join(data.get('matched_terms', []))}"
        )
        lines.append(f"    path={data.get('memory_path', '?')}")
        if data.get("excerpt"):
            lines.append(f"    excerpt={data.get('excerpt')}")
    return lines


def format_tool_approval_preview(preview: ToolApprovalPreview | Mapping[str, Any]) -> list[str]:
    data = _as_mapping(preview)
    lines = [
        f"[tool] {data.get('tool_name', '?')} agent={data.get('agent_name', '?')} risk={data.get('risk_level', '?')} mode={data.get('permission_mode', '?')}",
        f"  allowed={'yes' if data.get('allowed') else 'no'} confirm={'yes' if data.get('requires_confirmation') else 'no'} reason={data.get('reason', '?')}",
    ]
    if data.get("description"):
        lines.append(f"  desc={data.get('description')}")
    if data.get("arguments_preview"):
        lines.append(f"  args={data.get('arguments_preview')}")
    impact_preview = data.get("impact_preview")
    if impact_preview is not None:
        lines.extend(tool_impact_service.render_lines(impact_preview))
    if data.get("confirmation_prompt"):
        lines.append(f"  prompt={data.get('confirmation_prompt')}")
    return lines


def format_tool_impact_preview(preview: Any) -> list[str]:
    if preview is None:
        return ["[impact] sin datos de impacto"]
    return tool_impact_service.render_lines(preview)


def format_tool_catalog(lines: list[str]) -> list[str]:
    if not lines:
        return ["[tools] no hay tools registradas"]
    return ["[tools] catálogo de tools:", *[f"  {line}" for line in lines]]


def format_worker_list(workers: list[CoordinatorWorker] | list[Mapping[str, Any]]) -> list[str]:
    if not workers:
        return ["[workers] no hay workers coordinados en esta sesión"]
    lines = [f"[workers] total={len(workers)}"]
    for worker in workers:
        data = _as_mapping(worker)
        lines.append(
            f"  - {data.get('worker_id', '?')} name={data.get('worker_name', '?')} agent={data.get('agent_name', '?')} status={data.get('status', '?')}"
        )
        if data.get("parent_worker_id"):
            lines.append(f"    parent={data.get('parent_worker_id')}")
        if data.get("metadata"):
            lines.append(f"    metadata={data.get('metadata')}")
    return lines


def format_worker_detail(worker: CoordinatorWorker | Mapping[str, Any]) -> list[str]:
    data = _as_mapping(worker)
    lines = [
        f"[worker] {data.get('worker_id', '?')} name={data.get('worker_name', '?')} agent={data.get('agent_name', '?')} status={data.get('status', '?')}",
        f"  session={data.get('session_id', '?')} created_at_ms={data.get('created_at_ms', '?')} updated_at_ms={data.get('updated_at_ms', '?')}",
    ]
    if data.get("parent_worker_id"):
        lines.append(f"  parent={data.get('parent_worker_id')}")
    if data.get("metadata"):
        lines.append(f"  metadata={data.get('metadata')}")
    return lines


def format_worker_messages(messages: list[CoordinatorMessage] | list[Mapping[str, Any]]) -> list[str]:
    if not messages:
        return ["[mailbox] no hay mensajes coordinados en esta sesión"]
    lines = [f"[mailbox] mensajes={len(messages)}"]
    for message in messages:
        data = _as_mapping(message)
        content = str(data.get('content', '')).replace("\n", " ⏎ ")
        if len(content) > 120:
            content = content[:117] + "…"
        lines.append(
            f"  - {data.get('created_at_ms', '?')} {data.get('sender', '?')} → {data.get('recipient', '?')} [{data.get('kind', '?')}] {content}"
        )
    return lines


__all__ = [
    "format_background_task_state",
    "format_background_task_summary",
    "format_chat_block",
    "format_cli_chrome",
    "format_cli_prompt",
    "format_cli_status",
    "format_agent_roster",
    "format_session_selector",
    "format_bookmark_detail",
    "format_bookmark_list",
    "format_command_detail",
    "format_command_registry",
    "format_context_budget",
    "format_inspection_help",
    "format_memory_search_results",
    "format_prompt_snapshot",
    "format_prompt_snapshot_list",
    "format_shell_frame",
    "format_replay_timeline",
    "format_session_transcript",
    "format_session_artifact",
    "format_transcript_blocks",
    "format_session_banner",
    "format_tool_approval_preview",
    "format_tool_catalog",
    "format_tool_impact_preview",
    "format_worker_detail",
    "format_worker_list",
    "format_worker_messages",
]
