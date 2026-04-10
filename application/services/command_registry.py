"""Registry de comandos slash para CLI.

Concentra metadata, aliases y agrupación para que la ayuda sea autogenerada
y el TUI futuro pueda reutilizar la misma fuente.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SlashCommandSpec:
    name: str
    summary: str
    usage: str
    group: str
    aliases: tuple[str, ...] = ()
    hidden: bool = False


class SlashCommandRegistry:
    def __init__(self, commands: Iterable[SlashCommandSpec]) -> None:
        self._commands = list(commands)
        self._by_name = {command.name: command for command in self._commands}
        self._aliases = {alias: command.name for command in self._commands for alias in command.aliases}

    def list_commands(self) -> list[SlashCommandSpec]:
        return [command for command in self._commands if not command.hidden]

    def grouped(self) -> dict[str, list[SlashCommandSpec]]:
        grouped: dict[str, list[SlashCommandSpec]] = {}
        for command in self.list_commands():
            grouped.setdefault(command.group, []).append(command)
        return grouped

    def get(self, name: str) -> SlashCommandSpec | None:
        canonical = self.resolve_name(name)
        return self._by_name.get(canonical) if canonical else None

    def resolve_name(self, name: str) -> str | None:
        if name in self._by_name:
            return name
        return self._aliases.get(name)

    def search(self, needle: str) -> list[SlashCommandSpec]:
        needle = needle.lower().strip()
        if not needle:
            return self.list_commands()
        return [
            command
            for command in self.list_commands()
            if needle in command.name.lower() or needle in command.summary.lower() or any(needle in alias.lower() for alias in command.aliases)
        ]


COMMAND_REGISTRY = SlashCommandRegistry(
    [
        SlashCommandSpec("help", "muestra la ayuda autogenerada", "/help", "general", aliases=("?",)),
        SlashCommandSpec("inspect", "resume tasks y artifact de la sesión actual", "/inspect", "sesión", aliases=("status",)),
        SlashCommandSpec("context", "muestra el presupuesto de contexto y el estado de la sesión", "/context [agente]", "sesión", aliases=("state",)),
        SlashCommandSpec("tasks", "muestra el resumen de background tasks", "/tasks", "delegación"),
        SlashCommandSpec("task", "muestra el estado detallado de una task", "/task <id>", "delegación"),
        SlashCommandSpec("cancel", "cancela una task en ejecución", "/cancel <id>", "delegación"),
        SlashCommandSpec("retryable", "muestra tareas fallidas o canceladas", "/retryable", "delegación"),
        SlashCommandSpec("artifact", "exporta y muestra el resumen del artifact actual", "/artifact", "sesión"),
        SlashCommandSpec("prompts", "lista snapshots de prompts persistidos", "/prompts", "prompts"),
        SlashCommandSpec("prompt", "muestra el snapshot del prompt de un agente", "/prompt <agente>", "prompts"),
        SlashCommandSpec("replay", "muestra la línea de tiempo unificada de la sesión", "/replay [session_id]", "sesión"),
        SlashCommandSpec("memory", "busca memoria destilada en sesiones previas", "/memory [buscar texto]", "memoria"),
        SlashCommandSpec("bookmarks", "lista checkpoints guardados en la sesión", "/bookmarks", "sesión", aliases=("checkpoints",)),
        SlashCommandSpec("bookmark", "guarda un checkpoint de la sesión actual", "/bookmark [nombre]", "sesión"),
        SlashCommandSpec("checkpoint", "muestra un checkpoint guardado", "/checkpoint <id>", "sesión"),
        SlashCommandSpec("tools", "lista catálogo de tools y riesgos", "/tools", "tools"),
        SlashCommandSpec("tool", "muestra la vista previa de aprobación de una tool", "/tool <name> [json_args|key=value ...]", "tools"),
        SlashCommandSpec("impact", "muestra el impacto estimado de una tool", "/impact <name> [json_args|key=value ...]", "tools"),
        SlashCommandSpec("workers", "lista workers coordinados de la sesión", "/workers", "coordinación"),
        SlashCommandSpec("worker", "muestra el detalle de un worker coordinado", "/worker <id>", "coordinación"),
        SlashCommandSpec("mailbox", "muestra los mensajes coordinados de la sesión", "/mailbox", "coordinación"),
    ]
)


__all__ = ["SlashCommandSpec", "SlashCommandRegistry", "COMMAND_REGISTRY"]
