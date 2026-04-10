"""Ensamblado del prompt del supervisor."""
from __future__ import annotations

from dataclasses import dataclass

from application.services.coordinator_mode import build_coordinator_prompt_assembly, is_coordinator_mode_enabled
from application.services.agent_registry import AGENT_NAMES, build_supervisor_agent_lines


@dataclass(frozen=True)
class SupervisorPromptAssembly:
    """Prompt listo para el chain del supervisor."""

    system_prompt: str
    agent_lines: str


def build_supervisor_prompt_assembly() -> SupervisorPromptAssembly:
    if is_coordinator_mode_enabled():
        coordinator_assembly = build_coordinator_prompt_assembly()
        return SupervisorPromptAssembly(
            system_prompt=coordinator_assembly.system_prompt,
            agent_lines=coordinator_assembly.worker_lines,
        )

    agent_lines = build_supervisor_agent_lines()
    system_prompt = (
        "Eres un supervisor que coordina un equipo de agentes especializados.\n\n"
        f"Tienes acceso a {len(AGENT_NAMES)} agentes:\n"
        f"{agent_lines}\n\n"
        "Elige el agente más adecuado para la solicitud. "
        "Si no estás seguro, elige el que mejor se ajuste."
    )
    return SupervisorPromptAssembly(system_prompt=system_prompt, agent_lines=agent_lines)


__all__ = ["SupervisorPromptAssembly", "build_supervisor_prompt_assembly"]
