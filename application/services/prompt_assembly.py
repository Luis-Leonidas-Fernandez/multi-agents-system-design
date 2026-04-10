"""Ensamblado de prompts para agentes.

Centraliza cómo se combinan el prompt base, el inventario de tools y las
líneas de permisos para que las factories no tengan que armar strings a mano.
"""
from __future__ import annotations

from dataclasses import dataclass

from application.services.prompt_loader import load_agent_prompt
from application.services.prompt_versioning import PromptSnapshot, prompt_version_service
from application.services.tool_registry import build_agent_permission_lines, build_agent_tool_lines


@dataclass(frozen=True)
class AgentPromptAssembly:
    """Resultado listo para pasar al constructor ReAct."""

    agent_name: str
    system_prompt: str
    extra_context: str
    prompt_version: str = ""
    prompt_hash: str = ""
    prompt_snapshot: PromptSnapshot | None = None


def build_agent_prompt_extra_context(agent_name: str) -> str:
    tool_lines = build_agent_tool_lines(agent_name)
    permission_lines = build_agent_permission_lines(agent_name)
    return f"Herramientas:\n{tool_lines}\n\nPermisos:\n{permission_lines}"


def build_agent_prompt_assembly(agent_name: str) -> AgentPromptAssembly:
    extra_context = build_agent_prompt_extra_context(agent_name)
    system_prompt = load_agent_prompt(agent_name, extra_context=extra_context)
    snapshot = prompt_version_service.save_snapshot(agent_name, system_prompt, extra_context)
    return AgentPromptAssembly(
        agent_name=agent_name,
        system_prompt=system_prompt,
        extra_context=extra_context,
        prompt_version=snapshot.prompt_version,
        prompt_hash=snapshot.prompt_hash,
        prompt_snapshot=snapshot,
    )


__all__ = ["AgentPromptAssembly", "build_agent_prompt_assembly", "build_agent_prompt_extra_context"]
