"""Contrato de modo coordinador para el sistema multi-agentes.

Este módulo concentra la forma en que el sistema describe la coordinación
central cuando el modo coordinador está habilitado. Por ahora reutiliza el
catálogo estático de agentes como workers, pero deja explícita la frontera
para evolucionar hacia spawn dinámico y mensajería entre workers.
"""
from __future__ import annotations

from dataclasses import dataclass
import os

from application.services.agent_registry import get_agent_specs


COORDINATOR_MODE_ENV = "COORDINATOR_MODE"


@dataclass(frozen=True)
class CoordinatorPromptAssembly:
    """Prompt listo para el chain del coordinador."""

    system_prompt: str
    worker_lines: str


@dataclass(frozen=True)
class CoordinatorModeContract:
    """Contrato liviano del modo coordinador."""

    enabled: bool
    worker_names: tuple[str, ...]
    worker_lines: str
    system_prompt: str


def is_coordinator_mode_enabled() -> bool:
    """Retorna True si el modo coordinador está habilitado por env."""
    return os.getenv(COORDINATOR_MODE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def build_coordinator_worker_lines() -> str:
    """Lista legible de workers disponibles para el coordinador."""
    specs = get_agent_specs()
    return "\n".join(
        f"- {spec.name}: {spec.routing_description} (risk={spec.risk_level})"
        for spec in specs
    )


def build_coordinator_prompt_assembly() -> CoordinatorPromptAssembly:
    worker_lines = build_coordinator_worker_lines()
    system_prompt = (
        "Eres un coordinador que orquesta trabajos entre workers especializados.\n\n"
        "## Tu rol\n"
        "- Descomponer el problema en trabajos paralelos cuando sea posible.\n"
        "- Mantener el control de la sesión y sintetizar resultados.\n"
        "- Responder directamente cuando no haga falta delegar.\n\n"
        "## Estrategia\n"
        "- Si la consulta pide información de internet, lanzá workers en paralelo para probar fuentes distintas.\n"
        "- Usá distintas variantes de búsqueda o extracción para comparar evidencia.\n"
        "- Elegí la respuesta mejor respaldada por datos concretos, no la más verbosa.\n"
        "- Si dos fuentes coinciden, priorizá esa coincidencia.\n\n"
        "## Workers disponibles\n"
        f"{worker_lines}\n\n"
        "## Reglas\n"
        "- No inventes workers que no existen.\n"
        "- No superpongas trabajos que toquen los mismos archivos sin necesidad.\n"
        "- Si la coordinación no aporta valor, actuá como supervisor secuencial.\n"
    )
    return CoordinatorPromptAssembly(system_prompt=system_prompt, worker_lines=worker_lines)


def build_coordinator_mode_contract() -> CoordinatorModeContract:
    assembly = build_coordinator_prompt_assembly()
    return CoordinatorModeContract(
        enabled=is_coordinator_mode_enabled(),
        worker_names=tuple(spec.name for spec in get_agent_specs()),
        worker_lines=assembly.worker_lines,
        system_prompt=assembly.system_prompt,
    )


__all__ = [
    "COORDINATOR_MODE_ENV",
    "CoordinatorModeContract",
    "CoordinatorPromptAssembly",
    "build_coordinator_mode_contract",
    "build_coordinator_prompt_assembly",
    "build_coordinator_worker_lines",
    "is_coordinator_mode_enabled",
]
