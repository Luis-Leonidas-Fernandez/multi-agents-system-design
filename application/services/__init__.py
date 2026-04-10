"""Servicios de aplicación y orquestación.

Este paquete queda liviano para evitar ciclos de importación y startup pesado.
Importá los módulos concretos directamente.
"""

from importlib import import_module


_LAZY_SUBMODULES = {
    "agent_registry",
    "background_tasks",
    "command_registry",
    "context_budget",
    "coordinator_mode",
    "coordinator_workers",
    "memory_retrieval",
    "prompt_assembly",
    "prompt_loader",
    "prompt_versioning",
    "runtime",
    "session_artifacts",
    "session_bookmarks",
    "session_gateway",
    "session_inspection",
    "session_memory",
    "session_persistence",
    "session_replay",
    "supervisor_prompt",
    "tool_approval",
    "tool_audit",
    "tool_audit_store",
    "tool_execution",
    "tool_impact",
    "tool_registry",
    "trace_context",
}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
