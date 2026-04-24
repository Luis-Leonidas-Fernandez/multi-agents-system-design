"""Servicios de aplicación y orquestación.

Este paquete queda liviano para evitar ciclos de importación y startup pesado.
Importá los módulos concretos directamente.
"""

from importlib import import_module


_LAZY_SUBMODULES = {
    "agent_registry": f"{__name__}.agent_registry",
    "background_tasks": "features.sessions.application.background_tasks",
    "command_registry": f"{__name__}.command_registry",
    "context_budget": "features.sessions.application.context_budget",
    "coordinator_mode": f"{__name__}.coordinator_mode",
    "coordinator_workers": "features.sessions.application.coordinator_workers",
    "memory_retrieval": "features.sessions.application.memory_retrieval",
    "prompt_assembly": f"{__name__}.prompt_assembly",
    "prompt_loader": f"{__name__}.prompt_loader",
    "prompt_versioning": "features.sessions.application.prompt_versioning",
    "runtime": f"{__name__}.runtime",
    "session_artifacts": "features.sessions.application.session_artifacts",
    "session_bookmarks": "features.sessions.application.session_bookmarks",
    "session_gateway": "features.sessions.application.session_gateway",
    "session_inspection": "features.sessions.application.session_inspection",
    "session_memory": "features.sessions.application.session_memory",
    "session_persistence": "features.sessions.application.session_persistence",
    "session_replay": "features.sessions.application.session_replay",
    "supervisor_prompt": f"{__name__}.supervisor_prompt",
    "tool_approval": f"{__name__}.tool_approval",
    "tool_audit": f"{__name__}.tool_audit",
    "tool_audit_store": "features.sessions.application.tool_audit_store",
    "tool_execution": f"{__name__}.tool_execution",
    "tool_impact": f"{__name__}.tool_impact",
    "tool_registry": f"{__name__}.tool_registry",
    "trace_context": f"{__name__}.trace_context",
}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = import_module(_LAZY_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
