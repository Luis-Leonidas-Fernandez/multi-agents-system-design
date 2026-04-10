"""Caso de uso para el guard de entrada.

Genera request_id, ejecuta el middleware de seguridad y devuelve el patch
de estado listo para el grafo.
"""
from typing import Callable, Optional, Any, Mapping

from domain.models import AgentState
from application.helpers.trace_flow_helpers import get_or_create_request_id


async def run_input_guard(
    state: Mapping[str, Any],
    guard_fn: Callable[[Mapping[str, Any]], Optional[dict]],
    request_id_factory: Callable[[], str],
) -> dict[str, Any]:
    rid = get_or_create_request_id(state, request_id_factory)
    blocked = guard_fn(state)
    if blocked:
        return {**blocked, "request_id": rid}
    return {"request_id": rid}


__all__ = ["run_input_guard"]
