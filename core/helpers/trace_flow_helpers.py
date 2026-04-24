"""Helpers compartidos de trazabilidad de flujo."""

from typing import Callable, Any, Mapping


def get_or_create_request_id(state: Mapping[str, Any], request_id_factory: Callable[[], str]) -> str:
    rid = state.get("request_id", "")
    return rid if isinstance(rid, str) and rid else request_id_factory()


__all__ = ["get_or_create_request_id"]
