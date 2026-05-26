"""Contexto runtime por turno para toggles dinámicos de herramientas."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass(frozen=True)
class RequestRuntimeConfig:
    session_id: str = ""
    request_id: str = ""
    enabled_mcps: tuple[str, ...] = ()


_REQUEST_RUNTIME = ContextVar("request_runtime_config", default=RequestRuntimeConfig())


def _normalize_mcps(enabled_mcps: Iterable[str] | None) -> tuple[str, ...]:
    if not enabled_mcps:
        return ()
    normalized = {str(item).strip() for item in enabled_mcps if str(item).strip()}
    return tuple(sorted(normalized))


def get_request_runtime_config() -> RequestRuntimeConfig:
    return _REQUEST_RUNTIME.get()


def get_enabled_mcps() -> tuple[str, ...]:
    return get_request_runtime_config().enabled_mcps


def get_request_tool_signature() -> tuple[str, ...]:
    return get_enabled_mcps()


def is_mcp_enabled(mcp_key: str) -> bool:
    return mcp_key in get_enabled_mcps()


@contextmanager
def use_request_runtime(
    *,
    session_id: str = "",
    request_id: str = "",
    enabled_mcps: Iterable[str] | None = None,
) -> Iterator[RequestRuntimeConfig]:
    config = RequestRuntimeConfig(
        session_id=session_id,
        request_id=request_id,
        enabled_mcps=_normalize_mcps(enabled_mcps),
    )
    token = _REQUEST_RUNTIME.set(config)
    try:
        yield config
    finally:
        _REQUEST_RUNTIME.reset(token)


__all__ = [
    "RequestRuntimeConfig",
    "get_enabled_mcps",
    "get_request_runtime_config",
    "get_request_tool_signature",
    "is_mcp_enabled",
    "use_request_runtime",
]
