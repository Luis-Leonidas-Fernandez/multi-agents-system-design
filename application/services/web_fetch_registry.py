"""Registry mínimo de providers para fetch web.

Inspirado en OpenClaw: el runtime resuelve un provider a partir de metadata,
pero en esta etapa mantenemos un único provider real y una interfaz estable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class WebFetchProviderSpec:
    name: str
    label: str
    kind: str
    requires_credential: bool = False


_DEFAULT_PROVIDER = WebFetchProviderSpec(
    name="default",
    label="Default Web Fetch",
    kind="llm_fetch",
    requires_credential=False,
)


def list_web_fetch_provider_specs() -> tuple[WebFetchProviderSpec, ...]:
    return (_DEFAULT_PROVIDER,)


def get_web_fetch_provider_spec(name: str) -> WebFetchProviderSpec:
    if name == _DEFAULT_PROVIDER.name:
        return _DEFAULT_PROVIDER
    valid = ", ".join(spec.name for spec in list_web_fetch_provider_specs())
    raise ValueError(f"Proveedor de web fetch no registrado: {name}. Validos: {valid}")


def resolve_web_fetch_provider_name(explicit_provider: Optional[str] = None) -> str:
    configured = (explicit_provider or os.getenv("WEB_FETCH_PROVIDER") or _DEFAULT_PROVIDER.name).strip().lower()
    if configured == _DEFAULT_PROVIDER.name:
        return _DEFAULT_PROVIDER.name
    return get_web_fetch_provider_spec(configured).name


__all__ = [
    "WebFetchProviderSpec",
    "get_web_fetch_provider_spec",
    "list_web_fetch_provider_specs",
    "resolve_web_fetch_provider_name",
]
