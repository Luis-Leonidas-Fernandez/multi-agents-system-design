"""Registry declarativo de providers para búsqueda web.

Este módulo sigue el patrón de OpenClaw: los providers se describen como
datos/config y el runtime resuelve cuál usar según credenciales o selección
explícita, sin acoplar la política al tool concreto.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from core.helpers.config_flow_helpers import get_web_search_runtime_config


@dataclass(frozen=True)
class WebSearchProviderSpec:
    name: str
    label: str
    kind: str
    env_vars: tuple[str, ...]
    auto_detect_order: int
    requires_credential: bool


def _load_policy() -> dict[str, Any]:
    policy_path = Path(__file__).resolve().parents[3] / "application" / "policies" / "web_search_provider_policy.json"
    with policy_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _web_search_provider_plugin_dir() -> Path:
    configured = (os.getenv("WEB_SEARCH_PROVIDER_PLUGIN_DIR") or "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[3] / "application" / "policies" / "web_search_provider_plugins"


def _load_provider_plugin_specs() -> list[dict[str, Any]]:
    plugin_dir = _web_search_provider_plugin_dir()
    if not plugin_dir.exists() or not plugin_dir.is_dir():
        return []

    provider_rows: list[dict[str, Any]] = []
    for plugin_path in sorted(plugin_dir.glob("*.json")):
        try:
            with plugin_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue

        if isinstance(payload, dict):
            rows = payload.get("providers") or []
            if isinstance(rows, list):
                provider_rows.extend(row for row in rows if isinstance(row, dict))
        elif isinstance(payload, list):
            provider_rows.extend(row for row in payload if isinstance(row, dict))

    return provider_rows


_WEB_SEARCH_PROVIDER_POLICY = _load_policy()


@lru_cache(maxsize=None)
def list_web_search_provider_specs() -> tuple[WebSearchProviderSpec, ...]:
    specs: list[WebSearchProviderSpec] = []
    provider_rows = list(_WEB_SEARCH_PROVIDER_POLICY.get("providers") or [])
    provider_rows.extend(_load_provider_plugin_specs())

    seen_names: set[str] = set()
    for raw in provider_rows:
        specs.append(
            WebSearchProviderSpec(
                name=str(raw.get("name") or ""),
                label=str(raw.get("label") or raw.get("name") or ""),
                kind=str(raw.get("kind") or raw.get("name") or ""),
                env_vars=tuple(str(value) for value in (raw.get("env_vars") or []) if value),
                auto_detect_order=int(raw.get("auto_detect_order") or 0),
                requires_credential=bool(raw.get("requires_credential", True)),
            )
        )
    deduped: list[WebSearchProviderSpec] = []
    for spec in sorted(specs, key=lambda spec: (spec.auto_detect_order, spec.name)):
        if not spec.name or spec.name in seen_names:
            continue
        seen_names.add(spec.name)
        deduped.append(spec)
    return tuple(deduped)


def get_web_search_provider_spec(name: str) -> WebSearchProviderSpec:
    for spec in list_web_search_provider_specs():
        if spec.name == name:
            return spec
    valid = ", ".join(spec.name for spec in list_web_search_provider_specs())
    raise ValueError(f"Proveedor de web search no registrado: {name}. Validos: {valid}")


def _provider_has_env_credential(spec: WebSearchProviderSpec) -> bool:
    return any(bool(os.getenv(env_var)) for env_var in spec.env_vars)


def _provider_is_ready(spec: WebSearchProviderSpec) -> bool:
    if spec.requires_credential:
        return _provider_has_env_credential(spec)
    if not spec.env_vars:
        return True
    return _provider_has_env_credential(spec)


def _ordered_web_search_specs() -> tuple[WebSearchProviderSpec, ...]:
    return tuple(sorted(list_web_search_provider_specs(), key=lambda spec: (spec.auto_detect_order, spec.name)))


def _prepend_ready_specs(first: WebSearchProviderSpec, specs: tuple[WebSearchProviderSpec, ...]) -> tuple[WebSearchProviderSpec, ...]:
    return (first,) + tuple(spec for spec in specs if spec.name != first.name and _provider_is_ready(spec))


def resolve_web_search_provider_candidates(
    explicit_provider: Optional[str] = None,
    runtime_selected_provider: Optional[str] = None,
    runtime_provider_configured: Optional[str] = None,
) -> tuple[WebSearchProviderSpec, ...]:
    specs = _ordered_web_search_specs()
    if not specs:
        raise ValueError("No hay providers de web search registrados")

    if explicit_provider:
        return (get_web_search_provider_spec(explicit_provider),)

    if runtime_selected_provider:
        return _prepend_ready_specs(get_web_search_provider_spec(runtime_selected_provider), specs)

    if runtime_provider_configured:
        return _prepend_ready_specs(get_web_search_provider_spec(runtime_provider_configured), specs)

    runtime_cfg = get_web_search_runtime_config()
    configured = (runtime_cfg.provider_configured or os.getenv("WEB_SEARCH_PROVIDER") or "").strip().lower()
    if configured:
        return _prepend_ready_specs(get_web_search_provider_spec(configured), specs)

    credentialed = tuple(spec for spec in specs if spec.requires_credential and _provider_is_ready(spec))
    if credentialed:
        keyless_after = tuple(spec for spec in specs if not spec.requires_credential and _provider_is_ready(spec))
        return credentialed + keyless_after

    keyless = tuple(spec for spec in specs if not spec.requires_credential and _provider_is_ready(spec))
    if keyless:
        return keyless

    fallback_provider = str(_WEB_SEARCH_PROVIDER_POLICY.get("fallback_provider") or specs[0].name)
    return (get_web_search_provider_spec(fallback_provider),)


def resolve_web_search_provider_name(
    explicit_provider: Optional[str] = None,
    runtime_selected_provider: Optional[str] = None,
    runtime_provider_configured: Optional[str] = None,
) -> str:
    return resolve_web_search_provider_candidates(
        explicit_provider=explicit_provider,
        runtime_selected_provider=runtime_selected_provider,
        runtime_provider_configured=runtime_provider_configured,
    )[0].name


def build_web_search_provider_lines() -> str:
    lines = []
    for spec in list_web_search_provider_specs():
        credential_hint = ", ".join(spec.env_vars) if spec.env_vars else "sin credencial"
        lines.append(f"- {spec.name} [{spec.kind} | {credential_hint}]")
    return "\n".join(lines)


def get_web_search_provider_kind(provider_name: Optional[str] = None) -> str:
    return get_web_search_provider_spec(resolve_web_search_provider_name(provider_name)).kind


__all__ = [
    "WebSearchProviderSpec",
    "build_web_search_provider_lines",
    "get_web_search_provider_kind",
    "get_web_search_provider_spec",
    "list_web_search_provider_specs",
    "resolve_web_search_provider_candidates",
    "resolve_web_search_provider_name",
]
