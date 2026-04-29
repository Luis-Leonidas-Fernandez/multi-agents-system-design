"""Registro unificado de tools y contratos de uso por agente.

La idea es centralizar qué tools existen, qué agente puede usarlas y
qué callable concreto se expone al motor ReAct. Esto evita que cada
factory vuelva a importar tools a mano y crea una frontera explícita
para futuras políticas o métricas por tool.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from core.domain.agent_roles import AGENT_NAMES as _AGENT_NAMES

from features.math.api import calculate
from features.analysis.api import analyze_data
from features.code.api import write_code
from features.price.api import get_crypto_price, extract_price_from_text
from features.web_scraping.infrastructure.search_tools import search_web
from features.web_scraping.infrastructure.scraping_tools import (
    scrape_website_simple,
    scrape_website_dynamic,
    scrape_website_with_json_capture,
    web_fetch,
    scrape_moodle_assignments,
)

from application.policies.tool_permissions import ToolPermissionMode


@dataclass(frozen=True)
class ToolSpec:
    """Contrato mínimo de una tool registrada."""

    name: str
    tool: Any
    agents: tuple[str, ...]
    category: str
    risk_level: str
    permission_mode: ToolPermissionMode
    description: str


_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="calculate",
        tool=calculate,
        agents=("math_agent",),
        category="math",
        risk_level="low",
        permission_mode="allow_all",
        description="Evalúa expresiones matemáticas seguras.",
    ),
    ToolSpec(
        name="analyze_data",
        tool=analyze_data,
        agents=("analysis_agent",),
        category="analysis",
        risk_level="low",
        permission_mode="allow_all",
        description="Analiza JSON, CSV o texto y devuelve estadísticas.",
    ),
    ToolSpec(
        name="write_code",
        tool=write_code,
        agents=("code_agent",),
        category="code",
        risk_level="high",
        permission_mode="confirm_high_risk",
        description="Genera esqueletos de código por lenguaje.",
    ),
    ToolSpec(
        name="get_crypto_price",
        tool=get_crypto_price,
        agents=("web_scraping_agent",),
        category="web",
        risk_level="low",
        permission_mode="allow_all",
        description="Obtiene el precio de criptomonedas desde CoinGecko.",
    ),
    ToolSpec(
        name="extract_price_from_text",
        tool=extract_price_from_text,
        agents=("web_scraping_agent",),
        category="web",
        risk_level="low",
        permission_mode="allow_all",
        description="Extrae un valor numérico tipo precio desde texto bruto.",
    ),
    ToolSpec(
        name="search_web",
        tool=search_web,
        agents=("web_scraping_agent",),
        category="web",
        risk_level="low",
        permission_mode="allow_all",
        description="Busca información en internet sin URL.",
    ),
    ToolSpec(
        name="scrape_website_simple",
        tool=scrape_website_simple,
        agents=("web_scraping_agent",),
        category="web",
        risk_level="high",
        permission_mode="confirm_high_risk",
        description="Extrae texto y enlaces de páginas estáticas.",
    ),
    ToolSpec(
        name="scrape_website_dynamic",
        tool=scrape_website_dynamic,
        agents=("web_scraping_agent",),
        category="web",
        risk_level="high",
        permission_mode="confirm_high_risk",
        description="Extrae contenido de páginas con JavaScript.",
    ),
    ToolSpec(
        name="scrape_website_with_json_capture",
        tool=scrape_website_with_json_capture,
        agents=("web_scraping_agent",),
        category="web",
        risk_level="high",
        permission_mode="confirm_high_risk",
        description="Extrae contenido dinámico y captura respuestas JSON.",
    ),
    ToolSpec(
        name="web_fetch",
        tool=web_fetch,
        agents=("web_scraping_agent",),
        category="web",
        risk_level="high",
        permission_mode="confirm_high_risk",
        description="Recupera una URL y sintetiza el contenido con un prompt explícito.",
    ),
    ToolSpec(
        name="scrape_moodle_assignments",
        tool=scrape_moodle_assignments,
        agents=("web_scraping_agent",),
        category="web",
        risk_level="high",
        permission_mode="confirm_high_risk",
        description="Inicia sesión en Moodle y extrae tareas pendientes del calendario.",
    ),
)

_TOOLS_BY_NAME = {spec.name: spec for spec in _TOOL_SPECS}


def list_tool_specs() -> tuple[ToolSpec, ...]:
    return _TOOL_SPECS


def get_tool_spec(name: str) -> ToolSpec:
    try:
        return _TOOLS_BY_NAME[name]
    except KeyError as exc:
        valid = ", ".join(sorted(_TOOLS_BY_NAME))
        raise ValueError(f"Tool no registrada: {name}. Validas: {valid}") from exc


def _tool_matches_agent(spec: ToolSpec, agent_name: str) -> bool:
    return agent_name in spec.agents


@lru_cache(maxsize=None)
def get_tools_for_agent(agent_name: str) -> tuple[Any, ...]:
    if agent_name not in _AGENT_NAMES:
        valid = ", ".join(_AGENT_NAMES)
        raise ValueError(f"Agente no soportado para tools: {agent_name}. Validos: {valid}")
    return tuple(spec.tool for spec in _TOOL_SPECS if _tool_matches_agent(spec, agent_name))


def build_agent_tool_lines(agent_name: str) -> str:
    get_tools_for_agent(agent_name)
    specs = [spec for spec in _TOOL_SPECS if _tool_matches_agent(spec, agent_name)]
    return "\n".join(f"- {spec.name} [{spec.risk_level}/{spec.permission_mode}]: {spec.description}" for spec in specs)


def build_agent_permission_lines(agent_name: str) -> str:
    get_tools_for_agent(agent_name)
    specs = [spec for spec in _TOOL_SPECS if _tool_matches_agent(spec, agent_name)]
    return "\n".join(f"- {spec.name}: {spec.permission_mode}" for spec in specs)


def build_tool_catalog_lines() -> str:
    return "\n".join(f"- {spec.name} [{spec.category} | {spec.risk_level} | {spec.permission_mode}]: {spec.description}" for spec in _TOOL_SPECS)


__all__ = [
    "ToolSpec",
    "build_agent_tool_lines",
    "build_agent_permission_lines",
    "build_tool_catalog_lines",
    "get_tool_spec",
    "get_tools_for_agent",
    "list_tool_specs",
]
