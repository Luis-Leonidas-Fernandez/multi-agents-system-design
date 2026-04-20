"""
Registro central de agentes del sistema multi-agentes.

Centraliza metadata de routing y expone factories lazy para agentes y nodos.
El módulo evita imports pesados en import-time para que domain.models y application/composition/graph.py
puedan leer los nombres registrados sin instanciar LLMs ni agentes.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from domain.agent_roles import AGENT_NAMES  # noqa: F401  — re-exported for callers


@dataclass(frozen=True)
class AgentSpec:
    name: str
    routing_description: str
    risk_level: str
    tags: tuple[str, ...]
    temperature: float | None = None


_AGENT_SPECS = {
    "math_agent": AgentSpec(
        name="math_agent",
        routing_description="problemas matematicos, calculos, algebra, estadistica numerica",
        risk_level="low",
        tags=("math", "agent"),
        temperature=0.1,
    ),
    "analysis_agent": AgentSpec(
        name="analysis_agent",
        routing_description="analisis de datos, insights, reportes, patrones en datasets",
        risk_level="low",
        tags=("analysis", "agent"),
        temperature=0.2,
    ),
    "code_agent": AgentSpec(
        name="code_agent",
        routing_description="escribir codigo, programacion, desarrollo de software",
        risk_level="high",
        tags=("code", "agent", "high_risk"),
        temperature=0.0,
    ),
    "web_scraping_agent": AgentSpec(
        name="web_scraping_agent",
        routing_description="extraer informacion de URLs, scraping, obtener datos de paginas web",
        risk_level="high",
        tags=("web_scraping", "agent", "high_risk"),
        temperature=0.0,
    ),
}


def get_agent_specs() -> tuple[AgentSpec, ...]:
    """Retorna los agentes en el orden oficial del sistema."""
    return tuple(_AGENT_SPECS[name] for name in AGENT_NAMES)


def get_agent_spec(name: str) -> AgentSpec:
    """Retorna la spec de un agente registrado o falla con error claro."""
    try:
        return _AGENT_SPECS[name]
    except KeyError as exc:
        valid = ", ".join(AGENT_NAMES)
        raise ValueError(f"Agente no registrado: {name}. Validos: {valid}") from exc


def get_agent_temperature(name: str) -> float | None:
    """Retorna la temperatura configurada para un agente, o None para heredar global."""
    return get_agent_spec(name).temperature


def build_supervisor_agent_lines() -> str:
    """Construye el bloque descriptivo de agentes usado por el supervisor."""
    return "\n".join(
        f"- {spec.name}: {spec.routing_description}"
        for spec in get_agent_specs()
    )


def _get_agent_factory(name: str):
    if name == "math_agent":
        from application.services.agents_factory import create_math_agent
        return create_math_agent
    if name == "analysis_agent":
        from application.services.agents_factory import create_analysis_agent
        return create_analysis_agent
    if name == "code_agent":
        from application.services.agents_factory import create_code_agent
        return create_code_agent
    if name == "web_scraping_agent":
        from application.services.agents_factory import create_web_scraping_agent
        return create_web_scraping_agent
    get_agent_spec(name)
    raise AssertionError("unreachable")


@lru_cache(maxsize=None)
def get_agent(name: str):
    """Instancia un agente solo la primera vez que se solicita."""
    return _get_agent_factory(name)()


@lru_cache(maxsize=None)
def get_node(name: str):
    """Construye el nodo LangGraph del agente con lazy init y cache."""
    agent = get_agent(name)

    if name == "math_agent":
        from nodes import make_math_node
        return make_math_node(agent)
    if name == "analysis_agent":
        from nodes import make_analysis_node
        return make_analysis_node(agent)
    if name == "code_agent":
        from nodes import make_code_node
        return make_code_node(agent)
    if name == "web_scraping_agent":
        from application.helpers.config_flow_helpers import get_llm
        from nodes import make_web_scraping_node
        return make_web_scraping_node(agent, get_llm)

    get_agent_spec(name)
    raise AssertionError("unreachable")


def get_registered_nodes() -> dict[str, object]:
    """Retorna el mapping name -> node callable para el grafo."""
    return {name: get_node(name) for name in AGENT_NAMES}


__all__ = [
    "AGENT_NAMES",
    "AgentSpec",
    "build_supervisor_agent_lines",
    "get_agent",
    "get_agent_spec",
    "get_agent_specs",
    "get_agent_temperature",
    "get_node",
    "get_registered_nodes",
]
