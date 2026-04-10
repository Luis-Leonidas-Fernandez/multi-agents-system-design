"""Modelos puros del dominio multi-agentes.

No dependen de LangGraph ni de infraestructura. Solo expresan el contrato de
estado y routing del sistema.
"""
from typing import TypedDict, Annotated, Literal

from pydantic import BaseModel

from application.services.agent_registry import AGENT_NAMES


# Python 3.9 no permite derivar Literal dinámicamente desde una tupla.
# Mantenemos el Literal explícito y validamos sincronía con el registry.
_REGISTERED_AGENT_NAMES = ("math_agent", "analysis_agent", "code_agent", "web_scraping_agent")
assert _REGISTERED_AGENT_NAMES == AGENT_NAMES, "AgentName Literal desincronizado con agent_registry.AGENT_NAMES"

AgentName = Literal["math_agent", "analysis_agent", "code_agent", "web_scraping_agent"]


class RoutingDecision(BaseModel):
    agent: AgentName
    reason: str


class AgentState(TypedDict):
    """Estado compartido entre todos los agentes del sistema."""

    messages: Annotated[list, lambda x, y: x + y]
    session_id: str
    next_agent: str
    risk_flag: bool
    blocked: bool
    request_id: str
    scrape_tracker: dict
    web_search_selected_provider: str
    web_search_provider_configured: str
    coordinator_worker_id: str
    coordinator_worker_agent: str
    coordinator_probe_best_source: str
    coordinator_probe_sources: list[str]


__all__ = ["AgentName", "RoutingDecision", "AgentState"]
