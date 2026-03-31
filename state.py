"""
Tipos compartidos del sistema multi-agentes.

Este módulo define el estado central (AgentState) y los tipos de routing.
No importa ningún módulo interno del proyecto para evitar dependencias circulares.
"""
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel


AgentName = Literal["math_agent", "analysis_agent", "code_agent", "web_scraping_agent"]


class RoutingDecision(BaseModel):
    agent: AgentName
    reason: str


class AgentState(TypedDict):
    """
    Estado compartido entre todos los agentes.

    Attributes:
        messages:       Historial de mensajes — append-only por diseño.
        next_agent:     Nombre del próximo agente a ejecutar.
        risk_flag:      True si algún turno previo activó una señal de riesgo.
        blocked:        True si input_guard bloqueó el mensaje actual.
        request_id:     UUID generado por input_guard_node — correlaciona nodos del mismo turno.
        scrape_tracker: Score acumulativo por categoría: {"crypto_price": {"score": -1, ...}}.
    """
    messages: Annotated[list, lambda x, y: x + y]
    next_agent: str
    risk_flag: bool
    blocked: bool
    request_id: str
    scrape_tracker: dict


__all__ = ["AgentName", "RoutingDecision", "AgentState"]
