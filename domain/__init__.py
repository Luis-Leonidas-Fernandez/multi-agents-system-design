"""Capa de dominio del sistema multi-agentes.

Arrancamos por los modelos puros compartidos y mantenemos compatibilidad con
los imports legacy mientras migramos el resto del repo por fases.
"""

from domain.models import AgentName, RoutingDecision, AgentState

__all__ = ["AgentName", "RoutingDecision", "AgentState"]
