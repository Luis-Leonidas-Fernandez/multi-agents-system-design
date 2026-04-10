"""Nodo fino del agente de web scraping.

Toda la coordinación vive en `application/use_cases/web_scraping_flow.py`.
Este módulo solo adapta dependencias concretas al caso de uso.
"""
from typing import Callable, Awaitable, Any, cast

from application.policies.agentdog import evaluate_trajectory_safe, _should_evaluate_guard
from application.use_cases.web_scraping_flow import run_web_scraping_flow
from ports.confirmation_port import ConfirmationPort
from ports.llm_port import LLMFactory
import application.policies.hitl_flow as hitl_flow
from application.policies.scrape_tracker import get_runtime_policy
from domain.models import AgentState

# ==================== FACTORY ====================

def make_web_scraping_node(
    agent,
    get_llm_fn: LLMFactory,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Retorna un adaptador fino que delega toda la lógica al caso de uso."""

    async def web_scraping_node(state: AgentState) -> dict[str, Any]:
        class _PatchedConfirmationAdapter:
            async def confirm(self, prompt: str) -> bool:
                return await hitl_flow.ask_confirmation(prompt)

        return await run_web_scraping_flow(
            state,
            agent,
            get_llm_fn,
            hitl_enabled=hitl_flow.HITL_ENABLED,
            confirmation_handler=cast(ConfirmationPort, _PatchedConfirmationAdapter()) if hitl_flow.HITL_ENABLED else None,
            ask_confirmation_compat=hitl_flow.ask_confirmation if hitl_flow.HITL_ENABLED else None,
            get_runtime_policy=get_runtime_policy,
            evaluate_trajectory_safe_fn=evaluate_trajectory_safe,
            should_evaluate_guard_fn=_should_evaluate_guard,
        )

    return web_scraping_node


__all__ = ["make_web_scraping_node"]
