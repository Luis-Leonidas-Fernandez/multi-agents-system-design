"""
Factory genérica para nodos de agentes especializados.

Reduce la duplicación entre math/analysis/code manteniendo:
- trazas y metadata por nodo
- guardrail AgentDoG post-ejecución
- HITL pre-ejecución opcional
"""

import time
import uuid
from typing import Any, Awaitable, Callable, Optional, Sequence

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig

from core.helpers.audit_flow_helpers import (
    _emit_node_outcome,
    _extract_followup,
    _extract_quality,
    _extract_tokens,
    _node_meta,
)
from application.policies.agentdog import _should_evaluate_guard, evaluate_trajectory_safe
from application.policies.hitl_flow import ConfirmationHandler
from core.domain.models import AgentState


def make_generic_agent_node(
    agent,
    *,
    node_name: str,
    agent_name: str,
    tags: Sequence[str],
    hitl_prompt_label: Optional[str] = None,
    confirmation_handler: Optional[ConfirmationHandler] = None,
    cancel_message: str = "Operación cancelada por el usuario.",
    rejected_reason: str = "hitl_rejected",
    blocked_reason: Optional[str] = None,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Crea un nodo async reutilizable para agentes especializados."""

    async def agent_node(state: AgentState) -> dict[str, Any]:
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        rid = state.get("request_id", str(uuid.uuid4()))
        t0 = time.time()

        if hitl_prompt_label and confirmation_handler is not None:
            preview = last_message[:120] + ("..." if len(last_message) > 120 else "")
            confirmed = await confirmation_handler.confirm(
                f"\n[HITL] {hitl_prompt_label} va a procesar: \"{preview}\"\n¿Confirmar? [s/n]: "
            )
            if not confirmed:
                _emit_node_outcome(
                    rid, node_name, "blocked", phase="pre_guard",
                    agent=agent_name,
                    duration_ms=int((time.time() - t0) * 1000),
                    reason=rejected_reason,
                )
                return {"messages": [AIMessage(content=cancel_message)]}

        try:
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=last_message)]},
                config=RunnableConfig(
                    tags=list(tags),
                    metadata={
                        "node": node_name,
                        "agent": agent_name,
                        "request_id": rid,
                        "input_chars": len(last_message),
                    },
                ),
            )

            tokens = _extract_tokens(result)
            quality = _extract_quality(result)
            followup = _extract_followup(result, "success")
            meta = _node_meta()

            if _should_evaluate_guard(node_name):
                combined = messages + result.get("messages", [])
                is_safe, _ = await evaluate_trajectory_safe(
                    {"messages": combined, "next_agent": state.get("next_agent", "")},
                    node_name,
                )
                if not is_safe:
                    blocked_kwargs = {
                        "phase": "post_guard",
                        "agent": agent_name,
                        "duration_ms": int((time.time() - t0) * 1000),
                        "followup_likely": True,
                        **tokens,
                        **quality,
                        **meta,
                    }
                    if blocked_reason is not None:
                        blocked_kwargs["reason"] = blocked_reason
                    _emit_node_outcome(rid, node_name, "blocked", **blocked_kwargs)
                    return {"messages": [AIMessage(content="Respuesta retenida por política de seguridad.")]}

            response_messages = result.get("messages", [])
            if response_messages:
                _emit_node_outcome(
                    rid, node_name, "success", phase="agent",
                    agent=agent_name,
                    duration_ms=int((time.time() - t0) * 1000),
                    output_msgs=len(response_messages),
                    **tokens, **quality, **followup, **meta,
                )
                return {"messages": response_messages}

            _emit_node_outcome(
                rid, node_name, "error", phase="agent",
                agent=agent_name,
                duration_ms=int((time.time() - t0) * 1000),
                reason="empty_response",
                followup_likely=True,
                **meta,
            )
            return {"messages": [AIMessage(content="No se pudo procesar la solicitud.")]}

        except Exception as e:
            _emit_node_outcome(
                rid, node_name, "error", phase="agent",
                agent=agent_name,
                duration_ms=int((time.time() - t0) * 1000),
                reason=str(e),
                followup_likely=True,
                **_node_meta(),
            )
            raise

    return agent_node


__all__ = ["make_generic_agent_node"]
