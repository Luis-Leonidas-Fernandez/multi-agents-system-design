"""
Nodo del agente de código con HITL pre-ejecución.
Factory pattern: make_code_node(agent) → async code_node(state).
"""
import time
import uuid
from typing import Callable, Awaitable

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig

from audit import (
    _emit_node_outcome,
    _extract_tokens,
    _extract_quality,
    _extract_followup,
    _node_meta,
)
from agentdog import evaluate_trajectory_safe, _should_evaluate_guard
from security import _HITL_ENABLED, _ask_confirmation
from state import AgentState


def make_code_node(agent) -> Callable[[AgentState], Awaitable[AgentState]]:
    """Retorna code_node con el agente inyectado como closure."""

    async def code_node(state: AgentState) -> AgentState:
        messages     = state["messages"]
        last_message = messages[-1].content if messages else ""
        rid          = state.get("request_id", str(uuid.uuid4()))
        t0           = time.time()

        if _HITL_ENABLED:
            preview   = last_message[:120] + ("..." if len(last_message) > 120 else "")
            confirmed = await _ask_confirmation(
                f"\n[HITL] code_agent va a procesar: \"{preview}\"\n¿Confirmar? [s/n]: "
            )
            if not confirmed:
                _emit_node_outcome(
                    rid, "code_node", "blocked", phase="pre_guard",
                    agent="code_agent",
                    duration_ms=int((time.time() - t0) * 1000),
                    reason="hitl_rejected",
                )
                return {"messages": [AIMessage(content="Operación cancelada por el usuario.")]}

        try:
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=last_message)]},
                config=RunnableConfig(
                    tags=["code", "agent", "high_risk"],
                    metadata={
                        "node":        "code_node",
                        "agent":       "code_agent",
                        "request_id":  rid,
                        "input_chars": len(last_message),
                    },
                ),
            )

            tokens   = _extract_tokens(result)
            quality  = _extract_quality(result)
            followup = _extract_followup(result, "success")
            meta     = _node_meta()

            if _should_evaluate_guard("code_node"):
                combined = messages + result.get("messages", [])
                is_safe, _ = await evaluate_trajectory_safe(
                    {"messages": combined, "next_agent": state.get("next_agent", "")},
                    "code_node",
                )
                if not is_safe:
                    _emit_node_outcome(
                        rid, "code_node", "blocked", phase="post_guard",
                        agent="code_agent",
                        duration_ms=int((time.time() - t0) * 1000),
                        reason="agentdog",
                        followup_likely=True,
                        **tokens, **quality, **meta,
                    )
                    return {"messages": [AIMessage(content="Respuesta retenida por política de seguridad.")]}

            response_messages = result.get("messages", [])
            if response_messages:
                _emit_node_outcome(
                    rid, "code_node", "success", phase="agent",
                    agent="code_agent",
                    duration_ms=int((time.time() - t0) * 1000),
                    output_msgs=len(response_messages),
                    **tokens, **quality, **followup, **meta,
                )
                return {"messages": response_messages}

            _emit_node_outcome(
                rid, "code_node", "error", phase="agent",
                agent="code_agent",
                duration_ms=int((time.time() - t0) * 1000),
                reason="empty_response",
                followup_likely=True,
                **meta,
            )
            return {"messages": [AIMessage(content="No se pudo procesar la solicitud.")]}

        except Exception as e:
            _emit_node_outcome(
                rid, "code_node", "error", phase="agent",
                agent="code_agent",
                duration_ms=int((time.time() - t0) * 1000),
                reason=str(e),
                followup_likely=True,
                **_node_meta(),
            )
            raise

    return code_node


__all__ = ["make_code_node"]
