"""
Nodo del agente de matemáticas.
Factory pattern: make_math_node(agent) → async math_node(state).
"""
import time
import uuid
from typing import Callable, Awaitable, Any

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
from state import AgentState


def make_math_node(agent) -> Callable[[AgentState], Awaitable[AgentState]]:
    """Retorna math_node con el agente inyectado como closure."""

    async def math_node(state: AgentState) -> AgentState:
        messages     = state["messages"]
        last_message = messages[-1].content if messages else ""
        rid          = state.get("request_id", str(uuid.uuid4()))
        t0           = time.time()

        try:
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=last_message)]},
                config=RunnableConfig(
                    tags=["math", "agent"],
                    metadata={
                        "node":        "math_node",
                        "agent":       "math_agent",
                        "request_id":  rid,
                        "input_chars": len(last_message),
                    },
                ),
            )

            tokens   = _extract_tokens(result)
            quality  = _extract_quality(result)
            followup = _extract_followup(result, "success")
            meta     = _node_meta()

            if _should_evaluate_guard("math_node"):
                combined = messages + result.get("messages", [])
                is_safe, _ = await evaluate_trajectory_safe(
                    {"messages": combined, "next_agent": state.get("next_agent", "")},
                    "math_node",
                )
                if not is_safe:
                    _emit_node_outcome(
                        rid, "math_node", "blocked", phase="post_guard",
                        agent="math_agent",
                        duration_ms=int((time.time() - t0) * 1000),
                        followup_likely=True,
                        **tokens, **quality, **meta,
                    )
                    return {"messages": [AIMessage(content="Respuesta retenida por política de seguridad.")]}

            response_messages = result.get("messages", [])
            if response_messages:
                _emit_node_outcome(
                    rid, "math_node", "success", phase="agent",
                    agent="math_agent",
                    duration_ms=int((time.time() - t0) * 1000),
                    output_msgs=len(response_messages),
                    **tokens, **quality, **followup, **meta,
                )
                return {"messages": response_messages}

            _emit_node_outcome(
                rid, "math_node", "error", phase="agent",
                agent="math_agent",
                duration_ms=int((time.time() - t0) * 1000),
                reason="empty_response",
                followup_likely=True,
                **meta,
            )
            return {"messages": [AIMessage(content="No se pudo procesar la solicitud.")]}

        except Exception as e:
            _emit_node_outcome(
                rid, "math_node", "error", phase="agent",
                agent="math_agent",
                duration_ms=int((time.time() - t0) * 1000),
                reason=str(e),
                followup_likely=True,
                **_node_meta(),
            )
            raise

    return math_node


__all__ = ["make_math_node"]
