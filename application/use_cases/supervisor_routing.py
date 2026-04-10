"""Ejecución del routing del supervisor."""
from typing import Any, Callable, cast

from langchain_core.messages import AIMessage

from application.helpers.message_flow_helpers import get_last_message_text
from application.use_cases.supervisor_shortcuts import should_route_to_web_scraping
from domain.models import AgentState, RoutingDecision


async def run_supervisor_routing(
    state: AgentState,
    chain_factory: Callable[[], Any],
) -> dict[str, Any]:
    messages = state["messages"]
    last_message = get_last_message_text(messages)

    if should_route_to_web_scraping(last_message):
        return {"next_agent": "web_scraping_agent"}

    chain = chain_factory()
    try:
        decision = cast(
            RoutingDecision,
            await cast(Any, chain).ainvoke({"input": last_message}),
        )
        return {"next_agent": decision.agent}
    except Exception as e:
        err_type = type(e).__name__
        print(f"[supervisor] routing falló ({err_type}: {e})")
        return {
            "messages": [AIMessage(content=(
                f"No pude procesar tu solicitud en este momento ({err_type}). "
                "Por favor, reintentá tu consulta."
            ))],
            "next_agent": "__error__",
        }


__all__ = ["run_supervisor_routing"]
