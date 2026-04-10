"""Tests del caso de uso de routing."""
from langchain_core.messages import HumanMessage
from typing import cast

from domain.models import AgentState


def test_decide_agent_route_con_agent_valido():
    from application.use_cases.routing_decision import decide_agent_route

    state = cast(AgentState, {
        "messages": [HumanMessage(content="Escribe código")],
        "next_agent": "code_agent",
        "risk_flag": False,
        "blocked": False,
        "request_id": "req-1",
        "scrape_tracker": {},
    })

    assert decide_agent_route(state, ("math_agent", "analysis_agent", "code_agent", "web_scraping_agent")) == "code_agent"


def test_decide_agent_route_reintenta_supervisor():
    from application.use_cases.routing_decision import decide_agent_route

    state = cast(AgentState, {
        "messages": [HumanMessage(content="¿qué hago?")],
        "next_agent": "",
        "risk_flag": False,
        "blocked": False,
        "request_id": "req-2",
        "scrape_tracker": {},
    })

    assert decide_agent_route(state, ("math_agent", "analysis_agent", "code_agent", "web_scraping_agent")) == "supervisor"


def test_decide_agent_route_finaliza_sin_mensajes():
    from application.use_cases.routing_decision import decide_agent_route

    assert decide_agent_route(cast(AgentState, {"messages": [], "next_agent": "", "risk_flag": False, "blocked": False, "request_id": "req-3", "scrape_tracker": {}}), ("math_agent",)) == "__end__"
