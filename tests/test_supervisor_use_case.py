"""Tests del caso de uso del supervisor."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import cast

from langchain_core.messages import HumanMessage
from domain.models import AgentState


@pytest.mark.asyncio
async def test_supervisor_use_case_routes_to_expected_agent():
    from application.use_cases.supervisor_routing import run_supervisor_routing
    from domain.models import RoutingDecision

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=RoutingDecision(agent="code_agent", reason="test"))

    state = cast(AgentState, {
        "messages": [HumanMessage(content="Escribe código")],
        "next_agent": "",
        "risk_flag": False,
        "blocked": False,
        "scrape_tracker": {},
    })

    result = await run_supervisor_routing(state, lambda: mock_chain)
    assert result["next_agent"] == "code_agent"


@pytest.mark.asyncio
async def test_supervisor_use_case_btc_shortcut():
    from application.use_cases.supervisor_routing import run_supervisor_routing

    state = cast(AgentState, {
        "messages": [HumanMessage(content="precio del bitcoin ahora")],
        "next_agent": "",
        "risk_flag": False,
        "blocked": False,
        "scrape_tracker": {},
    })

    result = await run_supervisor_routing(state, lambda: None)
    assert result["next_agent"] == "web_scraping_agent"
