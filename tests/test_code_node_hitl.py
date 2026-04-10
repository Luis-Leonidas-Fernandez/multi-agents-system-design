"""
Tests de comportamiento crítico del HITL en code_node.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import HumanMessage
from domain.models import AgentState


def _state(message: str = "Escribe código") -> AgentState:
    return {
        "messages": [HumanMessage(content=message)],
        "next_agent": "code_agent",
        "risk_flag": False,
        "blocked": False,
        "request_id": "req-hitl-1",
        "scrape_tracker": {},
    }


@pytest.mark.asyncio
async def test_code_node_hitl_rejects_before_agent_execution():
    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(return_value={"messages": [HumanMessage(content="no debería ejecutarse")]})

    fake_handler = MagicMock()
    fake_handler.confirm = AsyncMock(return_value=False)

    with (
        patch("application.policies.hitl_flow.HITL_ENABLED", True),
        patch("application.policies.hitl_flow.get_confirmation_handler", return_value=fake_handler),
    ):
        from nodes.code_node import make_code_node

        node = make_code_node(mock_agent)
        result = await node(_state())

    mock_agent.ainvoke.assert_not_called()
    fake_handler.confirm.assert_awaited_once()
    assert result["messages"][0].content.lower().startswith("operación cancelada")
