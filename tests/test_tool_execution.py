"""Tests para el contrato de ejecución de tools con permisos."""
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_execute_registered_tool_allows_low_risk_tool():
    from application.services.tool_execution import ToolExecutionContext, execute_registered_tool

    ctx = ToolExecutionContext(
        request_id="req-1",
        agent_name="math_agent",
        tool_name="calculate",
        arguments={"expression": "2 + 2"},
    )
    result = await execute_registered_tool(ctx)

    assert result.allowed is True
    assert result.tool_name == "calculate"
    assert "Resultado" in str(result.output)
    assert result.duration_ms >= 0


@pytest.mark.asyncio
async def test_execute_registered_tool_requiere_confirmacion_para_riesgo_alto():
    from application.services.tool_execution import ToolExecutionContext, execute_registered_tool

    ctx = ToolExecutionContext(
        request_id="req-2",
        agent_name="code_agent",
        tool_name="write_code",
        arguments={"task": "ordenar una lista", "language": "python"},
    )

    prompts = []

    async def confirm(prompt: str):
        prompts.append(prompt)
        return False

    result = await execute_registered_tool(ctx, confirm_fn=confirm)

    assert result.allowed is False
    assert result.decision.requires_confirmation is True
    assert "cancelada" in str(result.output).lower()
    assert prompts and "write_code" in prompts[0]
    assert "task" in prompts[0]


@pytest.mark.asyncio
async def test_execute_registered_tool_emite_audit_trail():
    from application.services.tool_execution import ToolExecutionContext, execute_registered_tool

    ctx = ToolExecutionContext(
        request_id="req-3",
        agent_name="math_agent",
        tool_name="calculate",
        arguments={"expression": "3 + 3"},
        session_id="sess-1",
        trace_id="trace-1",
    )

    with patch("application.services.tool_execution.tool_audit_service") as mock_audit:
        result = await execute_registered_tool(ctx)

    assert result.allowed is True
    assert mock_audit.requested.called
    assert mock_audit.decided.called
    assert mock_audit.completed.called


def test_tool_permission_policy_decide_tool_permission():
    from application.policies.tool_permissions import decide_tool_permission

    low = decide_tool_permission(risk_level="low", permission_mode="allow_all")
    high = decide_tool_permission(risk_level="high", permission_mode="confirm_high_risk")

    assert low.allowed is True
    assert low.requires_confirmation is False
    assert high.allowed is True
    assert high.requires_confirmation is True
