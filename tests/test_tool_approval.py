"""Tests para la vista previa de aprobación de tools."""


def test_tool_approval_service_builds_preview_for_high_risk_tool():
    from application.services.tool_approval import ToolApprovalService

    service = ToolApprovalService()
    preview = service.build_preview(
        agent_name="code_agent",
        tool_name="write_code",
        arguments={"task": "ordenar una lista", "language": "python"},
    )

    assert preview.tool_name == "write_code"
    assert preview.allowed is True
    assert preview.requires_confirmation is True
    assert "code_agent" in preview.confirmation_prompt
    assert "task" in preview.arguments_preview
    assert preview.impact_preview.tool_name == "write_code"
    assert preview.impact_preview.category == "code"


def test_tool_approval_service_render_lines_incluye_prompt():
    from application.services.tool_approval import ToolApprovalService

    service = ToolApprovalService()
    preview = service.build_preview(agent_name="math_agent", tool_name="calculate", arguments={"expression": "2+2"})

    lines = service.render_preview_lines(preview)

    assert any("calculate" in line for line in lines)
    assert any("args=" in line for line in lines)


def test_session_inspection_tool_formatters():
    from application.services.session_inspection import format_tool_approval_preview, format_tool_catalog, format_tool_impact_preview

    lines = format_tool_approval_preview(
        {
            "tool_name": "write_code",
            "agent_name": "code_agent",
            "risk_level": "high",
            "permission_mode": "confirm_high_risk",
            "allowed": True,
            "requires_confirmation": True,
            "reason": "confirm_high_risk",
            "description": "Genera esqueletos de código por lenguaje.",
            "arguments_preview": "{task='x'}",
            "confirmation_prompt": "[APPROVAL] ...",
            "impact_preview": {"tool_name": "write_code", "agent_name": "code_agent", "category": "code", "scope": "new-file", "confidence": "medium", "estimated_diff_lines": "~12-30 líneas", "affected_files": ["tests/test_x.py"], "side_effects": "cambios locales en el repositorio", "risk_notes": ["Revisar imports"], "generated_at_ms": 1},
        }
    )
    impact_lines = format_tool_impact_preview({"tool_name": "scrape_website_with_json_capture", "agent_name": "web_scraping_agent", "category": "web", "scope": "artifact-write", "confidence": "high", "estimated_diff_lines": "0 líneas de código, 1 artefacto de datos", "affected_files": ["data_trading/example.json"], "side_effects": "escribe bundles JSON en data_trading/", "risk_notes": ["Puede persistir respuestas JSON"], "generated_at_ms": 1})

    assert any("write_code" in line for line in lines)
    assert any("confirm=yes" in line for line in lines)
    assert any("[impact]" in line for line in lines)
    assert any("data_trading" in line for line in impact_lines)
    assert format_tool_catalog(["- calculate"])[0].startswith("[tools]")


def test_handle_inspection_command_tool_and_tools(capsys):
    from application.services.runtime import AgentRuntime
    from main import _handle_inspection_command
    from unittest.mock import MagicMock

    runtime = AgentRuntime(gateway=MagicMock())
    runtime.tool_catalog = MagicMock(return_value=["- calculate [math | low | allow_all]: Evalúa expresiones matemáticas seguras."])  # type: ignore[assignment]
    runtime.preview_tool = MagicMock(return_value={
        "tool_name": "write_code",
        "agent_name": "cli",
        "risk_level": "high",
        "permission_mode": "confirm_high_risk",
        "allowed": True,
        "requires_confirmation": True,
        "reason": "confirm_high_risk",
        "description": "desc",
        "arguments_preview": "{}",
        "confirmation_prompt": "prompt",
        "impact_preview": {"tool_name": "write_code", "agent_name": "cli", "category": "code", "scope": "new-file", "confidence": "medium", "estimated_diff_lines": "~12-30 líneas", "affected_files": ["tests/test_write_code.py"], "side_effects": "cambios locales en el repositorio", "risk_notes": ["Revisar imports"], "generated_at_ms": 1},
    })  # type: ignore[assignment]
    lifecycle = type("L", (), {"session_id": "sess-x"})()

    assert _handle_inspection_command("/tools", lifecycle, runtime=runtime) is True
    assert _handle_inspection_command("/tool write_code", lifecycle, runtime=runtime) is True
    assert _handle_inspection_command("/impact write_code", lifecycle, runtime=runtime) is True
    output = capsys.readouterr().out
    assert "calculate" in output or "write_code" in output
    assert "[impact]" in output


def test_handle_inspection_command_tool_con_json_args(capsys):
    from application.services.runtime import AgentRuntime
    from main import _handle_inspection_command
    from unittest.mock import MagicMock

    runtime = AgentRuntime(gateway=MagicMock())
    captured = {}
    def preview_tool(tool_name, arguments=None):
        captured["tool_name"] = tool_name
        captured["arguments"] = arguments
        return {
            "tool_name": tool_name,
            "agent_name": "cli",
            "risk_level": "high",
            "permission_mode": "confirm_high_risk",
            "allowed": True,
            "requires_confirmation": True,
            "reason": "confirm_high_risk",
            "description": "desc",
            "arguments_preview": "{}",
            "confirmation_prompt": "prompt",
            "impact_preview": {"tool_name": tool_name, "agent_name": "cli", "category": "code", "scope": "new-file", "confidence": "medium", "estimated_diff_lines": "~12-30 líneas", "affected_files": ["tests/test_write_code.py"], "side_effects": "cambios locales en el repositorio", "risk_notes": ["Revisar imports"], "generated_at_ms": 1},
        }

    runtime.preview_tool = MagicMock(side_effect=preview_tool)  # type: ignore[assignment]
    lifecycle = type("L", (), {"session_id": "sess-x"})()

    assert _handle_inspection_command('/tool write_code {"language":"python","task":"ordenar"}', lifecycle, runtime=runtime) is True
    output = capsys.readouterr().out
    assert captured["tool_name"] == "write_code"
    assert captured["arguments"] == {"language": "python", "task": "ordenar"}
    assert "write_code" in output
    assert "[impact]" in output


def test_handle_inspection_command_tool_con_key_value_args(capsys):
    from application.services.runtime import AgentRuntime
    from main import _handle_inspection_command
    from unittest.mock import MagicMock

    runtime = AgentRuntime(gateway=MagicMock())
    captured = {}

    def preview_tool(tool_name, arguments=None):
        captured["tool_name"] = tool_name
        captured["arguments"] = arguments
        return {
            "tool_name": tool_name,
            "agent_name": "cli",
            "risk_level": "high",
            "permission_mode": "confirm_high_risk",
            "allowed": True,
            "requires_confirmation": True,
            "reason": "confirm_high_risk",
            "description": "desc",
            "arguments_preview": "{}",
            "confirmation_prompt": "prompt",
            "impact_preview": {"tool_name": tool_name, "agent_name": "cli", "category": "code", "scope": "new-file", "confidence": "medium", "estimated_diff_lines": "~12-30 líneas", "affected_files": ["tests/test_write_code.py"], "side_effects": "cambios locales en el repositorio", "risk_notes": ["Revisar imports"], "generated_at_ms": 1},
        }

    runtime.preview_tool = MagicMock(side_effect=preview_tool)  # type: ignore[assignment]
    lifecycle = type("L", (), {"session_id": "sess-x"})()

    assert _handle_inspection_command("/tool write_code language=python task=ordenar retries=2 dry_run=true", lifecycle, runtime=runtime) is True
    output = capsys.readouterr().out
    assert captured["tool_name"] == "write_code"
    assert captured["arguments"] == {"language": "python", "task": "ordenar", "retries": 2, "dry_run": True}
    assert "write_code" in output
    assert "[impact]" in output


def test_handle_inspection_command_tool_rechaza_json_invalido(capsys):
    from application.services.runtime import AgentRuntime
    from main import _handle_inspection_command
    from unittest.mock import MagicMock

    runtime = AgentRuntime(gateway=MagicMock())
    runtime.preview_tool = MagicMock(return_value={})  # type: ignore[assignment]
    lifecycle = type("L", (), {"session_id": "sess-x"})()

    handled = _handle_inspection_command("/tool write_code {not-json}", lifecycle, runtime=runtime)
    output = capsys.readouterr().out

    assert handled is True
    assert "JSON inválido" in output


def test_handle_inspection_command_tool_rechaza_formato_invalido(capsys):
    from application.services.runtime import AgentRuntime
    from main import _handle_inspection_command
    from unittest.mock import MagicMock

    runtime = AgentRuntime(gateway=MagicMock())
    runtime.preview_tool = MagicMock(return_value={})  # type: ignore[assignment]
    lifecycle = type("L", (), {"session_id": "sess-x"})()

    handled = _handle_inspection_command("/tool write_code language python", lifecycle, runtime=runtime)
    output = capsys.readouterr().out

    assert handled is True
    assert "formato inválido" in output
