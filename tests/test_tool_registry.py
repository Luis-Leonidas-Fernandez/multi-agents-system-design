"""Tests para el registro unificado de tools."""
from unittest.mock import patch


def test_tool_registry_expone_tools_por_agente():
    from application.services.tool_registry import get_tools_for_agent

    assert [tool.name for tool in get_tools_for_agent("math_agent")] == ["calculate"]
    assert [tool.name for tool in get_tools_for_agent("analysis_agent")] == ["analyze_data"]
    assert [tool.name for tool in get_tools_for_agent("code_agent")] == ["write_code"]
    web_tools = [tool.name for tool in get_tools_for_agent("web_scraping_agent")]
    assert web_tools == [
        "get_crypto_price",
        "extract_price_from_text",
        "search_web",
        "scrape_website_simple",
        "scrape_website_dynamic",
        "scrape_website_with_json_capture",
        "web_fetch",
    ]


def test_tool_registry_catalogo_tiene_nombres_estables():
    from application.services.tool_registry import build_agent_permission_lines, list_tool_specs, get_tool_spec

    names = [spec.name for spec in list_tool_specs()]
    assert "calculate" in names
    assert "write_code" in names
    assert get_tool_spec("calculate").category == "math"
    assert get_tool_spec("write_code").permission_mode == "confirm_high_risk"
    assert "confirm_high_risk" in build_agent_permission_lines("code_agent")


def test_agents_factory_usa_el_registry_de_tools():
    from application.services import agents_factory
    from application.services.prompt_assembly import AgentPromptAssembly

    with patch("application.services.agents_factory.get_tools_for_agent", return_value=("tool-a",)) as mock_get_tools, \
         patch("application.services.agents_factory.get_llm") as mock_get_llm, \
         patch("application.services.agents_factory.build_agent_prompt_assembly", return_value=AgentPromptAssembly("math_agent", "prompt", "extra")) as mock_prompt_assembly, \
         patch("application.services.agents_factory.create_react_agent", return_value="agent"):
        agents_factory.create_math_agent()

    mock_get_tools.assert_called_once_with("math_agent")
    mock_get_llm.assert_called_once_with(temperature=0.1)
    mock_prompt_assembly.assert_called_once_with("math_agent")
