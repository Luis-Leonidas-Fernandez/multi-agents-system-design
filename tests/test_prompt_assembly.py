"""Tests para el seam de ensamblado de prompts."""
from unittest.mock import patch


def test_build_agent_prompt_extra_context_arma_bloques_estables():
    from application.services.prompt_assembly import build_agent_prompt_extra_context

    with patch("application.services.prompt_assembly.build_agent_tool_lines", return_value="tool-lines"), \
         patch("application.services.prompt_assembly.build_agent_permission_lines", return_value="permission-lines"):
        extra = build_agent_prompt_extra_context("math_agent")

    assert extra == "Herramientas:\ntool-lines\n\nPermisos:\npermission-lines"


def test_build_agent_prompt_assembly_usa_loader():
    from application.services.prompt_assembly import build_agent_prompt_assembly
    from features.sessions.application.prompt_versioning import PromptSnapshot

    with patch("application.services.prompt_assembly.build_agent_tool_lines", return_value="tool-lines"), \
         patch("application.services.prompt_assembly.build_agent_permission_lines", return_value="permission-lines"), \
         patch("application.services.prompt_assembly.load_agent_prompt", return_value="base prompt"), \
         patch("application.services.prompt_assembly.prompt_version_service.save_snapshot", return_value=PromptSnapshot("math_agent", "v123", "abc", 1, "base prompt", "extra")):
        assembly = build_agent_prompt_assembly("math_agent")

    assert assembly.agent_name == "math_agent"
    assert assembly.extra_context == "Herramientas:\ntool-lines\n\nPermisos:\npermission-lines"
    assert assembly.system_prompt == "base prompt"
    assert assembly.prompt_version == "v123"
    assert assembly.prompt_hash == "abc"
    assert assembly.prompt_snapshot is not None
