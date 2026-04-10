"""Tests del ensamblado del prompt del supervisor."""
from unittest.mock import patch


def test_build_supervisor_prompt_assembly_arma_prompt_estable():
    from application.services.supervisor_prompt import build_supervisor_prompt_assembly

    with patch("application.services.supervisor_prompt.build_supervisor_agent_lines", return_value="agent-lines"):
        assembly = build_supervisor_prompt_assembly()

    assert assembly.agent_lines == "agent-lines"
    assert "Eres un supervisor" in assembly.system_prompt
    assert "agent-lines" in assembly.system_prompt


def test_build_supervisor_prompt_assembly_usa_coordinator_mode(monkeypatch):
    from application.services.supervisor_prompt import build_supervisor_prompt_assembly

    monkeypatch.setenv("COORDINATOR_MODE", "true")

    with patch("application.services.supervisor_prompt.build_coordinator_prompt_assembly") as build_coord:
        build_coord.return_value.system_prompt = "coord-prompt"
        build_coord.return_value.worker_lines = "worker-lines"

        assembly = build_supervisor_prompt_assembly()

    assert assembly.system_prompt == "coord-prompt"
    assert assembly.agent_lines == "worker-lines"
