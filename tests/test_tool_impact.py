"""Tests para la vista previa de impacto repo-aware."""


def test_tool_impact_service_detecta_archivos_reales_del_repo(tmp_path):
    from application.services.tool_impact import ToolImpactService

    (tmp_path / "application" / "services").mkdir(parents=True)
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "application" / "services" / "session_bookmarks.py").write_text("# bookmark service\nclass SessionBookmarkService: ...\n", encoding="utf-8")
    (tmp_path / "tests" / "test_session_bookmarks.py").write_text("# tests\n", encoding="utf-8")
    (tmp_path / "application" / "services" / "session_inspection.py").write_text("# inspection\n", encoding="utf-8")

    service = ToolImpactService()
    preview = service.build_preview(
        agent_name="code_agent",
        tool_name="write_code",
        arguments={"task": "add checkpoint bookmarks"},
        repo_root=tmp_path,
    )

    assert preview.discovery_basis == "repo-aware"
    assert any("session_bookmarks.py" in file for file in preview.repo_matches)
    assert any("test_session_bookmarks.py" in file for file in preview.affected_files)
    assert preview.confidence == "high"
    assert any(symbol == "SessionBookmarkService" for symbol in preview.matched_symbols)
    assert any("symbols=" in line for line in service.render_lines(preview))


def test_tool_impact_service_detecta_simbolos_reales_del_repo(tmp_path):
    from application.services.tool_impact import ToolImpactService

    (tmp_path / "application" / "services").mkdir(parents=True)
    (tmp_path / "application" / "services" / "runtime.py").write_text(
        "from application.services.session_bookmarks import SessionBookmarkService\n\nclass AgentRuntime:\n    def create_bookmark(self):\n        return SessionBookmarkService()\n",
        encoding="utf-8",
    )
    (tmp_path / "application" / "services" / "session_bookmarks.py").write_text(
        "class SessionBookmarkService:\n    pass\n",
        encoding="utf-8",
    )

    service = ToolImpactService()
    preview = service.build_preview(
        agent_name="cli",
        tool_name="write_code",
        arguments={"task": "add bookmark service"},
        repo_root=tmp_path,
    )

    assert preview.discovery_basis in {"repo-aware", "repo-aware+symbols"}
    assert any(symbol == "SessionBookmarkService" for symbol in preview.matched_symbols)
    assert any("runtime.py" in file for file in preview.repo_matches)


def test_tool_impact_service_para_web_json_capture_estima_artifacto(tmp_path):
    from application.services.tool_impact import ToolImpactService

    service = ToolImpactService()
    preview = service.build_preview(
        agent_name="web_scraping_agent",
        tool_name="scrape_website_with_json_capture",
        arguments={"url": "https://example.com/prices"},
        repo_root=tmp_path,
    )

    assert preview.discovery_basis in {"heuristic", "repo-aware"}
    assert preview.scope in {"network", "artifact-write"}
    assert any("data_trading/" in file for file in preview.affected_files) or preview.affected_files == tuple()
