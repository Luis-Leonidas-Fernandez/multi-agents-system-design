"""Tests para el retrieval inteligente de memoria."""


def test_memory_retrieval_service_busca_por_terminos(tmp_path):
    from application.services.memory_retrieval import MemoryRetrievalService

    service = MemoryRetrievalService(sessions_dir=tmp_path)
    session_a = tmp_path / "sess-a"
    session_a.mkdir()
    (session_a / "MEMORY.md").write_text("- Preferencia: usar CLI\n- Resultado: replay unificado", encoding="utf-8")
    session_b = tmp_path / "sess-b"
    session_b.mkdir()
    (session_b / "MEMORY.md").write_text("- Tarea: mejorar memoria\n- Resultado: search helper", encoding="utf-8")

    hits = service.search("replay CLI", limit=2)

    assert hits
    assert hits[0].session_id == "sess-a"
    assert "replay" in hits[0].excerpt.lower() or "cli" in hits[0].excerpt.lower()


def test_memory_retrieval_service_summarize_y_lista_sessions(tmp_path):
    from application.services.memory_retrieval import MemoryRetrievalService

    service = MemoryRetrievalService(sessions_dir=tmp_path)
    (tmp_path / "sess-c").mkdir()
    (tmp_path / "sess-c" / "MEMORY.md").write_text("- Aprendizaje: search inteligente", encoding="utf-8")

    summary = service.summarize("search inteligente")

    assert summary["total"] == 1
    assert summary["hits"][0]["session_id"] == "sess-c"
    assert service.list_sessions() == ["sess-c"]


def test_format_memory_search_results_muestra_hallazgos():
    from application.services.session_inspection import format_memory_search_results

    lines = format_memory_search_results(
        "replay",
        [
            {
                "session_id": "sess-a",
                "score": 10.0,
                "memory_path": "sessions/sess-a/MEMORY.md",
                "excerpt": "- Resultado: replay unificado",
                "matched_terms": ["replay"],
            }
        ],
    )

    assert any("sess-a" in line for line in lines)
    assert any("replay" in line for line in lines)


def test_handle_inspection_command_memory(monkeypatch, capsys):
    from application.services.runtime import AgentRuntime
    from main import _handle_inspection_command

    runtime = AgentRuntime(gateway=None)
    runtime._memory_retrieval = type("Stub", (), {  # type: ignore[attr-defined]
        "list_sessions": lambda self: ["sess-a", "sess-b"],
        "search": lambda self, query, limit=5: [
            type("Hit", (), {"session_id": "sess-a", "score": 12.0, "memory_path": "sessions/sess-a/MEMORY.md", "excerpt": "- replay unificado", "matched_terms": ["replay"], "line_count": 1, "char_count": 10, "modified_at_ms": 1})(),
        ],
    })()  # type: ignore[attr-defined]

    lifecycle = type("L", (), {"session_id": "sess-z"})()

    handled = _handle_inspection_command("/memory replay", lifecycle, runtime=runtime)
    output = capsys.readouterr().out

    assert handled is True
    assert "sess-a" in output


def test_handle_inspection_command_memory_sin_query(capsys):
    from application.services.runtime import AgentRuntime
    from main import _handle_inspection_command

    runtime = AgentRuntime(gateway=None)
    runtime._memory_retrieval = type("Stub", (), {  # type: ignore[attr-defined]
        "list_sessions": lambda self: ["sess-a"],
        "search": lambda self, query, limit=5: [],
    })()  # type: ignore[attr-defined]
    lifecycle = type("L", (), {"session_id": "sess-z"})()

    handled = _handle_inspection_command("/memory", lifecycle, runtime=runtime)
    output = capsys.readouterr().out

    assert handled is True
    assert "sess-a" in output


def test_runtime_and_lifecycle_memory_helpers_delegan():
    from application.services.runtime import AgentRuntime, SessionLifecycle

    runtime = AgentRuntime(gateway=None)
    runtime._memory_retrieval = type("Stub", (), {  # type: ignore[attr-defined]
        "list_sessions": lambda self: ["sess-a"],
        "search": lambda self, query, limit=5: ["match"],
    })()  # type: ignore[attr-defined]

    lifecycle = SessionLifecycle(runtime=runtime, session_id="sess-x")

    assert runtime.list_memory_sessions() == ["sess-a"]
    assert runtime.search_memory("replay") == ["match"]
    assert lifecycle.list_memory_sessions() == ["sess-a"]
    assert lifecycle.search_memory("replay") == ["match"]
