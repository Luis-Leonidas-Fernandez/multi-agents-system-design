"""Tests para versionado y snapshots de prompts."""


def test_prompt_version_service_calcula_hash_estable():
    from application.services.prompt_versioning import PromptVersionService

    service = PromptVersionService()
    snapshot_a = service.build_snapshot("math_agent", "system prompt", "extra")
    snapshot_b = service.build_snapshot("math_agent", "system prompt", "extra")

    assert snapshot_a.prompt_hash == snapshot_b.prompt_hash
    assert snapshot_a.prompt_version.startswith("v")


def test_prompt_version_store_persiste_historial(tmp_path):
    from application.services.prompt_versioning import PromptSnapshotStore, PromptVersionService

    store = PromptSnapshotStore(base_dir=tmp_path)
    service = PromptVersionService(store=store)

    snapshot = service.save_snapshot("code_agent", "base", "ctx")

    loaded = service.load_snapshot("code_agent")
    history = service.load_history("code_agent")

    assert loaded is not None
    assert loaded["prompt_hash"] == snapshot.prompt_hash
    assert history and history[-1]["prompt_version"] == snapshot.prompt_version
    assert service.list_agents() == ["code_agent"]
    assert str(service.snapshot_path("code_agent")).endswith("PROMPT_SNAPSHOT.json")
    assert str(service.history_path("code_agent")).endswith("PROMPT_HISTORY.jsonl")


def test_session_inspection_format_prompt_snapshot():
    from application.services.session_inspection import format_prompt_snapshot, format_prompt_snapshot_list

    lines = format_prompt_snapshot(
        {
            "agent_name": "math_agent",
            "prompt_version": "v123",
            "prompt_hash": "abcdef1234567890",
            "created_at_ms": 1,
            "extra_context": "x" * 5,
            "system_prompt": "y" * 9,
        }
    )

    assert any("math_agent" in line for line in lines)
    assert any("v123" in line for line in lines)
    assert any("extra_context_chars=5" in line for line in lines)
    assert format_prompt_snapshot_list([])[0].startswith("[prompt] no hay")


def test_session_inspection_format_prompt_snapshot_incluye_paths():
    from application.services.session_inspection import format_prompt_snapshot

    lines = format_prompt_snapshot(
        {
            "agent_name": "math_agent",
            "prompt_version": "v123",
            "prompt_hash": "abcdef1234567890",
            "created_at_ms": 1,
            "extra_context": "x",
            "system_prompt": "y",
            "snapshot_path": "prompts/math_agent/PROMPT_SNAPSHOT.json",
            "history_path": "prompts/math_agent/PROMPT_HISTORY.jsonl",
        }
    )

    assert any("snapshot_path=" in line for line in lines)
    assert any("history_path=" in line for line in lines)


def test_prompt_version_service_expone_rutas(tmp_path):
    from application.services.prompt_versioning import PromptSnapshotStore, PromptVersionService

    service = PromptVersionService(store=PromptSnapshotStore(base_dir=tmp_path))

    assert str(service.snapshot_path("math_agent")).endswith("PROMPT_SNAPSHOT.json")
    assert str(service.history_path("math_agent")).endswith("PROMPT_HISTORY.jsonl")
