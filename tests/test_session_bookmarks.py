"""Tests para checkpoints/bookmarks de sesión."""
from unittest.mock import MagicMock


def test_session_bookmark_store_persiste_y_lee(tmp_path):
    from features.sessions.application.session_bookmarks import SessionBookmark, SessionBookmarkStore

    store = SessionBookmarkStore(base_dir=tmp_path)
    bookmark = SessionBookmark(
        checkpoint_id="chk-1",
        session_id="sess-1",
        label="base",
        created_at_ms=1,
        note="",
        message_count=2,
        has_memory=True,
        artifact_path="data/sessions/sess-1/SESSION_ARTIFACT.json",
        replay_item_count=7,
        context_budget={"report": {"scope": "session"}},
        prompt_agents=["math_agent"],
    )

    store.save(bookmark)

    listed = store.list("sess-1")
    assert listed[0]["checkpoint_id"] == "chk-1"
    loaded = store.load("sess-1", "chk-1")
    assert loaded is not None
    assert loaded["label"] == "base"


def test_session_bookmark_service_crea_ids_estables():
    from features.sessions.application.session_bookmarks import SessionBookmarkService, SessionBookmarkStore

    store = MagicMock(spec=SessionBookmarkStore)
    service = SessionBookmarkService(store=store)
    store.list.return_value = []  # type: ignore[attr-defined]

    bookmark = service.create(
        session_id="sess-2",
        label="mi checkpoint",
        artifact_path="data/sessions/sess-2/SESSION_ARTIFACT.json",
        message_count=3,
        has_memory=False,
        replay_item_count=4,
        context_budget={"report": {"scope": "session"}},
        prompt_agents=["math_agent"],
    )

    assert bookmark.session_id == "sess-2"
    assert bookmark.label == "mi checkpoint"
    assert bookmark.checkpoint_id.startswith("mi-checkpoint-")
    store.save.assert_called_once()  # type: ignore[attr-defined]


def test_runtime_bookmark_delega_en_servicio():
    from application.services.runtime import AgentRuntime, SessionLifecycle
    from features.sessions.application.context_budget import ContextBudgetItem, SessionContextBudget
    from types import SimpleNamespace

    runtime = AgentRuntime(gateway=MagicMock())
    runtime._bookmarks = MagicMock()  # type: ignore[attr-defined]
    runtime._bookmarks.list.return_value = [{"checkpoint_id": "chk-1"}]  # type: ignore[attr-defined]
    runtime._bookmarks.describe.return_value = {"checkpoint_id": "chk-1"}  # type: ignore[attr-defined]
    runtime._bookmarks.create.return_value = {"checkpoint_id": "chk-2"}  # type: ignore[attr-defined]
    runtime._artifacts = MagicMock()  # type: ignore[attr-defined]
    runtime._artifacts.export_artifact.return_value = SimpleNamespace(message_count=1, has_memory=False, prompt_snapshots=[])
    runtime._artifacts.artifact_path.return_value = "data/sessions/sess-3/SESSION_ARTIFACT.json"  # type: ignore[attr-defined]
    runtime.build_session_replay = MagicMock(return_value=SimpleNamespace(items=[1, 2]))  # type: ignore[attr-defined]
    runtime.build_context_budget = MagicMock(return_value=SessionContextBudget(  # type: ignore[attr-defined]
        session_id="sess-3",
        generated_at_ms=1,
        budget_chars=1000,
        estimated_context_chars=100,
        estimated_remaining_chars=900,
        estimated_tokens=25,
        status="ok",
        scope="session",
        transcript_message_count=1,
        memory_present=False,
        items=[ContextBudgetItem(section="transcript", role="included", chars=10, detail="2 mensajes")],
    ))
    lifecycle = SessionLifecycle(runtime=runtime, session_id="sess-3")

    created = lifecycle.create_bookmark(label="base")
    listed = lifecycle.list_bookmarks()
    described = lifecycle.describe_bookmark("chk-1")

    assert created == {"checkpoint_id": "chk-2"}
    assert listed == [{"checkpoint_id": "chk-1"}]
    assert described == {"checkpoint_id": "chk-1"}
    runtime._bookmarks.create.assert_called_once()  # type: ignore[attr-defined]
