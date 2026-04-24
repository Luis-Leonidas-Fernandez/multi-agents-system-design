import pytest
from unittest.mock import AsyncMock, patch

from features.sessions.application.coordinator_workers import CoordinatorWorkerService, CoordinatorWorkerStore
from features.sessions.application.coordinator_workers import CoordinatorRuntimeService


def test_spawn_list_and_message_persistence(tmp_path):
    store = CoordinatorWorkerStore(base_dir=tmp_path)
    service = CoordinatorWorkerService(store=store, clock=lambda: 1_000.0)

    worker = service.spawn_worker("sess-1", "math_agent", worker_name="math-one")

    assert worker.session_id == "sess-1"
    assert worker.agent_name == "math_agent"
    assert worker.worker_name == "math-one"

    workers = service.list_workers("sess-1")
    assert len(workers) == 1
    assert workers[0].worker_id == worker.worker_id

    message = service.send_message("sess-1", "coordinator", worker.worker_id, "hola worker")
    assert message.recipient == worker.worker_id

    persisted = service.load_messages("sess-1")
    assert len(persisted) == 1
    assert persisted[0]["content"] == "hola worker"


def test_broadcast_message_reaches_all_workers(tmp_path):
    store = CoordinatorWorkerStore(base_dir=tmp_path)
    service = CoordinatorWorkerService(store=store, clock=lambda: 1_000.0)

    w1 = service.spawn_worker("sess-2", "math_agent")
    w2 = service.spawn_worker("sess-2", "analysis_agent")

    messages = service.broadcast_message("sess-2", "coordinator", "trabajen")

    assert len(messages) == 2
    assert {msg.recipient for msg in messages} == {w1.worker_id, w2.worker_id}


def test_refine_query_removes_noise_and_deduplicates():
    from features.sessions.application.coordinator_workers import CoordinatorWorkerService

    refined = CoordinatorWorkerService._refine_query("dame las últimas noticias de noticias sobre IA en internet hoy")

    assert refined == "últimas noticias sobre IA"


@pytest.mark.asyncio
async def test_parallel_probe_round_synthesizes_best_result(tmp_path):
    store = CoordinatorWorkerStore(base_dir=tmp_path)
    service = CoordinatorWorkerService(store=store, clock=lambda: 1_000.0)
    runtime = CoordinatorRuntimeService(worker_service=service)

    with patch.object(runtime, "spawn_worker", AsyncMock(side_effect=[
        type("Worker", (), {"worker_id": "w-search", "worker_name": "search_direct_probe", "agent_name": "web_scraping_agent"})(),
        type("Worker", (), {"worker_id": "w-refine", "worker_name": "search_refined_probe", "agent_name": "web_scraping_agent"})(),
        type("Worker", (), {"worker_id": "w-extract", "worker_name": "extraction_probe", "agent_name": "web_scraping_agent"})(),
    ])), patch("features.sessions.application.coordinator_workers._search_web_query", side_effect=[
        "Search direct: la teoría de la relatividad especial https://example.com/relatividad",
        "Search refined: la teoría de la relatividad general",
        "Extraction: dato verificable",
    ]):
        result = await runtime.execute_parallel_probe_round(
            "sess-3",
            "general",
            "buscame en internet qué es la teoría de la relatividad",
        )

    assert result["best_source"] == "search_direct"
    assert result["worker_ids"] == ["w-search", "w-refine", "w-extract"]
    assert "teoría de la relatividad" in result["response"]
    assert "Sources:" in result["response"]
    assert "https://example.com/relatividad" in result["response"]
