"""Orquestación de probes coordinados para workers."""
from __future__ import annotations

import asyncio
from typing import Any, Mapping

from langchain_core.messages import HumanMessage

from core.helpers.url_helpers import _extract_web_fetch_redirect_url
from application.policies.security_flow import input_guard
from application.services.agent_registry import get_agent
from features.sessions.application.background_tasks import BackgroundTaskService, background_task_service
from features.web_scraping.domain.text_utils import _extract_urls_from_text


def _search_web_query(query: str) -> str:
    from features.web_scraping.infrastructure.search_tools import search_web

    from core.helpers.config_flow_helpers import get_web_search_runtime_config

    runtime_cfg = get_web_search_runtime_config()
    runtime_args = {
        "runtime_selected_provider": runtime_cfg.selected_provider or None,
        "runtime_provider_configured": runtime_cfg.provider_configured or None,
    }
    return str(search_web.invoke({"query": query, **{k: v for k, v in runtime_args.items() if v}}))


async def _extract_url_query(url: str) -> dict[str, Any]:
    from features.web_scraping.infrastructure.scraping_tools import fetch_web_page

    result = await fetch_web_page(url=url, prompt="Extraé y resumí la información relevante de esta página web.", use_dynamic=True)
    if isinstance(result, str):
        redirect_url = _extract_web_fetch_redirect_url(result)
        if redirect_url:
            result = await fetch_web_page(url=redirect_url, prompt="Extraé y resumí la información relevante de esta página web.", use_dynamic=True)
    return {"main_text": result if isinstance(result, str) else str(result)}


class CoordinatorProbeOrchestrator:
    def __init__(self, worker_service, background_task_backend: BackgroundTaskService | None = None) -> None:
        self._workers = worker_service
        self._background_tasks = background_task_backend or background_task_service

    def _search_web_query(self, query: str) -> str:
        return _search_web_query(query)

    async def _extract_url_query(self, url: str) -> dict[str, Any]:
        return await _extract_url_query(url)

    async def spawn_worker(self, session_id: str, agent_name: str, *, worker_name: str | None = None, parent_worker_id: str | None = None, metadata: Mapping[str, Any] | None = None):
        return self._workers.spawn_worker(
            session_id,
            agent_name,
            worker_name=worker_name,
            parent_worker_id=parent_worker_id,
            metadata=metadata,
        )

    async def send_message(self, session_id: str, worker_id: str, content: str, *, sender: str = "coordinator") -> dict[str, Any]:
        worker = self._workers.get_worker(session_id, worker_id)
        if worker is None:
            raise ValueError(f"Worker no encontrado: {worker_id}")

        self._workers.send_message(session_id, sender, worker_id, content)
        self._workers.update_worker_status(session_id, worker_id, "running")

        async def _runner() -> dict[str, Any]:
            guard_state = {"messages": [HumanMessage(content=content)]}
            guard_result = input_guard(guard_state)
            if guard_result and guard_result.get("blocked"):
                return {"worker_id": worker_id, "agent_name": worker.agent_name, "response": "Solicitud bloqueada por política de seguridad."}

            agent = get_agent(worker.agent_name)
            result = await agent.ainvoke({"messages": [HumanMessage(content=content)]})
            messages = result.get("messages", []) if isinstance(result, dict) else []
            last_message = messages[-1] if messages else None
            response_text = getattr(last_message, "content", "") if last_message is not None else ""
            self._workers.send_message(session_id, worker.worker_id, sender, response_text or "(sin respuesta)", kind="result")
            self._workers.update_worker_status(session_id, worker_id, "idle", metadata=worker.metadata)
            return {"worker_id": worker_id, "agent_name": worker.agent_name, "response": response_text}

        task = await self._background_tasks.submit(
            session_id,
            f"worker:{worker.worker_name}",
            _runner,
            metadata={"worker_id": worker_id, "agent_name": worker.agent_name, "kind": "coordinator_worker_turn"},
        )
        return {"task_id": task.task_id, "worker_id": worker_id, "agent_name": worker.agent_name}

    async def execute_worker_turn(self, session_id: str, worker_id: str, content: str, *, sender: str = "coordinator") -> dict[str, Any]:
        worker = self._workers.get_worker(session_id, worker_id)
        if worker is None:
            raise ValueError(f"Worker no encontrado: {worker_id}")

        self._workers.send_message(session_id, sender, worker_id, content)
        self._workers.update_worker_status(session_id, worker_id, "running")

        guard_state = {"messages": [HumanMessage(content=content)]}
        guard_result = input_guard(guard_state)
        if guard_result and guard_result.get("blocked"):
            return {"worker_id": worker_id, "agent_name": worker.agent_name, "response": "Solicitud bloqueada por política de seguridad."}

        agent = get_agent(worker.agent_name)
        result = await agent.ainvoke({"messages": [HumanMessage(content=content)]})
        messages = result.get("messages", []) if isinstance(result, dict) else []
        last_message = messages[-1] if messages else None
        response_text = getattr(last_message, "content", "") if last_message is not None else ""
        self._workers.send_message(session_id, worker.worker_id, sender, response_text or "(sin respuesta)", kind="result")
        self._workers.update_worker_status(session_id, worker_id, "idle", metadata=worker.metadata)
        return {"worker_id": worker_id, "agent_name": worker.agent_name, "response": response_text}

    async def execute_parallel_probe_round(self, session_id: str, category: str, question: str, *, sender: str = "coordinator") -> dict[str, Any]:
        specs = self._workers.build_probe_specs(session_id, category, question)
        if not specs:
            return {"response": "", "best_source": "", "worker_ids": [], "probe_results": []}

        async def _run(spec) -> dict[str, Any]:
            worker = await self.spawn_worker(
                session_id,
                spec.agent_name,
                worker_name=spec.worker_name,
                metadata={"source_name": spec.source_name, "category": category, "probe": True},
            )
            self._workers.send_message(session_id, sender, worker.worker_id, spec.query)
            self._workers.update_worker_status(session_id, worker.worker_id, "running")

            guard_text = spec.query if not spec.target_url else f"{spec.query}\n{spec.target_url}"
            guard_state = {"messages": [HumanMessage(content=guard_text)]}
            guard_result = input_guard(guard_state)
            if guard_result and guard_result.get("blocked"):
                blocked = {"worker_id": worker.worker_id, "worker_name": worker.worker_name, "agent_name": worker.agent_name, "source_name": spec.source_name, "response": "Solicitud bloqueada por política de seguridad.", "source_url": spec.target_url or "", "urls": [spec.target_url] if spec.target_url else []}
                self._workers.send_message(session_id, worker.worker_id, sender, blocked["response"], kind="result")
                self._workers.update_worker_status(session_id, worker.worker_id, "idle", metadata=getattr(worker, "metadata", None))
                return blocked

            if spec.target_url:
                execution_text = await self._extract_url_query(spec.target_url)
                execution = {"response": execution_text.get("main_text", "") if isinstance(execution_text, dict) else str(execution_text), "source_url": spec.target_url, "urls": [spec.target_url]}
            else:
                loop = asyncio.get_running_loop()
                search_query = spec.query if spec.source_name == "search_direct" else f"{spec.query}"
                execution_text = await loop.run_in_executor(None, lambda: self._search_web_query(search_query))
                execution = {"response": execution_text, "source_url": "", "urls": _extract_urls_from_text(execution_text)}

            self._workers.send_message(session_id, worker.worker_id, sender, execution.get("response", "") or "(sin respuesta)", kind="result")
            self._workers.update_worker_status(session_id, worker.worker_id, "idle", metadata=getattr(worker, "metadata", None))
            return {"worker_id": worker.worker_id, "worker_name": worker.worker_name, "agent_name": worker.agent_name, "source_name": spec.source_name, "response": execution.get("response", ""), "source_url": execution.get("source_url", ""), "urls": execution.get("urls", [])}

        raw = await asyncio.gather(*(_run(spec) for spec in specs), return_exceptions=True)
        probe_results = [r for r in raw if isinstance(r, dict)]
        synthesis = self._workers.synthesize_probe_results(category, question, probe_results)
        return {**synthesis, "worker_ids": [result["worker_id"] for result in probe_results], "probe_results": probe_results}

    def list_workers(self, session_id: str):
        return self._workers.list_workers(session_id)

    def list_messages(self, session_id: str):
        return self._workers.load_messages(session_id)


__all__ = ["CoordinatorProbeOrchestrator", "_search_web_query", "_extract_url_query"]
