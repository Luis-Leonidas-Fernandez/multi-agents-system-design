"""Registro y mensajería de workers para el modo coordinador.

Este módulo no ejecuta LLMs por sí mismo; define la frontera estable para
spawn, inbox/outbox y persistencia de workers por sesión. La ejecución real
puede engancharse más tarde a background tasks, pero el contrato ya queda
separado y testeable.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from langchain_core.messages import HumanMessage

from infra.stores import AppendOnlyStore

from application.services.agent_registry import get_agent_spec
from application.helpers.config_flow_helpers import get_web_search_runtime_config
from application.helpers.url_helpers import _extract_web_fetch_redirect_url
from domain.web_text_utils import _extract_urls_from_text
from application.services.background_tasks import background_task_service, BackgroundTaskService


def _search_web_query(query: str) -> str:
    from tools import search_web

    runtime_cfg = get_web_search_runtime_config()
    runtime_args = {
        "runtime_selected_provider": runtime_cfg.selected_provider or None,
        "runtime_provider_configured": runtime_cfg.provider_configured or None,
    }
    return str(search_web.invoke({"query": query, **{k: v for k, v in runtime_args.items() if v}}))


async def _extract_url_query(url: str) -> dict[str, Any]:
    from tools.scraping_tools import fetch_web_page

    result = await fetch_web_page(url=url, prompt="Extraé y resumí la información relevante de esta página web.", use_dynamic=True)
    if isinstance(result, str):
        redirect_url = _extract_web_fetch_redirect_url(result)
        if redirect_url:
            result = await fetch_web_page(url=redirect_url, prompt="Extraé y resumí la información relevante de esta página web.", use_dynamic=True)
    return {"main_text": result if isinstance(result, str) else str(result)}


@dataclass(frozen=True)
class CoordinatorWorker:
    session_id: str
    worker_id: str
    worker_name: str
    agent_name: str
    status: str
    created_at_ms: int
    updated_at_ms: int
    parent_worker_id: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CoordinatorMessage:
    session_id: str
    message_id: str
    sender: str
    recipient: str
    content: str
    created_at_ms: int
    kind: str = "direct"


@dataclass(frozen=True)
class CoordinatorProbeSpec:
    session_id: str
    source_name: str
    worker_name: str
    query: str
    target_url: str | None = None
    agent_name: str = "web_scraping_agent"


class CoordinatorWorkerStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or Path("sessions")

    def _session_dir(self, session_id: str) -> Path:
        return self._base_dir / session_id

    def _workers_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "COORDINATOR_WORKERS.json"

    def _messages_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "COORDINATOR_MESSAGES.jsonl"

    def save_workers(self, session_id: str, workers: list[CoordinatorWorker]) -> None:
        path = self._workers_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(worker) for worker in workers]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_workers(self, session_id: str) -> list[dict[str, Any]]:
        path = self._workers_path(session_id)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []
        return [record for record in payload if isinstance(record, dict)]

    def append_message(self, message: CoordinatorMessage) -> None:
        AppendOnlyStore.append(self._messages_path(message.session_id), message)

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        return AppendOnlyStore.load_lines(self._messages_path(session_id))


class CoordinatorWorkerService:
    def __init__(self, store: CoordinatorWorkerStore | None = None, clock: Callable[[], float] | None = None) -> None:
        self._store = store or CoordinatorWorkerStore()
        self._clock = clock or time.time
        self._workers: dict[str, dict[str, CoordinatorWorker]] = {}

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    def _load_session_workers(self, session_id: str) -> dict[str, CoordinatorWorker]:
        if session_id not in self._workers:
            loaded = {}
            for record in self._store.load_workers(session_id):
                try:
                    worker = CoordinatorWorker(
                        session_id=str(record.get("session_id") or session_id),
                        worker_id=str(record.get("worker_id") or ""),
                        worker_name=str(record.get("worker_name") or "worker"),
                        agent_name=str(record.get("agent_name") or ""),
                        status=str(record.get("status") or "idle"),
                        created_at_ms=int(record.get("created_at_ms") or self._now_ms()),
                        updated_at_ms=int(record.get("updated_at_ms") or self._now_ms()),
                        parent_worker_id=record.get("parent_worker_id"),
                        metadata=record.get("metadata"),
                    )
                except Exception:
                    continue
                if worker.worker_id:
                    loaded[worker.worker_id] = worker
            self._workers[session_id] = loaded
        return self._workers[session_id]

    def _persist(self, session_id: str) -> None:
        self._store.save_workers(session_id, list(self._load_session_workers(session_id).values()))

    def spawn_worker(
        self,
        session_id: str,
        agent_name: str,
        *,
        worker_name: str | None = None,
        parent_worker_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CoordinatorWorker:
        get_agent_spec(agent_name)
        workers = self._load_session_workers(session_id)
        worker_id = str(uuid.uuid4())[:8]
        worker = CoordinatorWorker(
            session_id=session_id,
            worker_id=worker_id,
            worker_name=worker_name or agent_name,
            agent_name=agent_name,
            status="idle",
            created_at_ms=self._now_ms(),
            updated_at_ms=self._now_ms(),
            parent_worker_id=parent_worker_id,
            metadata=metadata,
        )
        workers[worker_id] = worker
        self._persist(session_id)
        return worker

    def list_workers(self, session_id: str) -> list[CoordinatorWorker]:
        return list(self._load_session_workers(session_id).values())

    def get_worker(self, session_id: str, worker_id: str) -> CoordinatorWorker | None:
        return self._load_session_workers(session_id).get(worker_id)

    def update_worker_status(self, session_id: str, worker_id: str, status: str, *, metadata: Mapping[str, Any] | None = None) -> CoordinatorWorker | None:
        workers = self._load_session_workers(session_id)
        current = workers.get(worker_id)
        if current is None:
            return None
        updated = CoordinatorWorker(
            session_id=current.session_id,
            worker_id=current.worker_id,
            worker_name=current.worker_name,
            agent_name=current.agent_name,
            status=status,
            created_at_ms=current.created_at_ms,
            updated_at_ms=self._now_ms(),
            parent_worker_id=current.parent_worker_id,
            metadata=metadata if metadata is not None else current.metadata,
        )
        workers[worker_id] = updated
        self._persist(session_id)
        return updated

    def send_message(self, session_id: str, sender: str, recipient: str, content: str, *, kind: str = "direct") -> CoordinatorMessage:
        message = CoordinatorMessage(
            session_id=session_id,
            message_id=str(uuid.uuid4())[:8],
            sender=sender,
            recipient=recipient,
            content=content,
            created_at_ms=self._now_ms(),
            kind=kind,
        )
        self._store.append_message(message)
        return message

    def broadcast_message(self, session_id: str, sender: str, content: str) -> list[CoordinatorMessage]:
        messages: list[CoordinatorMessage] = []
        for worker in self.list_workers(session_id):
            messages.append(self.send_message(session_id, sender, worker.worker_id, content, kind="broadcast"))
        return messages

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        return self._store.load_messages(session_id)

    def build_probe_specs(self, session_id: str, category: str, question: str) -> list[CoordinatorProbeSpec]:
        question = question.strip()
        if not question:
            return []

        urls = re.findall(r"https?://\S+", question)
        if urls:
            return [
                CoordinatorProbeSpec(
                    session_id=session_id,
                    source_name=f"url_{idx + 1}",
                    worker_name=f"url_probe_{idx + 1}",
                    query=question,
                    target_url=url,
                )
                for idx, url in enumerate(urls[:3])
            ] + [
                CoordinatorProbeSpec(
                    session_id=session_id,
                    source_name="search_direct",
                    worker_name="search_direct_probe",
                    query=question,
                ),
                CoordinatorProbeSpec(
                    session_id=session_id,
                    source_name="search_refined",
                    worker_name="search_refined_probe",
                    query=self._refine_query(question),
                ),
            ]

        specs = [
            CoordinatorProbeSpec(
                session_id=session_id,
                source_name="search_direct",
                worker_name="search_direct_probe",
                query=question,
            ),
            CoordinatorProbeSpec(
                session_id=session_id,
                source_name="search_refined",
                worker_name="search_refined_probe",
                query=self._refine_query(question),
            ),
            CoordinatorProbeSpec(
                session_id=session_id,
                source_name="extraction",
                worker_name="extraction_probe",
                query=question,
            ),
        ]
        return specs

    @staticmethod
    def _refine_query(question: str) -> str:
        cleaned = re.sub(r"https?://\S+", " ", question)
        cleaned = re.sub(r"[\t\n\r]+", " ", cleaned)
        cleaned = re.sub(r"[¿?¡!.,;:()\[\]{}<>|/\\]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        fillers = {
            "dame", "buscame", "buscá", "buscar", "please", "por", "favor",
            "en", "internet", "web", "online", "me", "de", "la", "el",
            "los", "las", "un", "una", "unos", "unas", "del", "al", "y",
            "qué", "que", "es", "son", "cuál", "cual", "cuáles", "cuales",
            "cómo", "como", "cuándo", "cuando", "dónde", "donde", "hoy",
            "actual", "actuales", "actualmente", "info", "información", "informacion",
        }

        words: list[str] = []
        seen: set[str] = set()
        for raw_word in cleaned.split():
            word = raw_word.strip()
            if not word:
                continue
            normalized = word.lower()
            if normalized in fillers:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            words.append(word)

        if not words:
            return question.strip()

        if len(words) > 10:
            words = words[:10]

        return " ".join(words).strip()

    @staticmethod
    def _score_probe_response(category: str, text: str, source_name: str) -> int:
        lower = text.lower()
        score = 0
        if not text.strip():
            return -100
        if any(marker in lower for marker in ("no pude", "sin dato", "no encontré", "no logré")):
            score -= 8
        if re.search(r"\b\d+\s*-\s*\d+\b", text):
            score += 8
        for kw in ("respuesta", "verificable", "evidencia", "fuente", "dato", "hecho", "confirmado"):
            if kw in lower:
                score += 1
        normalized_source = source_name.replace("_", " ").lower()
        if normalized_source in lower:
            score += 2
        if source_name.startswith("search_direct"):
            score += 2
        elif source_name.startswith("search_refined"):
            score += 1
        if len(text.split()) < 8:
            score -= 2
        if len(text.split()) > 80:
            score -= 1
        return score

    @staticmethod
    def _extract_unique_score_lines(text: str) -> list[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        unique: list[str] = []
        seen: set[str] = set()
        score_pat = re.compile(r"\b.+?\b\d+\s*-\s*\d+\b.+")
        for line in lines:
            normalized = " ".join(line.split())
            if normalized in seen:
                continue
            if score_pat.search(normalized):
                seen.add(normalized)
                unique.append(normalized)
        return unique

    def synthesize_probe_results(
        self,
        category: str,
        question: str,
        probe_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        scored: list[dict[str, Any]] = []
        for result in probe_results:
            response = str(result.get("response") or "")
            source_name = str(result.get("source_name") or result.get("worker_name") or "worker")
            score = self._score_probe_response(category, response, source_name)
            urls = result.get("urls") or []
            source_url = str(result.get("source_url") or "")
            scored.append({
                **result,
                "response": response,
                "score": score,
                "source_name": source_name,
                "urls": list(urls) if isinstance(urls, list) else [],
                "source_url": source_url,
            })

        if not scored:
            return {"response": "No pude obtener resultados claros para esta consulta.", "best_source": "", "probe_results": []}

        ranked = sorted(scored, key=lambda item: (item["score"], len(item["response"])), reverse=True)
        best = ranked[0]
        best_response = best["response"].strip()

        best_lines = self._extract_unique_score_lines(best_response)
        corroboration: list[str] = []
        if best_lines:
            corroboration.extend(best_lines)
        for candidate in ranked[1:]:
            for line in self._extract_unique_score_lines(candidate["response"]):
                if line not in corroboration:
                    corroboration.append(line)
                if len(corroboration) >= 4:
                    break
            if len(corroboration) >= 4:
                break

        if corroboration:
            response_lines = [f"Fuente más confiable: {best['source_name']}"]
            response_lines.extend(f"- {line}" for line in corroboration[:4])
            response = "\n".join(response_lines)
        else:
            response = best_response or "No pude obtener resultados claros para esta consulta."

        sources: list[dict[str, str]] = []
        for item in ranked:
            candidate_urls = list(item.get("urls") or [])
            source_url = str(item.get("source_url") or "")
            if source_url:
                candidate_urls.insert(0, source_url)
            for url in candidate_urls:
                if url and not any(existing["url"] == url for existing in sources):
                    sources.append({"title": str(item.get("source_name") or "source"), "url": url})

        if sources:
            response = f"{response}\n\nSources:\n" + "\n".join(
                f"- [{source['title']}]({source['url']})" for source in sources[:6]
            )

        if len(response.split()) <= 12 and len(ranked) > 1:
            second = ranked[1]["response"].strip()
            if second and second != response:
                response = f"{response}\n\nRespaldo de {ranked[1]['source_name']}: {second}"

        return {
            "response": response,
            "best_source": best["source_name"],
            "probe_results": ranked,
        }

class CoordinatorRuntimeService:
    """Ejecuta workers coordinados usando el registry persistido y background tasks."""

    def __init__(
        self,
        worker_service: CoordinatorWorkerService | None = None,
        background_task_backend: BackgroundTaskService | None = None,
    ) -> None:
        self._workers = worker_service or coordinator_worker_service
        self._background_tasks = background_task_backend or background_task_service

    async def spawn_worker(self, session_id: str, agent_name: str, *, worker_name: str | None = None, parent_worker_id: str | None = None, metadata: Mapping[str, Any] | None = None) -> CoordinatorWorker:
        worker = self._workers.spawn_worker(
            session_id,
            agent_name,
            worker_name=worker_name,
            parent_worker_id=parent_worker_id,
            metadata=metadata,
        )
        return worker

    async def send_message(self, session_id: str, worker_id: str, content: str, *, sender: str = "coordinator") -> dict[str, Any]:
        worker = self._workers.get_worker(session_id, worker_id)
        if worker is None:
            raise ValueError(f"Worker no encontrado: {worker_id}")

        self._workers.send_message(session_id, sender, worker_id, content)
        self._workers.update_worker_status(session_id, worker_id, "running")

        async def _runner() -> dict[str, Any]:
            from application.policies.security_flow import input_guard
            from application.services.agent_registry import get_agent

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
            return {
                "worker_id": worker_id,
                "agent_name": worker.agent_name,
                "response": response_text,
            }

        task = await self._background_tasks.submit(
            session_id,
            f"worker:{worker.worker_name}",
            _runner,
            metadata={"worker_id": worker_id, "agent_name": worker.agent_name, "kind": "coordinator_worker_turn"},
        )
        return {
            "task_id": task.task_id,
            "worker_id": worker_id,
            "agent_name": worker.agent_name,
        }

    async def execute_worker_turn(self, session_id: str, worker_id: str, content: str, *, sender: str = "coordinator") -> dict[str, Any]:
        worker = self._workers.get_worker(session_id, worker_id)
        if worker is None:
            raise ValueError(f"Worker no encontrado: {worker_id}")

        self._workers.send_message(session_id, sender, worker_id, content)
        self._workers.update_worker_status(session_id, worker_id, "running")

        from application.policies.security_flow import input_guard
        from application.services.agent_registry import get_agent

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
        return {
            "worker_id": worker_id,
            "agent_name": worker.agent_name,
            "response": response_text,
        }

    async def execute_parallel_probe_round(
        self,
        session_id: str,
        category: str,
        question: str,
        *,
        sender: str = "coordinator",
    ) -> dict[str, Any]:
        specs = self._workers.build_probe_specs(session_id, category, question)
        if not specs:
            return {"response": "", "best_source": "", "worker_ids": [], "probe_results": []}

        async def _run(spec: CoordinatorProbeSpec) -> dict[str, Any]:
            worker = await self.spawn_worker(
                session_id,
                spec.agent_name,
                worker_name=spec.worker_name,
                metadata={"source_name": spec.source_name, "category": category, "probe": True},
            )
            self._workers.send_message(session_id, sender, worker.worker_id, spec.query)
            self._workers.update_worker_status(session_id, worker.worker_id, "running")

            if spec.target_url:
                execution_text = await _extract_url_query(spec.target_url)
                execution = {
                    "response": execution_text.get("main_text", "") if isinstance(execution_text, dict) else str(execution_text),
                    "source_url": spec.target_url,
                    "urls": [spec.target_url],
                }
            else:
                loop = asyncio.get_running_loop()
                search_query = spec.query if spec.source_name == "search_direct" else f"{spec.query}"
                execution_text = await loop.run_in_executor(None, lambda: _search_web_query(search_query))
                execution = {
                    "response": execution_text,
                    "source_url": "",
                    "urls": _extract_urls_from_text(execution_text),
                }

            self._workers.send_message(session_id, worker.worker_id, sender, execution.get("response", "") or "(sin respuesta)", kind="result")
            self._workers.update_worker_status(session_id, worker.worker_id, "idle", metadata=getattr(worker, "metadata", None))
            return {
                "worker_id": worker.worker_id,
                "worker_name": worker.worker_name,
                "agent_name": worker.agent_name,
                "source_name": spec.source_name,
                "response": execution.get("response", ""),
                "source_url": execution.get("source_url", ""),
                "urls": execution.get("urls", []),
            }

        raw = await asyncio.gather(*(_run(spec) for spec in specs), return_exceptions=True)
        probe_results = [r for r in raw if isinstance(r, dict)]
        synthesis = self._workers.synthesize_probe_results(category, question, probe_results)
        return {
            **synthesis,
            "worker_ids": [result["worker_id"] for result in probe_results],
            "probe_results": probe_results,
        }

    def list_workers(self, session_id: str) -> list[CoordinatorWorker]:
        return self._workers.list_workers(session_id)

    def list_messages(self, session_id: str) -> list[dict[str, Any]]:
        return self._workers.load_messages(session_id)


coordinator_worker_store = CoordinatorWorkerStore()
coordinator_worker_service = CoordinatorWorkerService()
coordinator_runtime_service = CoordinatorRuntimeService()


__all__ = [
    "CoordinatorMessage",
    "CoordinatorProbeSpec",
    "CoordinatorWorker",
    "CoordinatorWorkerService",
    "CoordinatorWorkerStore",
    "CoordinatorRuntimeService",
    "coordinator_worker_service",
    "coordinator_worker_store",
    "coordinator_runtime_service",
]
