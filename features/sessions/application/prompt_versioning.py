"""Versionado y snapshots de prompts por agente.

Guarda una huella estable del prompt final (hash + versión) para poder
inspeccionar qué prompt fue usado en cada build sin depender del source.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any


_PROMPT_DIR = Path("agents") / "snapshots"


@dataclass(frozen=True)
class PromptSnapshot:
    agent_name: str
    prompt_version: str
    prompt_hash: str
    created_at_ms: int
    system_prompt: str
    extra_context: str


class PromptSnapshotStore:
    """Persistencia simple de snapshots por agente."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or _PROMPT_DIR

    def _agent_dir(self, agent_name: str) -> Path:
        return self._base_dir / agent_name

    def _snapshot_path(self, agent_name: str) -> Path:
        return self._agent_dir(agent_name) / "PROMPT_SNAPSHOT.json"

    def _history_path(self, agent_name: str) -> Path:
        return self._agent_dir(agent_name) / "PROMPT_HISTORY.jsonl"

    def snapshot_path(self, agent_name: str) -> Path:
        return self._snapshot_path(agent_name)

    def history_path(self, agent_name: str) -> Path:
        return self._history_path(agent_name)

    def save(self, snapshot: PromptSnapshot) -> None:
        agent_dir = self._agent_dir(snapshot.agent_name)
        agent_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(snapshot), ensure_ascii=False, indent=2)
        self._snapshot_path(snapshot.agent_name).write_text(payload, encoding="utf-8")
        with self._history_path(snapshot.agent_name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(snapshot), ensure_ascii=False) + "\n")

    def load(self, agent_name: str) -> dict[str, Any] | None:
        path = self._snapshot_path(agent_name)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def load_history(self, agent_name: str) -> list[dict[str, Any]]:
        path = self._history_path(agent_name)
        if not path.exists():
            return []
        history: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return history

    def list_agents(self) -> list[str]:
        if not self._base_dir.exists():
            return []
        agents: list[str] = []
        for agent_dir in self._base_dir.iterdir():
            if agent_dir.is_dir() and (agent_dir / "PROMPT_SNAPSHOT.json").exists():
                agents.append(agent_dir.name)
        return sorted(agents)


class PromptVersionService:
    """Calcula y persiste versiones estables de prompts."""

    def __init__(self, store: PromptSnapshotStore | None = None) -> None:
        self._store = store or PromptSnapshotStore()

    def _hash_payload(self, agent_name: str, system_prompt: str, extra_context: str) -> str:
        payload = "\n".join([agent_name.strip(), system_prompt.strip(), extra_context.strip()])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def build_snapshot(self, agent_name: str, system_prompt: str, extra_context: str) -> PromptSnapshot:
        prompt_hash = self._hash_payload(agent_name, system_prompt, extra_context)
        prompt_version = f"v{prompt_hash[:12]}"
        return PromptSnapshot(
            agent_name=agent_name,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            created_at_ms=int(time.time() * 1000),
            system_prompt=system_prompt,
            extra_context=extra_context,
        )

    def save_snapshot(self, agent_name: str, system_prompt: str, extra_context: str) -> PromptSnapshot:
        snapshot = self.build_snapshot(agent_name, system_prompt, extra_context)
        self._store.save(snapshot)
        return snapshot

    def load_snapshot(self, agent_name: str) -> dict[str, Any] | None:
        return self._store.load(agent_name)

    def load_history(self, agent_name: str) -> list[dict[str, Any]]:
        return self._store.load_history(agent_name)

    def list_agents(self) -> list[str]:
        return self._store.list_agents()

    def snapshot_path(self, agent_name: str) -> Path:
        return self._store.snapshot_path(agent_name)

    def history_path(self, agent_name: str) -> Path:
        return self._store.history_path(agent_name)


prompt_snapshot_store = PromptSnapshotStore()
prompt_version_service = PromptVersionService()


__all__ = [
    "PromptSnapshot",
    "PromptSnapshotStore",
    "PromptVersionService",
    "prompt_snapshot_store",
    "prompt_version_service",
]
