"""Primitivos de persistencia reutilizables para stores append-only JSONL."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


class AppendOnlyStore:
    """Operaciones de bajo nivel para archivos JSONL append-only."""

    @staticmethod
    def append(path: Path, record: Any) -> None:
        """Serializa *record* (dataclass) como línea JSON y lo agrega al archivo."""
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(record), ensure_ascii=False, default=str)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    @staticmethod
    def load_lines(path: Path) -> list[dict[str, Any]]:
        """Lee todas las líneas JSON válidas del archivo. Ignora líneas vacías o malformadas."""
        if not path.exists():
            return []
        result: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                result.append(payload)
        return result


__all__ = ["AppendOnlyStore"]
