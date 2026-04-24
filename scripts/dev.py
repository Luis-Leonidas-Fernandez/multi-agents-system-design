#!/usr/bin/env python3
"""One-command development runner.

Starts the Vite frontend and the frontend WebSocket bridge. Vite handles HMR
for frontend changes, while this script restarts the backend bridge whenever
Python files change.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_CMD = ["npm", "--prefix", "frontend", "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"]
BACKEND_CMD = [sys.executable, "main.py", "--frontend-bridge"]
POLL_SECONDS = 0.75
IGNORE_PARTS = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist"}
WATCH_ROOTS = [ROOT / "main.py", ROOT / "application", ROOT / "core", ROOT / "features", ROOT / "agents", ROOT / "config", ROOT / "integrations", ROOT / "tests"]


def _watch_snapshot() -> dict[Path, int]:
    snapshot: dict[Path, int] = {}
    for base in WATCH_ROOTS:
        if base.is_file() and base.suffix == ".py":
            try:
                snapshot[base] = base.stat().st_mtime_ns
            except FileNotFoundError:
                continue
            continue

        if not base.exists():
            continue

        for path in base.rglob("*.py"):
            if any(part in IGNORE_PARTS for part in path.parts):
                continue
            try:
                snapshot[path] = path.stat().st_mtime_ns
            except FileNotFoundError:
                continue

    return snapshot


def _start_process(command: list[str], name: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print(f"[dev] starting {name}: {' '.join(command)}")
    return subprocess.Popen(command, cwd=ROOT, env=env)


def _stop_process(process: subprocess.Popen, name: str) -> None:
    if process.poll() is not None:
        return
    print(f"[dev] stopping {name}...")
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        print(f"[dev] forcing {name} shutdown...")
        process.kill()
        process.wait(timeout=8)


def _restart_backend(process: subprocess.Popen) -> subprocess.Popen:
    _stop_process(process, "backend")
    return _start_process(BACKEND_CMD, "backend")


def main() -> int:
    stopping = False

    def _handle_signal(_signum: int, _frame) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    frontend = _start_process(FRONTEND_CMD, "frontend")
    backend = _start_process(BACKEND_CMD, "backend")
    snapshot = _watch_snapshot()

    print("[dev] frontend en http://localhost:5173")
    print("[dev] backend bridge en ws://localhost:8787")
    print("[dev] el backend se reinicia solo cuando cambian archivos Python")

    try:
        while not stopping:
            time.sleep(POLL_SECONDS)

            if frontend.poll() is not None:
                print(f"[dev] frontend terminó con código {frontend.returncode}")
                stopping = True
                break

            if backend.poll() is not None:
                print(f"[dev] backend terminó con código {backend.returncode}; reiniciando...")
                backend = _start_process(BACKEND_CMD, "backend")
                snapshot = _watch_snapshot()
                continue

            current = _watch_snapshot()
            if current != snapshot:
                changed = sorted({str(path.relative_to(ROOT)) for path in set(current) | set(snapshot) if current.get(path) != snapshot.get(path)})
                print("[dev] cambios detectados → reiniciando backend:")
                for item in changed[:12]:
                    print(f"[dev]   - {item}")
                backend = _restart_backend(backend)
                snapshot = current
    finally:
        _stop_process(backend, "backend")
        _stop_process(frontend, "frontend")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
