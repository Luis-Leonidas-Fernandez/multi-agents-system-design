"""Bootstrap, REPL loop y lógica de arranque de UI para la CLI."""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

from core.helpers.config_flow_helpers import validate_env
from application.services.agent_registry import get_agent_specs
from application.services.cli_dispatch import dispatch_inspection_command
from application.services.runtime import AgentRuntime
from features.sessions.application.session_inspection import (
    format_chat_block,
    format_cli_prompt,
    format_session_selector,
    format_shell_frame,
)

_SEARXNG_DEFAULT_BASE_URL = "http://localhost:8888"
_SEARXNG_PROBE_PATH = "/search?q=multi-agents-healthcheck&format=json&categories=general"


# ── SearXNG bootstrap ──────────────────────────────────────────────────────────

def _searxng_auto_start_enabled() -> bool:
    value = (os.getenv("SEARXNG_AUTO_START") or "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _searxng_base_url() -> str:
    return (os.getenv("SEARXNG_BASE_URL") or _SEARXNG_DEFAULT_BASE_URL).strip()


def _searxng_is_local_url(base_url: str) -> bool:
    lowered = base_url.strip().lower()
    return lowered.startswith((
        "http://localhost",
        "https://localhost",
        "http://127.0.0.1",
        "https://127.0.0.1",
        "http://host.docker.internal",
        "https://host.docker.internal",
    ))


def _wait_for_searxng(base_url: str, timeout_seconds: int = 45) -> bool:
    import urllib.request

    probe_url = f"{base_url.rstrip('/')}{_SEARXNG_PROBE_PATH}"
    deadline = time.monotonic() + timeout_seconds
    last_error: Optional[Exception] = None
    probe_headers = {
        "User-Agent": "Mozilla/5.0 (Multi-Agents)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "X-Forwarded-For": "127.0.0.1",
        "X-Real-IP": "127.0.0.1",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Site": "same-origin",
    }

    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(probe_url, headers=probe_headers)
            with urllib.request.urlopen(request, timeout=3) as response:
                if 200 <= int(getattr(response, "status", 0)) < 300:
                    print(f"[searxng] listo en {base_url}")
                    return True
        except Exception as exc:  # pragma: no cover - depends on docker startup timing
            last_error = exc
            time.sleep(2)

    print(f"[searxng] no respondió a tiempo en {base_url}: {last_error}")
    return False


def bootstrap_searxng() -> None:
    if not _searxng_auto_start_enabled():
        return

    base_url = _searxng_base_url()
    os.environ.setdefault("SEARXNG_BASE_URL", base_url)

    if not _searxng_is_local_url(base_url):
        print(f"[searxng] SEARXNG_BASE_URL apunta a un host remoto ({base_url}); no auto-levanto un contenedor local.")
        return

    compose_file = Path(__file__).resolve().parents[2] / "docker-compose.yml"
    if not compose_file.exists():
        print("[searxng] No encontré docker-compose.yml; no puedo auto-levantar SearXNG.")
        return

    try:
        print("[searxng] Levantando SearXNG con docker compose...")
        subprocess.run(
            ["docker", "compose", "up", "-d", "searxng"],
            cwd=str(compose_file.parent),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("[searxng] Docker no está instalado o no está en PATH; seguí con un SearXNG externo.")
        return
    except subprocess.CalledProcessError as exc:
        output = (exc.stdout or "").strip()
        error_output = (exc.stderr or "").strip()
        detail = "\n".join(part for part in [output, error_output] if part)
        print(f"[searxng] No se pudo levantar SearXNG:\n{detail}")
        return

    _wait_for_searxng(base_url)


# ── Dashboard ─────────────────────────────────────────────────────────────────

def start_dashboard_watcher() -> None:
    try:
        from features.analytics.infrastructure.build_dashboard import _watch
        t = threading.Thread(
            target=_watch,
            args=(os.getenv("AGENTDOG_AUDIT_LOG", "./data/logs/agentdog_audit.jsonl"), "features/analytics/dist/index.html"),
            daemon=True,
        )
        t.start()
        print("[dashboard] Live dashboard en http://localhost:8765")
    except Exception as e:
        print(f"[dashboard] No se pudo iniciar el watcher: {e}")


# ── UI helpers ────────────────────────────────────────────────────────────────

def _default_node_bin() -> Optional[str]:
    candidate = Path.home() / ".local" / "bin" / "node"
    return str(candidate) if candidate.exists() else None


def _ensure_tui_deps() -> None:
    import importlib.util
    missing = [name for name in ("rich", "textual") if importlib.util.find_spec(name) is None]
    if not missing:
        return
    print(f"[ui] instalando dependencias faltantes: {', '.join(missing)}")
    subprocess.run([sys.executable, "-m", "pip", "install", "--user", *missing], check=True)


def start_ui(runtime: AgentRuntime) -> None:
    node_bin = _default_node_bin()
    project_root = Path(__file__).resolve().parents[2]
    ui_entry = project_root / "ui" / "index.js"
    if node_bin and ui_entry.exists():
        env = os.environ.copy()
        env["PYTHON_BRIDGE"] = sys.executable
        env["MAIN_PY"] = str(project_root / "main.py")
        subprocess.run([node_bin, str(ui_entry)], cwd=str(project_root), env=env)
        return
    try:
        _ensure_tui_deps()
        from application.ui.claude_app import ClaudeApp
        ClaudeApp(runtime, runtime.start_session_lifecycle(None)).run()
    except Exception as exc:
        print(f"[ui] No se pudo iniciar la UI React/Ink ni la TUI: {exc}")
        traceback.print_exc()
        raise


# ── REPL helpers ──────────────────────────────────────────────────────────────

def _read_input(prompt: str) -> Optional[str]:
    try:
        return input(prompt).strip()
    except EOFError:
        return None
    except KeyboardInterrupt:
        return "salir"


def _resolve_session_choice(session_input: str, sessions: list[str]) -> str:
    if not session_input:
        return ""
    if session_input.isdigit():
        index = int(session_input) - 1
        if 0 <= index < len(sessions):
            return sessions[index]
    return session_input


def _clear_screen() -> None:
    if os.getenv("NO_COLOR") is None and os.isatty(1):
        print("\033[2J\033[H", end="")


def _render_shell(runtime: AgentRuntime, lifecycle, transcript_limit: int = 4) -> None:
    session_view = lifecycle.view()
    context_report = lifecycle.context_budget()
    transcript_artifact = runtime.build_session_artifact(lifecycle.session_id)
    _clear_screen()
    for line in format_shell_frame(
        session_view.snapshot,
        context_report,
        list(get_agent_specs()),
        transcript_artifact.transcript,
        session_view.prompt_hint,
        transcript_limit=transcript_limit,
    ):
        print(line)


# ── REPL loop ─────────────────────────────────────────────────────────────────

async def run_legacy_loop(runtime: AgentRuntime, lifecycle) -> None:
    _render_shell(runtime, lifecycle)
    start_dashboard_watcher()

    loop = asyncio.get_running_loop()

    while True:
        session_view = lifecycle.view()
        user_input = await loop.run_in_executor(None, _read_input, format_cli_prompt(session_view.snapshot))

        if user_input is None:
            print("\nEntrada cerrada. Saliendo...")
            closure = await lifecycle.close()
            await runtime.shutdown()
            if closure.memory_written:
                print("Memoria destilada y persistida.")
            print(
                f"\nSesión guardada: {closure.after.message_count} mensajes previos"
                f"{', con memoria' if closure.after.has_memory else ''}. ¡Hasta luego!"
            )
            break

        if user_input.lower() in {"salir", "exit", "quit"}:
            print("\nDestilando memoria de sesión...")
            closure = await lifecycle.close()
            await runtime.shutdown()
            if closure.memory_written:
                print("Memoria destilada y persistida.")
            print(
                f"\nSesión guardada: {closure.after.message_count} mensajes previos"
                f"{', con memoria' if closure.after.has_memory else ''}. ¡Hasta luego!"
            )
            break

        if not user_input:
            continue

        result = dispatch_inspection_command(user_input, lifecycle, runtime)
        if result.handled:
            for line in result.lines:
                print(line)
            _render_shell(runtime, lifecycle)
            continue

        try:
            print()
            print("\n".join(format_chat_block("user", user_input)))
            print("\n[turno] procesando...\n")
            session = lifecycle.resolve(user_input)
            turn_context = session.turn_context
            if turn_context is None:
                continue
            turn = await runtime.execute_turn(turn_context)
            if turn.response:
                print("\n".join(format_chat_block("assistant", turn.response)))
                print()
            _render_shell(runtime, lifecycle)
        except Exception as e:
            print(f"\nError: {str(e)}\n")
            print("Por favor verifica tu configuración (API key, etc.)")


# ── Entry point helpers ───────────────────────────────────────────────────────

def bootstrap() -> None:
    bootstrap_searxng()
    validate_env()


def start_legacy_session(runtime: AgentRuntime) -> None:
    print("=" * 60)
    print("Sistema Multi-Agentes con LangGraph (Async)")
    print("=" * 60)

    sessions = runtime.list_sessions()
    if sessions:
        for line in format_session_selector(sessions):
            print(f"\n{line}")
        session_input = _read_input("  Session ID: ")
        session_id = _resolve_session_choice(session_input, sessions) if session_input is not None else None
        lifecycle = runtime.start_session_lifecycle(session_id)
    else:
        lifecycle = runtime.start_session_lifecycle(None)

    asyncio.run(run_legacy_loop(runtime, lifecycle))


__all__ = [
    "bootstrap",
    "bootstrap_searxng",
    "run_legacy_loop",
    "start_dashboard_watcher",
    "start_legacy_session",
    "start_ui",
]
