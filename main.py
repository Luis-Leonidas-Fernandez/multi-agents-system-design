"""
Punto de entrada principal para el sistema multi-agentes (Async)
"""
import asyncio
import json
import os
import shlex
import sys
import traceback
import threading
import importlib.util
import subprocess
from pathlib import Path
from typing import cast

from application.helpers.config_flow_helpers import validate_env
from application.services.agent_registry import get_agent_specs
from application.services.cli_dispatch import dispatch_inspection_command
from application.services.command_registry import COMMAND_REGISTRY
from application.services.runtime import AgentRuntime
from application.services.prompt_versioning import prompt_version_service
from application.services.session_inspection import (
    format_bookmark_detail,
    format_bookmark_list,
    format_command_detail,
    format_command_registry,
    format_chat_block,
    format_context_budget,
    format_background_task_state,
    format_background_task_summary,
    format_cli_chrome,
    format_cli_prompt,
    format_inspection_help,
    format_memory_search_results,
    format_tool_impact_preview,
    format_prompt_snapshot,
    format_prompt_snapshot_list,
    format_shell_frame,
    format_session_transcript,
    format_tool_approval_preview,
    format_tool_catalog,
    format_replay_timeline,
    format_session_artifact,
    format_session_selector,
)

TURN_TIMEOUT_SECONDS = float(os.getenv("TURN_TIMEOUT_SECONDS", "60"))


# ==================== DASHBOARD ====================

def _start_dashboard_watcher():
    """Arranca el dashboard en background. Muere solo cuando main.py termina."""
    try:
        from ops.build_dashboard import _watch
        t = threading.Thread(
            target=_watch,
            args=(os.getenv("AGENTDOG_AUDIT_LOG", "./logs/agentdog_audit.jsonl"), "dist/index.html"),
            daemon=True,
        )
        t.start()
        print("[dashboard] Live dashboard en http://localhost:8765")
    except Exception as e:
        print(f"[dashboard] No se pudo iniciar el watcher: {e}")


def _default_node_bin() -> str | None:
    candidate = Path.home() / ".local" / "bin" / "node"
    return str(candidate) if candidate.exists() else None


def _ui_state_payload(lifecycle, runtime: AgentRuntime, status: str) -> dict[str, object]:
    artifact = runtime.build_session_artifact(lifecycle.session_id)
    transcript: list[str] = []
    for item in artifact.transcript:
        role = str(item.get("role", "message")).lower()
        content = str(item.get("content", ""))
        transcript.append(f"{role}: {content}".rstrip())
    view = lifecycle.view()
    return {
        "session_id": lifecycle.session_id,
        "status": status,
        "prompt": view.prompt_hint,
        "transcript": transcript,
        "message_count": view.snapshot.message_count,
        "has_memory": view.snapshot.has_memory,
    }


def _emit_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _run_ui_bridge() -> None:
    validate_env()
    runtime = AgentRuntime()
    lifecycle = runtime.start_session_lifecycle(None)
    _emit_json({"type": "state", "state": _ui_state_payload(lifecycle, runtime, "listo")})
    try:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                _emit_json({"type": "error", "message": "JSON inválido"})
                continue

            action = str(message.get("action", "")).lower()
            if action == "exit":
                break
            if action != "submit":
                _emit_json({"type": "error", "message": f"Acción no soportada: {action}"})
                continue

            text = str(message.get("text", "")).strip()
            if not text:
                _emit_json({"type": "state", "state": _ui_state_payload(lifecycle, runtime, "listo")})
                continue

            if text.lower() in {"salir", "exit", "quit"}:
                asyncio.run(lifecycle.close())
                asyncio.run(runtime.shutdown())
                _emit_json({"type": "exit"})
                break

            if text.startswith("/"):
                result = dispatch_inspection_command(text, lifecycle, runtime)
                if result.handled:
                    base_state = _ui_state_payload(lifecycle, runtime, "listo")
                    base_transcript = list(cast(list[str], base_state.get("transcript", [])))
                    base_transcript.append(f"command: {text}")
                    base_transcript.extend(result.lines)
                    _emit_json({"type": "state", "state": {**base_state, "transcript": base_transcript}})
                    continue

            _emit_json({"type": "busy", "status": "procesando…"})
            session = lifecycle.resolve(text)
            if session.turn_context is None:
                _emit_json({"type": "state", "state": _ui_state_payload(lifecycle, runtime, "listo")})
                continue
            try:
                turn = asyncio.run(asyncio.wait_for(runtime.execute_turn(session.turn_context), timeout=TURN_TIMEOUT_SECONDS))
                _emit_json({"type": "state", "state": _ui_state_payload(lifecycle, runtime, "listo")})
            except asyncio.TimeoutError:
                _emit_json({"type": "error", "message": f"El turno superó {int(TURN_TIMEOUT_SECONDS)}s. Revisá proveedor/red/modelo."})
                _emit_json({"type": "state", "state": _ui_state_payload(lifecycle, runtime, "listo")})
            except Exception as exc:
                _emit_json({"type": "error", "message": str(exc)})
                _emit_json({"type": "state", "state": _ui_state_payload(lifecycle, runtime, "listo")})
    finally:
        try:
            asyncio.run(lifecycle.close())
        except Exception:
            pass
        try:
            asyncio.run(runtime.shutdown())
        except Exception:
            pass


def _handle_inspection_command(user_input: str, lifecycle, runtime: AgentRuntime) -> bool:
    command = user_input.strip()
    if not command.startswith("/"):
        return False

    parts = command.split()
    name = parts[0].lstrip("/").lower()
    resolved = COMMAND_REGISTRY.resolve_name(name)
    if resolved:
        name = resolved

    if name in {"help", "?"}:
        for line in format_inspection_help():
            print(line)
        return True

    if name == "commands":
        for line in format_command_registry():
            print(line)
        return True

    if name == "command":
        if len(parts) < 2:
            print("Usá /command <nombre>")
            return True
        spec = COMMAND_REGISTRY.get(parts[1].lstrip("/"))
        if spec is None:
            print(f"[command] no encontrado: {parts[1]}")
            return True
        for line in format_command_detail(spec):
            print(line)
        return True

    if name in {"context", "state"}:
        agent_name = parts[1] if len(parts) > 1 else None
        report = lifecycle.context_budget(agent_name)
        for line in format_context_budget(report):
            print(line)
        return True

    if name in {"inspect", "status"}:
        context_report = lifecycle.context_budget()
        for line in format_context_budget(context_report):
            print(line)
        summary = lifecycle.background_task_summary()
        artifact = lifecycle.export_artifact()
        for line in format_background_task_summary(summary):
            print(line)
        for line in format_session_artifact(artifact):
            print(line)
        tasks = lifecycle.list_background_tasks()
        if tasks:
            print("[tasks] detalle")
            for task in tasks[-5:]:
                print(f"  - {task.get('task_id', '?')} [{task.get('status', '?')}] {task.get('title', '')}")
        return True

    if name == "tasks":
        summary = lifecycle.background_task_summary()
        for line in format_background_task_summary(summary):
            print(line)
        tasks = lifecycle.list_background_tasks()
        if tasks:
            for task in tasks:
                for line in format_background_task_state(task):
                    print(line)
        else:
            print("[tasks] no hay tareas delegadas en esta sesión")
        return True

    if name == "task":
        if len(parts) < 2:
            print("Usá /task <id>")
            return True
        state = lifecycle.describe_background_task(parts[1])
        if state is None:
            print(f"[task] no encontrada: {parts[1]}")
            return True
        for line in format_background_task_state(state):
            print(line)
        return True

    if name == "cancel":
        if len(parts) < 2:
            print("Usá /cancel <id>")
            return True
        result = lifecycle.cancel_background_task(parts[1], reason="cancelado desde CLI")
        if asyncio.iscoroutine(result):
            asyncio.create_task(result)
        print(f"[task] cancel requested: {parts[1]}")
        return True

    if name == "retryable":
        tasks = lifecycle.list_retryable_background_tasks()
        if not tasks:
            print("[tasks] no hay tareas reintentables")
            return True
        for task in tasks:
            for line in format_background_task_state(task):
                print(line)
        return True

    if name == "artifact":
        artifact = lifecycle.export_artifact()
        for line in format_session_artifact(artifact):
            print(line)
        print(f"[artifact] path={lifecycle.artifact_path()}")
        print("[artifact] exportado y guardado")
        return True

    if name in {"bookmarks", "checkpoints"}:
        bookmarks = lifecycle.list_bookmarks()
        for line in format_bookmark_list(bookmarks):
            print(line)
        return True

    if name == "bookmark":
        raw = command[len(parts[0]):].strip()
        if raw:
            try:
                tokens = shlex.split(raw)
            except ValueError:
                print("[bookmark] argumento inválido")
                return True
            label = tokens[0] if tokens else None
            note = " ".join(tokens[1:]) if len(tokens) > 1 else ""
        else:
            label = None
            note = ""
        checkpoint = lifecycle.create_bookmark(label=label, note=note)
        for line in format_bookmark_detail(checkpoint):
            print(line)
        print("[bookmark] checkpoint guardado")
        return True

    if name == "checkpoint":
        if len(parts) < 2:
            print("Usá /checkpoint <id>")
            return True
        bookmark = lifecycle.describe_bookmark(parts[1])
        if bookmark is None:
            print(f"[checkpoint] no encontrado: {parts[1]}")
            return True
        for line in format_bookmark_detail(bookmark):
            print(line)
        return True

    if name == "prompts":
        agents = prompt_version_service.list_agents()
        for line in format_prompt_snapshot_list(agents):
            print(line)
        return True

    if name == "prompt":
        if len(parts) < 2:
            print("Usá /prompt <agente>")
            return True
        snapshot = prompt_version_service.load_snapshot(parts[1])
        if snapshot is None:
            print(f"[prompt] no encontrado: {parts[1]}")
            return True
        snapshot["snapshot_path"] = str(prompt_version_service.snapshot_path(parts[1]))
        snapshot["history_path"] = str(prompt_version_service.history_path(parts[1]))
        for line in format_prompt_snapshot(snapshot):
            print(line)
        history = prompt_version_service.load_history(parts[1])
        print(f"[prompt] historial={len(history)} versiones")
        return True

    if name == "replay":
        target_session_id = parts[1] if len(parts) > 1 else lifecycle.session_id
        replay = runtime.build_session_replay(target_session_id)
        for line in format_replay_timeline(replay):
            print(line)
        return True

    if name == "memory":
        if len(parts) < 2:
            sessions = runtime.list_memory_sessions()
            if sessions:
                print("[memory] sesiones con MEMORY.md:")
                for session_id in sessions:
                    print(f"  - {session_id}")
            else:
                print("[memory] no hay sesiones con memoria persistida")
            return True
        query = " ".join(parts[1:])
        results = runtime.search_memory(query)
        for line in format_memory_search_results(query, results):
            print(line)
        return True

    if name == "tools":
        for line in format_tool_catalog(runtime.tool_catalog()):
            print(line)
        return True

    if name == "tool":
        if len(parts) < 2:
            print("Usá /tool <nombre> [json_args|key=value ...]")
            return True
        tool_name = parts[1]
        raw_args = " ".join(parts[2:]).strip()
        arguments, error = _parse_tool_arguments(raw_args)
        if error:
            print(error)
            return True
        preview = runtime.preview_tool(tool_name, arguments=arguments)
        for line in format_tool_approval_preview(preview):
            print(line)
        return True

    if name == "impact":
        if len(parts) < 2:
            print("Usá /impact <nombre> [json_args|key=value ...]")
            return True
        tool_name = parts[1]
        raw_args = " ".join(parts[2:]).strip()
        arguments, error = _parse_tool_arguments(raw_args)
        if error:
            print(error)
            return True
        preview = runtime.preview_tool(tool_name, arguments=arguments)
        for line in format_tool_impact_preview(preview.get("impact_preview", preview) if isinstance(preview, dict) else getattr(preview, "impact_preview", preview)):
            print(line)
        return True

    print(f"Comando no reconocido: {command}")
    for line in format_inspection_help():
        print(line)
    return True


def _parse_tool_arguments(raw_args: str):
    if not raw_args:
        return {}, None
    if raw_args.startswith("{"):
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return {}, "[tool] JSON inválido en args"
        if not isinstance(parsed, dict):
            return {}, "[tool] los args deben ser un objeto JSON"
        return parsed, None

    arguments = {}
    try:
        tokens = shlex.split(raw_args)
    except ValueError:
        return {}, "[tool] args inválidos"

    for token in tokens:
        if "=" not in token:
            return {}, f"[tool] formato inválido: {token}. Usá key=value o JSON"
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            return {}, "[tool] key vacío en args"
        arguments[key] = _coerce_tool_argument(value)
    return arguments, None


def _coerce_tool_argument(value: str):
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _read_input(prompt: str):
    try:
        return input(prompt).strip()
    except EOFError:
        return None
    except KeyboardInterrupt:
        return "salir"


def _resolve_session_choice(session_input, sessions: list[str]) -> str:
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


def _ensure_tui_deps() -> None:
    missing = [name for name in ("rich", "textual") if importlib.util.find_spec(name) is None]
    if not missing:
        return
    print(f"[ui] instalando dependencias faltantes: {', '.join(missing)}")
    subprocess.run([sys.executable, "-m", "pip", "install", "--user", *missing], check=True)


def _render_shell(runtime: AgentRuntime, lifecycle, transcript_limit: int = 4) -> None:
    session_view = lifecycle.view()
    context_report = lifecycle.context_budget()
    transcript_artifact = runtime.build_session_artifact(lifecycle.session_id)
    _clear_screen()
    for line in format_shell_frame(session_view.snapshot, context_report, list(get_agent_specs()), transcript_artifact.transcript, session_view.prompt_hint, transcript_limit=transcript_limit):
        print(line)


# ==================== MAIN ====================

async def _run_legacy_loop(runtime: AgentRuntime, lifecycle) -> None:
    """Loop interactivo legado basado en print/input."""

    # --- Gateway (gestiona estado, persistence, memoria, LaneQueue) ---
    session_view = lifecycle.view()
    _render_shell(runtime, lifecycle)

    _start_dashboard_watcher()

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

        if user_input.lower() in ["salir", "exit", "quit"]:
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

        if _handle_inspection_command(user_input, lifecycle, runtime):
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


def main():
    """Bootstrap síncrono: UI React/Ink si está disponible, bridge para la conversación."""
    if "--ui-bridge" in sys.argv:
        _run_ui_bridge()
        return

    validate_env()
    runtime = AgentRuntime()
    use_legacy = os.getenv("CLI_MODE", "").lower() == "legacy"
    if not use_legacy:
        node_bin = _default_node_bin()
        ui_entry = Path(__file__).resolve().parent / "ui" / "index.js"
        if node_bin and ui_entry.exists():
            env = os.environ.copy()
            env["PYTHON_BRIDGE"] = sys.executable
            env["MAIN_PY"] = str(Path(__file__).resolve())
            subprocess.run([node_bin, str(ui_entry)], cwd=str(Path(__file__).resolve().parent), env=env)
            return
        try:
            _ensure_tui_deps()
            from application.ui.claude_app import ClaudeApp

            ClaudeApp(runtime, runtime.start_session_lifecycle(None)).run()
            return
        except Exception as exc:
            print(f"[ui] No se pudo iniciar la UI React/Ink ni la TUI: {exc}")
            traceback.print_exc()
            raise

    print("=" * 60)
    print("Sistema Multi-Agentes con LangGraph (Async)")
    print("=" * 60)

    sessions = runtime.list_sessions()
    if sessions:
        for line in format_session_selector(sessions):
            print(f"\n{line}")
        session_input = _read_input("  Session ID: ")
        lifecycle = runtime.start_session_lifecycle(_resolve_session_choice(session_input, sessions) if session_input is not None else None)
    else:
        lifecycle = runtime.start_session_lifecycle(None)

    asyncio.run(_run_legacy_loop(runtime, lifecycle))


if __name__ == "__main__":
    main()
