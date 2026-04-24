"""Punto de entrada principal para el sistema multi-agentes (Async)."""
import asyncio
import os
import sys

from application.services.cli_lifecycle import bootstrap, start_legacy_session, start_ui
from application.services.runtime import AgentRuntime


def _run_ui_bridge() -> None:
    from application.ui_bridge.runner import run_ui_bridge
    run_ui_bridge()


def _run_frontend_bridge() -> None:
    from application.frontend_bridge.server import serve_frontend_bridge
    asyncio.run(serve_frontend_bridge())


def main() -> None:
    if "--ui-bridge" in sys.argv:
        _run_ui_bridge()
        return

    if "--frontend-bridge" in sys.argv:
        _run_frontend_bridge()
        return

    bootstrap()
    runtime = AgentRuntime()

    if os.getenv("CLI_MODE", "").lower() == "legacy":
        start_legacy_session(runtime)
        return

    start_ui(runtime)


if __name__ == "__main__":
    main()
