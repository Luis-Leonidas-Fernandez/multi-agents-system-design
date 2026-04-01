"""
Punto de entrada principal para el sistema multi-agentes (Async)
"""
import asyncio
import os
import threading
import uuid

from config import validate_env
from gateway import AgentGateway
import persistence


# ==================== DASHBOARD ====================

def _start_dashboard_watcher():
    """Arranca el dashboard en background. Muere solo cuando main.py termina."""
    try:
        from build_dashboard import _watch
        t = threading.Thread(
            target=_watch,
            args=(os.getenv("AGENTDOG_AUDIT_LOG", "./logs/agentdog_audit.jsonl"), "dist/index.html"),
            daemon=True,
        )
        t.start()
        print("[dashboard] Live dashboard en http://localhost:8765")
    except Exception as e:
        print(f"[dashboard] No se pudo iniciar el watcher: {e}")


# ==================== MAIN ====================

async def main():
    """Función principal asíncrona"""
    validate_env()
    print("=" * 60)
    print("Sistema Multi-Agentes con LangGraph (Async)")
    print("=" * 60)

    # --- Selección de sesión ---
    sessions = persistence.list_sessions()
    if sessions:
        print(f"\nSesiones guardadas: {', '.join(sessions)}")
        print("Escribe un ID para continuar una sesión, o Enter para nueva sesión:")
        loop = asyncio.get_running_loop()
        session_input = await loop.run_in_executor(None, lambda: input("  Session ID: ").strip())
        session_id = session_input if session_input in sessions else str(uuid.uuid4())[:8]
    else:
        session_id = str(uuid.uuid4())[:8]

    # --- Gateway (gestiona estado, persistence, memoria, LaneQueue) ---
    gateway = AgentGateway()
    n_messages, has_memory = gateway.load_session_info(session_id)

    if has_memory:
        print(f"\n[sesión: {session_id}] {n_messages} mensajes previos + memoria cargada.")
    else:
        print(f"\n[sesión: {session_id}] {n_messages} mensajes previos cargados.")

    print("\nAgentes disponibles:")
    print("  - math_agent: Problemas matemáticos")
    print("  - analysis_agent: Análisis de datos")
    print("  - code_agent: Programación y código")
    print("  - web_scraping_agent: Extracción de información de páginas web")
    print("\nEscribe 'salir' para terminar\n")

    _start_dashboard_watcher()

    loop = asyncio.get_running_loop()

    while True:
        user_input = await loop.run_in_executor(None, lambda: input("\nTu pregunta: ").strip())

        if user_input.lower() in ["salir", "exit", "quit"]:
            print("\nDestilando memoria de sesión...")
            await gateway.close_session(session_id)
            await gateway.shutdown()
            print(f"\nSesión guardada. ¡Hasta luego!")
            break

        if not user_input:
            continue

        try:
            print("\nProcesando...\n")
            response = await gateway.send(session_id, user_input)
            if response:
                print(f"\nRespuesta:\n{response}\n")
        except Exception as e:
            print(f"\nError: {str(e)}\n")
            print("Por favor verifica tu configuración (API key, etc.)")


if __name__ == "__main__":
    asyncio.run(main())
