"""
Memoria destilada por sesión.

Mueve distill_memory() y load_memory_context() fuera de main.py
para que puedan ser reutilizadas por gateway.py y otros módulos.
"""
from pathlib import Path

from config import get_llm


def load_memory_context(session_id: str) -> str:
    """Carga MEMORY.md de la sesión si existe. Retorna '' si no."""
    memory_file = Path("sessions") / session_id / "MEMORY.md"
    if memory_file.exists():
        return memory_file.read_text(encoding="utf-8").strip()
    return ""


async def distill_memory(state: dict, session_id: str) -> None:
    """Destila los mensajes de la sesión en bullets y los guarda en sessions/{id}/MEMORY.md."""
    messages = state.get("messages", [])
    relevant = [
        m for m in messages
        if hasattr(m, "type") and m.type in ("human", "ai") and getattr(m, "content", "").strip()
    ]
    if len(relevant) < 4:
        return

    sample = relevant[-30:]
    transcript = "\n".join(
        f"{'Usuario' if m.type == 'human' else 'Agente'}: {m.content.strip()}"
        for m in sample
    )

    prompt = (
        "Eres un asistente que destila sesiones de conversación en memoria estructurada.\n"
        "Analiza la siguiente conversación y genera un resumen en bullets que incluya:\n"
        "- Preferencias del usuario\n"
        "- Resultados importantes obtenidos\n"
        "- URLs procesadas o scrapeadas\n"
        "- Decisiones tomadas\n"
        "- Errores encontrados\n\n"
        "Restricciones:\n"
        "- Máximo 10 bullets\n"
        "- Máximo 300 palabras\n"
        "- Estilo: conciso y accionable\n"
        "- Solo incluir información relevante para sesiones futuras\n"
        "- IMPORTANTE: si un valor (precio, cotización, etc.) aparece varias veces,\n"
        "  usar SIEMPRE el valor más reciente de la conversación, no el primero.\n\n"
        f"Conversación (ordenada cronológicamente, lo más reciente al final):\n{transcript}\n\n"
        "Resumen en bullets:"
    )

    try:
        llm = get_llm()
        response = await llm.ainvoke(prompt)
        content = response.content.strip() if hasattr(response, "content") else str(response).strip()

        memory_dir = Path("sessions") / session_id
        memory_dir.mkdir(parents=True, exist_ok=True)
        (memory_dir / "MEMORY.md").write_text(content, encoding="utf-8")
    except Exception:
        pass  # Fallas del LLM no interrumpen la salida
