"""
Middleware de seguridad y HITL del sistema multi-agentes.

Contiene:
- input_guard: middleware pre-ejecución de dos fases (stateful)
- _BLOCKED_PATTERNS / _RISK_SIGNALS: listas de patrones de seguridad
- HITL: confirmación humana antes de nodos de alto riesgo
"""
import os
import unicodedata
import uuid
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage

from audit import _emit_guard_audit


# ==================== PATRONES DE SEGURIDAD ====================

# Fase 1 — patrones que bloquean de inmediato en el último mensaje.
_BLOCKED_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "jailbreak",
    "bypass security",
    "act as dan",
    "forget your instructions",
]

# Fase 2 — señales de riesgo que activan revisión del historial completo.
# Son fragmentos más cortos / ambiguos que solos no justifican bloqueo.
_RISK_SIGNALS = [
    "ignore",
    "forget",
    "override",
    "pretend",
    "simulate",
    "as if you were",
    "new persona",
    "disregard",
]


# ==================== HELPERS INTERNOS ====================

def _extract_msg_text(msg) -> str:
    """Extrae el texto normalizado (NFKC) en minúsculas de un mensaje LangChain."""
    if hasattr(msg, "content") and isinstance(msg.content, str):
        return unicodedata.normalize("NFKC", msg.content).lower()
    return ""


def _check_patterns(text: str, patterns: list) -> Optional[str]:
    """Retorna el primer patrón encontrado en text, o None."""
    for pattern in patterns:
        if pattern in text:
            return pattern
    return None


def _emit_block(trajectory_id: str, pattern: str, source: str, approx_chars: int) -> None:
    _emit_guard_audit({
        "event_type":           "input_block",
        "trajectory_id":        trajectory_id,
        "guard_label":          "blocked",
        "guard_status":         "blocked_by_middleware",
        "verdict_source":       source,
        "node_name":            "input",
        "risk_level":           "low",
        "raw_response":         pattern,
        "model":                "middleware",
        "latency_ms":           0,
        "policy":               "block",
        "trajectory_steps_count": 0,
        "approx_chars":         approx_chars,
    })


def _get_human_history(messages: list, max_msgs: int = 10) -> list:
    """Retorna los últimos N mensajes humanos del historial (excluye AI y ToolMessages).
    Solo inputs humanos: más señal, menos ruido.
    """
    return [m for m in messages if isinstance(m, HumanMessage)][-max_msgs:]


# ==================== INPUT GUARD ====================

def input_guard(state) -> Optional[dict]:
    """
    Middleware pre-ejecución de dos fases con seguridad stateful.

    Fase 1 — último mensaje, siempre:
      Revisa messages[-1] contra _BLOCKED_PATTERNS. Sin costo.
      Si hay match → bloqueo inmediato.

    Fase 2 — historial solo si hay motivo:
      Se activa cuando CUALQUIERA de estas condiciones es verdadera:
        a) El último mensaje contiene un _RISK_SIGNAL (señal en este turno)
        b) state["risk_flag"] == True (algún turno anterior ya fue sospechoso)
      Revisa solo los últimos 10 mensajes HUMANOS (más señal, menos ruido).
      Si hay match → bloqueo + risk_flag sigue en True.

    Efecto stateful: una vez que risk_flag=True, cada turno siguiente
    pasa por la revisión de historial automáticamente — aunque el mensaje
    actual parezca inocente.
    """
    messages = state.get("messages", [])
    if not messages:
        return None

    trajectory_id = str(uuid.uuid4())
    last_text     = _extract_msg_text(messages[-1])

    # --- Fase 1: último mensaje ---
    hit = _check_patterns(last_text, _BLOCKED_PATTERNS)
    if hit:
        _emit_block(trajectory_id, hit, "last_message", len(last_text))
        return {
            "messages":  [AIMessage(content="Solicitud bloqueada por política de seguridad.")],
            "risk_flag": True,
            "blocked":   True,
        }

    # --- Fase 2: historial — activada por risk_signal O por risk_flag previo ---
    risk_signal = _check_patterns(last_text, _RISK_SIGNALS)
    prior_risk  = state.get("risk_flag", False)

    if risk_signal or prior_risk:
        human_history = _get_human_history(messages[:-1])
        if human_history:
            combined = " ".join(_extract_msg_text(m) for m in human_history)
            hit      = _check_patterns(combined, _BLOCKED_PATTERNS)
            if hit:
                source = "history_window_risk_flag" if prior_risk else "history_window_risk_signal"
                _emit_block(trajectory_id, hit, source, len(combined))
                return {
                    "messages":  [AIMessage(content="Solicitud bloqueada por política de seguridad.")],
                    "risk_flag": True,
                    "blocked":   True,
                }

        if risk_signal and not prior_risk:
            return {"risk_flag": True}

    return None


# ==================== HITL ====================

# Controla si se pide confirmación al usuario antes de nodos de alto riesgo.
_HITL_ENABLED = os.getenv("HITL_ENABLED", "true").strip().lower() == "true"


async def _ask_confirmation(prompt: str) -> bool:
    """Solicita confirmación al usuario de forma async (no bloquea el event loop)."""
    import asyncio
    loop   = asyncio.get_running_loop()
    answer = await loop.run_in_executor(None, lambda: input(prompt).strip().lower())
    return answer in ("s", "si", "sí", "y", "yes")


__all__ = [
    "input_guard",
]
