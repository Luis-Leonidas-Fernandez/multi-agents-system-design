"""
Observability y audit log del sistema multi-agentes.

Punto de escritura único para todos los eventos de auditoría (JSONL).
No importa ningún módulo interno del proyecto para evitar dependencias circulares.
"""
import json
import os
import time
from typing import Any, Dict, List, Optional, Literal

from langchain_core.messages import AIMessage


# ==================== CONSTANTES DE TRUNCADO ====================

_MAX_OBSERVATION_CHARS = 2000
_TRUNC_HEAD_CHARS      = 1200
_TRUNC_TAIL_CHARS      = 800
_RAW_RESPONSE_MAX_CHARS = 500

# ==================== CONSTANTES DE FOLLOW-UP ====================

# Frases que indican que el agente no pudo responder satisfactoriamente.
# Proxy heurístico — no ground truth, pero útil para detectar patrones a escala.
_FOLLOWUP_SIGNALS = [
    "no pude", "no encontré", "no tengo información", "no sé", "no puedo",
    "no hay datos", "no fue posible", "error", "intenta de nuevo",
    "could not", "unable to", "i don't know", "no information",
]

_FOLLOWUP_SHORT_THRESHOLD = 80  # chars — respuesta muy corta sugiere baja calidad

# ==================== TIPOS ====================

Outcome = Literal["success", "blocked", "error"]

# ==================== PRECIOS DE MODELOS ====================

# Precios por 1K tokens en USD — actualizar según tarifas vigentes.
# Ollama y modelos locales no tienen costo → no incluir aquí.
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o":                     {"input": 0.0025,  "output": 0.010},
    "gpt-4o-mini":                {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo":                {"input": 0.010,   "output": 0.030},
    "gpt-3.5-turbo":              {"input": 0.0005,  "output": 0.0015},
    "claude-3-5-sonnet-20241022": {"input": 0.003,   "output": 0.015},
    "claude-3-haiku-20240307":    {"input": 0.00025, "output": 0.00125},
}


# ==================== EMIT — punto de escritura único ====================

def _emit_guard_audit(log_data: Dict[str, Any]) -> None:
    """Escribe un evento al audit log (JSONL). Único punto de escritura de auditoría."""
    log_path = os.getenv("AGENTDOG_AUDIT_LOG", "").strip()
    payload  = json.dumps(log_data, ensure_ascii=False)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(payload + "\n")
    else:
        print(payload)


def _emit_node_outcome(request_id: str, node: str, outcome: Outcome, phase: str = "agent", **extra) -> None:
    """Emite un registro de outcome al audit log. Fuente de verdad para KPIs.

    Campos siempre presentes: request_id, node, outcome, phase, ts_ms.
    phase: "pre_guard" (HITL) | "agent" (ainvoke) | "post_guard" (AgentDoG)
    Extra: cualquier metadata adicional (agent, duration_ms, prompt_tokens, etc.)
    """
    _emit_guard_audit({
        "request_id": request_id,
        "node":        node,
        "outcome":     outcome,
        "phase":       phase,
        "ts_ms":       int(time.time() * 1000),
        **extra,
    })


# ==================== HELPERS DE MODELO ====================

def _get_model_name() -> str:
    """Retorna el nombre del modelo activo según LLM_PROVIDER (sin instanciar el LLM)."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider == "azure":
        return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL", "llama3")
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """Estima costo en USD. Retorna None si el modelo no está en MODEL_PRICING."""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return None
    return round(
        (prompt_tokens / 1000) * pricing["input"] +
        (completion_tokens / 1000) * pricing["output"],
        8,
    )


# ==================== HELPERS DE EXTRACCIÓN ====================

def _extract_tokens(result: dict) -> dict:
    """Extrae métricas de tokens + costo estimado del último AIMessage con usage_metadata.

    Siempre incluye tokens_available y model para que el audit log sea comparable:
    - tokens_available=True  → prompt/completion/total/cost son confiables
    - tokens_available=False → excluir de análisis de costos, pero model sigue presente
    """
    model = _get_model_name()
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                prompt     = usage.get("input_tokens", 0)
                completion = usage.get("output_tokens", 0)
                cost       = _estimate_cost(model, prompt, completion)
                data: Dict[str, Any] = {
                    "model":              model,
                    "tokens_available":   True,
                    "prompt_tokens":      prompt,
                    "completion_tokens":  completion,
                    "total_tokens":       prompt + completion,
                }
                if cost is not None:
                    data["estimated_cost_usd"] = cost
                return data
    return {"model": model, "tokens_available": False}


def _extract_quality(result: dict) -> dict:
    """Proxy de calidad de respuesta: longitud del output y número de tool calls.

    output_length: chars de la respuesta final → muy corto puede indicar baja calidad.
    tool_calls_count: cuántas herramientas invocó el agente en su razonamiento.
    """
    messages         = result.get("messages", [])
    tool_calls_count = sum(
        len(getattr(m, "tool_calls", None) or [])
        for m in messages
        if isinstance(m, AIMessage)
    )
    output_length = 0
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            output_length = len(msg.content)
            break
    return {"output_length": output_length, "tool_calls_count": tool_calls_count}


def _assess_followup_likely(outcome: str, output_length: int, content: str) -> bool:
    """Proxy de satisfacción funcional: estima si el usuario probablemente hará follow-up.

    Heurísticas (OR):
    - outcome no es success (blocked / error) → el agente no completó la tarea
    - respuesta muy corta (< _FOLLOWUP_SHORT_THRESHOLD chars) → contenido insuficiente
    - respuesta contiene frases de incapacidad / error → agente admite que no pudo
    """
    if outcome != "success":
        return True
    if output_length < _FOLLOWUP_SHORT_THRESHOLD:
        return True
    content_lower = content.lower()
    return any(signal in content_lower for signal in _FOLLOWUP_SIGNALS)


def _extract_followup(result: dict, outcome: str) -> dict:
    """Extrae el proxy de satisfacción funcional del resultado del agente."""
    content       = ""
    output_length = 0
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            content       = msg.content
            output_length = len(content)
            break
    return {"followup_likely": _assess_followup_likely(outcome, output_length, content)}


def _node_meta(parent_node: str = "supervisor", retry_count: int = 0) -> dict:
    """Campos estándar para correlación multi-nodo y A/B testing.

    parent_node: nodo que originó la ejecución — permite reconstruir el DAG.
    retry_count: número de reintentos; 0 en el flujo normal.
    experiment:  tag de experimento leído de la env var EXPERIMENT (vacío si no está).
    """
    meta: Dict[str, Any] = {"parent_node": parent_node, "retry_count": retry_count}
    exp = os.getenv("EXPERIMENT", "").strip()
    if exp:
        meta["experiment"] = exp
    return meta


# ==================== HELPERS DE TRUNCADO ====================

def _truncate_text(text: str, max_chars: int = _MAX_OBSERVATION_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    head = text[:_TRUNC_HEAD_CHARS]
    tail = text[-_TRUNC_TAIL_CHARS:]
    return head + "\n...[truncated]...\n" + tail


def _truncate_raw_response(text: str) -> str:
    if len(text) <= _RAW_RESPONSE_MAX_CHARS:
        return text
    return text[:_RAW_RESPONSE_MAX_CHARS] + "... [truncated]"


__all__ = [
    "Outcome",
    "_emit_guard_audit",
    "_emit_node_outcome",
    "_get_model_name",
    "MODEL_PRICING",
    "_estimate_cost",
    "_extract_tokens",
    "_extract_quality",
    "_assess_followup_likely",
    "_extract_followup",
    "_node_meta",
    "_truncate_text",
    "_truncate_raw_response",
    "_MAX_OBSERVATION_CHARS",
    "_TRUNC_HEAD_CHARS",
    "_TRUNC_TAIL_CHARS",
    "_RAW_RESPONSE_MAX_CHARS",
]
