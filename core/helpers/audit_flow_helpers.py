"""Helpers de observabilidad y audit heuristics."""
import json
import os
import time
from typing import Any, Dict, Optional, Literal

from langchain_core.messages import AIMessage

from core.domain.model_pricing import MODEL_PRICING
from core.helpers.text_truncation import truncate_head_tail, truncate_suffix


_MAX_OBSERVATION_CHARS = 2000
_TRUNC_HEAD_CHARS = 1200
_TRUNC_TAIL_CHARS = 800
_RAW_RESPONSE_MAX_CHARS = 500

_FOLLOWUP_SIGNALS = [
    "no pude", "no encontré", "no tengo información", "no sé", "no puedo",
    "no hay datos", "no fue posible", "error", "intenta de nuevo",
    "could not", "unable to", "i don't know", "no information",
]

_FOLLOWUP_SHORT_THRESHOLD = 80

Outcome = Literal["success", "blocked", "error", "retry", "low_confidence"]


def _emit_guard_audit(log_data: Dict[str, Any]) -> None:
    log_path = os.getenv("AGENTDOG_AUDIT_LOG", "").strip()
    payload = json.dumps(log_data, ensure_ascii=False)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(payload + "\n")
    else:
        print(payload)


def _emit_country_news_metrics(
    *,
    geography: Optional[str],
    resolution_path: str,
    domains_found: int,
    request_id: str = "",
) -> None:
    """Emite un evento de observabilidad para el sistema de noticias por país.

    resolution_path:
      "bootstrap"  — país curado en datos estáticos
      "dynamic"    — descubierto vía DefaultDynamicPressDiscovery
                     (sub-path exacto trazado por _web_debug en dynamic_press_discovery.py)
      "none"       — sin resolución (query fuera de scope, país desconocido, etc.)
    """
    _emit_guard_audit({
        "event_type": "country_news_resolution",
        "ts_ms": int(time.time() * 1000),
        "geography": geography,
        "resolution_path": resolution_path,
        "domains_found": domains_found,
        "request_id": request_id,
    })


def _emit_node_outcome(request_id: str, node: str, outcome: Outcome, phase: str = "agent", **extra) -> None:
    _emit_guard_audit({
        "event_type": "node_outcome",
        "request_id": request_id,
        "node": node,
        "outcome": outcome,
        "phase": phase,
        "ts_ms": int(time.time() * 1000),
        "risk_level": extra.pop("risk_level", "low"),
        "trajectory_summary": extra.pop("trajectory_summary", ""),
        **extra,
    })


def _get_model_name() -> str:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider == "azure":
        return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL", "llama3")
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return None
    return round(
        (prompt_tokens / 1000) * pricing["input"] +
        (completion_tokens / 1000) * pricing["output"],
        8,
    )


def _extract_tokens(result: dict) -> dict:
    model = _get_model_name()
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                prompt = usage.get("input_tokens", 0)
                completion = usage.get("output_tokens", 0)
                cost = _estimate_cost(model, prompt, completion)
                data: Dict[str, Any] = {
                    "model": model,
                    "tokens_available": True,
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": prompt + completion,
                }
                if cost is not None:
                    data["estimated_cost_usd"] = cost
                return data
    return {"model": model, "tokens_available": False}


def _extract_quality(result: dict) -> dict:
    messages = result.get("messages", [])
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
    if outcome != "success":
        return True
    if output_length < _FOLLOWUP_SHORT_THRESHOLD:
        return True
    content_lower = content.lower()
    return any(signal in content_lower for signal in _FOLLOWUP_SIGNALS)


def _extract_followup(result: dict, outcome: str) -> dict:
    content = ""
    output_length = 0
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            msg_content = msg.content
            if not isinstance(msg_content, str) or not msg_content:
                continue
            content = msg_content
            output_length = len(content)
            break
    return {"followup_likely": _assess_followup_likely(outcome, output_length, content)}


def _node_meta(parent_node: str = "supervisor", retry_count: int = 0) -> dict:
    meta: Dict[str, Any] = {"parent_node": parent_node, "retry_count": retry_count}
    exp = os.getenv("EXPERIMENT", "").strip()
    if exp:
        meta["experiment"] = exp
    return meta


def _truncate_text(text: str, max_chars: int = _MAX_OBSERVATION_CHARS) -> str:
    return truncate_head_tail(
        text,
        max_chars=max_chars,
        head_chars=_TRUNC_HEAD_CHARS,
        tail_chars=_TRUNC_TAIL_CHARS,
    )


def _truncate_raw_response(text: str) -> str:
    return truncate_suffix(text, max_chars=_RAW_RESPONSE_MAX_CHARS, suffix="... [truncated]")


__all__ = [
    "Outcome",
    "MODEL_PRICING",
    "_emit_guard_audit",
    "_emit_country_news_metrics",
    "_emit_node_outcome",
    "_extract_tokens",
    "_extract_quality",
    "_extract_followup",
    "_node_meta",
    "_get_model_name",
    "_truncate_text",
    "_truncate_raw_response",
]
