"""Caso de uso para el middleware de seguridad de entrada."""
import uuid
from typing import Optional

from langchain_core.messages import AIMessage

from application.helpers.audit_flow_helpers import _emit_guard_audit
from application.helpers.security_flow_helpers import (
    _extract_msg_text,
    _check_patterns,
    get_blocked_patterns,
    get_human_history,
    get_risk_signals,
)


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


def input_guard(state) -> Optional[dict]:
    messages = state.get("messages", [])
    if not messages:
        return None

    trajectory_id = str(uuid.uuid4())
    last_text = _extract_msg_text(messages[-1])

    blocked_patterns = get_blocked_patterns()
    risk_signals = get_risk_signals()

    hit = _check_patterns(last_text, blocked_patterns)
    if hit:
        _emit_block(trajectory_id, hit, "last_message", len(last_text))
        return {
            "messages": [AIMessage(content="Solicitud bloqueada por política de seguridad.")],
            "risk_flag": True,
            "blocked": True,
        }

    risk_signal = _check_patterns(last_text, risk_signals)
    prior_risk = state.get("risk_flag", False)

    if risk_signal or prior_risk:
        human_history = get_human_history(messages[:-1])
        if human_history:
            combined = " ".join(_extract_msg_text(m) for m in human_history)
            hit = _check_patterns(combined, blocked_patterns)
            if hit:
                source = "history_window_risk_flag" if prior_risk else "history_window_risk_signal"
                _emit_block(trajectory_id, hit, source, len(combined))
                return {
                    "messages": [AIMessage(content="Solicitud bloqueada por política de seguridad.")],
                    "risk_flag": True,
                    "blocked": True,
                }

        if risk_signal and not prior_risk:
            return {"risk_flag": True}

    return None


__all__ = ["input_guard"]
