"""Helpers puros para el middleware de seguridad.

Se mantienen fuera de `application.policies.security_flow` para que el
caso de uso quede fino y el parsing/policy sea reutilizable y testeable.
"""
import os
import unicodedata
from typing import Optional, Sequence

from langchain_core.messages import HumanMessage


_BLOCKED_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "jailbreak",
    "bypass security",
    "act as dan",
    "forget your instructions",
]

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


def _parse_env_patterns(env_var: str) -> list[str]:
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return []
    parts = [part.strip().lower() for part in raw.replace("\n", ",").split(",")]
    return [part for part in parts if part]


def _merge_patterns(defaults: Sequence[str], extras: Sequence[str]) -> list[str]:
    merged: list[str] = []
    seen = set()
    for pattern in list(defaults) + list(extras):
        if pattern and pattern not in seen:
            merged.append(pattern)
            seen.add(pattern)
    return merged


def _extract_msg_text(msg) -> str:
    if hasattr(msg, "content") and isinstance(msg.content, str):
        return unicodedata.normalize("NFKC", msg.content).lower()
    return ""


def _check_patterns(text: str, patterns: list) -> Optional[str]:
    for pattern in patterns:
        if pattern in text:
            return pattern
    return None


def get_blocked_patterns() -> list[str]:
    return _merge_patterns(_BLOCKED_PATTERNS, _parse_env_patterns("SECURITY_BLOCKED_PATTERNS"))


def get_risk_signals() -> list[str]:
    return _merge_patterns(_RISK_SIGNALS, _parse_env_patterns("SECURITY_RISK_SIGNALS"))


def get_human_history(messages: list, max_msgs: int = 10) -> list:
    return [m for m in messages if isinstance(m, HumanMessage)][-max_msgs:]


__all__ = [
    "_BLOCKED_PATTERNS",
    "_RISK_SIGNALS",
    "_parse_env_patterns",
    "_merge_patterns",
    "_extract_msg_text",
    "_check_patterns",
    "get_blocked_patterns",
    "get_risk_signals",
    "get_human_history",
]
