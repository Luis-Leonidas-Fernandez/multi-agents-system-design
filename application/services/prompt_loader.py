"""Carga y cache de prompts por agente."""
from __future__ import annotations

from pathlib import Path
import os
import logging


_PROMPT_CACHE: dict[str, str] = {}
_MISSING_PROMPTS: set[str] = set()
_log = logging.getLogger(__name__)


def load_agent_prompt(agent_name: str, extra_context: str = "") -> str:
    """
    Carga el system prompt desde agents/{agent_name}.md.

    Fallback:
    - Si el archivo no existe → retornar extra_context o ""
    - Nunca lanzar excepción
    """
    hot_reload = os.getenv("AGENT_HOT_RELOAD", "false").lower() == "true"
    warn_missing = os.getenv("AGENT_PROMPT_WARN_MISSING", "false").lower() == "true"

    prompt_path = Path(__file__).parent / "agents" / f"{agent_name}.md"
    cache_key = agent_name
    if prompt_path.exists():
        try:
            cache_key = f"{agent_name}|{prompt_path.stat().st_mtime_ns}"
        except Exception:
            cache_key = agent_name

    if not hot_reload and cache_key in _PROMPT_CACHE:
        prompt = _PROMPT_CACHE[cache_key]
    else:
        if prompt_path.exists():
            try:
                prompt = prompt_path.read_text(encoding="utf-8").strip()
                if not hot_reload:
                    _PROMPT_CACHE[cache_key] = prompt
            except Exception:
                _log.warning("Error al leer prompt: %s.md", agent_name)
                prompt = ""
        else:
            if warn_missing and agent_name not in _MISSING_PROMPTS:
                _MISSING_PROMPTS.add(agent_name)
                _log.warning("Prompt no encontrado: %s.md", agent_name)
            prompt = extra_context or ""
            return prompt

    if extra_context:
        return f"{prompt}\n\n---\n{extra_context}"
    return prompt


__all__ = ["load_agent_prompt"]
