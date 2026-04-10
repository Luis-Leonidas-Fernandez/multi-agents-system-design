"""Compat layer para el supervisor.

La lógica real vive en `supervisor_chain.py`, `supervisor_shortcuts.py` y
`supervisor_routing.py`.
"""
from application.use_cases.supervisor_chain import build_supervisor_chain
from application.use_cases.supervisor_routing import run_supervisor_routing

__all__ = ["build_supervisor_chain", "run_supervisor_routing"]
