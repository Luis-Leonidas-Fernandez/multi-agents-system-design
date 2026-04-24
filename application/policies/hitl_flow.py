"""Helpers de HITL para nodos de alto riesgo."""
import os

from core.ports.confirmation_port import ConfirmationPort


ConfirmationHandler = ConfirmationPort


class InputConfirmationHandler(ConfirmationPort):
    """Implementación por defecto basada en input()."""

    async def confirm(self, prompt: str) -> bool:
        import asyncio

        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(None, lambda: input(prompt).strip().lower())
        return answer in ("s", "si", "sí", "y", "yes")


DEFAULT_CONFIRMATION_HANDLER = InputConfirmationHandler()
_HITL_ENABLED = os.getenv("HITL_ENABLED", "true").strip().lower() == "true"


async def ask_confirmation(prompt: str) -> bool:
    return await DEFAULT_CONFIRMATION_HANDLER.confirm(prompt)


def get_confirmation_handler() -> ConfirmationPort:
    return DEFAULT_CONFIRMATION_HANDLER


HITL_ENABLED = _HITL_ENABLED

__all__ = ["HITL_ENABLED", "ask_confirmation", "get_confirmation_handler", "ConfirmationHandler", "InputConfirmationHandler", "DEFAULT_CONFIRMATION_HANDLER"]
