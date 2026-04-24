"""Puerto para confirmación humana."""
from abc import ABC, abstractmethod


class ConfirmationPort(ABC):
    @abstractmethod
    async def confirm(self, prompt: str) -> bool:
        raise NotImplementedError


__all__ = ["ConfirmationPort"]
