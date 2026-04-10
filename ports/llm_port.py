"""Puerto para fábricas de LLM."""
from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMFactory(Protocol):
    def __call__(self, temperature: Optional[float] = None) -> Any: ...


__all__ = ["LLMFactory"]
