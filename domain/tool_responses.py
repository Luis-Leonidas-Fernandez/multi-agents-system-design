"""Modelos de respuesta estructurada compartidos entre tools y supervisión.

No dependen de LangGraph ni infraestructura. Solo expresan contratos de salida
para tools que devuelven datos consumibles programáticamente.
"""
from __future__ import annotations

import time
from typing import Optional

from pydantic import BaseModel, field_validator


class ToolResponse(BaseModel):
    """Base común para todas las respuestas de tools estructuradas."""

    schema_version: str = "1.0"
    source: Optional[str] = None
    timestamp: int = 0

    def model_post_init(self, __context: object, /) -> None:  # noqa: ARG002
        if self.timestamp == 0:
            object.__setattr__(self, "timestamp", int(time.time()))


class PriceToolResponse(ToolResponse):
    """Respuesta estructurada de get_crypto_price."""

    asset: str
    asset_id: str
    price: Optional[float]
    currency: str
    confidence: str
    change_24h_pct: Optional[float] = None
    updated_at: Optional[str] = None
    error: Optional[str] = None

    @field_validator("price")
    @classmethod
    def price_must_be_positive_if_present(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError(f"price debe ser > 0, recibido: {v}. Usa None para errores.")
        return v

    @field_validator("currency")
    @classmethod
    def currency_must_be_nonempty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("currency no puede estar vacío")
        return v.upper()

    def is_valid_price(self, min_price: float = 1.0, max_price: float = 1_000_000.0) -> bool:
        return (
            self.error is None
            and isinstance(self.price, float)
            and min_price < self.price < max_price
        )


__all__ = ["ToolResponse", "PriceToolResponse"]
