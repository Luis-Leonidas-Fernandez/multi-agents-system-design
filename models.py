"""
Modelos de respuesta estructurada compartidos entre agentes y supervisor.

Todas las herramientas (tools) que devuelven datos a consumir programáticamente
heredan de ToolResponse. Esto garantiza:
  - schema_version consistente → compatibilidad futura sin parseo a ciegas
  - timestamp en todas las respuestas → detección de datos stale
  - source explícito → comparación de fuentes

Flujo:
  tool.run() → PriceToolResponse(...).model_dump_json()
  supervisor → PriceToolResponse.model_validate(json.loads(msg.content))
"""
from __future__ import annotations

import time
from typing import Optional

from pydantic import BaseModel, field_validator


class ToolResponse(BaseModel):
    """Base común para todas las respuestas de tools estructuradas."""

    schema_version: str = "1.0"
    source:         Optional[str] = None
    timestamp:      int = 0          # unix epoch, se rellena en __init__ si falta

    def model_post_init(self, __context: object, /) -> None:  # noqa: ARG002
        if self.timestamp == 0:
            # Asignar timestamp actual si no se proporcionó
            object.__setattr__(self, "timestamp", int(time.time()))


class PriceToolResponse(ToolResponse):
    """Respuesta estructurada de get_crypto_price.

    Ejemplo:
        {
          "schema_version": "1.0",
          "asset":          "BTC",
          "asset_id":       "bitcoin",
          "price":          95234.56,
          "currency":       "USD",
          "confidence":     "high",
          "source":         "CoinGecko",
          "timestamp":      1710763200,
          "change_24h_pct": 2.1,
          "updated_at":     "2026-03-20T18:00:00Z"
        }
    """

    asset:          str
    asset_id:       str
    price:          Optional[float]        # None en casos de error (price_unavailable)
    currency:       str
    confidence:     str                    # "high" | "low" | "none"
    change_24h_pct: Optional[float] = None
    updated_at:     Optional[str]   = None
    error:          Optional[str]   = None  # "price_unavailable" etc.

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
        """True si el precio está dentro del rango de sanidad y no hay error."""
        return (
            self.error is None
            and isinstance(self.price, float)
            and min_price < self.price < max_price
        )
