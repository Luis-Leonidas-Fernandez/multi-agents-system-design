"""Herramienta de extracción de precios para el sistema multi-agentes."""

import re
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field


@tool
def extract_price_from_text(
    text: Annotated[str, Field(description="Texto crudo del que extraer un precio numérico")],
) -> str:
    """Extrae un número tipo precio desde un texto y devuelve un valor normalizado."""
    if not text:
        return "No hay texto para extraer precio."

    m = re.search(r'([0-9]{1,3}(?:[,\.\s][0-9]{3})*(?:[,\.\s][0-9]{2,8})|[0-9]+(?:[,\.\s][0-9]{2,8})?)', text)
    if not m:
        return "No encontré un número de precio en el texto."

    raw = m.group(1).strip()
    if "." in raw and "," in raw:
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    else:
        raw = raw.replace(" ", "")
        if raw.count(",") == 1 and raw.count(".") == 0:
            parts = raw.split(",")
            if len(parts[-1]) in (2, 3, 4, 5, 6, 7, 8):
                raw = raw.replace(",", ".")

    return f"Precio detectado: {raw}"


__all__ = ["extract_price_from_text"]
