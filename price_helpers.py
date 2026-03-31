"""
Helpers de precio para el fast path de criptomonedas.

Extrae, valida y formatea precios desde respuestas de API y mensajes LangChain.
"""
import json
import re
from typing import Optional

from langchain_core.messages import ToolMessage

from agents import _get_crypto_price_fn  # función subyacente del tool, sin overhead de LangChain
from models import PriceToolResponse


# ==================== CONSTANTES ====================

_PRICE_CONTEXT_RE = re.compile(
    r'(?:precio|price|value|valor|usd)\s*[:\s]*\$?\s*'
    r'([\d]{1,3}(?:[,\s]\d{3})*(?:\.\d{1,4})?[kKmM]?|\d+(?:\.\d{1,4})?[kKmM]?)',
    re.IGNORECASE,
)
_PRICE_SANITY_MIN = 1.0
_PRICE_SANITY_MAX = 1_000_000.0

# Mapa de keywords → coin_id (CoinGecko). Permite detectar la moneda sin LLM.
_QUERY_COIN_MAP: dict = {
    "bitcoin":   "bitcoin",       "btc":      "bitcoin",
    "ethereum":  "ethereum",      "eth":      "ethereum",
    "solana":    "solana",        "sol":      "solana",
    "cardano":   "cardano",       "ada":      "cardano",
    "dogecoin":  "dogecoin",      "doge":     "dogecoin",
    "ripple":    "ripple",        "xrp":      "ripple",
    "polkadot":  "polkadot",      "dot":      "polkadot",
    "chainlink": "chainlink",     "link":     "chainlink",
    "litecoin":  "litecoin",      "ltc":      "litecoin",
    "avalanche": "avalanche-2",   "avax":     "avalanche-2",
    "matic":     "matic-network", "polygon":  "matic-network",
    "uniswap":   "uniswap",       "uni":      "uniswap",
}

_KNOWN_SCHEMA_VERSIONS = {"1.0"}


# ==================== EXTRACCIÓN ====================

def _extract_structured_price(text: str) -> Optional[float]:
    """Extrae un valor de precio de una respuesta de API estructurada.

    Busca números en contexto semántico de precio (palabras clave: precio, price,
    value, USD). Elimina falsos positivos como "error 404", "2024", "123 usuarios".

    Soporta shorthand: "71k" → 71000, "$71.2k" → 71200, "1.5M" → 1500000.
    Aplica sanity check de rango (_PRICE_SANITY_MIN, _PRICE_SANITY_MAX).
    """
    m = _PRICE_CONTEXT_RE.search(text)
    if not m:
        return None
    raw = m.group(1).replace(",", "").replace(" ", "")
    multiplier = 1.0
    if raw and raw[-1].lower() == "k":
        multiplier, raw = 1_000.0, raw[:-1]
    elif raw and raw[-1].lower() == "m":
        multiplier, raw = 1_000_000.0, raw[:-1]
    try:
        val = float(raw) * multiplier
        return val if _PRICE_SANITY_MIN < val < _PRICE_SANITY_MAX else None
    except ValueError:
        return None


def _extract_price_from_messages(result: dict) -> Optional[dict]:
    """Extrae el payload JSON de precio directamente del ToolMessage de get_crypto_price.

    Buscar en el ToolMessage es más fiable que parsear el texto formateado por el LLM,
    ya que capturamos los datos antes de la capa de lenguaje.

    Retorna el dict con {"asset", "price", "currency", "confidence", "source", ...}
    o None si no hay ToolMessage de get_crypto_price en el resultado.
    """
    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "get_crypto_price":
            try:
                data = json.loads(msg.content)
                if not isinstance(data, dict):
                    return None

                schema_version = data.get("schema_version", "1.0")
                if schema_version not in _KNOWN_SCHEMA_VERSIONS:
                    return None

                if data.get("error"):
                    return None
                price    = data.get("price")
                currency = data.get("currency", "")
                if not isinstance(price, (int, float)):
                    return None
                price_f = float(price)
                if price_f <= 0:
                    return None
                if not (_PRICE_SANITY_MIN < price_f < _PRICE_SANITY_MAX):
                    return None
                if not isinstance(currency, str) or not currency:
                    return None

                data["price"] = price_f
                return data
            except (json.JSONDecodeError, ValueError, TypeError):
                return None
    return None


# ==================== DETECCIÓN Y FORMATO ====================

def _detect_coin_from_query(text: str) -> str:
    """Detecta el coin_id de CoinGecko en el texto de la query. Default: 'bitcoin'."""
    tl = text.lower()
    for keyword, coin_id in _QUERY_COIN_MAP.items():
        if keyword in tl:
            return coin_id
    return "bitcoin"


def _format_price_response(data: dict) -> str:
    """Formatea en Markdown el payload JSON de PriceToolResponse.

    Reemplaza la capa de lenguaje del LLM para el fast path — misma información,
    sin costo ni latencia de inferencia.
    """
    asset    = data.get("asset", "?")
    price    = data.get("price")
    currency = data.get("currency", "USD")
    source   = data.get("source", "API")
    lines    = [f"**{asset}** — ${price:,.2f} {currency}"]
    change   = data.get("change_24h_pct")
    if change is not None:
        sign = "+" if change >= 0 else ""
        lines.append(f"Cambio 24h: {sign}{change:.2f}%")
    updated = data.get("updated_at")
    if updated:
        lines.append(f"Actualizado: {updated}")
    lines.append(f"Fuente: {source}")
    return "\n".join(lines)


__all__: list = []  # todo el módulo son helpers internos; importar explícitamente
