"""Feature-level application API for price tools."""

from datetime import datetime, timezone
from typing import Annotated, Any, Dict, Optional

from langchain_core.tools import tool
from pydantic import Field

from core.domain.tool_responses import PriceToolResponse


_COIN_ALIASES: Dict[str, str] = {
    "btc": "bitcoin",
    "eth": "ethereum",
    "ether": "ethereum",
    "bnb": "binancecoin",
    "sol": "solana",
    "ada": "cardano",
    "xrp": "ripple",
    "dot": "polkadot",
    "doge": "dogecoin",
    "avax": "avalanche-2",
    "matic": "matic-network",
    "link": "chainlink",
    "usdt": "tether",
    "usdc": "usd-coin",
    "ltc": "litecoin",
    "atom": "cosmos",
    "near": "near",
    "algo": "algorand",
}

_COIN_TICKER: Dict[str, str] = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "binancecoin": "BNB",
    "solana": "SOL",
    "cardano": "ADA",
    "ripple": "XRP",
    "polkadot": "DOT",
    "dogecoin": "DOGE",
    "avalanche-2": "AVAX",
    "matic-network": "MATIC",
    "chainlink": "LINK",
    "litecoin": "LTC",
    "cosmos": "ATOM",
    "near": "NEAR",
    "algorand": "ALGO",
}

_API_TIMEOUT = 7


def _price_coingecko(coin_id: str, vs_currency: str) -> Optional[Dict[str, Any]]:
    import requests

    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": coin_id, "vs_currencies": vs_currency,
                "include_24hr_change": "true", "include_last_updated_at": "true",
            },
            timeout=_API_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        data = r.json()
        if not data or coin_id not in data:
            return None
        entry = data[coin_id]
        price = entry.get(vs_currency)
        if price is None:
            return None
        return {
            "price": float(price),
            "change_24h": entry.get(f"{vs_currency}_24h_change"),
            "updated_at": entry.get("last_updated_at"),
            "source": "CoinGecko",
        }
    except Exception:
        return None


def _price_binance(ticker: str, vs_currency: str) -> Optional[Dict[str, Any]]:
    import requests

    if vs_currency.lower() not in ("usd", "usdt"):
        return None
    symbol = f"{ticker}USDT"
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/24hr",
            params={"symbol": symbol},
            timeout=_API_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        data = r.json()
        price = float(data["lastPrice"])
        if price == 0:
            return None
        return {
            "price": price,
            "change_24h": float(data.get("priceChangePercent", 0)),
            "updated_at": None,
            "source": "Binance",
        }
    except Exception:
        return None


def _price_coinbase(ticker: str, vs_currency: str) -> Optional[Dict[str, Any]]:
    import requests

    pair = f"{ticker}-{vs_currency.upper()}"
    try:
        r = requests.get(
            f"https://api.coinbase.com/v2/prices/{pair}/spot",
            timeout=_API_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        data = r.json().get("data", {})
        price = float(data.get("amount", 0))
        if price == 0:
            return None
        return {
            "price": price,
            "change_24h": None,
            "updated_at": None,
            "source": "Coinbase",
        }
    except Exception:
        return None


def _build_price_payload(coin_id: str, coin_input: str, vs_currency: str, result: Dict[str, Any]) -> str:
    ticker = _COIN_TICKER.get(coin_id, coin_input.upper())
    change = result.get("change_24h")
    updated_at = None
    if result.get("updated_at"):
        updated_at = datetime.fromtimestamp(result["updated_at"], tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    response = PriceToolResponse(
        asset=ticker,
        asset_id=coin_id,
        price=float(result["price"]),
        currency=vs_currency,
        confidence="high",
        source=result["source"],
        change_24h_pct=round(change, 4) if change is not None else None,
        updated_at=updated_at,
    )
    return response.model_dump_json(exclude_none=False)


@tool
def get_crypto_price(
    coin: Annotated[str, Field(description="Nombre o símbolo de la criptomoneda, ej: 'bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol'")],
    vs_currency: Annotated[str, Field(description="Moneda de cotización, ej: 'usd', 'eur'")] = "usd",
) -> str:
    """Obtiene el precio actual de una criptomoneda usando APIs públicas."""
    coin_id = _COIN_ALIASES.get(coin.lower().strip(), coin.lower().strip())
    ticker = _COIN_TICKER.get(coin_id, coin_id.upper())

    for fn in (_price_coingecko, _price_binance, _price_coinbase):
        try:
            result = fn(coin_id if fn is _price_coingecko else ticker, vs_currency)
            if result:
                return _build_price_payload(coin_id, coin, vs_currency, result)
        except Exception:
            continue

    return PriceToolResponse(
        asset=ticker,
        asset_id=coin_id,
        price=None,
        currency=vs_currency,
        confidence="none",
        error="price_unavailable",
    ).model_dump_json(exclude_none=False)


@tool
def extract_price_from_text(
    text: Annotated[str, Field(description="Texto crudo del que extraer un precio numérico")],
) -> str:
    """Extrae un número tipo precio desde un texto y devuelve un valor normalizado."""
    import re

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


_get_crypto_price_fn = getattr(get_crypto_price, "func")


CRYPTO_KEYWORDS: frozenset = frozenset({
    "bitcoin", "btc", "ethereum", "eth", "ether", "solana", "sol",
    "cardano", "ada", "dogecoin", "doge", "ripple", "xrp",
    "polkadot", "dot", "chainlink", "link", "litecoin", "ltc",
    "avalanche", "avax", "matic", "polygon", "binancecoin", "bnb",
    "uniswap", "uni", "cosmos", "atom", "near", "algorand", "algo",
    "precio", "price", "cotiza", "cotización", "cotizacion",
})


__all__ = [
    "_COIN_ALIASES",
    "_COIN_TICKER",
    "_get_crypto_price_fn",
    "CRYPTO_KEYWORDS",
    "get_crypto_price",
    "extract_price_from_text",
]
