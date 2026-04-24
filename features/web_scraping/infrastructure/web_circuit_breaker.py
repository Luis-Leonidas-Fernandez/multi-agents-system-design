"""Circuit breaker para providers de web search.

Evita reintentar providers que fallaron con errores no-retriables
(ej: Tavily con plan excedido). TTL: 5 minutos por proceso.
"""

import time

_PROVIDER_CIRCUIT_BREAKER: dict[str, float] = {}
_CIRCUIT_BREAKER_TTL = 300


def _is_non_retryable_provider_error(error: Exception) -> bool:
    msg = str(error).lower()
    return any(kw in msg for kw in ["forbidden", "usage limit", "403", "rate limit", "exceeded", "unauthorized", "invalid api key"])


def _circuit_trip(provider: str) -> None:
    _PROVIDER_CIRCUIT_BREAKER[provider] = time.time()


def _circuit_open(provider: str) -> bool:
    tripped_at = _PROVIDER_CIRCUIT_BREAKER.get(provider)
    if tripped_at is None:
        return False
    return (time.time() - tripped_at) < _CIRCUIT_BREAKER_TTL
