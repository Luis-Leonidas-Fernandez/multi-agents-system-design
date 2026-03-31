"""
Definición de agentes especializados para el sistema multi-agentes

Este módulo usa langgraph.prebuilt.create_react_agent para crear agentes
que pueden usar herramientas de forma autónoma siguiendo el patrón ReAct.
"""
# pyright: reportDeprecated=false
# Nota: El linter sugiere usar langchain.agents.create_agent, pero esa función
# no existe con la misma firma. langgraph.prebuilt.create_react_agent es correcto
# para agentes basados en LangGraph que devuelven CompiledStateGraph.
#
# TODO(tech-debt): Evaluar migración a langchain.agents.create_agent cuando:
#   - exista soporte equivalente para CompiledStateGraph con la misma firma
#   - haya tests de comportamiento end-to-end que validen parity de routing
#   Tracking: buscar "create_react_agent" en este archivo para localizar todos los call sites.
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from config import get_llm
from typing import Optional, Dict, Tuple, Any, List, Annotated
from pydantic import Field
from models import PriceToolResponse
from pathlib import Path
from datetime import datetime, timezone
import os
import time
import hashlib
import json
import asyncio


# ==================== PROMPT LOADER ====================

_PROMPT_CACHE: dict[str, str] = {}


def load_agent_prompt(agent_name: str, extra_context: str = "") -> str:
    """
    Carga el system prompt desde agents/{agent_name}.md.

    Fallback:
    - Si el archivo no existe → retornar extra_context o ""
    - Nunca lanzar excepción
    """
    hot_reload = os.getenv("AGENT_HOT_RELOAD", "false").lower() == "true"

    if not hot_reload and agent_name in _PROMPT_CACHE:
        prompt = _PROMPT_CACHE[agent_name]
    else:
        prompt_path = Path(__file__).parent / "agents" / f"{agent_name}.md"
        if prompt_path.exists():
            try:
                prompt = prompt_path.read_text(encoding="utf-8").strip()
                if not hot_reload:
                    _PROMPT_CACHE[agent_name] = prompt
            except Exception:
                print(f"[WARN] Error al leer prompt: {agent_name}.md")
                prompt = ""
        else:
            print(f"[WARN] Prompt no encontrado: {agent_name}.md")
            prompt = extra_context or ""
            return prompt

    if extra_context:
        return f"{prompt}\n\n---\n{extra_context}"
    return prompt


# ==================== HERRAMIENTAS ====================
# ==================== AGENTES ESPECIALIZADO ============= scrape_website_with_json_capture):

@tool
def extract_price_from_text(
    text: Annotated[str, Field(description="Texto crudo del que extraer un precio numérico")],
) -> str:
    """Extrae un número tipo precio desde un texto y devuelve un valor normalizado."""
    import re

    if not text:
        return "No hay texto para extraer precio."

    # Captura números con separadores de miles/decimales
    m = re.search(r'([0-9]{1,3}(?:[,\.\s][0-9]{3})*(?:[,\.\s][0-9]{2,8})|[0-9]+(?:[,\.\s][0-9]{2,8})?)', text)
    if not m:
        return "No encontré un número de precio en el texto."

    raw = m.group(1).strip()

    # Normalización heurística:
    # Si tiene "." y "," => el último separador suele ser decimal
    if "." in raw and "," in raw:
        if raw.rfind(",") > raw.rfind("."):
            # 41.234,56 -> 41234.56
            raw = raw.replace(".", "").replace(",", ".")
        else:
            # 41,234.56 -> 41234.56
            raw = raw.replace(",", "")
    else:
        raw = raw.replace(" ", "")
        # Si solo hay coma y parece decimal: 1234,56 -> 1234.56
        if raw.count(",") == 1 and raw.count(".") == 0:
            parts = raw.split(",")
            if len(parts[-1]) in (2, 3, 4, 5, 6, 7, 8):
                raw = raw.replace(",", ".")

    return f"Precio detectado: {raw}"


@tool
def calculate(
    expression: Annotated[str, Field(description="Expresión matemática a evaluar, ej: '2 + 2', 'sqrt(16)', 'sin(pi/2)'")],
) -> str:
    """Evalúa una expresión matemática y retorna el resultado."""
    try:
        import math
        from simpleeval import simple_eval, FeatureNotAvailable, InvalidExpression
        result = simple_eval(
            expression,
            names={"pi": math.pi, "e": math.e},
            functions={
                "sqrt": math.sqrt,
                "sin":  math.sin,
                "cos":  math.cos,
                "tan":  math.tan,
                "log":  math.log,
                "exp":  math.exp,
                "abs":  abs,
                "round": round,
            },
        )
        return f"Resultado: {result}"
    except (FeatureNotAvailable, InvalidExpression) as e:
        return f"Expresión no permitida: {str(e)}"
    except Exception as e:
        return f"Error al calcular: {str(e)}"


@tool
def analyze_data(
    data: Annotated[str, Field(
        description=(
            "Datos a analizar. Puede ser: "
            "(1) JSON array de números, ej: [10, 20, 30], "
            "(2) JSON array de objetos, ej: [{\"ventas\": 100, \"mes\": \"ene\"}, ...], "
            "(3) CSV con encabezado, ej: 'mes,ventas\\nene,100\\nfeb,200', "
            "(4) descripción textual si no hay datos estructurados."
        )
    )],
) -> str:
    """
    Analiza datos estructurados (JSON, CSV) y retorna estadísticas reales.
    Si los datos no son estructurados, retorna un framework de análisis.
    """
    import json
    import statistics
    import csv
    import io

    # --- Intento 1: JSON array de números ---
    try:
        parsed = json.loads(data)
        if isinstance(parsed, list) and parsed and all(isinstance(x, (int, float)) for x in parsed):
            n = len(parsed)
            mean = statistics.mean(parsed)
            return (
                f"Análisis estadístico — {n} valores numéricos:\n"
                f"- Mínimo:              {min(parsed)}\n"
                f"- Máximo:              {max(parsed)}\n"
                f"- Media:               {mean:.4f}\n"
                f"- Mediana:             {statistics.median(parsed):.4f}\n"
                f"- Desviación estándar: {statistics.stdev(parsed):.4f}\n"
                f"- Suma:                {sum(parsed)}\n"
                f"- Rango:               {max(parsed) - min(parsed)}"
            )
    except (json.JSONDecodeError, TypeError, statistics.StatisticsError):
        pass

    # --- Intento 2: JSON array de objetos (tabla) ---
    try:
        parsed = json.loads(data)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            columns = list(parsed[0].keys())
            lines = [f"Dataset tabular: {len(parsed)} filas × {len(columns)} columnas", ""]
            for col in columns:
                values = [row[col] for row in parsed if isinstance(row.get(col), (int, float))]
                if len(values) >= 2:
                    lines.append(
                        f"  {col}: min={min(values):.2f}, max={max(values):.2f}, "
                        f"media={statistics.mean(values):.2f}, std={statistics.stdev(values):.2f}"
                    )
                elif values:
                    lines.append(f"  {col}: valor único = {values[0]}")
                else:
                    unique = list({str(row.get(col)) for row in parsed if row.get(col) is not None})
                    lines.append(f"  {col} (categórico): {len(unique)} valores únicos → {unique[:5]}")
            return "\n".join(lines)
    except (json.JSONDecodeError, TypeError, KeyError, statistics.StatisticsError):
        pass

    # --- Intento 3: CSV ---
    try:
        reader = csv.DictReader(io.StringIO(data.strip()))
        rows = list(reader)
        if rows:
            columns = list(rows[0].keys())
            lines = [f"Dataset CSV: {len(rows)} filas × {len(columns)} columnas", ""]
            for col in columns:
                values = []
                for row in rows:
                    try:
                        values.append(float(row[col]))
                    except (ValueError, KeyError):
                        pass
                if len(values) >= 2:
                    lines.append(
                        f"  {col}: min={min(values):.2f}, max={max(values):.2f}, "
                        f"media={statistics.mean(values):.2f}"
                    )
                else:
                    unique = list({row.get(col, "") for row in rows})
                    lines.append(f"  {col} (texto): {len(unique)} valores únicos")
            return "\n".join(lines)
    except Exception:
        pass

    # --- Fallback: descripción textual ---
    return (
        f"Datos recibidos (formato no estructurado):\n{data}\n\n"
        f"Para un análisis completo, considera:\n"
        f"1. Distribución y estadísticas descriptivas\n"
        f"2. Valores atípicos y datos faltantes\n"
        f"3. Correlaciones entre variables\n"
        f"4. Tendencias temporales (si aplica)\n"
        f"5. Segmentación por categorías clave"
    )


@tool
def write_code(
    task: Annotated[str, Field(description="Descripción clara de la funcionalidad a implementar")],
    language: Annotated[str, Field(description="Lenguaje de programación, ej: 'python', 'javascript', 'typescript'")] = "python",
) -> str:
    """
    Genera un esqueleto de código válido para la tarea indicada.
    Para Python: valida sintaxis con compile(). Retorna código listo para completar.
    """
    import re
    import textwrap

    # Generar nombre de función a partir de la tarea
    slug   = re.sub(r"[^a-z0-9]+", "_", task.lower())[:40].strip("_") or "solution"

    if language.lower() == "python":
        code = textwrap.dedent(f"""
            def {slug}(*args, **kwargs):
                \"\"\"
                {task}

                Args:
                    Definir parámetros según los requerimientos.

                Returns:
                    Resultado de la implementación.

                Raises:
                    NotImplementedError: hasta que se complete la implementación.
                \"\"\"
                # TODO: implementar lógica de '{task}'
                raise NotImplementedError("Implementación pendiente")
        """).strip()

        try:
            compile(code, "<generated>", "exec")
            validation = "✅ Sintaxis Python validada."
        except SyntaxError as e:
            validation = f"⚠️ Error de sintaxis detectado: {e}"

        return f"```python\n{code}\n```\n\n{validation}"

    # Otros lenguajes: skeleton genérico
    templates = {
        "javascript": f"function {slug}() {{\n  // TODO: {task}\n  throw new Error('Not implemented');\n}}",
        "typescript": f"function {slug}(): void {{\n  // TODO: {task}\n  throw new Error('Not implemented');\n}}",
        "java":       f"public static void {slug}() {{\n    // TODO: {task}\n    throw new UnsupportedOperationException();\n}}",
        "go":         f"func {slug}() {{\n\t// TODO: {task}\n\tpanic(\"not implemented\")\n}}",
    }
    template = templates.get(language.lower(), f"// {task}\n// Implementar en {language}")
    return f"```{language}\n{template}\n```"


@tool
def search_web(
    query: Annotated[str, Field(description="Consulta de búsqueda en lenguaje natural, ej: 'bitcoin price usd today', 'precio actual ethereum'")],
) -> str:
    """Busca información en internet usando DuckDuckGo. No requiere URL. Útil para precios, noticias y datos actuales."""
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun().run(query)
    except Exception as e:
        return f"Error en búsqueda: {str(e)}"


# ==================== CRYPTO PRICE APIs ====================
# Tres fuentes públicas con fallback automático: CoinGecko → Binance → Coinbase.
# Orden elegido por: completitud de datos > disponibilidad > velocidad.

# Alias de entrada → ID canónico de CoinGecko
_COIN_ALIASES: Dict[str, str] = {
    "btc":    "bitcoin",
    "eth":    "ethereum",
    "ether":  "ethereum",
    "bnb":    "binancecoin",
    "sol":    "solana",
    "ada":    "cardano",
    "xrp":    "ripple",
    "dot":    "polkadot",
    "doge":   "dogecoin",
    "avax":   "avalanche-2",
    "matic":  "matic-network",
    "link":   "chainlink",
    "usdt":   "tether",
    "usdc":   "usd-coin",
    "ltc":    "litecoin",
    "atom":   "cosmos",
    "near":   "near",
    "algo":   "algorand",
}

# CoinGecko ID → símbolo de ticker (para Binance y Coinbase)
_COIN_TICKER: Dict[str, str] = {
    "bitcoin":      "BTC",
    "ethereum":     "ETH",
    "binancecoin":  "BNB",
    "solana":       "SOL",
    "cardano":      "ADA",
    "ripple":       "XRP",
    "polkadot":     "DOT",
    "dogecoin":     "DOGE",
    "avalanche-2":  "AVAX",
    "matic-network": "MATIC",
    "chainlink":    "LINK",
    "litecoin":     "LTC",
    "cosmos":       "ATOM",
    "near":         "NEAR",
    "algorand":     "ALGO",
}

_API_TIMEOUT = 7  # segundos por intento


def _price_coingecko(coin_id: str, vs_currency: str) -> Optional[Dict[str, Any]]:
    """CoinGecko: precio + cambio 24h + timestamp. Soporta cualquier moneda."""
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
            "price":      float(price),
            "change_24h": entry.get(f"{vs_currency}_24h_change"),
            "updated_at": entry.get("last_updated_at"),
            "source":     "CoinGecko",
        }
    except Exception:
        return None


def _price_binance(ticker: str, vs_currency: str) -> Optional[Dict[str, Any]]:
    """Binance 24hr ticker: precio + cambio 24h. Solo funciona para USDT (≈USD)."""
    import requests
    if vs_currency.lower() not in ("usd", "usdt"):
        return None  # Binance no tiene pares vs EUR/ARS directamente
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
            "price":      price,
            "change_24h": float(data.get("priceChangePercent", 0)),
            "updated_at": None,
            "source":     "Binance",
        }
    except Exception:
        return None


def _price_coinbase(ticker: str, vs_currency: str) -> Optional[Dict[str, Any]]:
    """Coinbase spot price: solo precio, sin cambio 24h."""
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
            "price":      price,
            "change_24h": None,
            "updated_at": None,
            "source":     "Coinbase",
        }
    except Exception:
        return None


def _build_price_payload(coin_id: str, coin_input: str, vs_currency: str, result: Dict[str, Any]) -> str:
    """Devuelve el precio como JSON validado por PriceToolResponse (capa de datos).

    El LLM que recibe este tool output formatea la respuesta en lenguaje natural.
    Separar datos y lenguaje permite validar precios en el supervisor sin parsear texto.
    """
    ticker     = _COIN_TICKER.get(coin_id, coin_input.upper())
    change     = result.get("change_24h")
    updated_at = None
    if result.get("updated_at"):
        updated_at = datetime.fromtimestamp(
            result["updated_at"], tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

    response = PriceToolResponse(
        asset          = ticker,
        asset_id       = coin_id,
        price          = float(result["price"]),
        currency       = vs_currency,
        confidence     = "high",
        source         = result["source"],
        change_24h_pct = round(change, 4) if change is not None else None,
        updated_at     = updated_at,
    )
    return response.model_dump_json(exclude_none=False)


@tool
def get_crypto_price(
    coin: Annotated[str, Field(description="Nombre o símbolo de la criptomoneda, ej: 'bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol'")],
    vs_currency: Annotated[str, Field(description="Moneda de cotización, ej: 'usd', 'eur'")] = "usd",
) -> str:
    """Obtiene el precio actual de una criptomoneda usando APIs públicas.

    Intenta en orden: CoinGecko → Binance → Coinbase.
    Más rápido y confiable que scraping. No requiere API key."""
    coin_id = _COIN_ALIASES.get(coin.lower().strip(), coin.lower().strip())
    ticker  = _COIN_TICKER.get(coin_id, coin_id.upper())

    for fn, _ in [
        (_price_coingecko, "CoinGecko"),
        (_price_binance,   "Binance"),
        (_price_coinbase,  "Coinbase"),
    ]:
        try:
            result = fn(coin_id if fn is _price_coingecko else ticker, vs_currency)
            if result:
                return _build_price_payload(coin_id, coin, vs_currency, result)
        except Exception:
            continue

    return PriceToolResponse(
        asset      = ticker,
        asset_id   = coin_id,
        price      = None,
        currency   = vs_currency,
        confidence = "none",
        error      = "price_unavailable",
    ).model_dump_json(exclude_none=False)


_CACHE_TTL_SECONDS = 60
_SCRAPE_CACHE_MAX  = 256
_SCRAPE_CACHE: Dict[str, Tuple[float, str]] = {}
_PLAYWRIGHT = None
_BROWSER = None

# ==================== DATA TRADING HELPERS ====================
DATA_TRADING_DIR = Path(__file__).parent / "data_trading"


def _ensure_data_trading_dir() -> Path:
    """Crea la carpeta data_trading si no existe."""
    DATA_TRADING_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_TRADING_DIR


def _safe_slug(s: str, max_len: int = 80) -> str:
    """Genera un slug seguro para nombres de archivo."""
    safe = "".join(c if c.isalnum() else "_" for c in s)[:max_len]
    return safe.strip("_") or "page"


def _save_json_bundle(page_url: str, captured: List[Dict[str, Any]]) -> str:
    """Guarda el bundle JSON de respuestas capturadas."""
    _ensure_data_trading_dir()
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    h = hashlib.sha256(page_url.encode("utf-8")).hexdigest()[:10]
    slug = _safe_slug(page_url)
    filename = f"{slug}_{h}_{int(time.time())}.json"
    out_path = DATA_TRADING_DIR / filename

    payload = {
        "page_url": page_url,
        "captured_at": now,
        "responses": captured,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def _is_json_content_type(ct: str) -> bool:
    """Verifica si el content-type es JSON (application/json o application/*+json)."""
    ct = (ct or "").lower()
    return ("application/json" in ct) or ct.endswith("+json") or ("+json;" in ct)


def _cache_key(url: str, params: Dict[str, Any]) -> str:
    key_parts = [url] + [f"{k}={params[k]}" for k in sorted(params.keys())]
    return "|".join(key_parts)


def _get_cache(key: str) -> Optional[str]:
    entry = _SCRAPE_CACHE.get(key)
    if not entry:
        return None
    timestamp, value = entry
    if time.time() - timestamp > _CACHE_TTL_SECONDS:
        _SCRAPE_CACHE.pop(key, None)
        return None
    return value


def _set_cache(key: str, value: str) -> None:
    if len(_SCRAPE_CACHE) >= _SCRAPE_CACHE_MAX:
        # Evictar la entrada más antigua (primer item insertado en el dict)
        oldest_key = next(iter(_SCRAPE_CACHE))
        _SCRAPE_CACHE.pop(oldest_key, None)
    _SCRAPE_CACHE[key] = (time.time(), value)


def _build_headers() -> Dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }


def _fetch_html(url: str, timeout_seconds: int = 10) -> bytes:
    import requests
    response = requests.get(url, headers=_build_headers(), timeout=timeout_seconds)
    response.raise_for_status()
    return response.content


def _clean_text(text: str) -> str:
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return "\n".join(chunk for chunk in chunks if chunk)


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [texto truncado]"


def _extract_text(soup, max_chars: int, extract_selector: Optional[str] = None) -> str:
    if extract_selector:
        target = soup.select_one(extract_selector)
        if not target:
            return ""
        return _truncate_text(_clean_text(target.get_text()), max_chars)
    for script in soup(["script", "style", "noscript"]):
        script.decompose()
    return _truncate_text(_clean_text(soup.get_text()), max_chars)


def _extract_links(soup, base_url: str, max_links: int = 20) -> Tuple[int, str]:
    from urllib.parse import urljoin
    links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        link_text = link.get_text(strip=True)
        if href.startswith("/"):
            href = urljoin(base_url, href)
        links.append(f"- {link_text}: {href}")
    total = len(links)
    return total, "\n".join(links[:max_links])


def _build_result(url: str, text: Optional[str], links: Optional[str], total_links: int) -> str:
    parts = [f"URL: {url}\n"]
    if text:
        parts.append(f"\nTexto extraido:\n{text}")
    if links:
        parts.append(
            f"\n\nEnlaces encontrados ({total_links} total, mostrando primeros 20):\n"
        )
        parts.append(links)
    return "\n".join(parts)


def _get_playwright():
    global _PLAYWRIGHT
    if _PLAYWRIGHT is None:
        import atexit
        from playwright.sync_api import sync_playwright  # pyright: ignore[reportMissingImports]
        _PLAYWRIGHT = sync_playwright().start()
        atexit.register(_shutdown_playwright)
    return _PLAYWRIGHT


def _get_browser():
    global _BROWSER
    if _BROWSER is None:
        _BROWSER = _get_playwright().chromium.launch(headless=True)
    return _BROWSER


def _shutdown_playwright() -> None:
    """Cierra browser y playwright al terminar el proceso (registrado via atexit)."""
    global _BROWSER, _PLAYWRIGHT
    try:
        if _BROWSER is not None:
            _BROWSER.close()
            _BROWSER = None
    except Exception:
        pass
    try:
        if _PLAYWRIGHT is not None:
            _PLAYWRIGHT.stop()
            _PLAYWRIGHT = None
    except Exception:
        pass


def _configure_page(page, block_resources: bool = True) -> None:
    if not block_resources:
        return

    def route_handler(route):
        request = route.request
        if request.resource_type in {"image", "media", "font"}:
            route.abort()
        else:
            route.continue_()

    page.route("**/*", route_handler)


@tool
def scrape_website_simple(
    url: Annotated[str, Field(description="URL completa incluyendo https://, para páginas estáticas (blogs, docs, noticias)")],
    extract_text: Annotated[bool, Field(description="Si True, extrae el texto principal de la página")] = True,
    extract_links: Annotated[bool, Field(description="Si True, extrae los enlaces encontrados en la página")] = False,
    max_chars: Annotated[int, Field(description="Límite de caracteres del texto extraído", ge=100, le=10000)] = 2000,
) -> str:
    """Extrae información de una página web estática usando requests + BeautifulSoup. Rápida, sin JavaScript."""
    try:
        from bs4 import BeautifulSoup

        html = _fetch_html(url)
        soup = BeautifulSoup(html, "html.parser")
        text = None
        links_text = None
        total_links = 0

        if extract_text:
            text = _extract_text(soup, max_chars)

        if extract_links:
            total_links, links_text = _extract_links(soup, url)

        return _build_result(url, text, links_text, total_links)
    except Exception as e:
        return f"Error al procesar la pagina web: {str(e)}"


@tool
def scrape_website_dynamic(
    url: Annotated[str, Field(description="URL completa incluyendo https://, para páginas con JavaScript (precios, dashboards, SPAs)")],
    wait_for_selector: Annotated[Optional[str], Field(description="Selector CSS a esperar antes de extraer, ej: '.price', '#content'")] = None,
    extract_selector: Annotated[Optional[str], Field(description="Selector CSS del bloque específico a extraer, ej: 'main', '.article-body'")] = None,
    extract_text: Annotated[bool, Field(description="Si True, extrae el texto principal de la página")] = True,
    extract_links: Annotated[bool, Field(description="Si True, extrae los enlaces encontrados")] = False,
    max_chars: Annotated[int, Field(description="Límite de caracteres del texto extraído", ge=100, le=10000)] = 2000,
    block_resources: Annotated[bool, Field(description="Si True, bloquea imágenes y fonts para mayor velocidad")] = True,
    use_cache: Annotated[bool, Field(description="Si True, usa caché de 60s por URL para evitar requests repetidos")] = True,
) -> str:
    """Extrae información de páginas web con JavaScript usando Playwright (sync). Sin captura de JSON de APIs."""
    cache_params = {
        "wait_for_selector": wait_for_selector,
        "extract_selector": extract_selector,
        "extract_text": extract_text,
        "extract_links": extract_links,
        "max_chars": max_chars,
        "block_resources": block_resources,
    }
    cache_key = _cache_key(url, cache_params)
    if use_cache:
        cached = _get_cache(cache_key)
        if cached:
            return cached

    try:
        from bs4 import BeautifulSoup

        browser = _get_browser()
        page = browser.new_page()
        _configure_page(page, block_resources=block_resources)

        page.goto(url, wait_until="domcontentloaded")
        if wait_for_selector:
            page.wait_for_selector(wait_for_selector, timeout=10000)

        html = page.content()
        page.close()

        soup = BeautifulSoup(html, "html.parser")
        text = None
        links_text = None
        total_links = 0

        if extract_text:
            text = _extract_text(soup, max_chars, extract_selector=extract_selector)

        if extract_links:
            total_links, links_text = _extract_links(soup, url)

        result = _build_result(url, text, links_text, total_links)
        if use_cache:
            _set_cache(cache_key, result)
        return result
    except Exception as e:
        return f"Error al procesar la pagina web: {str(e)}"


# ==================== ASYNC SCRAPING CON CAPTURA JSON ====================

async def _scrape_dynamic_async(
    url: str,
    wait_for_selector: Optional[str] = None,
    extract_selector: Optional[str] = None,
    text_limit: int = 2000,
    timeout_ms: int = 20000,
    block_resources: bool = True,
    capture_json: bool = True,
    max_json_responses: int = 50,
    max_total_json_bytes: int = 2_000_000,  # 2MB total aprox
) -> Dict[str, Any]:
    """
    Scrapea una página con Playwright async y captura respuestas JSON.
    
    Guarda un bundle JSON en data_trading/ con todas las respuestas capturadas.
    """
    from playwright.async_api import async_playwright  # pyright: ignore[reportMissingImports]
    from bs4 import BeautifulSoup

    captured: List[Dict[str, Any]] = []
    tasks: List[asyncio.Task] = []
    total_bytes = 0
    json_bundle_path: Optional[str] = None

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()

        if block_resources:
            async def route_handler(route):
                rtype = route.request.resource_type
                if rtype in {"image", "font", "media"}:
                    await route.abort()
                else:
                    await route.continue_()
            await page.route("**/*", route_handler)

        # --- JSON capture handler ---
        async def _capture_response(resp):
            nonlocal total_bytes
            if len(captured) >= max_json_responses:
                return
            try:
                headers = await resp.all_headers()
                ct = headers.get("content-type", "")
                if not _is_json_content_type(ct):
                    return

                body = await resp.body()
                if not body:
                    data = None
                    body_len = 0
                else:
                    body_len = len(body)
                    if (total_bytes + body_len) > max_total_json_bytes:
                        return
                    total_bytes += body_len

                    try:
                        data = json.loads(body.decode("utf-8", errors="replace"))
                    except Exception:
                        data = body.decode("utf-8", errors="replace")[:5000]

                captured.append({
                    "url": resp.url,
                    "status": resp.status,
                    "headers": {"content-type": ct},
                    "data": data,
                })
            except Exception:
                return

        def on_response(resp):
            if not capture_json:
                return
            tasks.append(asyncio.create_task(_capture_response(resp)))

        if capture_json:
            page.on("response", on_response)

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

            if wait_for_selector:
                await page.wait_for_selector(wait_for_selector, timeout=timeout_ms)
            else:
                await page.wait_for_timeout(800)

            # Esperar para capturar calls tardías
            await page.wait_for_timeout(600)

            # Asegurar que terminaron tasks de captura
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Guardar bundle SIEMPRE (aunque esté vacío)
            json_bundle_path = _save_json_bundle(page.url, captured)

            # --- Extracción HTML normal ---
            html = await page.content()
            title = await page.title()

            soup = BeautifulSoup(html, "html.parser")
            node = soup.select_one(extract_selector) if extract_selector else soup
            main_text = node.get_text(" ", strip=True) if node else ""
            main_text = " ".join(main_text.split())
            if len(main_text) > text_limit:
                main_text = main_text[:text_limit] + "... [texto truncado]"

            links = []
            if node:
                for a in node.select("a[href]")[:30]:
                    t = " ".join(a.get_text(" ", strip=True).split())[:80]
                    href = a.get("href", "")
                    if href:
                        links.append({"text": t, "href": href})

            return {
                "requested_url": url,
                "url": page.url,
                "title": title,
                "rendered": True,
                "main_text": main_text,
                "links": links,
                "json_bundle_path": json_bundle_path,
                "json_captured_count": len(captured),
                "json_total_bytes": total_bytes,
            }

        finally:
            await context.close()
            await browser.close()


@tool
async def scrape_website_with_json_capture(
    url: Annotated[str, Field(description="URL completa incluyendo https://, ideal para páginas con APIs/endpoints JSON (trading, precios, datos en tiempo real)")],
    wait_for_selector: Annotated[Optional[str], Field(description="Selector CSS a esperar antes de extraer, ej: '.price', '#ticker'")] = None,
    extract_selector: Annotated[Optional[str], Field(description="Selector CSS del bloque específico a extraer")] = None,
    max_chars: Annotated[int, Field(description="Límite de caracteres del texto extraído", ge=100, le=10000)] = 2000,
    capture_json: Annotated[bool, Field(description="Si True, intercepta y guarda respuestas JSON de APIs en data_trading/")] = True,
) -> str:
    """Extrae información de páginas web con JavaScript y captura endpoints JSON automáticamente en data_trading/."""
    try:
        result = await _scrape_dynamic_async(
            url=url,
            wait_for_selector=wait_for_selector,
            extract_selector=extract_selector,
            text_limit=max_chars,
            capture_json=capture_json,
        )

        # Construir respuesta legible
        parts = [f"URL: {result['url']}"]
        if result.get("title"):
            parts.append(f"Titulo: {result['title']}")
        parts.append(f"\nTexto extraido:\n{result['main_text']}")

        if result.get("links"):
            links_str = "\n".join([f"- {l['text']}: {l['href']}" for l in result["links"][:20]])
            parts.append(f"\n\nEnlaces encontrados ({len(result['links'])} total):\n{links_str}")

        if result.get("json_bundle_path"):
            parts.append(f"\n\n[JSON Capturado]")
            parts.append(f"Archivo: {result['json_bundle_path']}")
            parts.append(f"Respuestas capturadas: {result['json_captured_count']}")
            parts.append(f"Total bytes JSON: {result['json_total_bytes']}")

        return "\n".join(parts)
    except Exception as e:
        return f"Error al procesar la pagina web: {str(e)}"


# ==================== AGENTES ESPECIALIZADOS ====================
# Usando langgraph.prebuilt.create_react_agent para crear agentes
# que siguen el patrón ReAct (Reasoning + Acting)

def create_math_agent():
    """
    Crea un agente especializado en matemáticas.
    
    Usa create_react_agent que devuelve un CompiledStateGraph
    que puede ser invocado directamente con mensajes.
    """
    llm = get_llm()
    
    system_prompt = load_agent_prompt("math_agent")
    
    # create_react_agent devuelve un grafo compilado listo para usar
    return create_react_agent(
        model=llm,
        tools=[calculate],
        prompt=system_prompt,
        name="math_agent"
    )


def create_analysis_agent():
    """
    Crea un agente especializado en análisis de datos.
    
    Este agente puede analizar descripciones de datos y generar insights.
    """
    llm = get_llm()
    
    system_prompt = load_agent_prompt("analysis_agent")
    
    return create_react_agent(
        model=llm,
        tools=[analyze_data],
        prompt=system_prompt,
        name="analysis_agent"
    )


def create_code_agent():
    """
    Crea un agente especializado en programación.
    
    Este agente puede escribir código en varios lenguajes.
    """
    llm = get_llm()
    
    system_prompt = load_agent_prompt("code_agent")
    
    return create_react_agent(
        model=llm,
        tools=[write_code],
        prompt=system_prompt,
        name="code_agent"
    )


def create_web_scraping_agent():
    """
    Crea un agente especializado en web scraping.
    
    Este agente puede extraer información de páginas web y capturar endpoints JSON.
    """
    llm = get_llm()
    
    system_prompt = load_agent_prompt("web_scraping_agent")

    return create_react_agent(
        model=llm,
        tools=[get_crypto_price, search_web, scrape_website_simple, scrape_website_dynamic, scrape_website_with_json_capture, extract_price_from_text],
        prompt=system_prompt,
        name="web_scraping_agent"
    )


# Alias explícito para acceder a la función subyacente sin overhead del tool wrapper.
# Usado por price_helpers.py para el fast path de API price.
_get_crypto_price_fn = get_crypto_price.func
