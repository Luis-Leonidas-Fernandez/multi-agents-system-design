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
from pathlib import Path
import os
import asyncio
from crypto_price import get_crypto_price
from scraping_infra import (
    _validate_url, _cache_key, _get_cache, _set_cache,
    _build_headers, _fetch_html, _extract_text, _extract_links, _build_result,
    _get_playwright, _get_browser, _configure_page, _scrape_dynamic_async,
    _is_json_content_type, DATA_TRADING_DIR,
)


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




@tool
def scrape_website_simple(
    url: Annotated[str, Field(description="URL completa incluyendo https://, para páginas estáticas (blogs, docs, noticias)")],
    extract_text: Annotated[bool, Field(description="Si True, extrae el texto principal de la página")] = True,
    extract_links: Annotated[bool, Field(description="Si True, extrae los enlaces encontrados en la página")] = False,
    max_chars: Annotated[int, Field(description="Límite de caracteres del texto extraído", ge=100, le=10000)] = 2000,
) -> str:
    """Extrae información de una página web estática usando requests + BeautifulSoup. Rápida, sin JavaScript."""
    url_error = _validate_url(url)
    if url_error:
        return f"URL rechazada: {url_error}"
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
    url_error = _validate_url(url)
    if url_error:
        return f"URL rechazada: {url_error}"
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



@tool
async def scrape_website_with_json_capture(
    url: Annotated[str, Field(description="URL completa incluyendo https://, ideal para páginas con APIs/endpoints JSON (trading, precios, datos en tiempo real)")],
    wait_for_selector: Annotated[Optional[str], Field(description="Selector CSS a esperar antes de extraer, ej: '.price', '#ticker'")] = None,
    extract_selector: Annotated[Optional[str], Field(description="Selector CSS del bloque específico a extraer")] = None,
    max_chars: Annotated[int, Field(description="Límite de caracteres del texto extraído", ge=100, le=10000)] = 2000,
    capture_json: Annotated[bool, Field(description="Si True, intercepta y guarda respuestas JSON de APIs en data_trading/")] = True,
) -> str:
    """Extrae información de páginas web con JavaScript y captura endpoints JSON automáticamente en data_trading/."""
    url_error = _validate_url(url)
    if url_error:
        return f"URL rechazada: {url_error}"
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


