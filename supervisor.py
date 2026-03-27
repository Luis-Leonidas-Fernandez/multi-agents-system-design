"""
Sistema supervisor multi-agentes usando LangGraph

Este módulo implementa el patrón Supervisor donde un agente coordinador
decide qué agente especializado debe manejar cada solicitud.
Los agentes ahora usan langgraph.prebuilt.create_react_agent.
"""
from typing import TypedDict, Annotated, Literal, Any, Dict, List, Optional, Tuple
import json
import math
import os
import random
import re
import time
import uuid
import httpx
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel
from config import get_llm
from agents import create_math_agent, create_analysis_agent, create_code_agent, create_web_scraping_agent, get_crypto_price


AgentName = Literal["math_agent", "analysis_agent", "code_agent", "web_scraping_agent"]


class RoutingDecision(BaseModel):
    agent: AgentName
    reason: str


def _flatten_messages_text(msgs: List[Any]) -> str:
    parts: List[str] = []
    for m in msgs:
        # contenido normal
        c = getattr(m, "content", None)
        if isinstance(c, str) and c:
            parts.append(c)

        # tool calls (donde viene el url normalmente)
        tool_calls = getattr(m, "tool_calls", None) or []
        for tc in tool_calls:
            try:
                # args puede ser dict o string
                args = tc.get("args")
                if isinstance(args, dict):
                    parts.append(json.dumps(args, ensure_ascii=False))
                elif isinstance(args, str):
                    parts.append(args)
            except Exception:
                pass

        # ToolMessage también puede contener algo
        # (no siempre trae el url, pero suma)
        if hasattr(m, "tool_call_id"):
            # ya lo cubre content, pero ok
            pass

    return " ".join(parts).lower()


def _is_allowed_public_price_request(msgs: List[Any], node: str) -> bool:
    txt = _flatten_messages_text(msgs)

    wants_price = any(k in txt for k in ["precio", "price", "cotiza", "cotización", "cotizacion"])
    is_btc = any(k in txt for k in ["bitcoin", "btc"])

    # dominios que querés permitir (opcional)
    allowed_domains = ["coingecko.com", "coinbase.com", "kraken.com"]
    has_allowed_domain = any(d in txt for d in allowed_domains)

    # ✅ regla robusta: si estamos en el nodo de scraping y la intención es “precio btc”, permitir igual
    if node == "web_scraping_node" and wants_price and is_btc:
        return True

    # si querés ser más estricto fuera del scraping node:
    return wants_price and is_btc and has_allowed_domain


# ==================== MIDDLEWARE PRE-EJECUCIÓN ====================

# Fase 1 — patrones que bloquean de inmediato en el último mensaje.
_BLOCKED_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "jailbreak",
    "bypass security",
    "act as dan",
    "forget your instructions",
]

# Fase 2 — señales de riesgo que activan revisión del historial completo.
# Son fragmentos más cortos / ambiguos que solos no justifican bloqueo.
_RISK_SIGNALS = [
    "ignore",
    "forget",
    "override",
    "pretend",
    "simulate",
    "as if you were",
    "new persona",
    "disregard",
]


def _extract_text(msg) -> str:
    return msg.content.lower() if hasattr(msg, "content") and isinstance(msg.content, str) else ""


def _check_patterns(text: str, patterns: list) -> Optional[str]:
    """Retorna el primer patrón encontrado en text, o None."""
    for pattern in patterns:
        if pattern in text:
            return pattern
    return None


def _emit_block(trajectory_id: str, pattern: str, source: str, approx_chars: int) -> None:
    _emit_guard_audit({
        "trajectory_id": trajectory_id,
        "guard_label": "blocked",
        "guard_status": "blocked_by_middleware",
        "verdict_source": source,
        "node_name": "input",
        "raw_response": pattern,
        "model": "middleware",
        "latency_ms": 0,
        "policy": "block",
        "trajectory_steps_count": 0,
        "approx_chars": approx_chars,
    })


def _get_human_history(messages: list, max_msgs: int = 10) -> list:
    """Retorna los últimos N mensajes humanos del historial (excluye AI y ToolMessages).
    Solo inputs humanos: más señal, menos ruido.
    """
    return [m for m in messages if isinstance(m, HumanMessage)][-max_msgs:]


def input_guard(state: "AgentState") -> Optional["AgentState"]:
    """
    Middleware pre-ejecución de dos fases con seguridad stateful.

    Fase 1 — último mensaje, siempre:
      Revisa messages[-1] contra _BLOCKED_PATTERNS. Sin costo.
      Si hay match → bloqueo inmediato.

    Fase 2 — historial solo si hay motivo:
      Se activa cuando CUALQUIERA de estas condiciones es verdadera:
        a) El último mensaje contiene un _RISK_SIGNAL (señal en este turno)
        b) state["risk_flag"] == True (algún turno anterior ya fue sospechoso)
      Revisa solo los últimos 10 mensajes HUMANOS (más señal, menos ruido).
      Si hay match → bloqueo + risk_flag sigue en True.

    Efecto stateful: una vez que risk_flag=True, cada turno siguiente
    pasa por la revisión de historial automáticamente — aunque el mensaje
    actual parezca inocente.
    """
    messages = state.get("messages", [])
    if not messages:
        return None

    trajectory_id = str(uuid.uuid4())
    last_text = _extract_text(messages[-1])

    # --- Fase 1: último mensaje ---
    hit = _check_patterns(last_text, _BLOCKED_PATTERNS)
    if hit:
        _emit_block(trajectory_id, hit, "last_message", len(last_text))
        return {
            "messages": [AIMessage(content="Solicitud bloqueada por política de seguridad.")],
            "risk_flag": True,
            "blocked": True,   # flag tipado — no comparación de strings
        }

    # --- Fase 2: historial — activada por risk_signal O por risk_flag previo ---
    risk_signal = _check_patterns(last_text, _RISK_SIGNALS)
    prior_risk = state.get("risk_flag", False)

    if risk_signal or prior_risk:
        human_history = _get_human_history(messages[:-1])  # excluye el último ya revisado
        if human_history:
            combined = " ".join(_extract_text(m) for m in human_history)
            hit = _check_patterns(combined, _BLOCKED_PATTERNS)
            if hit:
                source = "history_window_risk_flag" if prior_risk else "history_window_risk_signal"
                _emit_block(trajectory_id, hit, source, len(combined))
                return {
                    "messages": [AIMessage(content="Solicitud bloqueada por política de seguridad.")],
                    "risk_flag": True,
                    "blocked": True,
                }

        if risk_signal and not prior_risk:
            return {"risk_flag": True}

    return None


# ==================== HITL ====================

# Controla si se pide confirmación al usuario antes de ejecutar nodos de alto riesgo.
# Puede desactivarse con HITL_ENABLED=false en .env (útil en tests o modo batch).
_HITL_ENABLED = os.getenv("HITL_ENABLED", "true").strip().lower() == "true"


async def _ask_confirmation(prompt: str) -> bool:
    """Solicita confirmación al usuario de forma async (no bloquea el event loop)."""
    import asyncio
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, lambda: input(prompt).strip().lower())
    return answer in ("s", "si", "sí", "y", "yes")


# ==================== ESTADO COMPARTIDO ====================

class AgentState(TypedDict):
    """
    Estado compartido entre todos los agentes.
    
    Attributes:
        messages: Historial de mensajes de la conversación
        next_agent: Nombre del próximo agente a ejecutar
    """
    messages: Annotated[list, lambda x, y: x + y]  # Historial de mensajes
    next_agent: str   # Próximo agente a ejecutar
    risk_flag: bool   # True si algún turno previo activó una señal de riesgo
    blocked: bool     # True si input_guard bloqueó el mensaje actual
    request_id: str   # UUID generado por input_guard_node — correlaciona todos los nodos del mismo turno
    scrape_tracker: dict  # Score acumulativo por categoría: {"crypto_price": {"score": -1, "last_turn": 3}, ...}


# ==================== AGENTES ====================

# Crear instancias de los agentes (usando create_react_agent)
# Cada agente es ahora un CompiledStateGraph que recibe y devuelve mensajes
math_agent = create_math_agent()
analysis_agent = create_analysis_agent()
code_agent = create_code_agent()
web_scraping_agent = create_web_scraping_agent()


# ==================== GUARDRAIL AGENTDOG ====================

_MAX_OBSERVATION_CHARS = 2000
_TRUNC_HEAD_CHARS = 1200
_TRUNC_TAIL_CHARS = 800
_RAW_RESPONSE_MAX_CHARS = 500

# Nodos considerados de alto riesgo para políticas de seguridad
HIGH_RISK_NODES = frozenset({"code_node", "web_scraping_node"})


def is_high_risk(node_name: str) -> bool:
    """Retorna True si el nodo es considerado de alto riesgo."""
    return node_name in HIGH_RISK_NODES


Outcome = Literal["success", "blocked", "error"]


def _emit_node_outcome(request_id: str, node: str, outcome: Outcome, phase: str = "agent", **extra) -> None:
    """Emite un registro de outcome al audit log. Fuente de verdad para KPIs.

    Campos siempre presentes: request_id, node, outcome, phase, ts_ms.
    phase: "pre_guard" (HITL) | "agent" (ainvoke) | "post_guard" (AgentDoG)
    Extra: cualquier metadata adicional (agent, duration_ms, prompt_tokens, etc.)
    """
    _emit_guard_audit({
        "request_id": request_id,
        "node": node,
        "outcome": outcome,      # "success" | "blocked" | "error"
        "phase": phase,          # "pre_guard" | "agent" | "post_guard"
        "ts_ms": int(time.time() * 1000),
        **extra,
    })


def _get_model_name() -> str:
    """Retorna el nombre del modelo activo según LLM_PROVIDER (sin instanciar el LLM)."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider == "azure":
        return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL", "llama3")
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# Precios por 1K tokens en USD — actualizar según tarifas vigentes.
# Ollama y modelos locales no tienen costo → no incluir aquí.
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o":                     {"input": 0.0025,  "output": 0.010},
    "gpt-4o-mini":                {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo":                {"input": 0.010,   "output": 0.030},
    "gpt-3.5-turbo":              {"input": 0.0005,  "output": 0.0015},
    "claude-3-5-sonnet-20241022": {"input": 0.003,   "output": 0.015},
    "claude-3-haiku-20240307":    {"input": 0.00025, "output": 0.00125},
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """Estima costo en USD. Retorna None si el modelo no está en MODEL_PRICING."""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return None
    return round(
        (prompt_tokens / 1000) * pricing["input"] +
        (completion_tokens / 1000) * pricing["output"],
        8,
    )


def _extract_tokens(result: dict) -> dict:
    """Extrae métricas de tokens + costo estimado del último AIMessage con usage_metadata.

    Siempre incluye tokens_available y model para que el audit log sea comparable:
    - tokens_available=True  → prompt/completion/total/cost son confiables
    - tokens_available=False → excluir de análisis de costos, pero model sigue presente
    """
    model = _get_model_name()
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                prompt = usage.get("input_tokens", 0)
                completion = usage.get("output_tokens", 0)
                cost = _estimate_cost(model, prompt, completion)
                data: Dict[str, Any] = {
                    "model": model,
                    "tokens_available": True,
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": prompt + completion,
                }
                if cost is not None:
                    data["estimated_cost_usd"] = cost
                return data
    return {"model": model, "tokens_available": False}


def _extract_quality(result: dict) -> dict:
    """Proxy de calidad de respuesta: longitud del output y número de tool calls.

    output_length: chars de la respuesta final → muy corto puede indicar baja calidad.
    tool_calls_count: cuántas herramientas invocó el agente en su razonamiento.
    """
    messages = result.get("messages", [])
    tool_calls_count = sum(
        len(getattr(m, "tool_calls", None) or [])
        for m in messages
        if isinstance(m, AIMessage)
    )
    output_length = 0
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            output_length = len(msg.content)
            break
    return {"output_length": output_length, "tool_calls_count": tool_calls_count}


# Frases que indican que el agente no pudo responder satisfactoriamente.
# Proxy heurístico — no ground truth, pero útil para detectar patrones a escala.
_FOLLOWUP_SIGNALS = [
    "no pude", "no encontré", "no tengo información", "no sé", "no puedo",
    "no hay datos", "no fue posible", "error", "intenta de nuevo",
    "could not", "unable to", "i don't know", "no information",
]

_FOLLOWUP_SHORT_THRESHOLD = 80  # chars — respuesta muy corta sugiere baja calidad

# ==================== SCRAPE TRACKER ====================
# Score acumulativo flotante por categoría. Rango: [-3.0, +3.0].
# Penaliza contenido, latencia y costo. Aplica decay y exploration.

_SCORE_DECAY_TURNS       = 3      # turnos sin actividad antes de resetear a 0
_SCORE_MAX               =  3.0
_SCORE_MIN               = -3.0
# Normalizadores para el delta continuo (ver _compute_delta)
_QUALITY_NORM_WORDS      = 120    # raw_words / 120, capped 1.0 → quality_score ∈ [0, 1]
_LATENCY_NORM_MS         = 15000  # duration_ms / 15000, capped 1.0 → latency ∈ [0, 1]
_COST_NORM_USD           = 0.001  # cost_usd / 0.001, capped 1.0 → cost ∈ [0, 1]
_W_LATENCY               = 0.5   # peso de la penalización de latencia en el delta
_W_COST                  = 0.5   # peso de la penalización de costo en el delta
_CONSECUTIVE_FAIL_EXTRA  = -0.5  # penalización extra cuando hay ≥ 2 fallas consecutivas
_COOLDOWN_TURNS          = 2     # turnos con "free" forzado después de ok_strong

# Categorías de query detectadas por keywords. "general" es el fallback.
_QUERY_CATEGORIES: Dict[str, List[str]] = {
    "crypto_price": ["bitcoin", "btc", "ethereum", "eth", "crypto", "precio", "price", "cotiz", "coin", "defi", "nft"],
    "finance":      ["stock", "accion", "bolsa", "mercado", "market", "nyse", "nasdaq", "s&p", "dow"],
    "news":         ["noticias", "noticia", "news", "article", "articulo", "headline", "titular"],
    "weather":      ["clima", "weather", "temperatura", "lluvia", "pronostico"],
}

# Policy table: categoría → score_range → acción dura.
# Elimina al LLM de la decisión: la estrategia es determinista dado el score.
# "free" → el agente decide; "prefer_search" / "force_search" → hint inyectado.
_STRATEGY_POLICY: Dict[str, Dict[str, str]] = {
    "crypto_price": {
        "very_low":   "force_search",   # score <= -2
        "low":        "prefer_search",  # -2 < score < 0
        "neutral":    "free",           # score >= 0
    },
    "finance": {
        "very_low":   "force_search",
        "low":        "prefer_search",
        "neutral":    "free",
    },
    "news": {
        "very_low":   "prefer_search",  # news: menos agresivo — scraping puede recuperarse
        "low":        "free",
        "neutral":    "free",
    },
    # "general" y "weather" usan el fallback
}
_DEFAULT_POLICY = {"very_low": "force_search", "low": "prefer_search", "neutral": "free"}


# Confianza mínima para respetar una estrategia promoted del policy.json.
# Debajo de este umbral (post-decay) se ignora y se cae al comportamiento heurístico.
_POLICY_MIN_CONFIDENCE = 0.50

# Cuántas estrategias promoted considerar como candidatas (top-k).
_POLICY_TOP_K = 2

# Tiempo de vida de la confianza: exp(-Δt / τ). Con τ=7 días:
#   1 día  → decay 0.87   (pequeño impacto)
#   7 días → decay 0.37   (confianza a la mitad)
#  14 días → decay 0.13   (policy obsoleta — casi ignorada)
_POLICY_DECAY_TAU = 7 * 24 * 3600   # segundos

# Temperatura base del softmax — se ajusta dinámicamente en _get_strategy().
# Definida como referencia; el valor real varía con el score.
_POLICY_SOFTMAX_TEMP_DEFAULT = 0.3

# UCB bonus: c × (1 - confidence) / √effective_runs
#
# Dos mejoras respecto a UCB clásico:
#   1. (1 - confidence): estrategias buenas explotan, malas exploran más.
#   2. effective_runs = runs + runs_last_24h: unifica "datos históricos" y
#      "sobreexplotación reciente" en una sola señal. A mayor uso reciente,
#      mayor effective_runs → UCB bonus más pequeño → menos sobreexplotación.
#      Esto reemplaza al usage_decay separado, eliminando tensión entre señales.
_POLICY_UCB_C = 0.1

# Score pressure: amplifica el peso cuando el sistema está rindiendo mal.
# weight *= (1 + α × |min(score, 0)|)
# α=0.2, score=-2 → factor 1.4 (urgencia de cambiar estrategia)
# α=0.2, score=+1 → factor 1.0 (sin efecto — sistema bien)
_POLICY_SCORE_PRESSURE_ALPHA = 0.2

# Epsilon-floor: garantía mínima de exploración en el softmax.
# Reducido a 0.02 porque UCB y temperatura dinámica ya cubren exploración.
# prob_final = (1 - ε) × softmax + ε/k
_POLICY_EPSILON = 0.02


def _load_runtime_policy() -> dict:
    """Carga policy.json versionado generado por analytics.py --train.

    Soporta dos formatos:
    - Nuevo (versionado): {"version": "...", "categories": { ... }}
    - Legacy (flat):      {"crypto_price": {"promoted": [...], ...}}

    Busca en:
    1. POLICY_CONFIG env var (path explícito)
    2. ./policy.json (directorio de trabajo)
    3. ./logs/policy.json (junto al audit log)

    Retorna {} si no existe — cae al comportamiento heurístico habitual.
    """
    candidates = [
        os.getenv("POLICY_CONFIG", ""),
        "policy.json",
        "logs/policy.json",
    ]
    for path in candidates:
        if not path:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            # Formato nuevo: tiene "categories" como key raíz
            if "categories" in raw:
                version      = raw.get("version", "unknown")
                generated_ts = raw.get("generated_ts", time.time())
                categories   = raw["categories"]
            else:
                # Formato legacy: dict plano por categoría
                version      = "legacy"
                generated_ts = time.time()   # sin ts → sin decay
                categories   = raw

            # Aplicar solo time decay al cargar.
            # El usage decay (sobreexplotación reciente) queda integrado en
            # effective_runs = runs + runs_last_24h dentro de _ucb_weight().
            delta_t    = time.time() - generated_ts
            time_decay = math.exp(-delta_t / _POLICY_DECAY_TAU)
            for cat_data in categories.values():
                for p in cat_data.get("promoted", []):
                    if isinstance(p, dict):
                        p["confidence"] = round(
                            p.get("confidence", 1.0) * time_decay, 3
                        )

            age_days = delta_t / 86400
            print(f"[supervisor] policy.json v{version} cargado desde {path} "
                  f"(age={age_days:.1f}d time_decay={time_decay:.2f})")
            return categories
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[supervisor] No se pudo cargar policy.json ({path}): {e}")
    return {}


# Cargado una sola vez al importar el módulo.
# Se puede recargar llamando _load_runtime_policy() y reasignando.
_RUNTIME_POLICY: dict = _load_runtime_policy()


_STRATEGY_HINTS: Dict[str, str] = {
    "api_price": (
        "[Sistema] Query de precio de criptomoneda detectada. "
        "Usa get_crypto_price directamente — es más rápido y confiable que scraping o search_web.\n\n"
    ),
    "force_search": (
        "[Sistema] Scraping falló repetidamente para este tipo de query. "
        "Usa search_web directamente — no intentes scraping de páginas.\n\n"
    ),
    "prefer_search": (
        "[Sistema] Scraping devolvió contenido insuficiente o lento antes. "
        "Intenta search_web primero; usa scraping solo si tienes una URL específica confiable.\n\n"
    ),
    "free": "",
}


# Niveles que disparan retry automático con force_search.
# ok_weak y ok_strong se devuelven directamente (suficiente contenido).
_RETRY_ON_RELIABILITY = frozenset({"unreliable", "low_content"})


def _scrape_reliability(raw_words: int) -> str:
    """4 niveles de confiabilidad según palabras extraídas.

    unreliable  → < 20  palabras  (model fallback / bloqueo total)
    low_content → 20–49 palabras  (contenido insuficiente)
    ok_weak     → 50–119 palabras (aceptable, pero no sólido)
    ok_strong   → ≥ 120 palabras  (resultado confiable)
    """
    if raw_words < 20:
        return "unreliable"
    if raw_words < 50:
        return "low_content"
    if raw_words < 120:
        return "ok_weak"
    return "ok_strong"


_PRICE_CONTEXT_RE = re.compile(
    r'(?:precio|price|value|valor|usd)\s*[:\s]*\$?\s*'
    r'([\d]{1,3}(?:[,\s]\d{3})*(?:\.\d{1,4})?[kKmM]?|\d+(?:\.\d{1,4})?[kKmM]?)',
    re.IGNORECASE,
)
_PRICE_SANITY_MIN = 1.0
_PRICE_SANITY_MAX = 1_000_000.0


def _extract_structured_price(text: str) -> Optional[float]:
    """Extrae un valor de precio de una respuesta de API estructurada.

    Busca números en contexto semántico de precio (palabras clave: precio, price,
    value, USD). Elimina falsos positivos como "error 404", "2024", "123 usuarios".

    Soporta shorthand: "71k" → 71000, "$71.2k" → 71200, "1.5M" → 1500000.
    Aplica sanity check de rango (_PRICE_SANITY_MIN, _PRICE_SANITY_MAX).

    Retorna:
        float dentro del rango válido si se encuentra un precio.
        None si no hay precio, el texto es ambiguo (ej. "price unavailable"),
        o el valor está fuera del rango de sanidad.
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
    _KNOWN_SCHEMA_VERSIONS = {"1.0"}

    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "get_crypto_price":
            try:
                data = json.loads(msg.content)
                if not isinstance(data, dict):
                    return None

                # Routing por schema_version — mantiene compatibilidad futura con v2.0+
                schema_version = data.get("schema_version", "1.0")
                if schema_version not in _KNOWN_SCHEMA_VERSIONS:
                    return None   # schema desconocido: no parsear a ciegas

                # --- v1.0: validación de campos requeridos ---
                if data.get("error"):                   return None   # API falló (price_unavailable, etc.)
                price    = data.get("price")
                currency = data.get("currency", "")
                if not isinstance(price, (int, float)):
                    return None   # cubre "N/A", None, strings inválidos
                price_f = float(price)
                if price_f <= 0:
                    return None   # cubre {"price": 0}
                if not (_PRICE_SANITY_MIN < price_f < _PRICE_SANITY_MAX):
                    return None   # fuera del rango de sanidad
                if not isinstance(currency, str) or not currency:
                    return None   # currency ausente o mal formada

                data["price"] = price_f   # normalizar a float
                return data
            except (json.JSONDecodeError, ValueError, TypeError):
                return None
    return None


# Estrategias que producen respuestas estructuradas (datos de API, JSON, tablas).
# Para éstas, la validación usa _extract_structured_price en vez de word count.
_STRUCTURED_SOURCE_STRATEGIES: set[str] = {"api_price"}


def _detect_query_category(text: str) -> str:
    """Detecta la categoría de la query por keywords. Retorna 'general' si no hay match."""
    text_lower = text.lower()
    for category, keywords in _QUERY_CATEGORIES.items():
        if any(k in text_lower for k in keywords):
            return category
    return "general"


def _get_category_score(tracker: dict, category: str, current_turn: int) -> float:
    """Retorna el score actual para una categoría, aplicando decay si corresponde."""
    entry = tracker.get(category)
    if not entry:
        return 0.0
    turns_ago = current_turn - entry.get("last_turn", 0)
    if turns_ago > _SCORE_DECAY_TURNS:
        return 0.0  # score decayed — olvidar la falla pasada
    return float(entry.get("score", 0.0))


def _score_to_policy_band(score: float) -> str:
    if score <= -2.0:
        return "very_low"
    if score < 0.0:
        return "low"
    return "neutral"


def _score_to_reliability(score: float) -> str:
    """Mapea score acumulativo al nivel de confiabilidad para el audit log."""
    band = _score_to_policy_band(score)
    if band == "very_low":
        return "unreliable"
    if band == "low":
        return "low_content"
    return "ok"


def _exploration_rate(score: float) -> float:
    """Tasa de exploration dinámica según certeza del score.

    Zona incierta (|score| < 0.5) → explorar más (20%).
    Score muy malo (≤ -2)         → explorar poco (5%) — evitar desperdiciar tokens.
    Default                       → 10%.
    """
    if abs(score) < 0.5:
        return 0.20
    if score <= -2.0:
        return 0.05
    return 0.10


# Descuento sobre latency/cost para fuentes estructuradas (APIs).
# API confiable > scraping rápido: no penalizar si la respuesta es autoritativa.
_STRUCTURED_LATENCY_DISCOUNT = 0.3   # latencia penalizada 70% menos para APIs
_API_VALIDATION_EPSILON      = 0.02  # 2% de turnos con api_price se redirigen a exploración


def _compute_delta(
    raw_words: int,
    duration_ms: int,
    cost_usd: Optional[float],
    consecutive_failures: int,
    *,
    source_type: str = "unstructured",
) -> float:
    """Delta normalizado y continuo para el score del tracker.

    Fórmula:
        quality_score   = min(raw_words / _QUALITY_NORM_WORDS, 1.0)   ∈ [0, 1]
        latency_penalty = min(duration_ms / _LATENCY_NORM_MS, 1.0)    ∈ [0, 1]
        cost_penalty    = min(cost_usd / _COST_NORM_USD, 1.0)         ∈ [0, 1]
        delta = quality_score
                - _W_LATENCY * latency_penalty
                - _W_COST    * cost_penalty
                [- _CONSECUTIVE_FAIL_EXTRA si consecutive_failures >= 2]

    Fuentes estructuradas (source_type="structured"):
        latency_penalty *= _STRUCTURED_LATENCY_DISCOUNT  (0.3)
        cost_penalty    *= _STRUCTURED_COST_DISCOUNT      (0.3)
        → API confiable > scraping rápido: penalizar menos latencia/costo.

    Rango natural del delta: [-1.0, +1.0] antes del bonus por fallas consecutivas.
    """
    quality_score   = min(raw_words / _QUALITY_NORM_WORDS, 1.0)
    latency_penalty = min(duration_ms / _LATENCY_NORM_MS, 1.0)
    cost_penalty    = min((cost_usd or 0.0) / _COST_NORM_USD, 1.0)

    if source_type == "structured":
        # API determinística = calidad máxima garantizada.
        # No tiene sentido penalizar por longitud de texto (la respuesta es corta por diseño).
        quality_score   = 1.0
        # Penalización de latencia y costo ya descontadas (0.3x), sin costo adicional.
        # cost_penalty → 0 cuando estimated_cost_usd=0.0 (fast path sin LLM).
        latency_penalty *= _STRUCTURED_LATENCY_DISCOUNT
        cost_penalty     = 0.0   # API sin LLM = sin costo facturable

    delta = quality_score - _W_LATENCY * latency_penalty - _W_COST * cost_penalty

    if consecutive_failures >= 2:
        delta += _CONSECUTIVE_FAIL_EXTRA  # acelera cambio de estrategia

    return round(delta, 3)


_EXPLORATION_WEIGHTS: dict[str, float] = {
    "prefer_search": 0.55,
    "force_search":  0.45,
}


def _get_strategy(tracker: dict, category: str, score: float, *, exploring: bool) -> str:
    """Retorna la acción de la policy table.

    Prioridades (en orden):
    1. Exploration — PRIMERO, antes del cooldown.
       El cooldown era la causa del bug: interceptaba exploring y devolvía "free"
       antes de llegar al weighted random, haciendo que la exploración fuera inútil.
    2. Cooldown activo tras ok_strong → "free"
    3. Runtime policy (policy.json)   → multi-promoted filtrado por confidence
    4. Policy table heurística basada en score band (respetando disabled)
    """
    if exploring:
        # Exploración: usa SOLO prefer_search / force_search.
        # "free" queda excluido porque ya es el default — incluirlo en exploración
        # produce el mismo output que no explorar, aportando cero señal de aprendizaje.
        # El objetivo es cubrir estrategias NO-default para descubrir si funcionan mejor.
        return random.choices(
            list(_EXPLORATION_WEIGHTS.keys()),
            weights=list(_EXPLORATION_WEIGHTS.values()),
        )[0]

    entry    = tracker.get(category) or {}
    cooldown = entry.get("cooldown_turns", 0)
    if cooldown > 0:
        return "free"

    # Consultar runtime policy (generada por analytics.py --train)
    rt_cat   = _RUNTIME_POLICY.get(category, {})
    disabled = rt_cat.get("disabled", [])
    promoted_raw = rt_cat.get("promoted", [])

    # Normalizar: soporta objetos {"strategy":..., "confidence":...} y strings legacy
    # Confidence ya tiene decay aplicado por _load_runtime_policy()
    promoted_candidates = []
    for p in promoted_raw[:_POLICY_TOP_K]:
        if isinstance(p, dict):
            if p.get("confidence", 0) >= _POLICY_MIN_CONFIDENCE:
                promoted_candidates.append(p)
        elif isinstance(p, str):
            promoted_candidates.append({"strategy": p, "confidence": 1.0, "aggressiveness": 1.0})

    if promoted_candidates:
        # 1. Temperatura dinámica según certeza del score
        #    Inseguro (|score|<0.5) → T alto → más exploración
        #    Muy malo (≤-2)        → T bajo → commit a estrategia más agresiva
        #    Normal               → T intermedia
        if abs(score) < 0.5:
            temp = 0.6
        elif score <= -2.0:
            temp = 0.2
        else:
            temp = _POLICY_SOFTMAX_TEMP_DEFAULT

        # 2. Peso con UCB confidence-dependiente + score pressure
        #    effective_runs = runs + runs_last_24h → unifica historial y uso reciente
        #    ucb_bonus = c × (1 - confidence) / √eff_runs
        #      → baja confidence + pocos datos → explorar más
        #      → alta confidence + uso reciente → bonus casi nulo
        #    score_pressure aplicada solo al ucb_bonus (no al base):
        #      weight = base + ucb_bonus × pressure
        #      → explotás bien lo que funciona (base intacto)
        #      → aumentás exploración cuando el sistema rinde mal (bonus amplificado)
        _AGG_PRIOR = {"force_search": 2.0, "prefer_search": 1.0, "free": 0.0}

        def _ucb_weight(p: dict) -> float:
            confidence = p.get("confidence", 0.0)
            eff_runs   = max(p.get("runs", 1) + p.get("runs_last_24h", 0), 1)
            ucb_bonus  = _POLICY_UCB_C * (1 - confidence) / math.sqrt(eff_runs)
            if score <= -1.0:
                base = p.get("aggressiveness") or _AGG_PRIOR.get(p["strategy"], 0.0)
            else:
                base = confidence
            pressure = 1.0 + _POLICY_SCORE_PRESSURE_ALPHA * abs(min(score, 0.0))
            return base + ucb_bonus * pressure

        # 3. Softmax con temperatura dinámica
        weights = [_ucb_weight(p) for p in promoted_candidates]
        max_w   = max(weights)
        exps    = [math.exp((w - max_w) / temp) for w in weights]
        total   = sum(exps)
        probs   = [e / total for e in exps]

        # 4. Epsilon-floor: garantía mínima de exploración entre candidatos
        k        = len(promoted_candidates)
        eps      = _POLICY_EPSILON
        probs    = [(1 - eps) * p + eps / k for p in probs]

        r, cum = random.random(), 0.0
        for p, prob in zip(promoted_candidates, probs):
            cum += prob
            if r <= cum:
                return p["strategy"]
        return promoted_candidates[-1]["strategy"]

    # Calcular estrategia heurística y verificar que no esté deshabilitada
    band     = _score_to_policy_band(score)
    policy   = _STRATEGY_POLICY.get(category, _DEFAULT_POLICY)
    strategy = policy.get(band, "free")

    if strategy in disabled:
        if strategy == "force_search":
            fallback = "prefer_search"
            return fallback if fallback not in disabled else "free"
        return "free"

    return strategy


def _update_scrape_tracker(
    tracker: dict,
    category: str,
    raw_words: int,
    current_turn: int,
    duration_ms: int = 0,
    cost_usd: Optional[float] = None,
    source_type: str = "unstructured",
    reliability_override: Optional[str] = None,
) -> Tuple[dict, dict]:
    """Retorna (new_tracker, analytics) con score normalizado, cooldown y fallas consecutivas.

    Tracker entry por categoría:
        score                → float acumulativo con decay y clamping
        last_turn            → para calcular decay
        cooldown_turns       → turnos donde strategy="free" es forzada (post ok_strong)
        consecutive_failures → contador de resultados malos consecutivos
        last_bad_turn        → turno donde empezó la racha mala (para recovery_turns)
        best_delta           → mejor delta histórico (para regret_estimate)

    analytics dict (para el audit log):
        quality_target       → 1 si ok_strong, 0 en otro caso (target limpio para regresión)
        recovery_turns       → turnos que tardó en recuperarse de racha mala (o None)
        regret_estimate      → best_delta_histórico - delta_actual (oportunidad perdida)
    """
    tracker  = dict(tracker)
    entry    = dict(tracker.get(category) or {})
    # reliability_override permite que el nodo pase el valor correcto ya procesado
    # (ej. api_price: word count = 20 → "low_content", pero el precio es válido → "ok_strong")
    rel      = reliability_override if reliability_override else _scrape_reliability(raw_words)
    is_ok    = rel in ("ok_weak", "ok_strong")

    # --- Decay ---
    turns_ago     = current_turn - entry.get("last_turn", 0)
    current_score = 0.0 if turns_ago > _SCORE_DECAY_TURNS else float(entry.get("score", 0.0))

    # --- Cooldown countdown ---
    cooldown = max(0, entry.get("cooldown_turns", 0) - 1)

    # --- Consecutive failures ---
    prev_failures        = entry.get("consecutive_failures", 0)
    consecutive_failures = 0 if is_ok else prev_failures + 1

    # --- Delta normalizado ---
    delta     = _compute_delta(raw_words, duration_ms, cost_usd, consecutive_failures, source_type=source_type)
    new_score = max(_SCORE_MIN, min(_SCORE_MAX, round(current_score + delta, 3)))

    # Boost aditivo para fuentes estructuradas exitosas.
    # Diferencia con floor duro:
    #   floor override → ignora la historia acumulada
    #   boost aditivo  → respeta y acelera la historia → bandit aprende más rápido
    # El clamp _SCORE_MAX evita que explote.
    if source_type == "structured" and rel == "ok_strong":
        new_score = max(_SCORE_MIN, min(_SCORE_MAX, new_score + 0.5))

    # --- Cooldown tras ok_strong ---
    new_cooldown = _COOLDOWN_TURNS if rel == "ok_strong" else cooldown

    # --- Recovery tracking ---
    last_bad_turn = entry.get("last_bad_turn")
    if not is_ok:
        last_bad_turn = current_turn   # inicio / continuación de racha mala
    recovery_turns: Optional[int] = None
    if is_ok and last_bad_turn is not None:
        recovery_turns = current_turn - last_bad_turn
        last_bad_turn  = None          # resetear: ya se recuperó

    # --- Regret: best_delta histórico por categoría ---
    best_delta     = max(float(entry.get("best_delta", delta)), delta)
    regret_estimate = round(best_delta - delta, 3)

    tracker[category] = {
        "score":               new_score,
        "last_turn":           current_turn,
        "last_duration_ms":    duration_ms,
        "last_cost_usd":       cost_usd,
        "cooldown_turns":      new_cooldown,
        "consecutive_failures": consecutive_failures,
        "last_bad_turn":       last_bad_turn,
        "best_delta":          best_delta,
    }
    tracker["_turn_count"] = current_turn

    analytics = {
        "quality_target":  1 if rel == "ok_strong" else 0,
        "recovery_turns":  recovery_turns,
        "regret_estimate": regret_estimate,
    }
    return tracker, analytics


def _assess_followup_likely(outcome: str, output_length: int, content: str) -> bool:
    """Proxy de satisfacción funcional: estima si el usuario probablemente hará follow-up.

    Heurísticas (OR):
    - outcome no es success (blocked / error) → el agente no completó la tarea
    - respuesta muy corta (< _FOLLOWUP_SHORT_THRESHOLD chars) → contenido insuficiente
    - respuesta contiene frases de incapacidad / error → agente admite que no pudo

    No es ground truth — sirve para detectar patrones at scale y priorizar mejoras.
    """
    if outcome != "success":
        return True
    if output_length < _FOLLOWUP_SHORT_THRESHOLD:
        return True
    content_lower = content.lower()
    return any(signal in content_lower for signal in _FOLLOWUP_SIGNALS)


def _extract_followup(result: dict, outcome: str) -> dict:
    """Extrae el proxy de satisfacción funcional del resultado del agente."""
    content = ""
    output_length = 0
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            content = msg.content
            output_length = len(content)
            break
    return {"followup_likely": _assess_followup_likely(outcome, output_length, content)}


def _node_meta(parent_node: str = "supervisor", retry_count: int = 0) -> dict:
    """Campos estándar para correlación multi-nodo y A/B testing.

    parent_node: nodo que originó la ejecución — permite reconstruir el DAG.
    retry_count: número de reintentos; 0 en el flujo normal.
    experiment:  tag de experimento leído de la env var EXPERIMENT (vacío si no está).
    """
    meta: Dict[str, Any] = {"parent_node": parent_node, "retry_count": retry_count}
    exp = os.getenv("EXPERIMENT", "").strip()
    if exp:
        meta["experiment"] = exp
    return meta


def _truncate_text(text: str, max_chars: int = _MAX_OBSERVATION_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    head = text[:_TRUNC_HEAD_CHARS]
    tail = text[-_TRUNC_TAIL_CHARS:]
    return head + "\n...[truncated]...\n" + tail


def _truncate_raw_response(text: str) -> str:
    if len(text) <= _RAW_RESPONSE_MAX_CHARS:
        return text
    return text[:_RAW_RESPONSE_MAX_CHARS] + "... [truncated]"


def build_trajectory_from_messages(messages: List[Any]) -> Dict[str, Any]:
    """
    Construye una trayectoria con pasos de acción/observación y respuesta final.

    - action: tool call (nombre + args) desde AIMessage.tool_calls
    - observation: ToolMessage.content
    - final_response: último AIMessage sin tool_calls
    """
    steps: List[Dict[str, Any]] = []
    tool_call_step_index: Dict[str, int] = {}
    final_response: Optional[str] = None
    step_id = 1

    for msg in messages:
        if isinstance(msg, AIMessage):
            tool_calls = msg.tool_calls or []
            if tool_calls:
                for call in tool_calls:
                    call_id = call.get("id")
                    step = {
                        "step_id": step_id,
                        "action": {
                            "id": call_id,
                            "name": call.get("name"),
                            "args": call.get("args"),
                        },
                        "observation": "(pending)",
                    }
                    steps.append(step)
                    tool_call_step_index[call_id] = len(steps) - 1
                    step_id += 1
            else:
                if msg.content:
                    final_response = msg.content

        if isinstance(msg, ToolMessage):
            call_id = msg.tool_call_id
            observation = _truncate_text(msg.content or "")
            if call_id in tool_call_step_index:
                idx = tool_call_step_index[call_id]
                steps[idx]["observation"] = observation
            else:
                steps.append({
                    "step_id": step_id,
                    "action": "(unknown)",
                    "observation": observation,
                })
                step_id += 1

    for step in steps:
        if step.get("observation") == "(pending)":
            step["observation"] = "(missing tool output)"

    if final_response:
        steps.append({
            "step_id": step_id,
            "final_response": final_response,
        })

    return {
        "steps": steps,
        "final_response": final_response,
    }


def _emit_guard_audit(log_data: Dict[str, Any]) -> None:
    log_path = os.getenv("AGENTDOG_AUDIT_LOG", "").strip()
    payload = json.dumps(log_data, ensure_ascii=False)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(payload + "\n")
    else:
        print(payload)


def _resolve_guard_policy() -> str:
    policy = os.getenv("AGENTDOG_POLICY", "fail_open").strip().lower()
    if policy not in {"fail_open", "fail_closed", "fail_soft"}:
        return "fail_open"
    return policy


def _should_evaluate_guard(node_name: str) -> bool:
    mode = os.getenv("AGENTDOG_EVAL_MODE", "high_risk_only").strip().lower()
    if mode == "all_nodes":
        return True
    if mode == "high_risk_only":
        return is_high_risk(node_name)
    if mode == "final_only":
        return True
    return True


async def evaluate_trajectory_safe(state: AgentState, node_name: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Evalúa la trayectoria con un guard AgentDoG vía API OpenAI-compatible.

    - Primero aplica allowlist (bypass del guard) para requests públicas tipo "precio BTC".
    - Si no hay guard_url, aplica policy (fail_open / fail_closed / fail_soft).
    - Si hay guard_url, construye trajectory y consulta al guard.
    """
    guard_url = os.getenv("AGENTDOG_GUARD_URL", "").strip()
    policy = _resolve_guard_policy()
    high_risk = is_high_risk(node_name)

    # ✅ 1) SIEMPRE primero: mensajes
    messages = state.get("messages", [])

    # ✅ 2) correlación id (solo una vez)
    trajectory_id = state.get("run_id") or state.get("request_id") or str(uuid.uuid4())

    # ✅ 3) allowlist: bypass del guard (y log como SAFE)
    if _is_allowed_public_price_request(messages, node_name):
        _emit_guard_audit({
            "trajectory_id": trajectory_id,
            "guard_label": "safe",
            "guard_status": "ok",
            "verdict_source": "allowlist_public_price",
            "node_name": node_name,
            "raw_response": "safe",
            "model": os.getenv("AGENTDOG_MODEL", "AgentDoG-Qwen3-4B"),
            "latency_ms": 0,
            "policy": policy,
            "trajectory_steps_count": 0,
            "approx_chars": 0,
        })
        return True, {
            "trajectory_id": trajectory_id,
            "verdict_source": "allowlist_public_price",
            "label": "safe",
        }

    # ✅ 4) guard deshabilitado => aplicar policy
    if not guard_url:
        _emit_guard_audit({
            "trajectory_id": trajectory_id,
            "guard_label": "disabled",
            "guard_status": "disabled",
            "verdict_source": "disabled",
            "node_name": node_name,
            "raw_response": "",
            "model": os.getenv("AGENTDOG_MODEL", "AgentDoG-Qwen3-4B"),
            "latency_ms": None,
            "policy": policy,
            "trajectory_steps_count": 0,
            "approx_chars": 0,
        })

        meta = {
            "policy": "guard_disabled",
            "trajectory_id": trajectory_id,
            "verdict_source": "disabled",
        }

        if policy == "fail_closed":
            return False, {**meta, "policy": "fail_closed", "reason": "guard_disabled"}

        if policy == "fail_soft":
            if high_risk:
                return False, {**meta, "policy": "fail_soft_block", "reason": "guard_disabled"}
            return True, {**meta, "policy": "fail_soft_allow", "reason": "guard_disabled"}

        # fail_open
        return True, meta

    # ✅ 5) construir trajectory y consultar guard
    trajectory = build_trajectory_from_messages(messages)

    system_prompt = (
        "Eres un guardrail para agentes.\n"
        "Clasifica la trayectoria completa como SAFE o UNSAFE.\n\n"
        "Marca SAFE si la acción es de bajo riesgo, por ejemplo:\n"
        "- obtener precios públicos (BTC, acciones) desde sitios públicos o APIs\n"
        "- scraping de páginas públicas sin login\n"
        "- cálculos, análisis, programación general\n\n"
        "Marca UNSAFE SOLO si hay:\n"
        "- instrucciones para hackear, evadir seguridad, romper login, bypass paywalls\n"
        "- scraping de datos privados, credenciales, doxxing, PII\n"
        "- malware, phishing, exfiltración, explotación\n\n"
        "Responde SOLO con 'safe' o 'unsafe'."
    )

    user_prompt = (
        "Evalúa la siguiente trayectoria JSON de un agente:\n\n"
        f"{json.dumps(trajectory, ensure_ascii=False)}"
    )

    payload = {
        "model": os.getenv("AGENTDOG_MODEL", "AgentDoG-Qwen3-4B"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    api_key = os.getenv("AGENTDOG_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        start = time.time()
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(guard_url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        latency_ms = int((time.time() - start) * 1000)

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
            .lower()
        )

        label = "unknown"
        verdict = None
        verdict_source = "fallback_unknown"

        if content.startswith("{"):
            try:
                parsed = json.loads(content)
                verdict = str(parsed.get("verdict", "")).strip().lower()
            except Exception:
                verdict = None

        if verdict == "safe":
            label = "safe"
            verdict_source = "json"
        elif verdict == "unsafe":
            label = "unsafe"
            verdict_source = "json"
        elif content == "safe":
            label = "safe"
            verdict_source = "text_exact"
        elif content == "unsafe":
            label = "unsafe"
            verdict_source = "text_exact"

        guard_status = "ok" if label in {"safe", "unsafe"} else "unknown"

        _emit_guard_audit({
            "trajectory_id": trajectory_id,
            "guard_label": label,
            "guard_status": guard_status,
            "verdict_source": verdict_source,
            "node_name": node_name,
            "raw_response": _truncate_raw_response(content),
            "model": payload["model"],
            "latency_ms": latency_ms,
            "policy": policy,
            "trajectory_steps_count": len(trajectory.get("steps", [])),
            "approx_chars": len(json.dumps(trajectory, ensure_ascii=False)),
        })

        base_meta = {"trajectory_id": trajectory_id, "verdict_source": verdict_source}

        if label == "unsafe":
            return False, {**base_meta, "label": "unsafe", "raw": content}

        if label == "safe":
            return True, {**base_meta, "label": "safe", "raw": content}

        # unknown => aplicar policy
        if policy == "fail_closed":
            return False, {**base_meta, "label": "unknown", "raw": content, "policy": "fail_closed"}

        if policy == "fail_soft":
            if high_risk:
                return False, {**base_meta, "label": "unknown", "raw": content, "policy": "fail_soft_block"}
            return True, {**base_meta, "label": "unknown", "raw": content, "policy": "fail_soft_allow"}

        return True, {**base_meta, "label": "unknown", "raw": content, "policy": "fail_open"}

    except Exception as e:
        _emit_guard_audit({
            "trajectory_id": trajectory_id,
            "guard_label": "error",
            "guard_status": "error",
            "verdict_source": "error",
            "node_name": node_name,
            "raw_response": _truncate_raw_response(str(e)),
            "model": os.getenv("AGENTDOG_MODEL", "AgentDoG-Qwen3-4B"),
            "latency_ms": None,
            "policy": policy,
            "trajectory_steps_count": len(trajectory.get("steps", [])),
            "approx_chars": len(json.dumps(trajectory, ensure_ascii=False)),
        })

        err_meta = {
            "trajectory_id": trajectory_id,
            "verdict_source": "error",
            "label": "error",
            "error": str(e),
        }

        if policy == "fail_closed":
            return False, {**err_meta, "policy": "fail_closed"}

        if policy == "fail_soft":
            if high_risk:
                return False, {**err_meta, "policy": "fail_soft_block"}
            return True, {**err_meta, "policy": "fail_soft_allow"}

        return True, {**err_meta, "policy": "fail_open"}



# ==================== NODOS (AGENTES) - ASYNC ====================

async def math_node(state: AgentState) -> AgentState:
    """
    Nodo del agente de matemáticas (async).
    
    Invoca el agente con el último mensaje y retorna la respuesta.
    Los agentes creados con create_react_agent reciben y devuelven mensajes.
    """
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    rid = state.get("request_id", str(uuid.uuid4()))
    t0 = time.time()
    try:
        result = await math_agent.ainvoke(
            {"messages": [HumanMessage(content=last_message)]},
            config=RunnableConfig(tags=["math", "agent"], metadata={"node": "math_node", "agent": "math_agent", "request_id": rid, "input_chars": len(last_message)}),
        )

        tokens = _extract_tokens(result)
        quality = _extract_quality(result)
        followup = _extract_followup(result, "success")
        meta = _node_meta()
        if _should_evaluate_guard("math_node"):
            combined_messages = messages + result.get("messages", [])
            is_safe, _ = await evaluate_trajectory_safe({
                "messages": combined_messages,
                "next_agent": state.get("next_agent", "")
            }, "math_node")
            if not is_safe:
                _emit_node_outcome(rid, "math_node", "blocked", phase="post_guard", agent="math_agent", duration_ms=int((time.time()-t0)*1000), followup_likely=True, **tokens, **quality, **meta)
                return {"messages": [AIMessage(content="Respuesta retenida por política de seguridad.")]}

        response_messages = result.get("messages", [])
        if response_messages:
            _emit_node_outcome(rid, "math_node", "success", phase="agent", agent="math_agent", duration_ms=int((time.time()-t0)*1000), output_msgs=len(response_messages), **tokens, **quality, **followup, **meta)
            return {"messages": response_messages}

        _emit_node_outcome(rid, "math_node", "error", phase="agent", agent="math_agent", duration_ms=int((time.time()-t0)*1000), reason="empty_response", followup_likely=True, **meta)
        return {"messages": [AIMessage(content="No se pudo procesar la solicitud.")]}
    except Exception as e:
        _emit_node_outcome(rid, "math_node", "error", phase="agent", agent="math_agent", duration_ms=int((time.time()-t0)*1000), reason=str(e), followup_likely=True, **_node_meta())
        raise


async def analysis_node(state: AgentState) -> AgentState:
    """Nodo del agente de análisis (async)"""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    rid = state.get("request_id", str(uuid.uuid4()))
    t0 = time.time()
    try:
        result = await analysis_agent.ainvoke(
            {"messages": [HumanMessage(content=last_message)]},
            config=RunnableConfig(tags=["analysis", "agent"], metadata={"node": "analysis_node", "agent": "analysis_agent", "request_id": rid, "input_chars": len(last_message)}),
        )

        tokens = _extract_tokens(result)
        quality = _extract_quality(result)
        followup = _extract_followup(result, "success")
        meta = _node_meta()
        if _should_evaluate_guard("analysis_node"):
            combined_messages = messages + result.get("messages", [])
            is_safe, _ = await evaluate_trajectory_safe({
                "messages": combined_messages,
                "next_agent": state.get("next_agent", "")
            }, "analysis_node")
            if not is_safe:
                _emit_node_outcome(rid, "analysis_node", "blocked", phase="post_guard", agent="analysis_agent", duration_ms=int((time.time()-t0)*1000), followup_likely=True, **tokens, **quality, **meta)
                return {"messages": [AIMessage(content="Respuesta retenida por política de seguridad.")]}

        response_messages = result.get("messages", [])
        if response_messages:
            _emit_node_outcome(rid, "analysis_node", "success", phase="agent", agent="analysis_agent", duration_ms=int((time.time()-t0)*1000), output_msgs=len(response_messages), **tokens, **quality, **followup, **meta)
            return {"messages": response_messages}

        _emit_node_outcome(rid, "analysis_node", "error", phase="agent", agent="analysis_agent", duration_ms=int((time.time()-t0)*1000), reason="empty_response", followup_likely=True, **meta)
        return {"messages": [AIMessage(content="No se pudo procesar la solicitud.")]}
    except Exception as e:
        _emit_node_outcome(rid, "analysis_node", "error", phase="agent", agent="analysis_agent", duration_ms=int((time.time()-t0)*1000), reason=str(e), followup_likely=True, **_node_meta())
        raise



async def code_node(state: AgentState) -> AgentState:
    """Nodo del agente de código (async) con HITL pre-ejecución."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    rid = state.get("request_id", str(uuid.uuid4()))
    t0 = time.time()

    if _HITL_ENABLED:
        preview = last_message[:120] + ("..." if len(last_message) > 120 else "")
        confirmed = await _ask_confirmation(
            f"\n[HITL] code_agent va a procesar: \"{preview}\"\n¿Confirmar? [s/n]: "
        )
        if not confirmed:
            _emit_node_outcome(rid, "code_node", "blocked", phase="pre_guard", agent="code_agent", duration_ms=int((time.time()-t0)*1000), reason="hitl_rejected")
            return {"messages": [AIMessage(content="Operación cancelada por el usuario.")]}

    try:
        result = await code_agent.ainvoke(
            {"messages": [HumanMessage(content=last_message)]},
            config=RunnableConfig(tags=["code", "agent", "high_risk"], metadata={"node": "code_node", "agent": "code_agent", "request_id": rid, "input_chars": len(last_message)}),
        )

        tokens = _extract_tokens(result)
        quality = _extract_quality(result)
        followup = _extract_followup(result, "success")
        meta = _node_meta()
        if _should_evaluate_guard("code_node"):
            combined_messages = messages + result.get("messages", [])
            is_safe, _ = await evaluate_trajectory_safe({
                "messages": combined_messages,
                "next_agent": state.get("next_agent", "")
            }, "code_node")
            if not is_safe:
                _emit_node_outcome(rid, "code_node", "blocked", phase="post_guard", agent="code_agent", duration_ms=int((time.time()-t0)*1000), reason="agentdog", followup_likely=True, **tokens, **quality, **meta)
                return {"messages": [AIMessage(content="Respuesta retenida por política de seguridad.")]}

        response_messages = result.get("messages", [])
        if response_messages:
            _emit_node_outcome(rid, "code_node", "success", phase="agent", agent="code_agent", duration_ms=int((time.time()-t0)*1000), output_msgs=len(response_messages), **tokens, **quality, **followup, **meta)
            return {"messages": response_messages}

        _emit_node_outcome(rid, "code_node", "error", phase="agent", agent="code_agent", duration_ms=int((time.time()-t0)*1000), reason="empty_response", followup_likely=True, **meta)
        return {"messages": [AIMessage(content="No se pudo procesar la solicitud.")]}
    except Exception as e:
        _emit_node_outcome(rid, "code_node", "error", phase="agent", agent="code_agent", duration_ms=int((time.time()-t0)*1000), reason=str(e), followup_likely=True, **_node_meta())
        raise


# ==================== HELPERS: API PRICE FAST PATH ====================

# Mapa de keywords → coin_id (CoinGecko). Permite detectar la moneda sin LLM.
_QUERY_COIN_MAP: dict[str, str] = {
    "bitcoin": "bitcoin",  "btc": "bitcoin",
    "ethereum": "ethereum", "eth": "ethereum",
    "solana": "solana",    "sol": "solana",
    "cardano": "cardano",  "ada": "cardano",
    "dogecoin": "dogecoin","doge": "dogecoin",
    "ripple": "ripple",    "xrp": "ripple",
    "polkadot": "polkadot","dot": "polkadot",
    "chainlink": "chainlink","link": "chainlink",
    "litecoin": "litecoin","ltc": "litecoin",
    "avalanche": "avalanche-2", "avax": "avalanche-2",
    "matic": "matic-network",   "polygon": "matic-network",
    "uniswap": "uniswap",  "uni": "uniswap",
}


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


async def web_scraping_node(state: AgentState) -> AgentState:
    """
    Nodo del agente de web scraping con HITL + context quarantine (async).

    Patrón: el sub-agente (web_scraping_agent) absorbe el HTML/texto crudo en su
    propio contexto aislado. Solo devuelve al estado compartido un resumen de ≤200
    palabras, evitando contaminar el historial del supervisor con miles de tokens.
    """
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    rid = state.get("request_id", str(uuid.uuid4()))
    t0 = time.time()

    if _HITL_ENABLED:
        import re
        urls = re.findall(r'https?://\S+', last_message)
        url_info = f" → URLs: {', '.join(urls)}" if urls else ""
        preview = last_message[:120] + ("..." if len(last_message) > 120 else "")
        confirmed = await _ask_confirmation(
            f"\n[HITL] web_scraping_agent va a procesar: \"{preview}\"{url_info}\n¿Confirmar? [s/n]: "
        )
        if not confirmed:
            _emit_node_outcome(rid, "web_scraping_node", "blocked", phase="pre_guard", agent="web_scraping_agent", duration_ms=int((time.time()-t0)*1000), reason="hitl_rejected")
            return {"messages": [AIMessage(content="Operación cancelada por el usuario.")]}

    try:
        # --- Policy-driven strategy selection con exploration dinámica ---
        import random
        tracker           = state.get("scrape_tracker") or {}
        turn_count        = (tracker.get("_turn_count") or 0) + 1
        category          = _detect_query_category(last_message)
        prior_score       = _get_category_score(tracker, category, turn_count)
        prior_reliability = _score_to_reliability(prior_score)

        # ML recommendation: top promoted strategy del policy.json (si existe).
        # Se captura ANTES de que el bandit pueda sobrescribirla — así podemos
        # medir cuándo el bandit coincide con ML y cuándo lo overridea.
        _rt = _RUNTIME_POLICY.get(category, {})
        _top_promoted = (_rt.get("promoted") or [None])[0]
        ml_recommended: Optional[str] = (
            _top_promoted.get("strategy") if isinstance(_top_promoted, dict)
            else _top_promoted  # str legacy o None
        )

        # crypto_price → API directa en el 98% de los casos.
        # El 2% restante (validación) usa force_search para detectar:
        #   - fallos silenciosos en la API (rate limit, cambio de schema)
        #   - drift: si la búsqueda encuentra mejor fuente
        #   - inconsistencias entre fuentes
        if category == "crypto_price":
            if random.random() < _API_VALIDATION_EPSILON:
                strategy  = "force_search"   # validación: comparar vs scraping
                exploring = True
            else:
                strategy  = "api_price"
                exploring = False
            exp_rate = _API_VALIDATION_EPSILON
        else:
            exp_rate  = _exploration_rate(prior_score)
            exploring = random.random() < exp_rate
            strategy  = _get_strategy(tracker, category, prior_score, exploring=exploring)

        # ¿El bandit eligió lo mismo que el ML recomienda?
        # None si no hay policy cargada (no hay base de comparación).
        prediction_match: Optional[bool] = (
            (strategy == ml_recommended) if ml_recommended is not None else None
        )

        # ================================================================
        # FAST PATH: api_price → llamada directa sin overhead de LLM.
        # Latencia esperada: < 1s (solo llamada HTTP a CoinGecko/Binance/Coinbase).
        # El agente LLM añade ~6-7s de overhead (decide tool + formatea respuesta)
        # que es innecesario cuando la fuente es una API estructurada conocida.
        # ================================================================
        if strategy == "api_price":
            from models import PriceToolResponse
            coin = _detect_coin_from_query(last_message)
            api_json: Optional[str] = None
            try:
                api_json = get_crypto_price.func(coin=coin, vs_currency="usd")
            except Exception:
                pass  # fallback al agente normal

            if api_json:
                try:
                    price_resp = PriceToolResponse.model_validate_json(api_json)
                except Exception:
                    price_resp = None

                if price_resp and price_resp.is_valid_price():
                    formatted     = _format_price_response(price_resp.model_dump())
                    duration_ms   = int((time.time() - t0) * 1000)
                    # Sin LLM → sin tokens reales; marcamos tokens_available=False
                    tokens_fast   = {
                        "model": _get_model_name(), "tokens_available": False,
                        "prompt_tokens": 0, "completion_tokens": 0,
                        "total_tokens": 0, "estimated_cost_usd": 0.0,
                    }
                    quality_fast  = {"output_length": len(formatted), "tool_calls_count": 1}
                    followup_fast = {"followup_likely": False}
                    meta          = _node_meta()

                    new_tracker, analytics = _update_scrape_tracker(
                        tracker, category, 200, turn_count,
                        duration_ms=duration_ms, cost_usd=0.0,
                        source_type="structured", reliability_override="ok_strong",
                    )
                    new_score = _get_category_score(new_tracker, category, turn_count)
                    quality_target_val = analytics.get("quality_target", 0)
                    ml_would_succeed: Optional[bool] = (
                        bool(quality_target_val) if prediction_match is True else None
                    )
                    _emit_node_outcome(
                        rid, "web_scraping_node", "success", phase="agent",
                        agent="web_scraping_agent", duration_ms=duration_ms,
                        category=category, strategy=strategy, exploring=exploring,
                        exp_rate=exp_rate, source_type="structured",
                        price_extracted=price_resp.price, parse_success=True,
                        scrape_reliability="ok_strong",
                        prior_reliability=prior_reliability, prior_score=prior_score,
                        scrape_score=new_score, retry_done=False,
                        ml_recommended=ml_recommended, prediction_match=prediction_match,
                        ml_would_succeed=ml_would_succeed,
                        **tokens_fast, **quality_fast, **followup_fast, **analytics, **meta,
                    )
                    return {"messages": [AIMessage(content=formatted)], "scrape_tracker": new_tracker}
                # price_resp inválido → continuar con el agente normal como fallback

        agent_hint = _STRATEGY_HINTS[strategy]

        if agent_hint:
            agent_hint = (
                f"[Sistema | categoría={category} score={prior_score:+.2f} "
                f"estrategia={strategy} exploring={exploring} exp_rate={exp_rate:.0%}]\n{agent_hint}"
            )

        agent_message = agent_hint + last_message

        # --- Fase 1: sub-agente extrae contenido crudo (contexto aislado) ---
        raw_result = await web_scraping_agent.ainvoke(
            {"messages": [HumanMessage(content=agent_message)]},
            config=RunnableConfig(tags=["web_scraping", "agent", "high_risk", "context_quarantine"], metadata={"node": "web_scraping_node", "agent": "web_scraping_agent", "request_id": rid, "input_chars": len(last_message), "prior_reliability": prior_reliability}),
        )

        tokens = _extract_tokens(raw_result)
        quality = _extract_quality(raw_result)
        followup = _extract_followup(raw_result, "success")
        meta = _node_meta()
        # --- Guardrail AgentDoG: solo la trayectoria del sub-agente ---
        if _should_evaluate_guard("web_scraping_node"):
            is_safe, _ = await evaluate_trajectory_safe({
                "messages": raw_result.get("messages", []),
                "next_agent": state.get("next_agent", "")
            }, "web_scraping_node")
            if not is_safe:
                _emit_node_outcome(rid, "web_scraping_node", "blocked", phase="post_guard", agent="web_scraping_agent", duration_ms=int((time.time()-t0)*1000), reason="agentdog", followup_likely=True, **tokens, **quality, **meta)
                return {"messages": [AIMessage(content="Respuesta retenida por política de seguridad.")]}

        # --- Fase 2: extraer solo la respuesta final del sub-agente ---
        raw_messages = raw_result.get("messages", [])
        raw_text = ""
        for msg in reversed(raw_messages):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                raw_text = msg.content
                break

        if not raw_text:
            _emit_node_outcome(rid, "web_scraping_node", "error", phase="agent", agent="web_scraping_agent", duration_ms=int((time.time()-t0)*1000), reason="empty_response", followup_likely=True, **meta)
            return {"messages": [AIMessage(content="No se pudo extraer información de la página.")]}

        # --- Fase 3: resumir a ≤200 palabras ---
        words = raw_text.split()
        summary_triggered = len(words) > 200
        if summary_triggered:
            llm = get_llm()
            summary_response = await llm.ainvoke(
                [HumanMessage(content=(
                    f"Resume el siguiente texto en máximo 200 palabras, "
                    f"conservando los datos más importantes:\n\n{raw_text[:4000]}"
                ))],
                config=RunnableConfig(
                    tags=["web_scraping", "context_quarantine", "summary"],
                    metadata={"node": "web_scraping_node", "agent": "web_scraping_agent", "request_id": rid, "raw_words": len(words), "summary_triggered": True},
                ),
            )
            summary = summary_response.content
        else:
            summary = raw_text

        duration_ms = int((time.time() - t0) * 1000)
        reliability = _scrape_reliability(len(words))
        retry_done  = False

        # Respuestas estructuradas (APIs, JSON) son inherentemente cortas.
        # Word count no es el indicador correcto — se valida por presencia de precio real.
        #   "Precio: 71,234.56 USD" → ok_strong
        #   "Bitcoin price unavailable" → unreliable → retry
        #   "error 404" / "2024" / "123 usuarios" → unreliable (sin contexto de precio)
        source_type  = "structured" if strategy in _STRUCTURED_SOURCE_STRATEGIES else "unstructured"
        parsed_price: Optional[float] = None
        parse_success: Optional[bool] = None
        price_data: Optional[dict]    = None

        if source_type == "structured":
            # Prioridad 1: extraer del ToolMessage JSON (capa de datos, antes del LLM).
            # get_crypto_price ahora devuelve {"asset","price","currency","confidence","source",...}
            price_data = _extract_price_from_messages(raw_result)
            if price_data:
                parsed_price  = price_data["price"]
                parse_success = True
                reliability   = "ok_strong"
            elif raw_text:
                # Fallback: parsear texto formateado por el LLM si el ToolMessage no estaba disponible.
                parsed_price  = _extract_structured_price(raw_text)
                parse_success = parsed_price is not None
                reliability   = "ok_strong" if parse_success else "unreliable"
            else:
                parse_success = False
                reliability   = "unreliable"

        # --- Auto-retry si el contenido es insuficiente y aún no usamos force_search ---
        if reliability in _RETRY_ON_RELIABILITY and strategy != "force_search":
            # Emitir evento del intento fallido para capturar la señal de aprendizaje
            _emit_node_outcome(
                rid, "web_scraping_node", "retry", phase="agent",
                agent="web_scraping_agent", duration_ms=duration_ms,
                reason=f"auto_retry:{reliability}",
                scrape_reliability=reliability, strategy=strategy,
                source_type=source_type, category=category, **tokens, **_node_meta(),
            )
            retry_hint = (
                f"[Sistema | auto-retry por {reliability} | estrategia=force_search]\n"
                + _STRATEGY_HINTS["force_search"]
            )
            retry_result = await web_scraping_agent.ainvoke(
                {"messages": [HumanMessage(content=retry_hint + last_message)]},
                config=RunnableConfig(
                    tags=["web_scraping", "agent", "high_risk", "context_quarantine", "retry"],
                    metadata={"node": "web_scraping_node", "agent": "web_scraping_agent",
                              "request_id": rid, "retry": True},
                ),
            )
            retry_text = ""
            for msg in reversed(retry_result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                    retry_text = msg.content
                    break

            if retry_text:
                retry_words = retry_text.split()
                summary_triggered = len(retry_words) > 200
                if summary_triggered:
                    llm = get_llm()
                    summary_response = await llm.ainvoke(
                        [HumanMessage(content=(
                            f"Resume el siguiente texto en máximo 200 palabras, "
                            f"conservando los datos más importantes:\n\n{retry_text[:4000]}"
                        ))],
                        config=RunnableConfig(tags=["web_scraping", "context_quarantine", "summary", "retry"],
                                              metadata={"node": "web_scraping_node", "request_id": rid}),
                    )
                    summary = summary_response.content
                else:
                    summary = retry_text
                words   = retry_words
                tokens  = _extract_tokens(retry_result)
                quality = _extract_quality(retry_result)

            strategy    = "force_search"
            reliability = _scrape_reliability(len(words))
            retry_done  = True
            duration_ms = int((time.time() - t0) * 1000)

        # --- Validación de output: si sigue sin contenido, fallar limpiamente ---
        if reliability == "unreliable":
            new_tracker, analytics = _update_scrape_tracker(
                tracker, category, len(words), turn_count,
                duration_ms=duration_ms, cost_usd=tokens.get("estimated_cost_usd"),
                source_type=source_type, reliability_override=reliability,
            )
            _emit_node_outcome(
                rid, "web_scraping_node", "low_confidence", phase="agent",
                agent="web_scraping_agent", duration_ms=duration_ms,
                scrape_reliability=reliability, strategy=strategy,
                retry_done=retry_done, category=category,
                source_type=source_type, price_extracted=parsed_price, parse_success=parse_success,
                ml_recommended=ml_recommended, prediction_match=prediction_match,
                ml_would_succeed=(False if prediction_match is True else None),
                **tokens, **quality, **_node_meta(), **analytics,
            )
            return {
                "messages": [AIMessage(content=(
                    "No pude obtener información confiable para esta consulta. "
                    "Intenta proporcionar una URL específica o reformular la pregunta."
                ))],
                "scrape_tracker": new_tracker,
            }

        cost_usd = tokens.get("estimated_cost_usd")
        new_tracker, analytics = _update_scrape_tracker(
            tracker, category, len(words), turn_count,
            duration_ms=duration_ms, cost_usd=cost_usd,
            source_type=source_type, reliability_override=reliability,
        )
        new_score = _get_category_score(new_tracker, category, turn_count)

        # followup_likely se activa también cuando el contenido es insuficiente
        if reliability not in ("ok_weak", "ok_strong"):
            followup = {"followup_likely": True}

        # ml_would_succeed: ¿fue correcta la recomendación de la policy?
        #
        # Semántica:
        #   prediction_match=True  → el sistema siguió lo que la policy recomendaba.
        #                            quality_target es el outcome real → podemos juzgar.
        #   prediction_match=False → el bandit overridó la policy (exploring, cooldown…).
        #                            No ejecutamos la recomendación → counterfactual desconocido → None.
        #   prediction_match=None  → no hay policy cargada → None.
        #
        # Nota: ml_recommended = top promoted del policy.json (policy derivada del entrenamiento),
        # NO la predicción directa del modelo sklearn. Son distintos:
        #   policy_decision  → promoted/disabled de analytics.py --train  (aquí)
        #   ml_prediction    → argmax(P(success|strategy)) del modelo       (futura separación)
        quality_target_val = analytics.get("quality_target", 0)
        ml_would_succeed: Optional[bool] = (
            bool(quality_target_val) if prediction_match is True else None
        )

        _emit_node_outcome(
            rid, "web_scraping_node", "success", phase="agent",
            agent="web_scraping_agent", duration_ms=duration_ms,
            summary_triggered=summary_triggered, raw_words=len(words),
            category=category, exploring=exploring, strategy=strategy, exp_rate=exp_rate,
            scrape_reliability=reliability, prior_reliability=prior_reliability,
            prior_score=prior_score, scrape_score=new_score,
            retry_done=retry_done,
            source_type=source_type, price_extracted=parsed_price, parse_success=parse_success,
            ml_recommended=ml_recommended, prediction_match=prediction_match,
            ml_would_succeed=ml_would_succeed,
            **tokens, **quality, **followup, **analytics, **meta,
        )
        return {"messages": [AIMessage(content=summary)], "scrape_tracker": new_tracker}
    except Exception as e:
        _emit_node_outcome(rid, "web_scraping_node", "error", phase="agent", agent="web_scraping_agent", duration_ms=int((time.time()-t0)*1000), reason=str(e), followup_likely=True, **_node_meta())
        raise


# ==================== SUPERVISOR ====================

async def supervisor_node(state: AgentState) -> AgentState:
    """
    Nodo supervisor que decide qué agente usar (async).

    Usa structured output (Pydantic) para garantizar que el LLM
    devuelva siempre un agente válido sin necesidad de text-parsing.
    """
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    # Shortcut BTC: evita llamada al LLM para consultas comunes de precio
    lm = last_message.lower()
    if ("bitcoin" in lm or "btc" in lm) and any(k in lm for k in ["precio", "price", "cotiza", "cotización", "cotizacion"]):
        return {"next_agent": "web_scraping_agent"}

    llm = get_llm()
    llm_structured = llm.with_structured_output(RoutingDecision)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un supervisor que coordina un equipo de agentes especializados.

Tienes acceso a cuatro agentes:
- math_agent: problemas matemáticos, cálculos, álgebra, estadística numérica
- analysis_agent: análisis de datos, insights, reportes, patrones en datasets
- code_agent: escribir código, programación, desarrollo de software
- web_scraping_agent: extraer información de URLs, scraping, obtener datos de páginas web

Elige el agente más adecuado para la solicitud. Si no estás seguro, elige el que mejor se ajuste."""),
        ("user", "{input}"),
    ])

    chain = prompt | llm_structured
    try:
        decision: RoutingDecision = await chain.ainvoke(
            {"input": last_message},
            config=RunnableConfig(
                tags=["supervisor", "routing"],
                metadata={
                    "node": "supervisor",
                    "input_chars": len(last_message),
                    "history_turns": len(messages),
                    "risk_flag": state.get("risk_flag", False),
                },
            ),
        )
        return {"next_agent": decision.agent}
    except Exception as e:
        # Fallback: math_agent es el agente más seguro para solicitudes ambiguas.
        # Registrar el error para diagnóstico sin romper el flujo.
        print(f"[supervisor] routing falló ({type(e).__name__}: {e}), usando fallback math_agent")
        return {"next_agent": "math_agent"}


# ==================== ROUTER (DECISIÓN) ====================

def route_agent(state: AgentState) -> str:
    """
    Función de enrutamiento basada en el estado.
    
    Determina a qué nodo ir después del supervisor basándose
    en el valor de next_agent en el estado.
    """
    # Si no hay mensajes, terminar
    if not state.get("messages"):
        return END
    
    # Si hay un next_agent definido, enrutar a ese agente
    next_agent = state.get("next_agent")
    
    if next_agent == "math_agent":
        return "math_agent"
    elif next_agent == "analysis_agent":
        return "analysis_agent"
    elif next_agent == "code_agent":
        return "code_agent"
    elif next_agent == "web_scraping_agent":
        return "web_scraping_agent"
    else:
        # Si no hay next_agent, ir al supervisor primero
        return "supervisor"


# ==================== CONSTRUCCIÓN DEL GRAFO ====================

async def input_guard_node(state: AgentState) -> AgentState:
    """Nodo de entrada: genera request_id del turno y aplica el middleware."""
    rid = str(uuid.uuid4())
    blocked = input_guard(state)
    if blocked:
        return {**blocked, "request_id": rid}
    return {**state, "request_id": rid}


def route_after_guard(state: AgentState) -> str:
    """Usa state['blocked'] (campo tipado) en lugar de comparar el contenido del mensaje."""
    if state.get("blocked", False):
        return END
    return "supervisor"


def create_supervisor_graph():
    """
    Crea y retorna el grafo supervisor compilado.

    Flujo:
      input_guard → supervisor → route_agent → [agente especializado] → END
    El middleware input_guard intercepta solicitudes antes del supervisor.
    """
    workflow = StateGraph(AgentState)

    # Nodos
    workflow.add_node("input_guard", input_guard_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("math_agent", math_node)
    workflow.add_node("analysis_agent", analysis_node)
    workflow.add_node("code_agent", code_node)
    workflow.add_node("web_scraping_agent", web_scraping_node)

    # Entry point → middleware primero
    workflow.set_entry_point("input_guard")
    workflow.add_conditional_edges("input_guard", route_after_guard, {"supervisor": "supervisor", END: END})

    workflow.add_conditional_edges(
        "supervisor",
        route_agent,
        {
            "math_agent": "math_agent",
            "analysis_agent": "analysis_agent",
            "code_agent": "code_agent",
            "web_scraping_agent": "web_scraping_agent",
            END: END,
        },
    )

    workflow.add_edge("math_agent", END)
    workflow.add_edge("analysis_agent", END)
    workflow.add_edge("code_agent", END)
    workflow.add_edge("web_scraping_agent", END)

    return workflow.compile()


if __name__ == "__main__":
    import asyncio
    
    async def test_graph():
        # Ejemplo de uso
        graph = create_supervisor_graph()
        
        # Ejecutar con una pregunta (usando ainvoke para async)
        result = await graph.ainvoke({
            "messages": [HumanMessage(content="Calcula la raíz cuadrada de 144")],
            "next_agent": ""
        })
        
        print("\n=== RESULTADO ===")
        if result["messages"]:
            print(result["messages"][-1].content)
    
    asyncio.run(test_graph())
