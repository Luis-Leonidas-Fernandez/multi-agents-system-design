"""
Tracker de estrategia de scraping con bandit UCB + softmax + epsilon-floor.

Solo usa stdlib de Python — sin dependencias externas.
La política de runtime (policy.json) se carga de forma lazy para evitar
I/O en el momento del import.
"""
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple


# ==================== CONSTANTES DE SCORE ====================

_SCORE_DECAY_TURNS      = 3      # turnos sin actividad antes de resetear a 0
_SCORE_MAX              =  3.0
_SCORE_MIN              = -3.0
# Normalizadores para el delta continuo (ver _compute_delta)
_QUALITY_NORM_WORDS     = 120    # raw_words / 120, capped 1.0 → quality_score ∈ [0, 1]
_LATENCY_NORM_MS        = 15000  # duration_ms / 15000, capped 1.0 → latency ∈ [0, 1]
_COST_NORM_USD          = 0.001  # cost_usd / 0.001, capped 1.0 → cost ∈ [0, 1]
_W_LATENCY              = 0.5    # peso de la penalización de latencia en el delta
_W_COST                 = 0.5    # peso de la penalización de costo en el delta
_CONSECUTIVE_FAIL_EXTRA = -0.5   # penalización extra cuando hay ≥ 2 fallas consecutivas
_COOLDOWN_TURNS         = 2      # turnos con "free" forzado después de ok_strong

# ==================== CATEGORÍAS Y POLICIES ====================

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
        "very_low":  "force_search",   # score <= -2
        "low":       "prefer_search",  # -2 < score < 0
        "neutral":   "free",           # score >= 0
    },
    "finance": {
        "very_low":  "force_search",
        "low":       "prefer_search",
        "neutral":   "free",
    },
    "news": {
        "very_low":  "prefer_search",  # news: menos agresivo — scraping puede recuperarse
        "low":       "free",
        "neutral":   "free",
    },
    # "general" y "weather" usan el fallback
}
_DEFAULT_POLICY = {"very_low": "force_search", "low": "prefer_search", "neutral": "free"}

# Confianza mínima para respetar una estrategia promoted del policy.json.
_POLICY_MIN_CONFIDENCE = 0.50

# Cuántas estrategias promoted considerar como candidatas (top-k).
_POLICY_TOP_K = 2

# Tiempo de vida de la confianza: exp(-Δt / τ). Con τ=7 días:
#   1 día  → decay 0.87   (pequeño impacto)
#   7 días → decay 0.37   (confianza a la mitad)
#  14 días → decay 0.13   (policy obsoleta — casi ignorada)
_POLICY_DECAY_TAU = 7 * 24 * 3600   # segundos

# Temperatura base del softmax — se ajusta dinámicamente en _get_strategy().
_POLICY_SOFTMAX_TEMP_DEFAULT = 0.3

# UCB bonus: c × (1 - confidence) / √effective_runs
_POLICY_UCB_C = 0.1

# Score pressure: amplifica el peso cuando el sistema está rindiendo mal.
_POLICY_SCORE_PRESSURE_ALPHA = 0.2

# Epsilon-floor: garantía mínima de exploración en el softmax.
_POLICY_EPSILON = 0.02


# ==================== RUNTIME POLICY (lazy init) ====================

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

            delta_t    = time.time() - generated_ts
            time_decay = math.exp(-delta_t / _POLICY_DECAY_TAU)
            for cat_data in categories.values():
                for p in cat_data.get("promoted", []):
                    if isinstance(p, dict):
                        p["confidence"] = round(
                            p.get("confidence", 1.0) * time_decay, 3
                        )

            age_days = delta_t / 86400
            print(f"[scrape_tracker] policy.json v{version} cargado desde {path} "
                  f"(age={age_days:.1f}d time_decay={time_decay:.2f})")
            return categories
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[scrape_tracker] No se pudo cargar policy.json ({path}): {e}")
    return {}


_RUNTIME_POLICY_CACHE: Optional[dict] = None


def get_runtime_policy() -> dict:
    """Retorna la runtime policy, cargándola de forma lazy en el primer acceso."""
    global _RUNTIME_POLICY_CACHE
    if _RUNTIME_POLICY_CACHE is None:
        _RUNTIME_POLICY_CACHE = _load_runtime_policy()
    return _RUNTIME_POLICY_CACHE


def reset_runtime_policy_cache() -> None:
    """Resetea el cache de la runtime policy. Útil en tests y recarga en caliente."""
    global _RUNTIME_POLICY_CACHE
    _RUNTIME_POLICY_CACHE = None


# ==================== HINTS Y NIVELES ====================

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
_RETRY_ON_RELIABILITY = frozenset({"unreliable", "low_content"})

# Estrategias que producen respuestas estructuradas (datos de API, JSON, tablas).
_STRUCTURED_SOURCE_STRATEGIES: set = {"api_price"}

# Descuento sobre latency/cost para fuentes estructuradas (APIs).
_STRUCTURED_LATENCY_DISCOUNT = 0.3
_API_VALIDATION_EPSILON      = 0.02  # 2% de turnos con api_price se redirigen a exploración


# ==================== HELPERS DE SCORING ====================

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
        return 0.0
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
        cost_penalty = 0.0  (API sin LLM = sin costo facturable)
    """
    quality_score   = min(raw_words / _QUALITY_NORM_WORDS, 1.0)
    latency_penalty = min(duration_ms / _LATENCY_NORM_MS, 1.0)
    cost_penalty    = min((cost_usd or 0.0) / _COST_NORM_USD, 1.0)

    if source_type == "structured":
        quality_score   = 1.0
        latency_penalty *= _STRUCTURED_LATENCY_DISCOUNT
        cost_penalty     = 0.0

    delta = quality_score - _W_LATENCY * latency_penalty - _W_COST * cost_penalty

    if consecutive_failures >= 2:
        delta += _CONSECUTIVE_FAIL_EXTRA

    return round(delta, 3)


_EXPLORATION_WEIGHTS: Dict[str, float] = {
    "prefer_search": 0.55,
    "force_search":  0.45,
}


def _get_strategy(tracker: dict, category: str, score: float, *, exploring: bool) -> str:
    """Retorna la acción de la policy table.

    Prioridades (en orden):
    1. Exploration — PRIMERO, antes del cooldown.
    2. Cooldown activo tras ok_strong → "free"
    3. Runtime policy (policy.json)   → multi-promoted filtrado por confidence
    4. Policy table heurística basada en score band (respetando disabled)
    """
    if exploring:
        return random.choices(
            list(_EXPLORATION_WEIGHTS.keys()),
            weights=list(_EXPLORATION_WEIGHTS.values()),
        )[0]

    entry    = tracker.get(category) or {}
    cooldown = entry.get("cooldown_turns", 0)
    if cooldown > 0:
        return "free"

    rt_cat        = get_runtime_policy().get(category, {})
    disabled      = rt_cat.get("disabled", [])
    promoted_raw  = rt_cat.get("promoted", [])

    promoted_candidates = []
    for p in promoted_raw[:_POLICY_TOP_K]:
        if isinstance(p, dict):
            if p.get("confidence", 0) >= _POLICY_MIN_CONFIDENCE:
                promoted_candidates.append(p)
        elif isinstance(p, str):
            promoted_candidates.append({"strategy": p, "confidence": 1.0, "aggressiveness": 1.0})

    if promoted_candidates:
        if abs(score) < 0.5:
            temp = 0.6
        elif score <= -2.0:
            temp = 0.2
        else:
            temp = _POLICY_SOFTMAX_TEMP_DEFAULT

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

        weights = [_ucb_weight(p) for p in promoted_candidates]
        max_w   = max(weights)
        exps    = [math.exp((w - max_w) / temp) for w in weights]
        total   = sum(exps)
        probs   = [e / total for e in exps]

        k     = len(promoted_candidates)
        eps   = _POLICY_EPSILON
        probs = [(1 - eps) * p + eps / k for p in probs]

        r, cum = random.random(), 0.0
        for p, prob in zip(promoted_candidates, probs):
            cum += prob
            if r <= cum:
                return p["strategy"]
        return promoted_candidates[-1]["strategy"]

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
        quality_target       → 1 si ok_strong, 0 en otro caso
        recovery_turns       → turnos que tardó en recuperarse de racha mala (o None)
        regret_estimate      → best_delta_histórico - delta_actual (oportunidad perdida)
    """
    tracker = dict(tracker)
    entry   = dict(tracker.get(category) or {})
    rel     = reliability_override if reliability_override else _scrape_reliability(raw_words)
    is_ok   = rel in ("ok_weak", "ok_strong")

    turns_ago     = current_turn - entry.get("last_turn", 0)
    current_score = 0.0 if turns_ago > _SCORE_DECAY_TURNS else float(entry.get("score", 0.0))

    cooldown             = max(0, entry.get("cooldown_turns", 0) - 1)
    prev_failures        = entry.get("consecutive_failures", 0)
    consecutive_failures = 0 if is_ok else prev_failures + 1

    delta     = _compute_delta(raw_words, duration_ms, cost_usd, consecutive_failures, source_type=source_type)
    new_score = max(_SCORE_MIN, min(_SCORE_MAX, round(current_score + delta, 3)))

    if source_type == "structured" and rel == "ok_strong":
        new_score = max(_SCORE_MIN, min(_SCORE_MAX, new_score + 0.5))

    new_cooldown = _COOLDOWN_TURNS if rel == "ok_strong" else cooldown

    last_bad_turn = entry.get("last_bad_turn")
    if not is_ok:
        last_bad_turn = current_turn
    recovery_turns: Optional[int] = None
    if is_ok and last_bad_turn is not None:
        recovery_turns = current_turn - last_bad_turn
        last_bad_turn  = None

    best_delta      = max(float(entry.get("best_delta", delta)), delta)
    regret_estimate = round(best_delta - delta, 3)

    tracker[category] = {
        "score":                new_score,
        "last_turn":            current_turn,
        "last_duration_ms":     duration_ms,
        "last_cost_usd":        cost_usd,
        "cooldown_turns":       new_cooldown,
        "consecutive_failures": consecutive_failures,
        "last_bad_turn":        last_bad_turn,
        "best_delta":           best_delta,
    }
    tracker["_turn_count"] = current_turn

    analytics = {
        "quality_target":  1 if rel == "ok_strong" else 0,
        "recovery_turns":  recovery_turns,
        "regret_estimate": regret_estimate,
    }
    return tracker, analytics


__all__ = [
    "get_runtime_policy",
    "reset_runtime_policy_cache",
    "_update_scrape_tracker",
    "_get_strategy",
    "_get_category_score",
    "_detect_query_category",
    "_exploration_rate",
    "_scrape_reliability",
    "_score_to_policy_band",
    "_score_to_reliability",
    "_compute_delta",
    "_STRATEGY_HINTS",
    "_RETRY_ON_RELIABILITY",
    "_STRUCTURED_SOURCE_STRATEGIES",
    "_API_VALIDATION_EPSILON",
    "_STRUCTURED_LATENCY_DISCOUNT",
]
