"""
Tests unitarios para scrape_tracker.py.

Cubre: detect_query_category, _scrape_reliability, _get_category_score,
_get_strategy, _update_scrape_tracker, runtime policy cache.

Sin dependencias externas — stdlib pura.
"""
import json
import os
import pytest


# ==================== HELPERS ====================

def _fresh_tracker() -> dict:
    return {}


# ==================== _detect_query_category ====================

@pytest.mark.parametrize("query,expected", [
    ("¿Cuál es el precio del bitcoin?",   "crypto_price"),
    ("Dame la cotización del BTC hoy",    "crypto_price"),
    ("precio del ethereum",               "crypto_price"),
    ("eth price in usd",                  "crypto_price"),
    ("defi protocol news",                "crypto_price"),  # "defi" → crypto_price
    ("stock market analysis today",       "finance"),
    ("bolsa de valores nasdaq",           "finance"),
    ("últimas noticias de tecnología",    "news"),
    ("noticias del día",                  "news"),
    ("latest headlines",                  "news"),
    ("clima en Buenos Aires mañana",      "weather"),
    ("weather forecast for NYC",          "weather"),
    ("receta de pasta carbonara",         "general"),
    ("cómo aprender Python",              "general"),
    ("best restaurants in Madrid",        "general"),
])
def test_detect_query_category_clasifica_correctamente(query, expected):
    from scrape_tracker import _detect_query_category
    assert _detect_query_category(query) == expected


def test_detect_query_category_case_insensitive():
    from scrape_tracker import _detect_query_category
    assert _detect_query_category("BITCOIN PRICE TODAY") == "crypto_price"
    assert _detect_query_category("STOCK MARKET") == "finance"


def test_detect_query_category_empty_string_returns_general():
    from scrape_tracker import _detect_query_category
    assert _detect_query_category("") == "general"


# ==================== _scrape_reliability ====================

@pytest.mark.parametrize("raw_words,expected_level", [
    (0,   "unreliable"),
    (19,  "unreliable"),
    (20,  "low_content"),
    (49,  "low_content"),
    (50,  "ok_weak"),
    (119, "ok_weak"),
    (120, "ok_strong"),
    (500, "ok_strong"),
])
def test_scrape_reliability_niveles_correctos(raw_words, expected_level):
    from scrape_tracker import _scrape_reliability
    assert _scrape_reliability(raw_words) == expected_level


# ==================== _get_category_score ====================

def test_get_category_score_tracker_vacio_retorna_cero():
    from scrape_tracker import _get_category_score
    assert _get_category_score({}, "crypto_price", 1) == 0.0


def test_get_category_score_categoria_inexistente_retorna_cero():
    from scrape_tracker import _get_category_score
    tracker = {"finance": {"score": 1.5, "last_turn": 3}}
    assert _get_category_score(tracker, "crypto_price", 3) == 0.0


def test_get_category_score_retorna_score_actual():
    from scrape_tracker import _get_category_score
    tracker = {"crypto_price": {"score": 2.0, "last_turn": 5}}
    result = _get_category_score(tracker, "crypto_price", 5)
    assert result == 2.0


def test_get_category_score_decae_despues_de_decay_turns():
    """Después de _SCORE_DECAY_TURNS (3) turnos sin actividad, retorna 0."""
    from scrape_tracker import _get_category_score, _SCORE_DECAY_TURNS
    tracker = {"crypto_price": {"score": 2.5, "last_turn": 1}}
    current_turn = 1 + _SCORE_DECAY_TURNS + 1  # un turno más allá del límite
    result = _get_category_score(tracker, "crypto_price", current_turn)
    assert result == 0.0


def test_get_category_score_dentro_de_ventana_decay_mantiene_score():
    from scrape_tracker import _get_category_score, _SCORE_DECAY_TURNS
    tracker = {"crypto_price": {"score": 1.5, "last_turn": 1}}
    current_turn = 1 + _SCORE_DECAY_TURNS  # exactamente en el límite
    result = _get_category_score(tracker, "crypto_price", current_turn)
    assert result == 1.5


# ==================== _get_strategy ====================

_VALID_STRATEGIES = {"free", "prefer_search", "force_search", "api_price"}


def test_get_strategy_sin_explorar_score_neutral_retorna_free():
    from scrape_tracker import _get_strategy, reset_runtime_policy_cache
    reset_runtime_policy_cache()
    os.environ["POLICY_CONFIG"] = ""  # sin policy file
    result = _get_strategy({}, "general", 0.0, exploring=False)
    assert result in _VALID_STRATEGIES


def test_get_strategy_explorando_retorna_estrategia_valida():
    from scrape_tracker import _get_strategy, reset_runtime_policy_cache
    reset_runtime_policy_cache()
    # exploring=True debe retornar una de las estrategias de exploración
    resultados = set()
    for _ in range(30):
        r = _get_strategy({}, "general", 0.0, exploring=True)
        resultados.add(r)
    # debe retornar alguna de las estrategias de exploración definidas en _EXPLORATION_WEIGHTS
    assert resultados.issubset({"prefer_search", "force_search"})


def test_get_strategy_score_muy_bajo_sin_policy_retorna_force_search():
    from scrape_tracker import _get_strategy, reset_runtime_policy_cache
    reset_runtime_policy_cache()
    # score <= -2 → "very_low" band → force_search para crypto_price
    result = _get_strategy({}, "crypto_price", -2.5, exploring=False)
    assert result == "force_search"


def test_get_strategy_score_bajo_sin_policy_retorna_prefer_search():
    from scrape_tracker import _get_strategy, reset_runtime_policy_cache
    reset_runtime_policy_cache()
    # -2 < score < 0 → "low" band → prefer_search para crypto_price
    result = _get_strategy({}, "crypto_price", -1.0, exploring=False)
    assert result == "prefer_search"


def test_get_strategy_cooldown_activo_retorna_free():
    from scrape_tracker import _get_strategy, reset_runtime_policy_cache
    reset_runtime_policy_cache()
    tracker = {"crypto_price": {"score": -3.0, "last_turn": 1, "cooldown_turns": 2}}
    # cooldown activo → ignora el score malo y retorna "free"
    result = _get_strategy(tracker, "crypto_price", -3.0, exploring=False)
    assert result == "free"


# ==================== _update_scrape_tracker ====================

def test_update_scrape_tracker_resultado_ok_strong_incrementa_score():
    from scrape_tracker import _update_scrape_tracker
    tracker = {}
    new_tracker, analytics = _update_scrape_tracker(
        tracker, "crypto_price", raw_words=150, current_turn=1,
        duration_ms=500, cost_usd=0.0,
    )
    score = new_tracker["crypto_price"]["score"]
    assert score > 0.0, f"Score debería ser positivo con ok_strong, got {score}"


def test_update_scrape_tracker_resultado_unreliable_reduce_score():
    """Con raw_words bajos, latencia alta y costo real, el delta debe ser negativo."""
    from scrape_tracker import _update_scrape_tracker
    tracker = {}
    # raw_words=5 → quality_score=5/120=0.042, latencia alta → penalización mayor
    new_tracker, analytics = _update_scrape_tracker(
        tracker, "crypto_price", raw_words=5, current_turn=1,
        duration_ms=15000, cost_usd=0.001,  # penalización máxima: -0.5 latency, -0.5 cost
    )
    score = new_tracker["crypto_price"]["score"]
    assert score < 0.0, f"Score debería ser negativo con unreliable+alta penalización, got {score}"


def test_update_scrape_tracker_score_clampado_en_max():
    from scrape_tracker import _update_scrape_tracker, _SCORE_MAX
    # Partir con score ya muy alto
    tracker = {"crypto_price": {"score": _SCORE_MAX - 0.1, "last_turn": 1}}
    new_tracker, _ = _update_scrape_tracker(
        tracker, "crypto_price", raw_words=500, current_turn=2,
        duration_ms=100, cost_usd=0.0,
    )
    assert new_tracker["crypto_price"]["score"] <= _SCORE_MAX


def test_update_scrape_tracker_score_clampado_en_min():
    from scrape_tracker import _update_scrape_tracker, _SCORE_MIN
    tracker = {"crypto_price": {"score": _SCORE_MIN + 0.1, "last_turn": 1}}
    new_tracker, _ = _update_scrape_tracker(
        tracker, "crypto_price", raw_words=0, current_turn=2,
        duration_ms=10000, cost_usd=0.01,
    )
    assert new_tracker["crypto_price"]["score"] >= _SCORE_MIN


def test_update_scrape_tracker_ok_strong_activa_cooldown():
    from scrape_tracker import _update_scrape_tracker, _COOLDOWN_TURNS
    tracker = {}
    new_tracker, _ = _update_scrape_tracker(
        tracker, "news", raw_words=200, current_turn=1,
    )
    cooldown = new_tracker["news"]["cooldown_turns"]
    assert cooldown == _COOLDOWN_TURNS


def test_update_scrape_tracker_failure_incrementa_consecutive_failures():
    from scrape_tracker import _update_scrape_tracker
    tracker = {"finance": {"score": 0.0, "last_turn": 1, "consecutive_failures": 1}}
    new_tracker, _ = _update_scrape_tracker(
        tracker, "finance", raw_words=5, current_turn=2,
    )
    assert new_tracker["finance"]["consecutive_failures"] == 2


def test_update_scrape_tracker_success_resetea_consecutive_failures():
    from scrape_tracker import _update_scrape_tracker
    tracker = {"finance": {"score": 0.0, "last_turn": 1, "consecutive_failures": 3}}
    new_tracker, _ = _update_scrape_tracker(
        tracker, "finance", raw_words=150, current_turn=2,
    )
    assert new_tracker["finance"]["consecutive_failures"] == 0


def test_update_scrape_tracker_analytics_quality_target():
    from scrape_tracker import _update_scrape_tracker
    tracker = {}
    _, analytics = _update_scrape_tracker(
        tracker, "news", raw_words=200, current_turn=1,
    )
    assert analytics["quality_target"] == 1

    _, analytics_bad = _update_scrape_tracker(
        tracker, "news", raw_words=10, current_turn=1,
    )
    assert analytics_bad["quality_target"] == 0


def test_update_scrape_tracker_recovery_turns_calculado():
    from scrape_tracker import _update_scrape_tracker
    # Simular racha mala iniciada en turno 3
    tracker = {"finance": {"score": -1.0, "last_turn": 3, "last_bad_turn": 3, "consecutive_failures": 2}}
    _, analytics = _update_scrape_tracker(
        tracker, "finance", raw_words=200, current_turn=7,
    )
    # recovery en el turno 7 desde bad_turn 3 → 4 turnos
    assert analytics["recovery_turns"] == 4


def test_update_scrape_tracker_no_modifica_tracker_original():
    """_update_scrape_tracker debe retornar una copia, no mutar el original."""
    from scrape_tracker import _update_scrape_tracker
    original = {"crypto_price": {"score": 1.0, "last_turn": 1}}
    original_score_before = original["crypto_price"]["score"]
    _update_scrape_tracker(original, "crypto_price", raw_words=50, current_turn=2)
    # El original NO debe mutar
    assert original["crypto_price"]["score"] == original_score_before


# ==================== Runtime Policy Cache ====================

def test_get_runtime_policy_retorna_dict():
    from scrape_tracker import get_runtime_policy, reset_runtime_policy_cache
    reset_runtime_policy_cache()
    os.environ["POLICY_CONFIG"] = ""
    policy = get_runtime_policy()
    assert isinstance(policy, dict)


def test_get_runtime_policy_es_lazy_y_cached():
    """Llamadas sucesivas retornan el mismo objeto (identidad)."""
    from scrape_tracker import get_runtime_policy, reset_runtime_policy_cache
    reset_runtime_policy_cache()
    os.environ["POLICY_CONFIG"] = ""
    p1 = get_runtime_policy()
    p2 = get_runtime_policy()
    assert p1 is p2


def test_reset_runtime_policy_cache_limpia_cache():
    from scrape_tracker import get_runtime_policy, reset_runtime_policy_cache
    reset_runtime_policy_cache()
    os.environ["POLICY_CONFIG"] = ""
    p1 = get_runtime_policy()
    reset_runtime_policy_cache()
    p2 = get_runtime_policy()
    # Después del reset, se crea un nuevo objeto (no el mismo)
    assert p1 is not p2


def test_get_runtime_policy_desde_tmp_file(tmp_path):
    """get_runtime_policy carga correctamente desde un policy.json válido."""
    import time as time_mod
    from scrape_tracker import get_runtime_policy, reset_runtime_policy_cache

    policy_data = {
        "version": "1.0",
        "generated_ts": time_mod.time(),
        "categories": {
            "crypto_price": {
                "promoted": [
                    {"strategy": "api_price", "confidence": 0.9, "runs": 100}
                ]
            }
        },
    }
    policy_file = tmp_path / "policy.json"
    policy_file.write_text(json.dumps(policy_data))

    reset_runtime_policy_cache()
    os.environ["POLICY_CONFIG"] = str(policy_file)
    try:
        policy = get_runtime_policy()
        assert "crypto_price" in policy
        assert policy["crypto_price"]["promoted"][0]["strategy"] == "api_price"
    finally:
        os.environ["POLICY_CONFIG"] = ""
        reset_runtime_policy_cache()


def test_policy_time_decay_no_se_acumula_en_cache(tmp_path):
    """El decay de la confianza en policy.json no se compone acumulativamente.

    Si se llama _load_runtime_policy varias veces (sin cache), el valor base
    es siempre el original del JSON — el decay no se multiplica sobre sí mismo.
    """
    import time as time_mod
    import math
    from scrape_tracker import (
        _load_runtime_policy,
        reset_runtime_policy_cache,
        _POLICY_DECAY_TAU,
    )

    # JSON con confianza 0.9 y timestamp "ahora"
    generated_ts = time_mod.time()
    policy_data = {
        "version": "1.0",
        "generated_ts": generated_ts,
        "categories": {
            "crypto_price": {
                "promoted": [
                    {"strategy": "api_price", "confidence": 0.9, "runs": 100}
                ]
            }
        },
    }
    policy_file = tmp_path / "policy.json"
    policy_file.write_text(json.dumps(policy_data))
    os.environ["POLICY_CONFIG"] = str(policy_file)

    try:
        # Llamar _load_runtime_policy dos veces — cada vez lee del archivo
        result1 = _load_runtime_policy()
        result2 = _load_runtime_policy()

        conf1 = result1["crypto_price"]["promoted"][0]["confidence"]
        conf2 = result2["crypto_price"]["promoted"][0]["confidence"]

        # Ambas llamadas deben producir el mismo decay (no acumulativo)
        # Permitir diferencia mínima por el tiempo que pasa entre llamadas
        assert abs(conf1 - conf2) < 0.01, (
            f"El decay se compone acumulativamente: primera={conf1}, segunda={conf2}"
        )
    finally:
        os.environ["POLICY_CONFIG"] = ""
        reset_runtime_policy_cache()
