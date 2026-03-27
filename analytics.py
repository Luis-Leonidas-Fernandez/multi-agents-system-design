#!/usr/bin/env python3
"""
Pipeline de analytics avanzado para el audit log del sistema multi-agentes.

  - Strategy ranking por categoría (success_rate, avg_delta, latency, cost, regret)
  - Learning curve: regret vs tiempo + success rate acumulativa
  - Regresión logística por categoría (con --train, requiere scikit-learn)
  - Predicción de estrategia óptima dado contexto actual

Uso:
    python analytics.py [audit.jsonl]
    python analytics.py --no-show          # solo imprime, sin ventana
    python analytics.py --train            # entrena logistic regression
    python analytics.py --train --no-show  # entrena + guarda PNG
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


# ==================== CARGA ====================

def _load_scrape_records(path: str) -> list:
    """Carga registros con quality_target (emitidos por web_scraping_node)."""
    records = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if "quality_target" in r and "strategy" in r:
                        records.append(r)
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        print(f"[analytics] Archivo no encontrado: {path}")
        sys.exit(1)
    return records


def _resolve_path() -> str:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        return args[0]
    return os.getenv("AGENTDOG_AUDIT_LOG", "./logs/agentdog_audit.jsonl")


# ==================== STRATEGY RANKING ====================

def strategy_ranking(records: list) -> list:
    """Ranking de estrategias por (categoría, estrategia) con todas las métricas."""
    import time as _time
    groups: dict = defaultdict(list)
    for r in records:
        key = (r.get("category", "unknown"), r.get("strategy", "unknown"))
        groups[key].append(r)

    now_ms   = _time.time() * 1000
    cutoff   = now_ms - 86_400_000   # últimas 24h en ms

    results = []
    for (category, strategy), recs in sorted(groups.items()):
        n           = len(recs)
        ok_strong   = sum(1 for r in recs if r.get("scrape_reliability") == "ok_strong")
        avg_delta   = sum((r.get("scrape_score") or 0) - (r.get("prior_score") or 0) for r in recs) / n
        avg_latency = sum(r.get("duration_ms") or 0 for r in recs) / n
        avg_cost    = sum(r.get("estimated_cost_usd") or 0 for r in recs) / n
        avg_regret  = sum(r.get("regret_estimate") or 0 for r in recs) / n
        avg_recovery = [r["recovery_turns"] for r in recs if r.get("recovery_turns") is not None]
        runs_last_24h = sum(1 for r in recs if (r.get("ts_ms") or 0) >= cutoff)

        # Penalizaciones normalizadas [0, 1] — mismos factores que supervisor.py
        avg_latency_penalty = min(avg_latency / 15000, 1.0)
        avg_cost_penalty    = min(avg_cost / 0.001, 1.0)
        # Agresividad aprendida: estrategias más lentas/caras son más "agresivas"
        aggressiveness = round(avg_latency_penalty + avg_cost_penalty, 3)

        results.append({
            "category":           category,
            "strategy":           strategy,
            "runs":               n,
            "runs_last_24h":      runs_last_24h,
            "success_rate":       ok_strong / n,
            "avg_delta":          round(avg_delta, 4),
            "avg_latency_ms":     round(avg_latency),
            "avg_cost_usd":       round(avg_cost, 7),
            "avg_regret":         round(avg_regret, 4),
            "avg_recovery_turns": round(sum(avg_recovery) / len(avg_recovery), 2) if avg_recovery else None,
            "avg_latency_penalty": round(avg_latency_penalty, 3),
            "avg_cost_penalty":    round(avg_cost_penalty, 4),
            "aggressiveness":      aggressiveness,
        })

    return sorted(results, key=lambda x: (-x["success_rate"], x["avg_regret"]))


# ==================== LOGISTIC REGRESSION ====================

# ==================== STRATEGY DOMINANCE ====================

# Umbrales para clasificar estrategias como dominantes o muertas
_DOMINANCE_SUCCESS_THRESHOLD = 0.70   # promoted si success_rate ≥ 70%
_DOMINANCE_DELTA_THRESHOLD   = 0.0    # promoted si avg_delta > 0
_DEAD_SUCCESS_THRESHOLD      = 0.25   # disabled si success_rate ≤ 25%
_DEAD_DELTA_THRESHOLD        = -0.15  # disabled si avg_delta < -0.15
_MIN_RUNS_FOR_JUDGMENT       = 8      # ignorar estrategias con pocos datos
_PROMOTED_MIN_CONFIDENCE     = 0.50   # no promover si confidence < 0.5 (pocos datos)


def _promotion_confidence(
    success_rate: float,
    runs: int,
    avg_latency_ms: float = 0.0,
    avg_cost_usd: float = 0.0,
) -> float:
    """Confidence efficiency-weighted: penaliza muestra pequeña Y estrategias caras/lentas.

    confidence = success_rate × sample_factor × efficiency

    sample_factor = min(√runs / √(2×_MIN_RUNS), 1.0)
      → 8 runs  → 0.71 (penalizado)
      → 16 runs → 1.00 (confianza plena)

    efficiency = 1 / (1 + latency_norm + cost_norm)
      → latencia=0, costo=0  → efficiency=1.00 (máxima)
      → latencia=0.5, costo=0.1 → efficiency=0.63
      → latencia=1.0, costo=1.0 → efficiency=0.33 (lenta y cara)

    Así promoted refleja: "esta estrategia es buena Y eficiente".
    """
    import math
    sample_factor = min(math.sqrt(runs) / math.sqrt(2 * _MIN_RUNS_FOR_JUDGMENT), 1.0)
    latency_norm  = min(avg_latency_ms / 15000, 1.0)
    cost_norm     = min(avg_cost_usd / 0.001, 1.0)
    efficiency    = 1.0 / (1.0 + latency_norm + cost_norm)
    return round(success_rate * sample_factor * efficiency, 3)


def strategy_dominance(ranking: list) -> dict:
    """Detecta estrategias dominantes (promoted) y muertas (disabled) por categoría.

    promoted: lista ordenada por confidence desc — objetos {"strategy": ..., "confidence": ...}
    disabled: lista de strings (no necesitan confidence)

    Retorna:
    {
      "crypto_price": {
        "promoted": [{"strategy": "force_search", "confidence": 0.82}],
        "disabled": ["free"],
      }
    }
    """
    by_category: dict = defaultdict(lambda: {"promoted": [], "disabled": []})

    for r in ranking:
        cat      = r["category"]
        strategy = r["strategy"]

        if r["runs"] < _MIN_RUNS_FOR_JUDGMENT:
            continue

        if (r["success_rate"] >= _DOMINANCE_SUCCESS_THRESHOLD
                and r["avg_delta"] > _DOMINANCE_DELTA_THRESHOLD):
            confidence = _promotion_confidence(
                r["success_rate"], r["runs"],
                avg_latency_ms=r.get("avg_latency_ms", 0),
                avg_cost_usd=r.get("avg_cost_usd", 0),
            )
            if confidence >= _PROMOTED_MIN_CONFIDENCE:
                by_category[cat]["promoted"].append({
                    "strategy":       strategy,
                    "confidence":     confidence,
                    "aggressiveness": r.get("aggressiveness", 0.0),
                    "runs":           r["runs"],
                    "runs_last_24h":  r.get("runs_last_24h", 0),
                })

        elif (r["success_rate"] <= _DEAD_SUCCESS_THRESHOLD
              and r["avg_delta"] < _DEAD_DELTA_THRESHOLD):
            by_category[cat]["disabled"].append(strategy)

    # Ordenar promoted por confidence desc (mejor primero)
    result = {}
    for cat, v in by_category.items():
        result[cat] = {
            "promoted": sorted(v["promoted"], key=lambda x: -x["confidence"]),
            "disabled": v["disabled"],
        }
    return result


# ==================== LOGISTIC REGRESSION ====================

_PRIOR_REL_ENCODING = {"unreliable": -1, "low_content": 0, "ok": 1, "ok_weak": 1, "ok_strong": 2}

_ALL_STRATEGIES = ["api_price", "force_search", "prefer_search", "free"]


def _build_feature_vector(r: dict, strategies: list) -> list:
    """Construye feature vector con one-hot de strategy + features de contexto.

    Features:
      strategy_* (one-hot, N columnas)   ← crítico: el modelo aprende por estrategia
      latency_norm                        ← [0, 1]
      cost_norm                           ← [0, 1]
      prior_score                         ← float acumulativo
      consecutive_failures_norm           ← min(failures/3, 1.0)
      prior_reliability_encoded           ← {-1, 0, 1, 2}
    """
    strategy = r.get("strategy", "free")
    one_hot  = [1 if strategy == s else 0 for s in strategies]

    return one_hot + [
        min((r.get("duration_ms") or 0) / 15000, 1.0),
        min((r.get("estimated_cost_usd") or 0) / 0.001, 1.0),
        float(r.get("prior_score") or 0.0),
        min((r.get("consecutive_failures") or 0) / 3, 1.0),
        _PRIOR_REL_ENCODING.get(r.get("prior_reliability", "ok"), 0),
    ]


def _feature_names(strategies: list) -> list:
    return [f"strategy_{s}" for s in strategies] + [
        "latency_norm", "cost_norm", "prior_score",
        "consecutive_failures_norm", "prior_reliability_enc",
    ]


def train_models(records: list) -> dict:
    """Entrena logistic regression por categoría con one-hot strategy + context features."""
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
        from sklearn.model_selection import cross_val_score  # type: ignore[import-untyped]
        import numpy as np
    except ImportError:
        print("[analytics] scikit-learn no instalado — pip install scikit-learn")
        return {}

    categories = sorted(set(r.get("category", "unknown") for r in records))
    models = {}

    for cat in categories:
        cat_recs   = [r for r in records if r.get("category") == cat]
        if len(cat_recs) < 10:
            print(f"[{cat}] insuficientes datos ({len(cat_recs)} < 10) — omitido")
            continue

        # Lista canónica de estrategias: garantiza encoding idéntico en
        # training, cross-val y predict_best_strategy (sin depender de los datos).
        strategies = _ALL_STRATEGIES
        X          = np.array([_build_feature_vector(r, strategies) for r in cat_recs])
        y          = np.array([r.get("quality_target", 0) for r in cat_recs])

        if len(set(y)) < 2:
            print(f"[{cat}] solo una clase en y — no se puede entrenar")
            continue

        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X, y)

        cv_score = None
        if len(cat_recs) >= 20:
            cv_score = cross_val_score(model, X, y, cv=min(5, len(cat_recs) // 4)).mean()

        feat_names = _feature_names(strategies)
        coefs      = dict(zip(feat_names, model.coef_[0]))
        print(f"[{cat}] entrenado | n={len(cat_recs)} accuracy={model.score(X, y):.2f}"
              f"{f'  cv={cv_score:.2f}' if cv_score else ''}")
        print(f"  coefs: {', '.join(f'{k}={v:+.3f}' for k, v in coefs.items())}")

        models[cat] = {"model": model, "strategies": strategies, "coefs": coefs}

    return models


def predict_best_strategy(
    models: dict,
    category: str,
    duration_ms: int = 5000,
    cost_usd: float = 0.0001,
    prior_score: float = 0.0,
    consecutive_failures: int = 0,
    prior_reliability: str = "ok",
    lam_cost: float = 0.3,
    mu_latency: float = 0.3,
    disabled: list = None,
) -> tuple:
    """Retorna (best_strategy, score) usando argmax(P(success) - λ*cost - μ*latency).

    Si no hay modelo, retorna ('free', None).
    disabled: estrategias a excluir (vienen del policy.json).
    """
    disabled = disabled or []

    if not models or category not in models:
        return "free", None

    try:
        import numpy as np
    except ImportError:
        return "free", None

    m          = models[category]
    model      = m["model"]
    strategies = [s for s in m["strategies"] if s not in disabled]
    if not strategies:
        strategies = m["strategies"]  # fallback si todas están deshabilitadas

    latency_norm = min(duration_ms / 15000, 1.0)
    cost_norm    = min(cost_usd / 0.001, 1.0)

    best_strategy, best_score = "free", -999.0
    for strategy in strategies:
        r_mock = {
            "strategy": strategy,
            "duration_ms": duration_ms,
            "estimated_cost_usd": cost_usd,
            "prior_score": prior_score,
            "consecutive_failures": consecutive_failures,
            "prior_reliability": prior_reliability,
        }
        x    = np.array([_build_feature_vector(r_mock, m["strategies"])])
        prob = model.predict_proba(x)[0][1]
        score = prob - lam_cost * cost_norm - mu_latency * latency_norm

        if score > best_score:
            best_score, best_strategy = score, strategy

    return best_strategy, round(best_score, 4)


# ==================== POLICY EXPORT ====================

def export_policy_config(dominance: dict, path: Path) -> None:
    """Exporta policy.json versionado para que supervisor.py lo cargue en runtime.

    Formato:
    {
      "version": "2026-03-19T18:00:00",
      "generated_ts": 1742400000,
      "categories": {
        "crypto_price": {
          "promoted": [{"strategy": "force_search", "confidence": 0.82}],
          "disabled": ["free"]
        }
      }
    }

    El campo "version" permite rollback, comparación de experimentos y auditoría.
    """
    import time
    from datetime import datetime, timezone

    now_ts  = int(time.time())
    version = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    policy = {
        "version":      version,
        "generated_ts": now_ts,
        "categories":   dominance,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2, ensure_ascii=False)
    print(f"[analytics] policy.json v{version} exportado → {path}")


# ==================== LEARNING CURVE ====================

def _rolling_avg(values: list, window: int = 3) -> list:
    return [
        sum(values[max(0, i - window + 1): i + 1]) / len(values[max(0, i - window + 1): i + 1])
        for i in range(len(values))
    ]


def plot_learning_curve(records: list, out_stem: Path, no_show: bool):
    import matplotlib.pyplot as plt

    # Ordenar por ts_ms
    timed = sorted(
        [r for r in records if "ts_ms" in r],
        key=lambda x: x["ts_ms"],
    )
    if not timed:
        print("[analytics] Sin ts_ms en registros — learning curve omitida")
        return

    categories = sorted(set(r.get("category", "unknown") for r in timed))
    palette    = plt.cm.Set2(range(len(categories)))
    color_map  = dict(zip(categories, palette))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Learning Curve — Regret & Success Rate over Time", fontsize=14, fontweight="bold")

    # --- Panel 1: regret vs run (rolling) ---
    for cat in categories:
        cat_data = [r for r in timed if r.get("category") == cat]
        if not cat_data:
            continue
        regrets  = [r.get("regret_estimate") or 0 for r in cat_data]
        rolling  = _rolling_avg(regrets, window=3)
        indices  = list(range(len(cat_data)))
        ax1.plot(indices, rolling, label=cat, color=color_map[cat], linewidth=2)
        ax1.scatter(indices, regrets, color=color_map[cat], alpha=0.25, s=20)

    ax1.axhline(0, color="black", linewidth=0.7, linestyle="--", label="regret=0 (óptimo)")
    ax1.set_title("Regret vs Run (rolling avg window=3)", fontweight="bold")
    ax1.set_xlabel("Run #")
    ax1.set_ylabel("Regret estimate")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # --- Panel 2: success rate acumulativa ---
    for cat in categories:
        cat_data = [r for r in timed if r.get("category") == cat and "quality_target" in r]
        if not cat_data:
            continue
        targets    = [r["quality_target"] for r in cat_data]
        cumulative = [sum(targets[: i + 1]) / (i + 1) for i in range(len(targets))]
        ax2.plot(cumulative, label=cat, color=color_map[cat], linewidth=2)

    ax2.axhline(0.8, color="green", linewidth=0.7, linestyle="--", label="target 80%")
    ax2.set_title("Success Rate acumulativa (quality_target=1 ↔ ok_strong)", fontweight="bold")
    ax2.set_xlabel("Run #")
    ax2.set_ylabel("Success rate")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = Path(f"{out_stem}_learning_curve.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[analytics] Learning curve → {out_path}")
    if not no_show:
        plt.show()


# ==================== MAIN ====================

def main():
    path     = _resolve_path()
    no_show  = "--no-show" in sys.argv
    do_train = "--train" in sys.argv

    records = _load_scrape_records(path)
    if not records:
        print(f"[analytics] Sin registros con quality_target en {path}")
        sys.exit(0)

    print(f"[analytics] {len(records)} registros cargados desde {path}")

    # --- Strategy ranking ---
    ranking = strategy_ranking(records)
    print("\n=== STRATEGY RANKING ===")
    header = f"{'CATEGORY':12s} {'STRATEGY':14s} {'RUNS':>4s} {'SUCCESS':>7s} {'AVG_DELTA':>9s} {'LATENCY':>8s} {'COST_USD':>10s} {'REGRET':>7s} {'RECOVERY':>9s}"
    print(header)
    print("-" * len(header))
    for r in ranking:
        rec_str = f"{r['avg_recovery_turns']:.1f}t" if r["avg_recovery_turns"] is not None else "  —"
        print(
            f"{r['category']:12s} {r['strategy']:14s} {r['runs']:>4d} "
            f"{r['success_rate']:>6.0%}  {r['avg_delta']:>+9.4f} "
            f"{r['avg_latency_ms']:>7.0f}ms ${r['avg_cost_usd']:>9.7f} "
            f"{r['avg_regret']:>7.4f} {rec_str:>9s}"
        )

    # --- Logistic regression (opcional) ---
    # La regresión es ADITIVA: si sklearn no está disponible, el sistema
    # sigue funcionando — policy.json se genera igualmente desde dominance.
    models = {}
    if do_train:
        print("\n=== LOGISTIC REGRESSION POR CATEGORÍA ===")
        models = train_models(records)

        if not models:
            print("  sklearn no disponible o datos insuficientes.")
            print("  policy.json se generará con dominance heurística (sin regresión).")
        else:
            print("\n=== OPTIMAL STRATEGY PREDICTIONS ===")
            test_contexts = [
                {"duration_ms": 3000,  "cost_usd": 0.00005, "prior_score": 0.0,  "label": "rápido, barato, score neutro"},
                {"duration_ms": 20000, "cost_usd": 0.0003,  "prior_score": -1.5, "label": "lento, caro, score malo"},
                {"duration_ms": 5000,  "cost_usd": 0.0001,  "prior_score": 0.8,  "label": "moderado, score bueno"},
            ]
            for cat in models:
                print(f"\n  [{cat}]")
                for ctx in test_contexts:
                    best, prob = predict_best_strategy(
                        models, cat,
                        duration_ms=ctx["duration_ms"],
                        cost_usd=ctx["cost_usd"],
                        prior_score=ctx["prior_score"],
                    )
                    print(f"    {ctx['label']:40s} → {best:14s} P(success)={prob:.2%}")

        # --- Strategy dominance + policy export ---
        # Siempre se ejecuta — no depende de sklearn.
        print("\n=== STRATEGY DOMINANCE ===")
        dominance = strategy_dominance(ranking)
        if dominance:
            for cat, d in dominance.items():
                promoted = d.get("promoted", [])
                disabled = d.get("disabled", [])
                if promoted:
                    promoted_str = ", ".join(
                        f"{p['strategy']} ({p['confidence']:.0%})" for p in promoted
                    )
                    print(f"  [{cat}] PROMOTED: {promoted_str}")
                if disabled:
                    print(f"  [{cat}] DISABLED: {', '.join(disabled)}")
        else:
            print("  (sin estrategias con datos suficientes para juzgar dominancia)")

        policy_path = Path(path).parent / "policy.json"
        export_policy_config(dominance, policy_path)

    # --- Learning curve ---
    out_stem = Path(path).with_suffix("")
    plot_learning_curve(records, out_stem, no_show)

    # --- Dataset export (para regresión externa) ---
    export_path = Path(f"{out_stem}_regression_dataset.jsonl")
    with open(export_path, "w", encoding="utf-8") as f:
        for r in records:
            row = {k: r.get(k) for k in [
                "quality_target", "strategy", "category",
                "raw_words", "duration_ms", "estimated_cost_usd",
                "prior_score", "scrape_score", "regret_estimate",
                "recovery_turns", "exploring", "exp_rate", "ts_ms",
            ]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n[analytics] Dataset de regresión exportado → {export_path}")
    print("  Columnas: quality_target (target), strategy+category+features (predictores)")


if __name__ == "__main__":
    main()
