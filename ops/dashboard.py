#!/usr/bin/env python3
"""
Dashboard de observabilidad para el audit log del sistema multi-agentes.

6 paneles (2×3):
  Fila 1 — estrategia / categoría / tiempo:
    1. Latencia por estrategia (boxplot, ms)
    2. Success rate por categoría (barras apiladas %)
    3. Regret en el tiempo (serie temporal + rolling avg)

  Fila 2 — agente / modelo / eficiencia:
    4. Outcomes por agente (barras apiladas %)
    5. Costo acumulado por modelo (USD)
    6. Eficiencia: tokens / ms por agente

Uso:
    python dashboard.py [ruta_al_audit.jsonl]
    python dashboard.py                        # lee AGENTDOG_AUDIT_LOG o ./logs/agentdog_audit.jsonl
    python dashboard.py --no-show              # guarda PNG sin abrir ventana
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ==================== CARGA ====================

def _load_node_outcomes(path: str) -> list:
    """Filtra registros de outcome de nodo: tienen request_id + outcome + duration_ms."""
    records = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if "request_id" in r and "outcome" in r and "duration_ms" in r:
                        records.append(r)
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        print(f"[dashboard] Archivo no encontrado: {path}")
        sys.exit(1)
    return records


def _resolve_path() -> str:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        return args[0]
    env = os.getenv("AGENTDOG_AUDIT_LOG", "").strip()
    return env if env else "./logs/agentdog_audit.jsonl"


# ==================== PREPARACIÓN ====================

def _build_series(records: list) -> dict:
    latency_by_agent    = defaultdict(list)   # agent    → [duration_ms]
    latency_by_strategy = defaultdict(list)   # strategy → [duration_ms]
    cost                = defaultdict(float)  # model    → USD acumulado
    outcomes_by_agent   = defaultdict(lambda: defaultdict(int))  # agent → outcome → count
    success_by_category = defaultdict(lambda: defaultdict(int))  # category → outcome → count
    efficiency          = defaultdict(list)   # agent    → [tokens/ms]

    regret_series: list[tuple[int, float]] = []  # (ts_ms, regret_estimate)
    run_idx = 0

    for r in sorted(records, key=lambda x: x.get("ts_ms", 0)):
        agent    = r.get("agent", r.get("node", "unknown"))
        outcome  = r.get("outcome", "unknown")
        dur      = r.get("duration_ms", 0) or 0
        strategy = r.get("strategy")
        category = r.get("category")

        if dur > 0:
            latency_by_agent[agent].append(dur)
            if strategy:
                latency_by_strategy[strategy].append(dur)

        outcomes_by_agent[agent][outcome] += 1

        if category:
            success_by_category[category][outcome] += 1

        if r.get("tokens_available") and dur > 0:
            model = r.get("model", "unknown")
            usd   = r.get("estimated_cost_usd")
            if usd is not None:
                cost[model] += usd
            total = r.get("total_tokens", 0) or 0
            if total > 0:
                efficiency[agent].append(total / dur)

        regret = r.get("regret_estimate")
        if regret is not None:
            run_idx += 1
            regret_series.append((run_idx, float(regret)))

    return {
        "latency_by_agent":    dict(latency_by_agent),
        "latency_by_strategy": dict(latency_by_strategy),
        "cost":                dict(cost),
        "outcomes_by_agent":   {k: dict(v) for k, v in outcomes_by_agent.items()},
        "success_by_category": {k: dict(v) for k, v in success_by_category.items()},
        "efficiency":          dict(efficiency),
        "regret_series":       regret_series,
    }


# ==================== HELPERS VISUALES ====================

_PALETTE    = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#64B5CD", "#E377C2"]
_OUT_COLOR  = {"success": "#55A868", "blocked": "#C44E52", "error": "#DD8452",
               "retry": "#DD8452", "low_confidence": "#8172B3"}
_AGENT_LABEL = lambda a: a.replace("_agent", "")


def _no_data(ax, msg="Sin datos"):
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, color="gray", fontsize=10)


def _stacked_bar(ax, groups: dict, title: str, xlabel: str):
    """Barras apiladas % de outcomes para un dict {group → {outcome → count}}."""
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("% de requests")
    if not groups:
        _no_data(ax)
        return
    keys   = sorted(groups)
    labels = [_AGENT_LABEL(k) for k in keys]
    bottoms = [0.0] * len(keys)
    all_outcomes = {o for g in groups.values() for o in g}
    order = ["success", "retry", "low_confidence", "blocked", "error"]
    sorted_outcomes = [o for o in order if o in all_outcomes] + \
                      [o for o in all_outcomes if o not in order]
    for outcome_key in sorted_outcomes:
        values = []
        for k in keys:
            total = sum(groups[k].values())
            pct   = groups[k].get(outcome_key, 0) / total * 100 if total else 0
            values.append(pct)
        color = _OUT_COLOR.get(outcome_key, _PALETTE[len(_OUT_COLOR) % len(_PALETTE)])
        ax.bar(labels, values, bottom=bottoms, label=outcome_key, color=color, alpha=0.85)
        for i, (v, b) in enumerate(zip(values, bottoms)):
            if v > 5:
                ax.text(i, b + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        bottoms = [b + v for b, v in zip(bottoms, values)]
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="x", rotation=10)


# ==================== PANELES ====================

def _plot_latency_by_strategy(ax, latency: dict):
    ax.set_title("Latencia por estrategia", fontweight="bold")
    ax.set_ylabel("ms")
    strategies = sorted(latency)
    if not strategies:
        _no_data(ax, "Sin datos de estrategia\n(solo registros sin scraping)")
        return
    data   = [latency[s] for s in strategies]
    bp = ax.boxplot(data, labels=strategies, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], _PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for i, s in enumerate(strategies, 1):
        median = np.median(latency[s])
        ymax   = max(latency[s])
        ax.text(i, median + ymax * 0.02, f"{median:.0f} ms",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel("Estrategia")
    ax.tick_params(axis="x", rotation=10)


def _plot_success_by_category(ax, success_by_category: dict):
    _stacked_bar(ax, success_by_category, "Success rate por categoría", "Categoría")


def _plot_regret(ax, regret_series: list):
    ax.set_title("Regret en el tiempo", fontweight="bold")
    ax.set_xlabel("Run #")
    ax.set_ylabel("Regret (best_delta − delta)")
    if not regret_series:
        _no_data(ax, "Sin datos de regret\n(requiere scraping con tracker activo)")
        return

    xs      = [r[0] for r in regret_series]
    ys      = [r[1] for r in regret_series]
    ax.scatter(xs, ys, s=18, alpha=0.35, color=_PALETTE[0], label="regret puntual")

    # Rolling average (ventana 5 o menos si hay pocos datos)
    window = min(5, len(ys))
    if window >= 2:
        rolling = np.convolve(ys, np.ones(window) / window, mode="valid")
        xs_roll = xs[window - 1:]
        ax.plot(xs_roll, rolling, color=_PALETTE[1], linewidth=2, label=f"rolling avg ({window})")

    # Línea en 0 → regret=0 es el óptimo teórico
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Anotación del promedio general
    avg = np.mean(ys)
    ax.text(0.02, 0.95, f"avg regret = {avg:.3f}", transform=ax.transAxes,
            fontsize=8, color=_PALETTE[1], va="top")


def _plot_latency_by_agent(ax, latency: dict):
    ax.set_title("Latencia por agente", fontweight="bold")
    ax.set_ylabel("ms")
    agents = sorted(latency)
    if not agents:
        _no_data(ax)
        return
    data   = [latency[a] for a in agents]
    labels = [_AGENT_LABEL(a) for a in agents]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], _PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for i, a in enumerate(agents, 1):
        median = np.median(latency[a])
        ax.text(i, median + max(latency[a]) * 0.02, f"{median:.0f} ms",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel("Agente")


def _plot_cost(ax, cost: dict):
    ax.set_title("Costo acumulado por modelo", fontweight="bold")
    ax.set_ylabel("USD")
    if not cost:
        _no_data(ax, "Sin datos de costo\n(tokens_available=false o modelo sin precio)")
        return
    models = sorted(cost)
    values = [cost[m] for m in models]
    bars   = ax.bar(models, values, color=_PALETTE[:len(models)], alpha=0.82)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"${v:.6f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)
    ax.set_xlabel("Modelo")


def _plot_efficiency(ax, efficiency: dict):
    ax.set_title("Eficiencia: tokens / ms", fontweight="bold")
    ax.set_ylabel("tokens / ms")
    if not efficiency:
        _no_data(ax, "Sin datos de tokens\n(tokens_available=false)")
        return
    agents = sorted(efficiency)
    labels = [_AGENT_LABEL(a) for a in agents]
    means  = [np.mean(efficiency[a]) for a in agents]
    stds   = [np.std(efficiency[a])  for a in agents]
    bars   = ax.bar(labels, means, yerr=stds, capsize=5,
                    color=_PALETTE[:len(agents)], alpha=0.82,
                    error_kw={"elinewidth": 1.5, "ecolor": "#333"})
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel("Agente")


# ==================== MAIN ====================

def main():
    path    = _resolve_path()
    records = _load_node_outcomes(path)
    no_show = "--no-show" in sys.argv

    if not records:
        print(f"[dashboard] Sin registros de outcome en {path}")
        sys.exit(0)

    print(f"[dashboard] {len(records)} registros cargados desde {path}")
    series = _build_series(records)

    # Resumen en consola
    total   = len(records)
    success = sum(1 for r in records if r.get("outcome") == "success")
    blocked = sum(1 for r in records if r.get("outcome") == "blocked")
    errors  = sum(1 for r in records if r.get("outcome") in ("error", "low_confidence"))
    regrets = [r.get("regret_estimate") for r in records if r.get("regret_estimate") is not None]
    followup = sum(1 for r in records if r.get("followup_likely") is True)
    print(f"  success={success}  blocked={blocked}  error/low={errors}  "
          f"followup_likely={followup}/{total} ({followup/total*100:.1f}%)")
    if regrets:
        print(f"  regret  avg={np.mean(regrets):.3f}  p95={np.percentile(regrets, 95):.3f}")

    # Figura 2×3
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Multi-Agent Observability Dashboard", fontsize=16, fontweight="bold", y=1.01)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.38)

    # Fila 1: estrategia / categoría / tiempo
    _plot_latency_by_strategy(fig.add_subplot(gs[0, 0]), series["latency_by_strategy"])
    _plot_success_by_category(fig.add_subplot(gs[0, 1]), series["success_by_category"])
    _plot_regret             (fig.add_subplot(gs[0, 2]), series["regret_series"])

    # Fila 2: agente / modelo / eficiencia
    _plot_latency_by_agent(fig.add_subplot(gs[1, 0]), series["latency_by_agent"])
    _plot_cost            (fig.add_subplot(gs[1, 1]), series["cost"])
    _plot_efficiency      (fig.add_subplot(gs[1, 2]), series["efficiency"])

    out_path = Path(path).with_suffix(".png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[dashboard] PNG guardado → {out_path}")

    if not no_show:
        plt.show()


if __name__ == "__main__":
    main()
