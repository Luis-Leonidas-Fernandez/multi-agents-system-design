#!/usr/bin/env python3
"""
Observabilidad del sistema de noticias por país.

Lee eventos `country_news_resolution` del audit log (JSONL) y produce:
  1. Tasa de resolución por path (bootstrap / dynamic / none)
  2. Top-N países no curados más frecuentes
  3. Top-N dominios dinámicos más descubiertos
  4. Tasa de error por path (resolution_path = "none")

Uso:
    python ops/country_news_analytics.py [audit.jsonl]
    python ops/country_news_analytics.py --no-show
"""
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ── Carga ──────────────────────────────────────────────────────────────────

def _load_resolution_events(path: str) -> list[dict]:
    events = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("event_type") == "country_news_resolution":
                        events.append(r)
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        print(f"[country_news_analytics] Archivo no encontrado: {path}")
        sys.exit(1)
    return events


def _resolve_path() -> str:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        return args[0]
    return os.getenv("AGENTDOG_AUDIT_LOG", "./logs/agentdog_audit.jsonl")


# ── Métricas ───────────────────────────────────────────────────────────────

def resolution_rates(events: list[dict]) -> dict[str, float]:
    """% por path: bootstrap / dynamic / none."""
    if not events:
        return {}
    counts: Counter = Counter(e.get("resolution_path", "unknown") for e in events)
    total = len(events)
    return {path: round(n / total * 100, 1) for path, n in counts.most_common()}


def top_uncurated_countries(events: list[dict], top_n: int = 10) -> list[tuple[str, int]]:
    """Top países que cayeron en dynamic o none (no estaban en bootstrap)."""
    dynamic_or_none = [
        e for e in events
        if e.get("resolution_path") in {"dynamic", "none"}
        and e.get("geography")
    ]
    counter: Counter = Counter(e["geography"] for e in dynamic_or_none)
    return counter.most_common(top_n)


def top_dynamic_domains(events: list[dict], top_n: int = 10) -> list[tuple[str, int]]:
    """Dominios descubiertos más veces por el path dinámico."""
    dynamic = [e for e in events if e.get("resolution_path") == "dynamic"]
    # domains_found es un int — no podemos listar dominios individuales sin
    # cambiar el schema, pero sí podemos ver cuántas veces se descubrió
    # cada país con éxito y cuántos dominios trajo de media.
    by_geo: dict[str, list[int]] = defaultdict(list)
    for e in dynamic:
        geo = e.get("geography") or "unknown"
        by_geo[geo].append(e.get("domains_found", 0))
    return sorted(
        [(geo, sum(counts)) for geo, counts in by_geo.items()],
        key=lambda x: -x[1],
    )[:top_n]


def error_rate_by_path(events: list[dict]) -> dict[str, dict]:
    """Para cada path, total y % que terminó en none (sin resultado)."""
    totals: Counter = Counter(e.get("resolution_path", "unknown") for e in events)
    nones = sum(1 for e in events if e.get("resolution_path") == "none")
    total = len(events)
    return {
        "total_events": total,
        "none_count": nones,
        "none_rate_pct": round(nones / total * 100, 1) if total else 0.0,
        "by_path": dict(totals),
    }


# ── Presentación ───────────────────────────────────────────────────────────

def _print_report(events: list[dict]) -> None:
    total = len(events)
    print(f"\n{'=' * 56}")
    print(f"  COUNTRY NEWS ANALYTICS  —  {total} eventos")
    print(f"{'=' * 56}")

    # 1. Tasa de resolución
    rates = resolution_rates(events)
    print("\n1. TASA DE RESOLUCIÓN POR PATH")
    print(f"   {'PATH':<22s} {'%':>6s}")
    print(f"   {'-' * 30}")
    for path, pct in rates.items():
        marker = " ✓" if path == "bootstrap" else (" ↗" if path == "dynamic" else " ✗")
        print(f"   {path:<22s} {pct:>5.1f}%{marker}")

    # 2. Top países no curados
    uncurated = top_uncurated_countries(events)
    print("\n2. TOP PAÍSES NO CURADOS (dynamic + none)")
    if uncurated:
        print(f"   {'PAÍS':<28s} {'REQUESTS':>8s}")
        print(f"   {'-' * 38}")
        for geo, n in uncurated:
            print(f"   {geo:<28s} {n:>8d}")
    else:
        print("   (sin datos)")

    # 3. Dominios dinámicos más activos
    dyn_domains = top_dynamic_domains(events)
    print("\n3. PAÍSES CON MÁS DOMINIOS DESCUBIERTOS DINÁMICAMENTE")
    if dyn_domains:
        print(f"   {'PAÍS':<28s} {'TOTAL DOMINIOS':>14s}")
        print(f"   {'-' * 44}")
        for geo, total_doms in dyn_domains:
            print(f"   {geo:<28s} {total_doms:>14d}")
    else:
        print("   (sin datos)")

    # 4. Tasa de error
    err = error_rate_by_path(events)
    print("\n4. TASA SIN RESOLUCIÓN")
    print(f"   none_rate : {err['none_rate_pct']}%  ({err['none_count']} / {err['total_events']})")
    print(f"   by_path   : {err['by_path']}")

    print(f"\n{'=' * 56}\n")


def plot_resolution_pie(events: list[dict], out_stem: Path, no_show: bool) -> None:
    """Gráfico de torta: distribución de resolution_path."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[country_news_analytics] matplotlib no instalado — pip install matplotlib")
        return

    rates = resolution_rates(events)
    if not rates:
        return

    labels = list(rates.keys())
    sizes = [rates[l] for l in labels]
    colors = {
        "bootstrap": "#4caf50",
        "dynamic": "#2196f3",
        "none": "#f44336",
    }
    pie_colors = [colors.get(l, "#9e9e9e") for l in labels]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 12},
    )
    ax.set_title("Country News — Resolución por Path", fontsize=14, fontweight="bold")

    out_path = Path(f"{out_stem}_country_news_resolution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[country_news_analytics] Gráfico → {out_path}")
    if not no_show:
        plt.show()
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    path = _resolve_path()
    no_show = "--no-show" in sys.argv

    events = _load_resolution_events(path)
    if not events:
        print(f"[country_news_analytics] Sin eventos 'country_news_resolution' en {path}")
        sys.exit(0)

    _print_report(events)

    out_stem = Path(path).with_suffix("")
    plot_resolution_pie(events, out_stem, no_show)


if __name__ == "__main__":
    main()
