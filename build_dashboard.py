#!/usr/bin/env python3
"""
Build web dashboard: lee audit JSONL → genera dist/index.html

Local:
    python build_dashboard.py [logs/agentdog_audit.jsonl]
    python build_dashboard.py --serve      # genera + abre browser
    python build_dashboard.py --watch      # reconstruye en vivo + server en :8765

Netlify:
    Build command : python build_dashboard.py
    Publish dir   : dist

El HTML resultante es auto-contenido (datos embebidos como JSON, charts via CDN).
No requiere servidor — es un archivo estático puro.
"""
import json
import math
import os
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def _percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def _load(path: str) -> list:
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
        print(f"[build] No encontrado: {path}", file=sys.stderr)
        sys.exit(1)
    return records


def _resolve_path() -> str:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        return args[0]
    env = os.getenv("AGENTDOG_AUDIT_LOG", "").strip()
    return env if env else "./logs/agentdog_audit.jsonl"


# ── data processing ───────────────────────────────────────────────────────────

def _build_payload(records: list) -> dict:
    latency_by_agent    = defaultdict(list)
    latency_by_strategy = defaultdict(list)
    cost                = defaultdict(float)
    outcomes_by_agent   = defaultdict(lambda: defaultdict(int))
    success_by_category = defaultdict(lambda: defaultdict(int))
    efficiency          = defaultdict(list)
    regret_series: list = []
    run_idx = 0
    total = len(records)
    n_success = n_blocked = n_errors = 0

    for r in sorted(records, key=lambda x: x.get("ts_ms", 0)):
        agent    = r.get("agent", r.get("node", "unknown"))
        outcome  = r.get("outcome", "unknown")
        dur      = r.get("duration_ms", 0) or 0
        strategy = r.get("strategy")
        category = r.get("category")

        if outcome == "success":
            n_success += 1
        elif outcome == "blocked":
            n_blocked += 1
        elif outcome in ("error", "low_confidence"):
            n_errors += 1

        if dur > 0:
            latency_by_agent[agent].append(dur)
            if strategy:
                latency_by_strategy[strategy].append(dur)

        outcomes_by_agent[agent][outcome] += 1
        if category:
            success_by_category[category][outcome] += 1

        usd = r.get("estimated_cost_usd")
        if usd is not None:
            cost[r.get("model", "unknown")] += usd

        if r.get("tokens_available") and dur > 0:
            total_tokens = r.get("total_tokens", 0) or 0
            if total_tokens > 0:
                efficiency[agent].append(total_tokens / dur)

        regret = r.get("regret_estimate")
        if regret is not None:
            run_idx += 1
            regret_series.append([run_idx, float(regret)])

    def lat_stats(values):
        if not values:
            return None
        return {
            "median": round(_percentile(values, 50)),
            "p95":    round(_percentile(values, 95)),
            "avg":    round(sum(values) / len(values)),
            "min":    round(min(values)),
            "max":    round(max(values)),
            "n":      len(values),
        }

    def eff_stats(values):
        if not values:
            return None
        mean = sum(values) / len(values)
        var  = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
        return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4), "n": len(values)}

    avg_regret = (
        round(sum(p[1] for p in regret_series) / len(regret_series), 4)
        if regret_series else None
    )

    return {
        "generated_at":       datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "total":              total,
        "success":            n_success,
        "blocked":            n_blocked,
        "errors":             n_errors,
        "avg_regret":         avg_regret,
        "latency_by_strategy": {k: lat_stats(v) for k, v in latency_by_strategy.items()},
        "latency_by_agent":    {k: lat_stats(v) for k, v in latency_by_agent.items()},
        "cost":                {k: round(v, 8) for k, v in cost.items()},
        "outcomes_by_agent":   {k: dict(v) for k, v in outcomes_by_agent.items()},
        "success_by_category": {k: dict(v) for k, v in success_by_category.items()},
        "efficiency":          {k: eff_stats(v) for k, v in efficiency.items()},
        "regret_series":       regret_series,
    }


# ── HTML template ─────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Multi-Agent Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:       #07070f;
      --surface:  #0d0d1a;
      --card:     #111120;
      --border:   #1c1c35;
      --text:     #e2e8f0;
      --muted:    #4a5568;
      --accent:   #818cf8;
      --success:  #4ade80;
      --error:    #f87171;
      --warning:  #fb923c;
      --purple:   #c084fc;
    }

    body {
      font-family: 'Inter', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 28px 24px 48px;
      max-width: 1400px;
      margin: 0 auto;
    }

    /* ── header ── */
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 36px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
    }
    header h1 {
      font-size: 1.25rem;
      font-weight: 600;
      letter-spacing: -0.02em;
    }
    header h1 .dot { color: var(--accent); }
    .header-meta { text-align: right; }
    .header-meta .ts { font-size: 0.72rem; color: var(--muted); display: block; }
    .live-badge {
      display: inline-block;
      font-size: 0.65rem;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      padding: 2px 8px;
      border-radius: 999px;
      background: #4ade8020;
      color: var(--success);
      border: 1px solid #4ade8040;
      margin-bottom: 6px;
    }

    /* ── stat cards ── */
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 14px;
      margin-bottom: 28px;
    }
    .stat-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px 20px;
      position: relative;
      overflow: hidden;
    }
    .stat-card::before {
      content: '';
      position: absolute;
      inset: 0;
      border-radius: 14px;
      background: var(--glow, transparent);
      pointer-events: none;
    }
    .stat-card.c-accent  { --glow: radial-gradient(ellipse at top left, #818cf810, transparent 60%); }
    .stat-card.c-success { --glow: radial-gradient(ellipse at top left, #4ade8010, transparent 60%); }
    .stat-card.c-warning { --glow: radial-gradient(ellipse at top left, #fb923c10, transparent 60%); }
    .stat-card.c-error   { --glow: radial-gradient(ellipse at top left, #f8717110, transparent 60%); }
    .stat-card.c-purple  { --glow: radial-gradient(ellipse at top left, #c084fc10, transparent 60%); }
    .stat-card .label {
      font-size: 0.68rem;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: var(--muted);
      margin-bottom: 10px;
    }
    .stat-card .value {
      font-size: 2.1rem;
      font-weight: 700;
      letter-spacing: -0.04em;
      line-height: 1;
    }
    .stat-card .sub {
      font-size: 0.7rem;
      color: var(--muted);
      margin-top: 6px;
    }
    .c-accent  .value { color: var(--accent); }
    .c-success .value { color: var(--success); }
    .c-warning .value { color: var(--warning); }
    .c-error   .value { color: var(--error); }
    .c-purple  .value { color: var(--purple); }

    /* ── charts grid ── */
    .charts-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 14px;
    }
    .chart-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px 22px 22px;
    }
    .chart-card h2 {
      font-size: 0.68rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 18px;
    }
    .chart-wrap {
      position: relative;
      height: 230px;
    }
    .no-data {
      height: 230px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      font-size: 0.8rem;
      border: 1px dashed var(--border);
      border-radius: 8px;
    }

    /* ── footer ── */
    footer {
      margin-top: 36px;
      text-align: center;
      font-size: 0.7rem;
      color: var(--muted);
      letter-spacing: 0.03em;
    }

    /* ── responsive ── */
    @media (max-width: 1100px) {
      .charts-grid { grid-template-columns: repeat(2, 1fr); }
      .stats-grid  { grid-template-columns: repeat(3, 1fr); }
    }
    @media (max-width: 700px) {
      .charts-grid { grid-template-columns: 1fr; }
      .stats-grid  { grid-template-columns: repeat(2, 1fr); }
    }
  </style>
</head>
<body>

  <header>
    <h1>Multi<span class="dot">·</span>Agent Dashboard</h1>
    <div class="header-meta">
      <span class="live-badge">● live</span>
      <span class="ts" id="ts"></span>
    </div>
  </header>

  <div class="stats-grid">
    <div class="stat-card c-accent">
      <div class="label">Total Requests</div>
      <div class="value" id="s-total">—</div>
      <div class="sub">sesiones registradas</div>
    </div>
    <div class="stat-card c-success">
      <div class="label">Success Rate</div>
      <div class="value" id="s-rate">—</div>
      <div class="sub" id="s-abs"></div>
    </div>
    <div class="stat-card c-warning">
      <div class="label">Blocked</div>
      <div class="value" id="s-blocked">—</div>
      <div class="sub">por guardrail</div>
    </div>
    <div class="stat-card c-error">
      <div class="label">Errors</div>
      <div class="value" id="s-errors">—</div>
      <div class="sub">error + low_conf</div>
    </div>
    <div class="stat-card c-purple">
      <div class="label">Avg Regret</div>
      <div class="value" id="s-regret">—</div>
      <div class="sub">best − actual delta</div>
    </div>
  </div>

  <div class="charts-grid">
    <div class="chart-card">
      <h2>Latencia por Estrategia</h2>
      <div id="w-lat-strat"><div class="chart-wrap"><canvas id="c-lat-strat"></canvas></div></div>
    </div>
    <div class="chart-card">
      <h2>Success Rate por Categoría</h2>
      <div id="w-success-cat"><div class="chart-wrap"><canvas id="c-success-cat"></canvas></div></div>
    </div>
    <div class="chart-card">
      <h2>Regret en el Tiempo</h2>
      <div id="w-regret"><div class="chart-wrap"><canvas id="c-regret"></canvas></div></div>
    </div>
    <div class="chart-card">
      <h2>Latencia por Agente</h2>
      <div id="w-lat-agent"><div class="chart-wrap"><canvas id="c-lat-agent"></canvas></div></div>
    </div>
    <div class="chart-card">
      <h2>Costo por Modelo</h2>
      <div id="w-cost"><div class="chart-wrap"><canvas id="c-cost"></canvas></div></div>
    </div>
    <div class="chart-card">
      <h2>Eficiencia · tokens / ms</h2>
      <div id="w-eff"><div class="chart-wrap"><canvas id="c-eff"></canvas></div></div>
    </div>
  </div>

  <footer>Multi-Agent Observability · generado con build_dashboard.py</footer>

  <script>
    const DATA = __DATA_PLACEHOLDER__;

    // ── Chart.js defaults ────────────────────────────────────────────────────
    Chart.defaults.color          = '#4a5568';
    Chart.defaults.borderColor    = '#1c1c35';
    Chart.defaults.font.family    = "'Inter', system-ui, sans-serif";
    Chart.defaults.font.size      = 11;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.pointStyleWidth = 8;

    const C = ['#818cf8','#4ade80','#fb923c','#f87171','#c084fc','#38bdf8','#fbbf24','#f472b6'];
    const OUTCOME_COLOR = {
      success: '#4ade80', blocked: '#f87171', error: '#f87171',
      retry: '#fb923c', low_confidence: '#c084fc'
    };

    function oc(o) { return OUTCOME_COLOR[o] || '#818cf8'; }

    function noData(wrapId, msg = 'Sin datos suficientes') {
      document.getElementById(wrapId).innerHTML =
        `<div class="no-data">${msg}</div>`;
    }

    const gridColor = '#1c1c35';
    const baseScales = {
      x: { grid: { color: gridColor } },
      y: { grid: { color: gridColor } },
    };

    // ── Stats ────────────────────────────────────────────────────────────────
    document.getElementById('ts').textContent = 'Actualizado: ' + DATA.generated_at;
    document.getElementById('s-total').textContent   = DATA.total;
    document.getElementById('s-blocked').textContent = DATA.blocked;
    document.getElementById('s-errors').textContent  = DATA.errors;
    document.getElementById('s-regret').textContent  =
      DATA.avg_regret != null ? DATA.avg_regret.toFixed(3) : '—';

    const sr = DATA.total > 0 ? Math.round(DATA.success / DATA.total * 100) : 0;
    document.getElementById('s-rate').textContent = sr + '%';
    document.getElementById('s-abs').textContent  = `${DATA.success} de ${DATA.total}`;

    // ── 1. Latencia por estrategia (P50 + P95 grouped bars) ──────────────────
    (function () {
      const lats = DATA.latency_by_strategy;
      const keys = Object.keys(lats).filter(k => lats[k]).sort(
        (a, b) => (lats[a]?.median || 0) - (lats[b]?.median || 0)
      );
      if (!keys.length) { noData('w-lat-strat'); return; }

      new Chart(document.getElementById('c-lat-strat'), {
        type: 'bar',
        data: {
          labels: keys,
          datasets: [
            {
              label: 'P50',
              data: keys.map(k => lats[k]?.median || 0),
              backgroundColor: '#818cf8aa',
              borderColor: '#818cf8',
              borderWidth: 1,
              borderRadius: 5,
            },
            {
              label: 'P95',
              data: keys.map(k => lats[k]?.p95 || 0),
              backgroundColor: '#fb923c55',
              borderColor: '#fb923c',
              borderWidth: 1,
              borderRadius: 5,
            },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { position: 'top' },
            tooltip: {
              callbacks: {
                afterBody: (items) => {
                  const k = keys[items[0].dataIndex];
                  const d = lats[k];
                  return d ? [`n=${d.n}  avg=${d.avg}ms`] : [];
                },
              },
            },
          },
          scales: {
            ...baseScales,
            y: { ...baseScales.y, title: { display: true, text: 'ms' } },
            x: { ...baseScales.x, ticks: { maxRotation: 20 } },
          },
        },
      });
    })();

    // ── 2. Success rate por categoría (stacked %) ────────────────────────────
    (function () {
      const cats = DATA.success_by_category;
      const keys = Object.keys(cats).sort();
      if (!keys.length) { noData('w-success-cat'); return; }

      const allOut = new Set(keys.flatMap(k => Object.keys(cats[k])));
      const ORDER  = ['success', 'retry', 'low_confidence', 'blocked', 'error'];
      const outcomes = [
        ...ORDER.filter(o => allOut.has(o)),
        ...[...allOut].filter(o => !ORDER.includes(o)),
      ];

      new Chart(document.getElementById('c-success-cat'), {
        type: 'bar',
        data: {
          labels: keys,
          datasets: outcomes.map(o => ({
            label: o,
            data: keys.map(k => {
              const total = Object.values(cats[k]).reduce((a, b) => a + b, 0);
              return total ? Math.round((cats[k][o] || 0) / total * 100) : 0;
            }),
            backgroundColor: oc(o) + 'bb',
            borderColor:     oc(o),
            borderWidth: 1,
            borderRadius: 3,
          })),
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { position: 'top' },
            tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.raw}%` } },
          },
          scales: {
            x: { ...baseScales.x, stacked: true, ticks: { maxRotation: 20 } },
            y: { ...baseScales.y, stacked: true, max: 100, title: { display: true, text: '%' } },
          },
        },
      });
    })();

    // ── 3. Regret en el tiempo ───────────────────────────────────────────────
    (function () {
      const series = DATA.regret_series;
      if (!series.length) { noData('w-regret', 'Sin datos de regret'); return; }

      const xs = series.map(p => p[0]);
      const ys = series.map(p => p[1]);
      const win = Math.min(5, ys.length);
      const rolling = [], rollingXs = [];
      for (let i = win - 1; i < ys.length; i++) {
        const sl = ys.slice(i - win + 1, i + 1);
        rolling.push(sl.reduce((a, b) => a + b, 0) / win);
        rollingXs.push(xs[i]);
      }

      new Chart(document.getElementById('c-regret'), {
        type: 'scatter',
        data: {
          datasets: [
            {
              label: 'Regret',
              data: xs.map((x, i) => ({ x, y: ys[i] })),
              backgroundColor: '#818cf855',
              borderColor:     '#818cf8',
              pointRadius: 4,
              pointHoverRadius: 6,
            },
            {
              label: `Rolling avg (${win})`,
              type: 'line',
              data: rollingXs.map((x, i) => ({ x, y: rolling[i] })),
              borderColor: '#fb923c',
              borderWidth: 2,
              pointRadius: 0,
              fill: false,
              tension: 0.35,
            },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { position: 'top' } },
          scales: {
            x: { ...baseScales.x, type: 'linear', title: { display: true, text: 'Run #' } },
            y: { ...baseScales.y, title: { display: true, text: 'Regret' } },
          },
        },
      });
    })();

    // ── 4. Latencia por agente ───────────────────────────────────────────────
    (function () {
      const lats = DATA.latency_by_agent;
      const keys = Object.keys(lats).filter(k => lats[k]).sort(
        (a, b) => (lats[a]?.median || 0) - (lats[b]?.median || 0)
      );
      if (!keys.length) { noData('w-lat-agent'); return; }
      const labels = keys.map(k => k.replace('_agent', ''));

      new Chart(document.getElementById('c-lat-agent'), {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            label: 'P50 (ms)',
            data: keys.map(k => lats[k]?.median || 0),
            backgroundColor: keys.map((_, i) => C[i % C.length] + 'aa'),
            borderColor:     keys.map((_, i) => C[i % C.length]),
            borderWidth: 1,
            borderRadius: 5,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                afterBody: items => {
                  const k = keys[items[0].dataIndex];
                  const d = lats[k];
                  return d ? [`P95: ${d.p95}ms  n=${d.n}`] : [];
                },
              },
            },
          },
          scales: {
            ...baseScales,
            y: { ...baseScales.y, title: { display: true, text: 'ms' } },
          },
        },
      });
    })();

    // ── 5. Costo por modelo (doughnut) ───────────────────────────────────────
    (function () {
      const cost = DATA.cost;
      const keys = Object.keys(cost).filter(k => cost[k] > 0);
      if (!keys.length) { noData('w-cost', 'Sin datos de costo'); return; }

      new Chart(document.getElementById('c-cost'), {
        type: 'doughnut',
        data: {
          labels: keys,
          datasets: [{
            data: keys.map(k => cost[k]),
            backgroundColor: C.map(c => c + 'bb'),
            borderColor:     C,
            borderWidth: 1,
            hoverOffset: 6,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { position: 'right' },
            tooltip: { callbacks: { label: ctx => `${ctx.label}: $${ctx.raw.toFixed(6)}` } },
          },
          cutout: '62%',
        },
      });
    })();

    // ── 6. Eficiencia ────────────────────────────────────────────────────────
    (function () {
      const eff  = DATA.efficiency;
      const keys = Object.keys(eff).filter(k => eff[k]).sort(
        (a, b) => (eff[b]?.mean || 0) - (eff[a]?.mean || 0)
      );
      if (!keys.length) { noData('w-eff', 'Sin datos de tokens'); return; }
      const labels = keys.map(k => k.replace('_agent', ''));

      new Chart(document.getElementById('c-eff'), {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            label: 'tokens/ms',
            data: keys.map(k => eff[k]?.mean || 0),
            backgroundColor: keys.map((_, i) => C[i % C.length] + 'aa'),
            borderColor:     keys.map((_, i) => C[i % C.length]),
            borderWidth: 1,
            borderRadius: 5,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                afterBody: items => {
                  const k = keys[items[0].dataIndex];
                  const d = eff[k];
                  return d ? [`±${d.std}  n=${d.n}`] : [];
                },
              },
            },
          },
          scales: {
            ...baseScales,
            y: { ...baseScales.y, title: { display: true, text: 'tok/ms' } },
          },
        },
      });
    })();
  </script>
</body>
</html>
"""


# ── build ─────────────────────────────────────────────────────────────────────

def build(src_path: str, out_path: str = "dist/index.html", *, live: bool = False) -> str:
    """Genera dist/index.html. live=True inyecta auto-refresh cada 5s."""
    records = _load(src_path)
    if not records:
        print(f"[build] Sin registros de outcome en {src_path}", file=sys.stderr)
        sys.exit(0)
    print(f"[build] {len(records)} registros  →  {out_path}")

    payload = _build_payload(records)
    html    = _HTML.replace("__DATA_PLACEHOLDER__", json.dumps(payload, ensure_ascii=False))

    if live:
        # Inyecta <meta refresh> para que el browser recargue automáticamente
        html = html.replace(
            "</head>",
            '  <meta http-equiv="refresh" content="5">\n</head>',
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html, encoding="utf-8")
    return out_path


def _watch(src_path: str, out_path: str, poll: float = 3.0):
    """Observa el JSONL y reconstruye el HTML cuando cambia."""
    import http.server
    import webbrowser

    # Servidor HTTP en un thread aparte (sirve dist/)
    dist_dir = str(Path(out_path).parent.resolve())
    port = 8765

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=dist_dir, **kw)
        def log_message(self, *_):   # silenciar logs del servidor
            pass

    server = http.server.HTTPServer(("", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    url = f"http://localhost:{port}"
    print(f"[watch] Servidor en {url}  (Ctrl+C para salir)")
    print(f"[watch] Observando {src_path} …")

    # Primera build + abrir browser
    build(src_path, out_path, live=True)
    webbrowser.open(url)

    last_mtime = Path(src_path).stat().st_mtime
    while True:
        time.sleep(poll)
        try:
            mtime = Path(src_path).stat().st_mtime
        except FileNotFoundError:
            continue
        if mtime != last_mtime:
            last_mtime = mtime
            print(f"[watch] Cambio detectado → reconstruyendo…")
            build(src_path, out_path, live=True)
            # El browser se recarga solo gracias al <meta refresh>


def main():
    src   = _resolve_path()
    out   = "dist/index.html"
    flags = set(sys.argv[1:])

    if "--watch" in flags:
        _watch(src, out)          # bloquea hasta Ctrl+C
        return

    out_path = build(src, out, live=False)

    if "--serve" in flags:
        import webbrowser
        webbrowser.open(f"file://{Path(out_path).resolve()}")
        print("[build] Abriendo en el browser…")


if __name__ == "__main__":
    main()
