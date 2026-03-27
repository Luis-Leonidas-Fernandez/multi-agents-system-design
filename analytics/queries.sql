-- analytics/queries.sql
-- Queries de análisis para el sistema multi-agentes.
-- Requiere DuckDB: pip install duckdb
--
-- Uso:
--   duckdb < analytics/queries.sql
--   duckdb -c ".read analytics/queries.sql"
--
-- Ajusta las rutas si tu audit log o DB están en otro path.

-- ==================== CONFIG ====================

-- Rutas (editar si es necesario)
-- AUDIT_LOG: logs/agentdog_audit.jsonl
-- SESSIONS_DB: sessions/sessions.db

-- ==================== 1. DEBUGGING ====================
-- Inputs del usuario donde el sistema falló (scrape_score negativo).
-- Útil para identificar queries que el sistema no sabe manejar bien.

SELECT
    m.content           AS user_input,
    a.strategy,
    a.scrape_score,
    a.scrape_reliability,
    a.outcome
FROM read_parquet('sessions/sessions.db')  -- DuckDB puede leer SQLite con extensión
-- Alternativa directa: attach 'sessions/sessions.db' AS sessions; SELECT * FROM sessions.messages
JOIN read_ndjson('logs/agentdog_audit.jsonl') a ON m.request_id = a.request_id
WHERE m.role = 'human'
  AND a.scrape_score < 0
ORDER BY a.scrape_score ASC;

-- ==================== 2. RANKING DE ESTRATEGIAS ====================
-- Score promedio, tasa de éxito y conteo por estrategia.
-- Base directa para mejorar la policy del bandit.

-- Segmentado por (category, strategy) para evitar conclusiones globales falsas.
-- Ejemplo: "force_search es mejor" puede ser cierto solo en crypto_price.
SELECT
    a.category,
    a.strategy,
    COUNT(*)                                    AS n,
    ROUND(AVG(a.scrape_score), 3)               AS avg_score,
    ROUND(AVG(a.quality_target), 3)             AS success_rate,
    ROUND(AVG(a.duration_ms), 0)                AS avg_latency_ms,
    ROUND(AVG(a.estimated_cost_usd) * 1000, 4)  AS avg_cost_musd
FROM read_ndjson('logs/agentdog_audit.jsonl') a
WHERE a.strategy IS NOT NULL
GROUP BY a.category, a.strategy
ORDER BY a.category, avg_score DESC;

-- ==================== 3. VISTA COMPLETA POR SESIÓN ====================
-- Input del usuario → decisión del sistema → outcome.
-- Debugging real + evaluación de policy.
-- Reemplaza <session_id> con el ID de la sesión a analizar.

ATTACH 'sessions/sessions.db' AS sessions;

SELECT
    m.role,
    m.content,
    a.strategy,
    a.scrape_score,
    a.quality_target,
    a.prediction_match,
    a.ml_would_succeed,
    a.outcome,
    a.scrape_reliability,
    a.duration_ms
FROM sessions.messages m
JOIN read_ndjson('logs/agentdog_audit.jsonl') a ON m.request_id = a.request_id
WHERE m.session_id = '<session_id>'
ORDER BY m.ts, m.id;

-- ==================== 4. MALAS DECISIONES DEL SISTEMA ====================
-- Turnos donde el sistema eligió una estrategia pero el outcome fue malo.
-- Útil para detectar fallas del bandit vs. fallas del ML.

SELECT
    m.content           AS user_input,
    a.strategy          AS chosen,
    a.ml_recommended    AS ml_said,
    a.prediction_match,
    a.scrape_score,
    a.outcome
FROM sessions.messages m
JOIN read_ndjson('logs/agentdog_audit.jsonl') a ON m.request_id = a.request_id
WHERE m.role       = 'human'
  AND a.outcome   != 'success'
ORDER BY a.ts DESC
LIMIT 50;

-- ==================== 5. COMPARATIVA API vs SCRAPING ====================
-- Latencia y costo según tipo de fuente.

SELECT
    a.source_type,
    COUNT(*)                                    AS n,
    ROUND(AVG(a.duration_ms), 0)                AS avg_latency_ms,
    ROUND(AVG(a.estimated_cost_usd) * 1000, 4)  AS avg_cost_musd,
    ROUND(AVG(a.quality_target), 3)             AS success_rate
FROM read_ndjson('logs/agentdog_audit.jsonl') a
WHERE a.node = 'web_scraping_node'
  AND a.source_type IS NOT NULL
GROUP BY a.source_type;

-- ==================== 7. COUNTERFACTUAL INSIGHT ====================
-- Casos donde el bandit eligió algo distinto al modelo Y el resultado fue exitoso.
-- Traducción: "el bandit le ganó al modelo" → indica sesgo o dato escaso en el ML.
-- Útil para:
--   - Mejorar features del modelo (¿por qué no predijo esto?)
--   - Detectar casos donde la exploración fue la decisión correcta
--   - Auditar sesgos de política

SELECT
    a.request_id,
    a.category,
    a.strategy          AS bandit_chose,
    a.ml_recommended    AS model_said,
    a.scrape_score,
    a.quality_target,
    a.exploring,
    a.exp_rate,
    a.ts_ms
FROM read_ndjson('logs/agentdog_audit.jsonl') a
WHERE a.prediction_match = false
  AND a.quality_target   = 1
ORDER BY a.ts_ms DESC;

-- ==================== 6. LEARNING CURVE (regret por turno) ====================
-- Evolución del regret a lo largo del tiempo.
-- Muestra si el sistema mejora con más datos.

SELECT
    ROW_NUMBER() OVER (ORDER BY a.ts) AS run,
    a.strategy,
    a.regret_estimate,
    AVG(a.quality_target) OVER (
        ORDER BY a.ts ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) AS rolling_success_10
FROM read_ndjson('logs/agentdog_audit.jsonl') a
WHERE a.node = 'web_scraping_node'
  AND a.regret_estimate IS NOT NULL
ORDER BY a.ts;
