# Fase 0 — Comportamiento actual del sistema de noticias por país

Documento de baseline previo al refactor. Describe exactamente cómo responde el sistema
hoy para los 5 tipos de caso relevantes. Sin este baseline, el refactor se hace a ciegas.

---

## Flujo principal

La clase `CountryRecentNewsStrategy` (`web_scraping_flow.py:236`) es la estrategia
responsable de noticias locales por país. Su ejecución sigue este orden:

```
_should_use_country_recent_news_strategy()   ← gate de entrada
  → detect_query_source_group()              ← ¿el país está en web_source_policy.json?
  → _discover_country_press_sources()        ← busca medios candidatos
      → _discover_country_press_sources_via_directory()  ← periodicos.com.ar
      → search fallback (Tavily site:periodicos.com.ar)
      → homepage fallback (periodicos.com.ar raíz)
  → _build_country_press_section_targets()   ← construye URLs a scrapear
      → _COUNTRY_PRESS_SECTION_PATHS[domain] ← paths curados por dominio
      → _GENERIC_SECTION_PATHS[topic]        ← fallback genérico
  → scrape de secciones + filtro de candidatos
```

**Gate de entrada** (`web_scraping_flow.py:1273`):

La estrategia aborta inmediatamente si:
- `query_source_group` es `None` → país NO registrado en `web_source_policy.json`
- El horizonte temporal no es `today`, `week` o `month`
- El query contiene palabras de deportes
- No hay palabras de noticias ni tópico reconocido (security/economy/politics)

---

## Estructuras de datos que gobiernan el comportamiento

| Estructura | Archivo | Entradas | Rol |
|---|---|---|---|
| `source_groups` | `web_source_policy.json` | 21 países | Gate primario: si el país no está acá, la estrategia no arranca |
| `_GEOGRAPHY_TERMS` | `web_scraping_flow.py:1290` | 85+ demonyms | Extrae nombre canónico del país desde el query |
| `_GEO_ENGLISH` | `web_scraping_flow.py:1461` | 45+ traducciones | Mapea nombre es → en para búsquedas anglófonas |
| `_PERIODICOS_CONTINENT_SLUG_BY_COUNTRY` | `web_scraping_flow.py:1496` | 62 países | Necesario para navegar el directorio periodicos.com.ar |
| `_COUNTRY_PRESS_SECTION_PATHS` | `web_scraping_flow.py:1874` | ~30 dominios | Paths curados por dominio+tópico; sólo existe para Italia y España actualmente |

---

## Caso 1 — País conocido + idioma español

**Ejemplo**: `"noticias de Argentina esta semana"`

### Flujo

1. `detect_query_source_group` → `"argentina"` ✓ (está en `source_groups`)
2. `_should_use_country_recent_news_strategy` → `True` (group + horizon "week" + "noticias")
3. `get_query_source_terms` → `["argentina", "argentino", "argentinos", "argentinas"]`
4. `_extract_query_geography` → `"Argentina"` ✓ (hit en `_GEOGRAPHY_TERMS`)
5. `_PERIODICOS_CONTINENT_SLUG_BY_COUNTRY["Argentina"]` → `"sudamerica"` ✓
6. Navega `periodicos.com.ar/periodicos/sudamerica/argentina/` → dominios locales
7. Por cada dominio:
   - `_COUNTRY_PRESS_SECTION_PATHS.get(domain)` → **vacío** (Argentina no tiene paths curados)
   - Usa `_GENERIC_SECTION_PATHS[topic]` como fallback
8. `get_group_language("argentina")` → `"es"` → sin traducción

### Estado

- **Fuentes**: descubiertas via directorio periodicos.com.ar ✓
- **Secciones**: genéricas (`/seguridad/`, `/politica/`, `/economia/`, `/noticias/`)
- **Traducción**: no requerida
- **Riesgo**: los paths genéricos pueden no existir en medios argentinos (ej. clarin.com no tiene `/seguridad/`)

---

## Caso 2 — País conocido + fuente curada

**Ejemplo**: `"noticias de seguridad en Italia esta semana"`

### Flujo

1. `detect_query_source_group` → `"italy"` ✓
2. `_should_use_country_recent_news_strategy` → `True` (group + horizon + tópico "security")
3. `_extract_query_geography` → `"Italia"` ✓
4. `_PERIODICOS_CONTINENT_SLUG_BY_COUNTRY["Italia"]` → `"europa"` ✓
5. Navega `periodicos.com.ar/periodicos/europa/italia/` → dominios locales
6. Por cada dominio:
   - `_COUNTRY_PRESS_SECTION_PATHS.get("ansa.it")` → `{security: [("/sito/notizie/cronaca/cronaca.shtml", "cronaca")], ...}` ✓
   - Paths curados disponibles para topic "security"
7. `get_group_language("italy")` → `"it"` → traducción requerida al español rioplatense

### Estado

- **Fuentes**: directorio + dominios curados en `source_groups` ✓
- **Secciones**: paths específicos por dominio+tópico ✓
- **Traducción**: activa (idioma `"it"`)
- **Cobertura curada**: ansa.it, repubblica.it, ilmessaggero.it, ilfattoquotidiano.it, ilfoglio.it, ilmanifesto.it, huffingtonpost.it

---

## Caso 3 — País desconocido

**Ejemplo**: `"noticias de Namibia esta semana"`

### Flujo

1. `detect_query_source_group` → `None` ✗ (Namibia no está en `source_groups`)
2. `_should_use_country_recent_news_strategy` → **`False`** ← salida inmediata
3. `CountryRecentNewsStrategy.execute` retorna `None`
4. El sistema cae al pipeline genérico de scraping (angle-based search)

### Lo que pasa en el pipeline genérico

- `_extract_query_geography("noticias de namibia esta semana")`:
  - `_GEOGRAPHY_TERMS` → sin match
  - Regex fallback: `\b(?:de|en|sobre)\s+([a-záéíóúüñ]{4,})\s+` → puede capturar `"namibia"` → retorna `"Namibia"`
- `_GEO_ENGLISH.get("Namibia")` → `None` → `geo_en = "Namibia"` (identidad, sin mapping)
- `_PERIODICOS_CONTINENT_SLUG_BY_COUNTRY.get("Namibia")` → `None` → directorio salteado
- Búsqueda genérica con ángulos en español + inglés usando el nombre tal como viene

### Estado

- **Fallo**: silencioso — no hay error, pero no hay estrategia de prensa local
- **Fallback**: búsqueda web genérica (Tavily) con queries de ángulo
- **Sin trazabilidad**: el sistema no registra por qué no encontró prensa local
- **Resultado observable**: el agente devuelve noticias internacionales (Reuters, AP) en lugar de prensa local namibia

---

## Caso 4 — País conocido + tópico no reconocido

**Ejemplo**: `"noticias de Japón sobre cultura esta semana"`

### Flujo

1. `detect_query_source_group` → `"japan"` ✓
2. `_detect_news_topic("noticias de japón sobre cultura")`:
   - No hay keywords de security/economy/politics
   - Retorna `"default"`
3. `_should_use_country_recent_news_strategy`:
   - `topic = "default"`, pero hay palabra "noticias" → retorna `True`
4. `_extract_query_geography` → `"Japón"` ✓
5. `_PERIODICOS_CONTINENT_SLUG_BY_COUNTRY["Japón"]` → `"asia"` ✓
6. Navega `periodicos.com.ar/periodicos/asia/japon/` → dominios `.jp`
7. Por cada dominio (ej. `nhk.or.jp`):
   - `_COUNTRY_PRESS_SECTION_PATHS.get("nhk.or.jp")` → **vacío** (Japón no tiene paths curados)
   - Usa `_GENERIC_SECTION_PATHS["default"]` → `[("/noticias/", "noticias"), ("/actualidad/", "actualidad"), ...]`
8. `get_group_language("japan")` → `"ja"` → traducción requerida

### Estado

- **Problema**: `_GENERIC_SECTION_PATHS` tiene paths en español (`/noticias/`, `/politica/`)
- Los dominios japoneses no tienen esas secciones → la mayoría retorna 404 o homepage
- **Degradación silenciosa**: el sistema scrapea homepages en japonés en lugar de secciones relevantes
- **Traducción**: activa, pero el contenido scrapeado puede ser incorrecto por paths erróneos

---

## Caso 5 — País ambiguo o mal escrito

**Ejemplo**: `"noticias de corea esta semana"` (sin especificar norte/sur)

### Flujo en `detect_query_source_group`

Revisa `source_groups` uno por uno. El grupo `south_korea` tiene terms:
```
["south korea", "korea del sur", "corea del sur", "surcorea", "surcoreano", "surcoreana", "coreano", "coreana"]
```
"corea" sola no matchea ninguno. El grupo `north_korea` tampoco. → `None`

1. `detect_query_source_group` → `None` ✗
2. `_should_use_country_recent_news_strategy` → `False`
3. `CountryRecentNewsStrategy.execute` retorna `None`

### En el pipeline genérico

- `_extract_query_geography("noticias de corea esta semana")`:
  - `_GEOGRAPHY_TERMS` tiene `("corea", "Corea")` → retorna `"Corea"`
- `_PERIODICOS_CONTINENT_SLUG_BY_COUNTRY.get("Corea")` → `"asia"` ✓ (sí está)
- Pero **este resultado no se usa**: la estrategia ya abortó en el gate
- Búsqueda genérica con "Corea" sin distinguir norte/sur

### Variante con demonym

`"noticias coreanas esta semana"` → `_GEOGRAPHY_TERMS` tiene `("coreano", "Corea")` pero
`detect_query_source_group` busca "coreano" en south_korea terms → **sí matchea** `"coreano"`
→ retorna `"south_korea"` → estrategia arranca correctamente.

### Estado

- Comportamiento **inconsistente**: "coreano" → south_korea; "corea" → ninguno
- No hay normalización de ambigüedad antes del gate
- Sin mensaje de error ni sugerencia al usuario

---

## Resumen de comportamientos

| Caso | Gate pasa | Directorio | Secciones curadas | Traducción | Resultado observable |
|---|---|---|---|---|---|
| País conocido + es | ✓ | ✓ | ✗ (genérico) | No | Noticias locales con paths genéricos |
| País conocido + curado | ✓ | ✓ | ✓ | Sí (si idioma ≠ es) | Noticias locales con secciones correctas |
| País desconocido | ✗ | ✗ | ✗ | N/A | Búsqueda genérica, sin prensa local |
| País conocido + tópico raro | ✓ | ✓ | ✗ (genérico español) | Sí | Paths erróneos en sitios no hispanos |
| País ambiguo | ✗ (si no demonym) | ✗ | ✗ | N/A | Búsqueda genérica, inconsistente |

---

## Puntos de quiebre identificados

1. **Gate duro en `query_source_group`**: si el país no está en `web_source_policy.json`, la
   estrategia no arranca. No hay degradación gradual ni fallback a directorio.

2. **`_COUNTRY_PRESS_SECTION_PATHS` sólo cubre Italia y España**: el resto de países
   usa `_GENERIC_SECTION_PATHS` con paths en español que fallan en medios no hispanos.

3. **Fallo silencioso**: cuando el país no existe, no hay log de motivo ni trazabilidad.
   El usuario recibe resultados genéricos sin saber por qué no obtuvo prensa local.

4. **Ambigüedad geográfica sin resolver**: "corea" y "coreano" producen comportamientos
   distintos. El sistema no normaliza antes del gate.

5. **`_GEO_ENGLISH` como fallback de identidad**: si el país no está en el dict, usa el
   nombre tal cual. Funciona para países con nombre igual en inglés (Namibia), pero
   puede generar búsquedas en español para países con nombre diferente.

6. **Cinco estructuras que deben estar sincronizadas**: agregar un país nuevo requiere
   actualizar `source_groups`, `_GEOGRAPHY_TERMS`, `_GEO_ENGLISH`,
   `_PERIODICOS_CONTINENT_SLUG_BY_COUNTRY` y opcionalmente `_COUNTRY_PRESS_SECTION_PATHS`.
   No hay validación de consistencia entre ellas.

---

## Criterios de aceptación para el refactor

El refactor se considera exitoso cuando, para los 5 casos documentados:

- [ ] Caso 1: comportamiento preservado (no regresión)
- [ ] Caso 2: comportamiento preservado (no regresión)
- [ ] Caso 3: el sistema intenta prensa local (directorio + LLM fallback) en lugar de abortar
- [ ] Caso 4: los paths se resuelven en el idioma del país, no en español genérico
- [ ] Caso 5: "corea" y "coreano" producen el mismo resultado; se registra la ambigüedad
- [ ] Todos: los fallos dejan trazabilidad del motivo (nivel de fallback alcanzado)
