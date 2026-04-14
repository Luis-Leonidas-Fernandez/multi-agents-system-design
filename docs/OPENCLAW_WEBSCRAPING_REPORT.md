# Informe: openclaw → Web Scraping útil para el proyecto

**Repo analizado:** `/Users/luis/Desktop/openclaw-main`
**Stack:** TypeScript, `undici`, `@mozilla/readability`, `linkedom` — sin Playwright ni BeautifulSoup.

---

## Lo que TIENEN ellos que NO tenemos nosotros

### 1. `stripInvisibleUnicode` — Anti-prompt injection a nivel HTML
**Archivo:** `src/agents/tools/web-fetch-visibility.ts:159`

```ts
const INVISIBLE_UNICODE_RE =
  /[\u200B-\u200F\u202A-\u202E\u2060-\u2064\u206A-\u206F\uFEFF\u{E0000}-\u{E007F}]/gu;

export function stripInvisibleUnicode(text: string): string {
  return text.replace(INVISIBLE_UNICODE_RE, "");
}
```

Tu `input_guard` bloquea inyección en el mensaje del usuario. Pero si un sitio web incrusta caracteres invisibles Unicode en el HTML para envenenar el contexto del LLM, te pasa por arriba. Este strip va al final de todo pipeline de extracción.

---

### 2. `sanitizeHtml` — Elimina elementos CSS-ocultos antes de parsear
**Archivo:** `src/agents/tools/web-fetch-visibility.ts:134`

Detecta y elimina del DOM (con `linkedom`) todos los nodos con:
- `display: none`, `visibility: hidden`, `opacity: 0`
- `class="sr-only"`, `"hidden"`, `"d-none"`, `"visually-hidden"`
- `aria-hidden="true"`, `hidden` attribute
- `transform: scale(0)`, `translateX(-9999px)`, `left: -9999px`
- `clip-path: inset(...)`

Esto es crítico. Muchos sitios de noticias meten contenido oculto (ads, tracking, injected scripts) que BeautifulSoup incluye alegremente.

---

### 3. `exceedsEstimatedHtmlNestingDepth` — Protección anti-stack overflow
**Archivo:** `src/agents/tools/web-fetch-utils.ts:116`

Heurística O(n) que cuenta depth de nesting HTML antes de pasarlo a Readability. Si supera 3000 niveles anidados → skip. Sin esto, HTML patológico puede reventar el parser o hacer OOM.

---

### 4. `@mozilla/readability + linkedom` — Extracción de artículos inteligente
**Archivo:** `src/agents/tools/web-fetch-utils.ts:226`

Lo que hace Firefox cuando activa "Modo Lectura". Extrae solo el artículo principal de la página, descarta nav, sidebars, footers, ads. Es la diferencia entre obtener el texto de la noticia vs 5KB de menú de navegación.

El proyecto actual usa BeautifulSoup, que no tiene este nivel de semántica. El equivalente en Python es `readability-lxml`.

---

### 5. Multi-extractor cascade con fallbacks
**Archivo:** `src/agents/tools/web-fetch.ts:461`

La cadena es:

```
Cloudflare Markdown → Readability → provider fallback → basic HTML fallback
```

Si Readability falla, llama a un provider externo (Jina, Browserless, etc). Si ese falla, cae a parseo básico. Si todo falla, lanza error claro.

El proyecto actual tiene Playwright o requests+BS4 como opciones separadas, sin fallback entre ellos ni cascade.

---

### 6. `readResponseText` con streaming byte cap
**Archivo:** `src/agents/tools/web-shared.ts:97`

Lee la response como `ReadableStream` con un cap exacto de bytes (default 2MB). En cuanto llega al límite, cancela el reader y libera la conexión. Sin esto, una página de 50MB se carga completa en memoria antes de recortarla.

---

### 7. SSRF protection con DNS pinning
**Archivo:** `src/infra/net/fetch-guard.ts`

Antes de conectar a cualquier URL, resuelve el hostname y verifica que no sea IP privada/RFC1918/loopback. Previene que el agente sea usado para atacar servicios internos. También detecta redirect loops y strips headers sensibles (`Authorization`, `Cookie`) en cross-origin redirects.

---

### 8. Content type dispatch diferenciado
**Archivo:** `src/agents/tools/web-fetch.ts:464`

Detecta `Content-Type` del response y aplica pipeline diferente:
- `text/markdown` → Cloudflare Markdown for Agents (directo, sin parseo)
- `text/html` → Readability pipeline
- `application/json` → `JSON.stringify` pretty-print

---

## Lo que ya tenemos y es redundante

| Ellos | Nosotros | Veredicto |
|-------|----------|-----------|
| Cache con TTL (`Map` + `expiresAt`) | Cache Playwright 60s | Similar, el nuestro es más simple pero funcional |
| User-Agent configurable | No configurado explícitamente | Gap menor |
| Timeout con AbortSignal | `requests` tiene timeout | Cubierto |
| Playwright (browser automation) | Playwright sync | Lo tenemos y ellos NO |
| JSON capture de APIs dinámicas | `scrape_website_with_json_capture` | Lo tenemos, ellos no |

---

## Prioridad de lo que conviene adoptar

| Prioridad | Qué | Por qué |
|-----------|-----|---------|
| 🔴 ALTA | `stripInvisibleUnicode` | Anti-prompt-injection a nivel contenido, una línea de código |
| 🔴 ALTA | `sanitizeHtml` (hidden elements) | Mejora drástica calidad del contenido extraído |
| 🟠 MEDIA | `readability-lxml` (equiv. Python) | Reemplaza/complementa BS4 para extracción de artículos |
| 🟠 MEDIA | Streaming byte cap en `requests` | Evita OOM en páginas pesadas |
| 🟡 BAJA | Cascade de extractores con fallback | Mejora resiliencia general, más trabajo de implementación |
| 🟡 BAJA | SSRF protection | Solo relevante si el sistema se expone a usuarios no confiables |

---

## Conclusión

El win más rápido y de mayor impacto: `stripInvisibleUnicode` + `sanitizeHtml`. Son dos funciones portables a Python en menos de 50 líneas, y cierran el gap de prompt injection a nivel contenido web — algo que el `input_guard` actual no cubre porque actúa antes de la llamada al LLM, no sobre el contenido scrapeado.
