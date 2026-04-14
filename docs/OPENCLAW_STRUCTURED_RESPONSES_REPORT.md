# Informe: openclaw → Manejo de Respuestas Estructuradas

**Repo analizado:** `/Users/luis/Desktop/openclaw-main`
**Stack:** TypeScript, Lit (Web Components), JSON declarativo de display specs.

Son **cuatro capas independientes** que trabajan en serie.

---

## Capa 1 — Clasificación y extracción de bloques

**Archivos:** `src/agents/content-blocks.ts`, `ui/src/ui/chat/tool-cards.ts`

Cuando llega un mensaje del LLM, lo clasifican por `content_block.type`:

```ts
// content-blocks.ts — extrae solo bloques de texto del array de contenido
if (rec.type === "text" && typeof rec.text === "string") { parts.push(rec.text) }
```

```ts
// tool-cards.ts — clasifica cada bloque como "call" o "result"
const isToolCall = isToolCallContentType(item.type) || resolveToolBlockArgs(item) != null;
cards.push({ kind: "call", name, args }) // o { kind: "result", name, text }
```

Cada mensaje se convierte en un array de `ToolCard[]` tipados antes de renderizar.

---

## Capa 2 — Resolución de display por tool (`ToolDisplaySpec`)

**Archivos:** `src/agents/tool-display-common.ts`, `ui/src/ui/tool-display.ts`, `apps/shared/.../tool-display.json`

El dato central es un **JSON declarativo** (`tool-display.json`) que define, por cada tool:
- `emoji` → ícono visual
- `title` → label legible (`"Web Fetch"`, `"Bash"`, etc.)
- `detailKeys` → qué args del tool call mostrar como "detalle"
- `actions` → sub-acciones con sus propios labels y detailKeys (ej: `browser.navigate`, `browser.screenshot`)

```json
"web_fetch": {
  "emoji": "📄",
  "title": "Web Fetch",
  "detailKeys": ["url", "extractMode", "maxChars"]
},
"browser": {
  "emoji": "🌐",
  "actions": {
    "navigate": { "label": "navigate", "detailKeys": ["targetUrl", "targetId"] },
    "screenshot": { "label": "screenshot", "detailKeys": ["targetUrl", "element"] }
  }
}
```

La función `resolveToolDisplay()` toma `{ name, args }` y devuelve un `ToolDisplay` con `{ icon, label, verb, detail }` listo para renderizar. Para tools como `read`, `write`, `web_search` tienen resolvers especializados que formatean el detail de forma legible:
- `"from line 42 in ~/foo.ts"`
- `'for "bitcoin price" (top 5)'`
- `"to ~/file.py (128 chars)"`

`coerceDisplayValue()` normaliza cualquier tipo de arg (string, number, boolean, array) a texto de display con truncado a 160 chars y máximo 3 entradas de array.

---

## Capa 3 — Renderizado de la card con modo inline/collapsed

**Archivos:** `ui/src/ui/chat/tool-cards.ts`, `ui/src/ui/chat/constants.ts`

```ts
export const TOOL_INLINE_THRESHOLD = 80;  // chars
export const PREVIEW_MAX_LINES     = 2;
export const PREVIEW_MAX_CHARS     = 100;
```

La decisión de layout es puramente por longitud del output:

| Output | Modo | Qué se muestra |
|--------|------|----------------|
| Sin texto | `isEmpty` | "Completed" + check icon |
| ≤ 80 chars | `showInline` | Texto completo en monospace |
| > 80 chars | `showCollapsed` | Preview truncado (2 líneas / 100 chars) + botón "View" |

Click en la card abre un **sidebar** con el output completo. Si el output es JSON válido, lo wrappea en ` ```json ` con pretty-print. Si no hay output, muestra el comando ejecutado con el label del tool.

```ts
// tool-helpers.ts
export function formatToolOutputForSidebar(text: string): string {
  const trimmed = text.trim();
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    try {
      const parsed = JSON.parse(trimmed);
      return "```json\n" + JSON.stringify(parsed, null, 2) + "\n```";
    } catch {
      // Not valid JSON, return as-is
    }
  }
  return text;
}
```

```ts
// getTruncatedPreview — preview de 2 líneas / 100 chars con "…"
export function getTruncatedPreview(text: string): string {
  const allLines = text.split("\n");
  const lines = allLines.slice(0, PREVIEW_MAX_LINES);
  const preview = lines.join("\n");
  if (preview.length > PREVIEW_MAX_CHARS) {
    return preview.slice(0, PREVIEW_MAX_CHARS) + "…";
  }
  return lines.length < allLines.length ? preview + "…" : preview;
}
```

---

## Capa 4 — Streaming con coalescing

**Archivos:** `src/auto-reply/reply/block-streaming.ts`, `block-reply-coalescer.ts`, `block-reply-pipeline.ts`

El texto de la respuesta no se manda token a token. Hay un **coalescer** que acumula chunks y los manda en bloques más grandes según:

```ts
DEFAULT_BLOCK_STREAM_MIN = 800   // chars mínimos antes de flushear
DEFAULT_BLOCK_STREAM_MAX = 1200  // chars máximos antes de forzar flush
DEFAULT_IDLE_MS          = 1000  // flush por timeout si no llegan más tokens
```

El `breakPreference` puede ser `"paragraph"` (default), `"newline"`, o `"sentence"`, determinando el joiner entre chunks.

El pipeline además:
- Deduplica payloads por **content key** (evita doble envío de la misma respuesta)
- Separa `isReasoning` vs `isCompactionNotice` vs respuesta normal — no los mezcla en el mismo bloque
- Maneja media (imágenes) **fuera** del coalescer — se mandan siempre de inmediato sin buffering
- Tiene timeout de delivery: si `onBlockReply` tarda más de `timeoutMs`, aborta el pipeline y loguea

```ts
// block-reply-coalescer.ts — lógica central del buffer
const nextText = bufferText ? `${bufferText}${joiner}${text}` : text;
if (nextText.length > maxChars) {
  if (bufferText) {
    void flush({ force: true }); // flush lo acumulado
    bufferText = text;           // empieza nuevo buffer con el chunk actual
    scheduleIdleFlush();
    return;
  }
  void onFlush(payload); // chunk único que ya supera el máximo, mandar directo
  return;
}
bufferText = nextText;
if (bufferText.length >= maxChars) {
  void flush({ force: true });
  return;
}
scheduleIdleFlush(); // timer de idle para flush tardío
```

---

## Arquitectura completa

```
LLM response
    │
    ▼
content_blocks[]
    │
    ├─ type: "text"        → coalescer (buffer 800-1200 chars)
    │                            │
    │                       ReplyPayload → canal (Slack, Discord, Telegram, etc.)
    │
    └─ type: "tool_use"    → ToolCard { kind: "call",   name, args }
       type: "tool_result" → ToolCard { kind: "result", name, text }
                                  │
                             resolveToolDisplay()
                             ← tool-display.json (declarativo)
                                  │
                             renderToolCardSidebar()
                                  │
                             ┌────┴────────────────────┐
                             │                         │
                        isEmpty           output existe
                        "Completed"            │
                                          ≤ 80 chars → inline monospace
                                          > 80 chars → collapsed preview + sidebar
                                                            └── JSON? → pretty-print
```

---

## Lo que nos sirve de todo esto

| Patrón | Aplicabilidad en nuestro proyecto |
|--------|----------------------------------|
| `tool-display.json` declarativo | Nuestro `/tool` y `/tools` ya hacen algo similar pero hardcodeado en Python — un JSON config por agente sería más limpio y extensible |
| Threshold inline/collapsed (80 chars) | El output de tools en el REPL podría adoptar este patrón en lugar de volcar todo |
| `formatToolOutputForSidebar` — JSON detection | `scrape_website_with_json_capture` podría formatearse así en el REPL |
| `coerceDisplayValue()` — normalización de args | Útil para mostrar los args de tool calls de forma legible en `/tool` |
| Content key deduplication en pipeline | Útil si se implementa streaming real en el futuro |
| Coalescing de streaming | No aplica directamente — usamos sync, no streaming de tokens |

---

## Conclusión

El patrón más valioso y portable al proyecto es el **JSON declarativo de display specs**. Actualmente los labels, iconos y formatos de los tools están hardcodeados en Python en varios lugares. Extraerlos a un `tool-display.json` por agente permitiría cambiar cómo se muestran sin tocar código, y abriría la puerta a una UI más rica con el mismo dato.

El segundo patrón valioso es la **lógica inline/collapsed con sidebar** — una convención simple (umbral de 80 chars) que resuelve el problema de outputs largos sin necesidad de truncar o perder información.
