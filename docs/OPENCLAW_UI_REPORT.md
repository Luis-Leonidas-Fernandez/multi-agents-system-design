# Informe: openclaw → Arquitectura de la UI

**Repo analizado:** `/Users/luis/Desktop/openclaw-main`
**Stack:** Lit (Web Components), CSS Custom Properties, Vite, i18n propio, DOMPurify + marked.

---

## Stack tecnológico

| Capa | Tecnología |
|------|-----------|
| Framework UI | **Lit** (Web Components nativos) |
| Build | **Vite** |
| Estilos | **CSS Custom Properties** puras (sin Tailwind, sin CSS-in-JS) |
| Markdown | **marked** + **DOMPurify** |
| i18n | Sistema propio con Lit Controller |
| Estado | `@state()` de Lit sobre un único componente raíz (`OpenClawApp`) |
| Comunicación backend | WebSocket (gateway) vía cliente custom |

---

## Componente raíz: `OpenClawApp`

**Archivo:** `ui/src/ui/app.ts`

Es un único `LitElement` que extiende sin Shadow DOM (`createRenderRoot() { return this }`). Todo el estado de la app vive en él como propiedades `@state()` — no hay Zustand, Redux ni Context. Son literalmente ~150 propiedades reactivas declaradas en la clase.

La lógica está fragmentada en módulos funcionales que reciben `this` como argumento:

```ts
// El componente delega todo — no tiene lógica propia
handleSendChat()  → handleSendChatInternal(this, ...)
setTab(next)      → setTabInternal(this, next)
connect()         → connectGatewayInternal(this)
render()          → renderApp(this as AppViewState)
```

El renderizado también es externo: `renderApp(this)` en `app-render.ts`. El componente es solo el contenedor de estado y el punto de entrada de eventos.

---

## Navegación por tabs

**Archivo:** `ui/src/ui/navigation.ts`

Hay 19 tabs organizadas en 4 grupos:

| Grupo | Tabs |
|-------|------|
| **chat** | chat |
| **control** | overview, channels, instances, sessions, usage, cron |
| **agent** | agents, skills, nodes, dreams |
| **settings** | config, communications, appearance, automation, infrastructure, aiAgents, debug, logs |

La navegación es por URL path (`/chat`, `/overview`, `/agents`, etc.) con soporte de `basePath` configurable. El tab activo se resuelve desde `window.location.pathname` al iniciar. No usa React Router ni ninguna librería — solo `history.pushState` + `popstate`.

---

## Sistema de temas

**Archivo:** `ui/src/ui/theme.ts`

Tres temas base: `"claw"` (default), `"knot"`, `"dash"`. Cada uno con modos `"light"` / `"dark"` / `"system"`. El sistema resuelve el modo OS via `matchMedia("(prefers-color-scheme: light)")`. Los temas se aplican como atributos en el `<html>` o el componente raíz. Legacy map incluido para compatibilidad con nombres viejos (`"openknot"`, `"fieldmanual"`, etc.).

```ts
export function resolveTheme(theme: ThemeName, mode: ThemeMode): ResolvedTheme {
  if (theme === "claw")  return mode === "light" ? "light" : "dark";
  if (theme === "knot")  return mode === "light" ? "openknot-light" : "openknot";
  return mode === "light" ? "dash-light" : "dash";
}
```

Los CSS usan Custom Properties (`--bg`, `--text`, `--accent`, `--border`, `--muted`, `--card`, `--radius-md`, `--mono`, etc.) que cambian por selector de tema. No hay ningún framework de theming.

---

## Internacionalización (i18n)

**Archivos:** `ui/src/i18n/`

Sistema propio con Lit Controller. Soporta 12 idiomas: en, es, de, fr, ja-JP, ko, pl, pt-BR, tr, uk, zh-CN, zh-TW. Las traducciones son archivos `.ts` con objetos tipados. La función `t("tabs.chat")` es la interfaz principal. El controlador `I18nController` se conecta al componente Lit y dispara re-renders automáticos al cambiar el locale.

---

## Renderizado del chat

**Archivos:** `ui/src/ui/chat/grouped-render.ts`, `ui/src/ui/views/chat.ts`

El chat NO renderiza mensaje a mensaje — agrupa por rol consecutivo en `MessageGroup[]`. Cada grupo tiene un avatar, nombre y timestamp compartidos, y N mensajes internos.

### Tipos de render por mensaje

```
mensaje
├── rol === "tool" / isToolResult
│     └── <details class="chat-tool-msg-collapse">
│           └── "Tool output" ▸ [tool names preview]
│                 └── body: texto/JSON/tool-cards
│
└── rol === "user" | "assistant"
      ├── imágenes (base64 o URL)
      ├── audio clips (<audio controls>)
      ├── reasoning/thinking (<div class="chat-thinking">)
      ├── JSON puro detectado → <details class="chat-json-collapse">
      ├── markdown → toSanitizedMarkdownHtml()
      └── tool cards → renderCollapsedToolCards()
```

### Detección de JSON en mensajes

```ts
const MAX_JSON_AUTOPARSE_CHARS = 20_000; // cap anti-DoS

function detectJson(text: string): { parsed: unknown; pretty: string } | null {
  if (t.length > MAX_JSON_AUTOPARSE_CHARS) return null;
  if ((t.startsWith("{") && t.endsWith("}")) || (t.startsWith("[") && t.endsWith("]"))) {
    try { return { parsed: JSON.parse(t), pretty: JSON.stringify(parsed, null, 2) } }
    catch { return null }
  }
  return null;
}
```

Si detecta JSON, lo renderiza en `<details>` con badge "JSON" y label legible:
- `Array (12 items)`
- `{ id, name, status }` (≤4 keys inline)
- `Object (18 keys)`

### Metadata por mensaje (tokens / costo / modelo)

Cada mensaje de assistant muestra inline:

```ts
↑128k   // input tokens
↓4.2k   // output tokens
R64k    // cache read
W12k    // cache write
$0.0042 // costo total
82% ctx // % de context window usado (en rojo si ≥90%, amarillo si ≥75%)
claude-3.5-sonnet // model (sin prefijo de provider)
```

Implementado en `renderMessageMeta()` con `fmtTokens()` que abrevia (`128000 → "128k"`, `1200000 → "1.2M"`).

### Streaming

Durante streaming hay dos estados:
1. `renderReadingIndicatorGroup()` — tres dots animados (mientras no llega ningún token)
2. `renderStreamingGroup(text, startedAt)` — burbuja con clase `streaming` que se actualiza con `chatStreamSegments`

---

## Renderizado de Markdown

**Archivo:** `ui/src/ui/markdown.ts`

Pipeline: `marked.parse()` → `DOMPurify.sanitize()` → `unsafeHTML()` en Lit.

Detalles importantes:

- **Cache LRU** de 200 entradas (máx 50k chars cada una) — evita re-parsear el mismo markdown
- **Cap de seguridad**: >140k chars → trunca; >40k chars → fallback a plain text sin parsear
- **HTML crudo escapado**: el renderer custom convierte HTML literal a texto escapado — si el LLM manda `<div>`, se muestra como texto, no como HTML
- **Imágenes**: solo se renderizan `data:image/` URLs — URLs externas se escapan (anti-phishing)
- **Bloques de código**: cada `<pre><code>` tiene header con botón "Copy" y `data-code` attribute. JSON dentro de code fences → `<details class="json-collapse">` automático
- **Extensión CJK**: custom tokenizer para evitar que URLs se traguen caracteres CJK adyacentes
- **Links**: DOMPurify hook agrega `rel="noreferrer noopener"` + `target="_blank"` en todos los `<a>`

```ts
// Allowed tags list — explícitamente restringida:
["a","b","blockquote","br","code","del","details","div","em","h1"..."h4",
 "hr","i","li","ol","p","pre","span","strong","summary","table"..."ul","img"]

// Allowed attrs — también restringida:
["class","href","rel","target","title","start","src","alt","data-code","type","aria-label"]
```

---

## CSS — Design System

**Archivos:** `ui/src/styles/`

Todo el sistema usa CSS Custom Properties definidas por tema. No hay un `reset.css` externo ni framework. Las variables clave:

```css
--bg, --bg-hover       /* fondos */
--text, --muted        /* tipografía */
--accent               /* color principal */
--border, --border-strong
--card, --secondary    /* superficies */
--ok, --warn, --error  /* estados */
--radius-sm, --radius-md /* bordes */
--mono                 /* font monoespaciada */
--duration-fast        /* transiciones */
```

Los archivos CSS están organizados por feature:
- `base.css` — reset y variables
- `chat/layout.css` — layout del chat
- `chat/tool-cards.css` — cards de tools (con responsive en 768px y 480px)
- `chat/sidebar.css` — panel derecho
- `chat/text.css` — tipografía del chat
- `chat/grouped.css` — grupos de mensajes
- `components.css` — botones, inputs, badges
- `layout.mobile.css` — overrides mobile

---

## Tool Cards UI

**Archivo:** `ui/src/styles/chat/tool-cards.css`

Las tool cards tienen `max-height: 120px` y `overflow: hidden`. En mobile (≤768px) baja a 100px, en small (≤480px) a 80px. El preview colapsado tiene su propio `max-height: 44px`.

Transiciones con `var(--duration-fast)`. Todos los estados (hover, focus, open) están manejados con CSS nativo usando `details[open]` selector y `::before` con `▸` que rota 90° al abrir.

```css
.chat-tools-collapse[open] > .chat-tools-summary::before {
  transform: rotate(90deg);
}
```

No hay JavaScript para el collapse — es HTML nativo `<details>/<summary>`.

---

## Sidebar de canvas

El sidebar es un panel dividido con ratio ajustable (40%–70%) via `ResizableDivider`. El estado es `sidebarOpen: boolean` + `sidebarContent: string | null`. Al cerrarse, el contenido se limpia con un delay de 200ms para que la transición CSS termine antes de vaciar el DOM.

```ts
handleCloseSidebar() {
  this.sidebarOpen = false;
  this.sidebarCloseTimer = window.setTimeout(() => {
    this.sidebarContent = null;
  }, 200); // espera la transición CSS
}
```

---

## Command Palette

`Cmd+K` / `Ctrl+K` abre una paleta de comandos (`paletteOpen`). Está implementada con event listener global en `connectedCallback`:

```ts
this.globalKeydownHandler = (e: KeyboardEvent) => {
  if ((e.metaKey || e.ctrlKey) && !e.shiftKey && e.key === "k") {
    e.preventDefault();
    this.paletteOpen = !this.paletteOpen;
  }
};
```

---

## Slash Commands

**Archivo:** `ui/src/ui/chat/slash-commands.ts`

El input del chat detecta el `/` y muestra completions con categorías. Los slash commands tienen handlers en `slash-command-executor.ts`. Entre ellos: `/export` (chat → markdown), `/refresh-tools-effective`, `/toggle-focus` (focus mode sin sidebar ni nav).

---

## Polling y WebSocket

La app se conecta al gateway por WebSocket. El estado `connected: boolean` controla si el input está habilitado. Hay polling periódico para `nodes`, `logs`, y `debug` (con `setInterval`). Las actualizaciones de chat llegan por WebSocket events.

---

## Lo que nos podría servir

| Patrón | Valor para nuestro proyecto |
|--------|----------------------------|
| **Detección de JSON en mensajes** con cap anti-DoS (20k chars) | Nuestro REPL muestra JSON crudo — implementar este collapse mejoraría drásticamente la legibilidad de `scrape_website_with_json_capture` |
| **Metadata de tokens/costo por mensaje** | Podría mostrarse en el REPL al final de cada respuesta del agente |
| **`<details>/<summary>` nativo para collapse** | Sin JS, sin librerías — aplicable en el output del REPL para herramientas con output largo |
| **Cache LRU de markdown parseado** | Si se implementa una UI real, evitar re-parsear el mismo contenido |
| **`max-height` + `overflow: hidden` en tool cards** | El output de tools en el REPL podría tener un límite visual con "expandir" |
| **Limpieza con delay en cierre de paneles** (200ms) | Patrón correcto para transiciones CSS sin artifacts visuales |
| **Cmd+K command palette** | Slash commands ya los tenemos — agregar atajo de teclado es directo |
| **Fragmentación de la lógica en módulos funcionales** | Nuestro `main.py` está creciendo — separar handlers por feature (chat, sessions, tools) seguiría este patrón |

---

## Conclusión

La UI de openclaw es un **monolito de estado plano** en un único Web Component, con toda la lógica externalizada en módulos funcionales. No hay framework de estado ni router externo. El sistema de temas, i18n y estilos es 100% propio y muy liviano.

El patrón más directamente aplicable a nuestro proyecto es la **detección y collapse de JSON** — una función de 10 líneas que transforma completamente la experiencia de ver respuestas estructuradas de los agentes.
