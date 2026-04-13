import React from 'react';
import {Box, Text} from 'ink';

const TERM_WIDTH = Math.max(Math.min(process.stdout.columns || 80, 120), 40);

// ─── normalización ────────────────────────────────────────────────────────

function extractSourceDomains(text) {
  const seen = new Set();
  const domains = [];
  for (const [, url] of text.matchAll(/\[.*?\]\((https?:\/\/[^)]+)\)/g)) {
    try {
      const host = new URL(url.split('|')[0]).hostname.replace(/^www\./, '');
      if (!seen.has(host)) { seen.add(host); domains.push(host); }
    } catch {}
  }
  for (const [url] of text.matchAll(/https?:\/\/[^\s)>\]]+/g)) {
    try {
      const host = new URL(url.split('|')[0]).hostname.replace(/^www\./, '');
      if (!seen.has(host)) { seen.add(host); domains.push(host); }
    } catch {}
  }
  return domains;
}

function normalizeText(line) {
  if (!(line.startsWith('assistant:') || line.startsWith('ai:'))) return line;
  const prefix = line.startsWith('assistant:') ? 'assistant:' : 'ai:';
  const body = line.slice(prefix.length).trimStart();
  const sourcesIdx = body.search(/(?:^|\n)#{0,3}\s*Sources[:\s]/);
  const mainText = sourcesIdx !== -1 ? body.slice(0, sourcesIdx) : body;
  const sourcesText = sourcesIdx !== -1 ? body.slice(sourcesIdx) : '';
  const cleaned = mainText
    .replace(/^\s*\d+\.\s+/gm, '• ')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .trimEnd();
  if (!sourcesText) return cleaned;
  const domains = extractSourceDomains(sourcesText);
  return cleaned + (domains.length > 0 ? `\nFuentes: ${domains.join(' · ')}` : '');
}

function isAssistant(line) {
  return line.startsWith('assistant:') || line.startsWith('ai:');
}

// ─── estimación de filas ──────────────────────────────────────────────────

/**
 * Estima cuántas filas de terminal ocupa un mensaje.
 * Sobrestima deliberadamente (ancho conservador) para no cortar mensajes.
 */
function estimateRows(line) {
  const text = normalizeText(line);
  const asst = isAssistant(line);
  const innerWidth = Math.max(TERM_WIDTH - (asst ? 8 : 4), 20);
  let rows = 0;
  for (const paragraph of text.split('\n')) {
    rows += Math.max(1, Math.ceil((paragraph.length || 1) / innerWidth));
  }
  return rows + (asst ? 2 : 0) + 1; // +2 borde, +1 margen entre mensajes
}

// ─── selección de entradas visibles ──────────────────────────────────────

/**
 * Decide qué entradas mostrar dado el viewport y el scroll.
 *
 * scrollOffset (en ENTRADAS): 0 = ver los más recientes.
 *   - anchorIdx = último mensaje a mostrar en la parte inferior.
 *   - Llena el viewport hacia atrás desde anchorIdx.
 *   - Si el primer mensaje visible no entra completo, muestra sus ÚLTIMAS líneas
 *     con un indicador "↑ continúa".
 *
 * Retorna:
 *   topEntry  : { idx, text, truncatedRows } — primer mensaje (posiblemente truncado)
 *   midEntries: [{ idx, text }]              — mensajes intermedios completos
 *   bottomEntry: { idx, text }              — último mensaje (completo)
 */
function buildViewport(lines, scrollOffset, viewportRows) {
  if (lines.length === 0) return {entries: [], hasMoreAbove: false, hasMoreBelow: false};

  // Ancla: el mensaje más reciente a mostrar (ajustado por scroll)
  const anchorIdx = Math.max(0, lines.length - 1 - scrollOffset);
  const hasMoreBelow = scrollOffset > 0;

  // Llenar el viewport de abajo hacia arriba
  const selected = []; // [{idx, truncatedLines: null | number}]
  let usedRows = 1; // línea de status

  for (let i = anchorIdx; i >= 0; i--) {
    const entryRows = estimateRows(lines[i]);
    const remaining = viewportRows - usedRows;

    if (entryRows <= remaining) {
      // Entra completo
      selected.unshift({idx: i, truncatedLines: null});
      usedRows += entryRows;
    } else if (selected.length === 0) {
      // El mensaje más reciente no entra completo — mostrarlo completo igual
      // (el viewport puede desbordarse levemente; es mejor que no ver nada)
      selected.unshift({idx: i, truncatedLines: null});
      usedRows += entryRows;
      break;
    } else if (remaining >= 3) {
      // Hay espacio para mostrar las últimas `remaining - 1` líneas del mensaje
      // (-1 por el indicador "↑ continúa")
      selected.unshift({idx: i, truncatedLines: remaining - 1});
      usedRows += remaining;
      break;
    } else {
      // Sin espacio útil, parar
      break;
    }
  }

  const hasMoreAbove = selected.length > 0 && selected[0].idx > 0;

  return {
    entries: selected,
    hasMoreAbove,
    hasMoreBelow,
    anchorIdx,
  };
}

/**
 * Recorta un texto a sus últimas `maxRows` líneas de terminal.
 */
function truncateToLastRows(text, maxRows, innerWidth) {
  const paragraphs = text.split('\n');
  const termLines = [];
  for (const p of paragraphs) {
    if (!p) { termLines.push(''); continue; }
    for (let s = 0; s < p.length; s += innerWidth) {
      termLines.push(p.slice(s, s + innerWidth));
    }
  }
  return termLines.slice(-maxRows).join('\n');
}

// ─── componente ──────────────────────────────────────────────────────────

function TranscriptList({lines, scrollOffset, height}) {
  const viewportRows = Math.max(height, 6);
  const {entries, hasMoreAbove, hasMoreBelow} = buildViewport(lines, scrollOffset, viewportRows);

  const atBottom = scrollOffset === 0;
  const statusText = lines.length === 0
    ? 'sin mensajes'
    : `${lines.length} msgs${atBottom ? '' : `  ·  ↑ ${scrollOffset} atrás`}`;

  return React.createElement(
    Box,
    {
      flexDirection: 'column',
      borderStyle: 'round',
      borderColor: atBottom ? 'gray' : 'yellow',
      paddingX: 1,
      paddingY: 0,
      height: viewportRows + 2,
    },

    // Status
    React.createElement(Text, {color: 'gray', dimColor: true}, statusText),

    // Indicador: hay contenido más arriba
    hasMoreAbove
      ? React.createElement(Text, {color: 'yellow', dimColor: true}, '↑ hay mensajes anteriores')
      : null,

    // Sin mensajes
    lines.length === 0
      ? React.createElement(Text, {color: 'gray'}, 'todavía no hay mensajes')
      : null,

    // Mensajes
    ...entries.map(({idx, truncatedLines}, ei) => {
      const line = lines[idx];
      const asst = isAssistant(line);
      const color = asst ? 'green' : 'magenta';
      const rawText = normalizeText(line);

      let displayText = rawText;
      if (truncatedLines !== null) {
        const innerWidth = Math.max(TERM_WIDTH - (asst ? 8 : 4), 20);
        displayText = truncateToLastRows(rawText, truncatedLines, innerWidth);
      }

      const isTruncated = truncatedLines !== null;

      return React.createElement(
        Box,
        {
          key: `e${idx}`,
          flexDirection: 'column',
          marginTop: ei === 0 ? 0 : 1,
        },
        // Indicador de truncado al tope del mensaje
        isTruncated
          ? React.createElement(Text, {color: 'yellow', dimColor: true}, '╌ continúa arriba ↑')
          : null,
        // Cuerpo del mensaje
        asst
          ? React.createElement(
              Box,
              {
                flexDirection: 'column',
                backgroundColor: '#1e293b',
                borderColor: isTruncated ? 'yellow' : 'cyanBright',
                borderStyle: 'round',
                paddingX: 1,
                paddingY: 0,
              },
              React.createElement(Text, {color, wrap: 'wrap'}, displayText)
            )
          : React.createElement(Text, {color, wrap: 'wrap'}, displayText)
      );
    }),

    // Indicador: hay mensajes más nuevos (cuando se scrolleó hacia atrás)
    hasMoreBelow
      ? React.createElement(Text, {color: 'yellow', dimColor: true}, '↓ hay mensajes más recientes')
      : null,
  );
}

export {TranscriptList};
