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

function isLog(line) {
  return line.startsWith('log:');
}

function normalizeText(line) {
  if (isLog(line)) return line.slice(4).trimStart();
  if (!(line.startsWith('assistant:') || line.startsWith('ai:'))) return line;
  const prefix = line.startsWith('assistant:') ? 'assistant:' : 'ai:';
  const body = line.slice(prefix.length).trimStart();
  const sourcesIdx = body.search(/(?:^|\n)#{0,3}\s*Sources[:\s]/);
  const mainText = sourcesIdx !== -1 ? body.slice(0, sourcesIdx) : body;
  const sourcesText = sourcesIdx !== -1 ? body.slice(sourcesIdx) : '';
  const cleaned = mainText
    .replace(/^\s*\d+\.\s+/gm, '• ')
    .replace(/^#{1,3}\s+\*?\*?(.*?)\*?\*?\s*$/gm, (_, t) => `\n◆ ${t.toUpperCase()}\n`)
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
  if (isLog(line)) return 1;
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
function buildViewport(lines, scrollOffset, viewportRows, expandedMessages = null) {
  if (lines.length === 0) return {entries: [], hasMoreAbove: false, hasMoreBelow: false};

  // Ancla: el mensaje más reciente a mostrar (ajustado por scroll)
  const anchorIdx = Math.max(0, lines.length - 1 - scrollOffset);
  const hasMoreBelow = scrollOffset > 0;

  // Llenar el viewport de abajo hacia arriba
  const selected = []; // [{idx, truncatedLines: null | number, isExpanded: bool}]
  let usedRows = 1; // línea de status

  for (let i = anchorIdx; i >= 0; i--) {
    const isExpanded = expandedMessages != null && expandedMessages.has(i);
    const entryRows = estimateRows(lines[i]);
    const remaining = viewportRows - usedRows;

    if (isExpanded) {
      // Expandido: ignorar restricciones del viewport, mostrar completo siempre.
      selected.unshift({idx: i, truncatedLines: null, isExpanded: true});
      usedRows += entryRows;
    } else if (entryRows <= remaining) {
      // Entra completo
      selected.unshift({idx: i, truncatedLines: null, isExpanded: false});
      usedRows += entryRows;
    } else if (selected.length === 0) {
      if (entryRows > viewportRows) {
        // Mensaje más grande que el viewport completo — mostrar primeras filas.
        // truncatedLines < 0 indica "truncar desde abajo" (mostrar inicio).
        const showRows = Math.max(viewportRows - 3, 4); // -1 status, -1 indicator, -1 buffer
        selected.unshift({idx: i, truncatedLines: -showRows, isExpanded: false});
        usedRows += showRows + 1;
      } else {
        // No entra del todo pero es menor que el viewport — mostrar completo
        // (desbordamiento leve aceptable).
        selected.unshift({idx: i, truncatedLines: null, isExpanded: false});
        usedRows += entryRows;
      }
      break;
    } else if (remaining >= 3) {
      // Hay espacio para mostrar las últimas `remaining - 1` líneas del mensaje
      // (-1 por el indicador "↑ continúa")
      selected.unshift({idx: i, truncatedLines: remaining - 1, isExpanded: false});
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

/**
 * Recorta un texto a sus primeras `maxRows` líneas de terminal.
 * Usado cuando el mensaje más reciente es más grande que el viewport.
 */
function truncateToFirstRows(text, maxRows, innerWidth) {
  const paragraphs = text.split('\n');
  const termLines = [];
  for (const p of paragraphs) {
    if (!p) { termLines.push(''); continue; }
    for (let s = 0; s < p.length; s += innerWidth) {
      termLines.push(p.slice(s, s + innerWidth));
      if (termLines.length >= maxRows) return termLines.join('\n');
    }
  }
  return termLines.slice(0, maxRows).join('\n');
}

// ─── componente ──────────────────────────────────────────────────────────

function TranscriptList({lines, scrollOffset, height, expandedMessages = null}) {
  const viewportRows = Math.max(height, 6);
  const {entries, hasMoreAbove, hasMoreBelow} = buildViewport(lines, scrollOffset, viewportRows, expandedMessages);

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
    ...entries.map(({idx, truncatedLines, isExpanded}, ei) => {
      const line = lines[idx];
      const asst = isAssistant(line);
      const color = asst ? 'green' : 'yellow';
      const rawText = normalizeText(line);

      let displayText = rawText;
      const isTruncatedAbove = truncatedLines !== null && truncatedLines > 0;
      const isTruncatedBelow = truncatedLines !== null && truncatedLines < 0;
      if (isTruncatedAbove) {
        const innerWidth = Math.max(TERM_WIDTH - (asst ? 8 : 4), 20);
        displayText = truncateToLastRows(rawText, truncatedLines, innerWidth);
      } else if (isTruncatedBelow) {
        const innerWidth = Math.max(TERM_WIDTH - (asst ? 8 : 4), 20);
        displayText = truncateToFirstRows(rawText, -truncatedLines, innerWidth);
      }

      const isTruncated = isTruncatedAbove || isTruncatedBelow;

      // ── Log line: gris/dim, sin caja, sin margen
      if (isLog(line)) {
        return React.createElement(
          Text,
          {key: `e${idx}`, color: 'gray', dimColor: true},
          `· ${displayText}`
        );
      }

      return React.createElement(
        Box,
        {
          key: `e${idx}`,
          flexDirection: 'column',
          marginTop: ei === 0 ? 0 : 1,
        },
        // Indicador: mensaje continúa hacia arriba (scroll)
        isTruncatedAbove
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
              React.createElement(Text, {color: 'gray', dimColor: true}, '📋 ai'),
              React.createElement(Text, {color, wrap: 'wrap'}, displayText)
            )
          : React.createElement(
              Box,
              {flexDirection: 'column'},
              React.createElement(Text, {color: 'yellow', dimColor: true}, '📋 usuario'),
              React.createElement(Text, {color, wrap: 'wrap'}, displayText)
            ),
        // Indicador: mensaje continúa hacia abajo — ofrecer expandir
        isTruncatedBelow
          ? React.createElement(Text, {color: 'yellow'}, '╌ continúa ↓   [ E · expandir ]')
          : isExpanded
            ? React.createElement(Text, {color: 'gray', dimColor: true}, '╌ expandido · E para colapsar')
            : null,
      );
    }),

    // Indicador: hay mensajes más nuevos (cuando se scrolleó hacia atrás)
    hasMoreBelow
      ? React.createElement(Text, {color: 'yellow', dimColor: true}, '↓ hay mensajes más recientes')
      : null,
  );
}

export {TranscriptList};
