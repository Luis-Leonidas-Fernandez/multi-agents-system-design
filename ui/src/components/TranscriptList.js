import React from 'react';
import {Box, Text} from 'ink';

function colorForLine(line) {
  if (line.startsWith('assistant:')) return 'green';
  if (line.startsWith('you:') || line.startsWith('command:')) return 'magenta';
  if (line.startsWith('error:')) return 'red';
  return 'white';
}

function lineSpacing(line, prevLine, index) {
  if (index === 0) return {marginTop: 0, marginBottom: 0};
  if (line.startsWith('human:')) return {marginTop: 1, marginBottom: 0};
  if (line.startsWith('assistant:') || line.startsWith('ai:')) {
    if (prevLine && prevLine.startsWith('human:')) return {marginTop: 1, marginBottom: 0};
    return {marginTop: 0, marginBottom: 0};
  }
  return {marginTop: 0, marginBottom: 0};
}

function messageSurface(line) {
  if (line.startsWith('assistant:') || line.startsWith('ai:')) {
    return {backgroundColor: '#1e293b', borderColor: 'cyanBright', borderStyle: 'round', paddingX: 1, paddingY: 0};
  }
  if (line.startsWith('human:')) {
    return {paddingX: 0, paddingY: 0};
  }
  return {paddingX: 0, paddingY: 0};
}

function extractSourceDomains(text) {
  const seen = new Set();
  const domains = [];
  // markdown links: [text](url) — extract hostname from url
  for (const [, url] of text.matchAll(/\[.*?\]\((https?:\/\/[^)]+)\)/g)) {
    try {
      const clean = url.split('|')[0];
      const host = new URL(clean).hostname.replace(/^www\./, '');
      if (!seen.has(host)) { seen.add(host); domains.push(host); }
    } catch {}
  }
  // bare URLs
  for (const [url] of text.matchAll(/https?:\/\/[^\s)>\]]+/g)) {
    try {
      const clean = url.split('|')[0];
      const host = new URL(clean).hostname.replace(/^www\./, '');
      if (!seen.has(host)) { seen.add(host); domains.push(host); }
    } catch {}
  }
  return domains;
}

function normalizeAssistantText(line) {
  if (!(line.startsWith('assistant:') || line.startsWith('ai:'))) return line;
  const prefix = line.startsWith('assistant:') ? 'assistant:' : 'ai:';
  const body = line.slice(prefix.length).trimStart();

  const sourcesMatch = body.search(/(?:^|\n)#{0,3}\s*Sources[:\s]/);
  const mainText = sourcesMatch !== -1 ? body.slice(0, sourcesMatch) : body;
  const sourcesText = sourcesMatch !== -1 ? body.slice(sourcesMatch) : '';

  const cleaned = mainText
    .replace(/^\s*\d+\.\s+/gm, '• ')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .trimEnd();

  if (!sourcesText) return cleaned;

  const domains = extractSourceDomains(sourcesText);
  const sourceLine = domains.length > 0 ? `\nFuentes: ${domains.join(' · ')}` : '';
  return cleaned + sourceLine;
}

function TranscriptList({lines, scrollOffset, height}) {
  const visibleHeight = Math.max(height, 4);
  const end = Math.max(lines.length - scrollOffset, 0);
  const start = Math.max(0, end - visibleHeight);
  const visible = lines.slice(start, end);

  return React.createElement(Box, {flexDirection: 'column', borderStyle: 'round', borderColor: 'gray', paddingX: 1, paddingY: 0, height: visibleHeight + 2},
    React.createElement(Text, {color: 'gray'}, `transcript ${lines.length ? `${start + 1}-${end}` : '0-0'} / ${lines.length}`),
    visible.length > 0
      ? visible.map((line, index) => React.createElement(Box, {key: `${start}-${end}-${index}-${line}`, flexDirection: 'column', ...lineSpacing(line, visible[index - 1], index)},
          React.createElement(Box, {flexDirection: 'column', ...messageSurface(line)},
            React.createElement(Text, {color: colorForLine(line)}, normalizeAssistantText(line))
          )
        ))
      : React.createElement(Text, {color: 'gray'}, 'todavía no hay mensajes')
  );
}

export {TranscriptList};
