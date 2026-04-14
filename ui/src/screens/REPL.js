import React from 'react';
import {execFileSync} from 'child_process';
import {Box, Text, useApp, useInput} from 'ink';
import {TextInput} from '../components/TextInput.js';
import {TranscriptList} from '../components/TranscriptList.js';
import {DebugPanel} from '../components/DebugPanel.js';
import {useBridgeSession} from '../hooks/useBridgeSession.js';
import {useMouseScroll} from '../hooks/useMouseScroll.js';

const UI_DEBUG = (process.env.NODE_UI_DEBUG || '0').toLowerCase() !== '0';

function debugUI(event, payload = {}) {
  if (!UI_DEBUG) return;
  console.error(`[node-ui][repl] ${event}`, payload);
}

function REPL() {
  const {exit: inkExit} = useApp();
  const [debugLines, setDebugLines] = React.useState([]);
  const [copyStatus, setCopyStatus] = React.useState('');

  const reportDebug = React.useCallback((scope, event, payload = {}) => {
    debugUI(`${scope}:${event}`, payload);
    const payloadText = Object.keys(payload).length > 0 ? ` ${JSON.stringify(payload)}` : '';
    setDebugLines(prev => [...prev.slice(-39), `[${scope}] ${event}${payloadText}`]);
  }, []);

  const {state, submit, exit} = useBridgeSession({onExit: inkExit, onDebug: reportDebug});
  const [prompt, setPrompt] = React.useState('');

  // scrollOffset en ENTRADAS (mensajes): 0 = ver los más recientes.
  // +1 = retrocede un mensaje. maxScroll = total mensajes - 1.
  const [scrollOffset, setScrollOffset] = React.useState(0);
  const maxScroll = Math.max(0, state.transcript.length - 1);

  // Índices de mensajes expandidos (Set para O(1) lookup).
  const [expandedMessages, setExpandedMessages] = React.useState(() => new Set());

  // Cuando llega contenido nuevo y estamos al fondo, quedarse al fondo.
  const prevLenRef = React.useRef(state.transcript.length);
  React.useEffect(() => {
    const newLen = state.transcript.length;
    if (newLen > prevLenRef.current) {
      // Si el usuario no había subido, seguir en el fondo
      if (scrollOffset === 0) {
        // ya estamos en 0, nada que hacer
      } else {
        // estaba scrolleado hacia atrás — no lo movemos, es intencional
      }
    }
    prevLenRef.current = newLen;
  }, [state.transcript.length, scrollOffset]);

  // Chrome: header(5) + marginTop(1) + input(3) + hints(1) + transcriptBorder(2) = 12
  const _debugRows = UI_DEBUG ? 13 : 0;
  const termRows = process.stdout.rows || 24;
  const viewportHeight = Math.max(termRows - 12 - _debugRows, 6);

  const copyDebugBuffer = React.useCallback(() => {
    const payload = debugLines.join('\n').trim();
    if (!payload) { setCopyStatus('sin logs para copiar'); return; }
    try {
      execFileSync('/usr/bin/pbcopy', {input: payload});
      setCopyStatus(`copiado (${debugLines.length} líneas)`);
    } catch (error) {
      setCopyStatus(`error al copiar: ${error?.message || 'desconocido'}`);
    }
  }, [debugLines]);

  const copyMessage = React.useCallback((idx) => {
    const line = state.transcript[idx];
    if (line == null) { setCopyStatus('sin mensaje para copiar'); return; }
    try {
      execFileSync('/usr/bin/pbcopy', {input: line});
      setCopyStatus('copiado ✓');
      setTimeout(() => setCopyStatus(''), 2000);
    } catch (error) {
      setCopyStatus(`error: ${error?.message || 'desconocido'}`);
    }
  }, [state.transcript]);

  useMouseScroll(React.useCallback((dir) => {
    if (dir === 'up') setScrollOffset(prev => Math.min(prev + 1, maxScroll));
    else              setScrollOffset(prev => Math.max(prev - 1, 0));
  }, [maxScroll]));

  useInput((input, key) => {
    // E → expandir/colapsar el mensaje anclado (el más reciente visible)
    if (!key.ctrl && !key.meta && input.toLowerCase() === 'e') {
      const anchorIdx = Math.max(0, state.transcript.length - 1 - scrollOffset);
      setExpandedMessages(prev => {
        const next = new Set(prev);
        if (next.has(anchorIdx)) next.delete(anchorIdx);
        else next.add(anchorIdx);
        return next;
      });
      return;
    }
    if (key.ctrl && input.toLowerCase() === 'y') {
      if (UI_DEBUG) {
        copyDebugBuffer();
      } else {
        const focusedIdx = Math.max(0, state.transcript.length - 1 - scrollOffset);
        copyMessage(focusedIdx);
      }
      return;
    }
    // ↑ → un mensaje más antiguo
    if (key.upArrow) {
      setScrollOffset(prev => Math.min(prev + 1, maxScroll));
      return;
    }
    // ↓ → un mensaje más reciente
    if (key.downArrow) {
      setScrollOffset(prev => Math.max(prev - 1, 0));
      return;
    }
    // PgUp → 5 mensajes más antiguos
    if (key.pageUp) {
      setScrollOffset(prev => Math.min(prev + 5, maxScroll));
      return;
    }
    // PgDn → 5 mensajes más recientes
    if (key.pageDown) {
      setScrollOffset(prev => Math.max(prev - 5, 0));
      return;
    }
    // Home → ir al inicio del historial
    if (key.home) {
      setScrollOffset(maxScroll);
      return;
    }
    // End → volver al fondo (mensajes más recientes)
    if (key.end) {
      setScrollOffset(0);
    }
  }, {isActive: process.stdin.isTTY === true});

  const handleSubmit = React.useCallback(value => {
    const text = value.trim();
    if (!text) return;
    if (['salir', 'exit', 'quit'].includes(text.toLowerCase())) {
      setPrompt('');
      exit();
      inkExit();
      return;
    }
    if (state.busy) return;
    setPrompt('');
    setScrollOffset(0); // volver al fondo al enviar
    submit(text);
  }, [exit, inkExit, submit, state.busy]);

  return React.createElement(
    Box,
    {flexDirection: 'column'},

    // ── Header ──────────────────────────────────────────────────────────────
    React.createElement(
      Box,
      {borderStyle: 'round', borderColor: 'cyan', paddingX: 1, flexDirection: 'column', flexShrink: 0},
      React.createElement(Text, {color: 'cyan', bold: true}, 'Multi-Agentes'),
      React.createElement(Text, {color: 'gray'}, state.status),
      React.createElement(Text, {color: 'gray'}, state.session_id ? `session ${state.session_id}` : 'sesión…')
    ),

    // ── Transcript ──────────────────────────────────────────────────────────
    React.createElement(
      Box,
      {marginTop: 1, flexDirection: 'column', flexShrink: 0},
      React.createElement(TranscriptList, {
        lines: state.transcript,
        scrollOffset,
        height: viewportHeight,
        expandedMessages,
      })
    ),

    // ── Debug (opcional) ────────────────────────────────────────────────────
    UI_DEBUG ? React.createElement(DebugPanel, {lines: debugLines, copyStatus}) : null,

    // ── Input ───────────────────────────────────────────────────────────────
    React.createElement(TextInput, {
      value: prompt,
      placeholder: state.prompt || 'Escribí un mensaje, /help o salir',
      busy: state.busy,
      onChange: setPrompt,
      onSubmit: handleSubmit,
      onDebug: reportDebug,
      onExit: () => { exit(); inkExit(); },
    }),

    // ── Hints ───────────────────────────────────────────────────────────────
    React.createElement(
      Box,
      {flexDirection: 'row', justifyContent: 'space-between'},
      React.createElement(
        Text,
        {color: 'gray'},
        'scroll · ↑↓ · PgUp/Dn · End/Home · E expandir · 📋 Ctrl+Y copia'
      ),
      copyStatus
        ? React.createElement(Text, {color: 'green'}, copyStatus)
        : null,
    ),
  );
}

export {REPL};
