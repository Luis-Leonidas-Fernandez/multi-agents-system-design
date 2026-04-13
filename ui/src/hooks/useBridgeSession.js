import React from 'react';
import {spawn} from 'child_process';

const PYTHON = process.env.PYTHON_BRIDGE || process.env.PYTHON || process.env.PYTHON_EXECUTABLE || '/Users/luis/.cache/06_multi_agents_venv/bin/python';
const BRIDGE_ARGS = [process.env.MAIN_PY || '/Users/luis/Desktop/PROYECTOS/06_multi_agents/main.py', '--ui-bridge'];
const UI_DEBUG = (process.env.NODE_UI_DEBUG || '0').toLowerCase() !== '0';

function debugUI(event, payload = {}) {
  if (!UI_DEBUG) return;
  console.error(`[node-ui][bridge] ${event}`, payload);
}

function parseLine(line) {
  try {
    return JSON.parse(line);
  } catch {
    return null;
  }
}

function normalizeState(raw) {
  const transcript = Array.isArray(raw?.transcript) ? raw.transcript.map(line => String(line)) : [];
  return {
    session_id: String(raw?.session_id || ''),
    status: String(raw?.status || 'listo'),
    prompt: String(raw?.prompt || ''),
    transcript,
    message_count: Number(raw?.message_count || 0),
    has_memory: Boolean(raw?.has_memory),
  };
}

function useBridgeSession(options = {}) {
  const {onExit, onDebug} = options;
  const [state, setState] = React.useState({session_id: '', status: 'conectando…', prompt: '', transcript: [], message_count: 0, has_memory: false, busy: false});
  const bufferRef = React.useRef('');
  const childRef = React.useRef(null);
  const exitedRef = React.useRef(false);

  const reportDebug = React.useCallback((event, payload = {}) => {
    debugUI(event, payload);
    if (typeof onDebug === 'function') onDebug('bridge', event, payload);
  }, [onDebug]);

  const notifyExit = React.useCallback(() => {
    if (exitedRef.current) return;
    exitedRef.current = true;
    if (typeof onExit === 'function') onExit();
  }, [onExit]);

  React.useEffect(() => {
    exitedRef.current = false;
    reportDebug('spawn:start', {python: PYTHON, args: BRIDGE_ARGS});
    const child = spawn(PYTHON, BRIDGE_ARGS, {stdio: ['pipe', 'pipe', 'inherit']});
    childRef.current = child;
    reportDebug('spawn:ok', {pid: child.pid});

    child.stdout.setEncoding('utf8');
    child.stdout.on('data', chunk => {
      reportDebug('stdout:chunk', {chunk});
      bufferRef.current += chunk;
      let idx;
      while ((idx = bufferRef.current.indexOf('\n')) !== -1) {
        const line = bufferRef.current.slice(0, idx).trim();
        bufferRef.current = bufferRef.current.slice(idx + 1);
        if (!line) continue;
        reportDebug('stdout:line', {line});
        const message = parseLine(line);
        if (!message) {
          reportDebug('stdout:json_invalid', {line});
          continue;
        }

        if (message.type === 'state') {
          reportDebug('message:state', {status: message?.state?.status, transcriptCount: message?.state?.transcript?.length});
          setState(prev => ({...prev, ...normalizeState(message.state), busy: false}));
          continue;
        }

        if (message.type === 'busy') {
          reportDebug('message:busy', {status: message.status});
          setState(prev => ({...prev, status: message.status || prev.status, busy: true}));
          continue;
        }

        if (message.type === 'status') {
          reportDebug('message:status', {status: message.status});
          setState(prev => ({...prev, status: message.status || prev.status}));
          continue;
        }

        if (message.type === 'error') {
          reportDebug('message:error', {message: message.message});
          setState(prev => ({...prev, status: `error: ${message.message || 'desconocido'}`, busy: false}));
          continue;
        }

        if (message.type === 'exit') {
          reportDebug('message:exit');
          notifyExit();
          child.kill();
        }
      }
    });

    child.on('error', error => {
      reportDebug('spawn:error', {message: error?.message || String(error)});
    });

    child.on('exit', () => {
      reportDebug('spawn:exit');
      childRef.current = null;
      notifyExit();
    });

    return () => {
      reportDebug('spawn:cleanup');
      child.kill();
      childRef.current = null;
      notifyExit();
    };
  }, [notifyExit, reportDebug]);

  const submit = React.useCallback(text => {
    reportDebug('submit:attempt', {text, hasChild: Boolean(childRef.current)});
    if (!childRef.current || !text) return;
    childRef.current.stdin.write(`${JSON.stringify({action: 'submit', text})}\n`);
    reportDebug('submit:sent', {text});
    if (!text.startsWith('/')) {
      setState(prev => ({...prev, busy: true, status: 'procesando…', transcript: [...prev.transcript, `you: ${text}`]}));
    }
  }, [reportDebug]);

  const exit = React.useCallback(() => {
    reportDebug('exit:attempt', {hasChild: Boolean(childRef.current)});
    if (childRef.current) {
      childRef.current.stdin.write(`${JSON.stringify({action: 'exit'})}\n`);
      childRef.current.kill();
    }
    notifyExit();
  }, [notifyExit, reportDebug]);

  return {state, submit, exit, setState};
}

export {useBridgeSession};
