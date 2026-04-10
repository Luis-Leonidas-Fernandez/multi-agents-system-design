import React from 'react';
import {spawn} from 'child_process';

const PYTHON = process.env.PYTHON_BRIDGE || process.env.PYTHON || process.env.PYTHON_EXECUTABLE || '/Users/luis/.cache/06_multi_agents_venv/bin/python';
const BRIDGE_ARGS = [process.env.MAIN_PY || '/Users/luis/Desktop/PROYECTOS/06_multi_agents/main.py', '--ui-bridge'];

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
  const {onExit} = options;
  const [state, setState] = React.useState({session_id: '', status: 'conectando…', prompt: '', transcript: [], message_count: 0, has_memory: false, busy: false});
  const bufferRef = React.useRef('');
  const childRef = React.useRef(null);
  const exitedRef = React.useRef(false);

  const notifyExit = React.useCallback(() => {
    if (exitedRef.current) return;
    exitedRef.current = true;
    if (typeof onExit === 'function') onExit();
  }, [onExit]);

  React.useEffect(() => {
    exitedRef.current = false;
    const child = spawn(PYTHON, BRIDGE_ARGS, {stdio: ['pipe', 'pipe', 'inherit']});
    childRef.current = child;

    child.stdout.setEncoding('utf8');
    child.stdout.on('data', chunk => {
      bufferRef.current += chunk;
      let idx;
      while ((idx = bufferRef.current.indexOf('\n')) !== -1) {
        const line = bufferRef.current.slice(0, idx).trim();
        bufferRef.current = bufferRef.current.slice(idx + 1);
        if (!line) continue;
        const message = parseLine(line);
        if (!message) continue;

        if (message.type === 'state') {
          setState(prev => ({...prev, ...normalizeState(message.state), busy: false}));
          continue;
        }

        if (message.type === 'busy') {
          setState(prev => ({...prev, status: message.status || prev.status, busy: true}));
          continue;
        }

        if (message.type === 'status') {
          setState(prev => ({...prev, status: message.status || prev.status}));
          continue;
        }

        if (message.type === 'error') {
          setState(prev => ({...prev, status: `error: ${message.message || 'desconocido'}`, busy: false}));
          continue;
        }

        if (message.type === 'exit') {
          notifyExit();
          child.kill();
        }
      }
    });

    child.on('exit', () => {
      childRef.current = null;
      notifyExit();
    });

    return () => {
      child.kill();
      childRef.current = null;
      notifyExit();
    };
  }, [notifyExit]);

  const submit = React.useCallback(text => {
    if (!childRef.current || !text) return;
    childRef.current.stdin.write(`${JSON.stringify({action: 'submit', text})}\n`);
    if (!text.startsWith('/')) {
      setState(prev => ({...prev, busy: true, status: 'procesando…', transcript: [...prev.transcript, `you: ${text}`]}));
    }
  }, []);

  const exit = React.useCallback(() => {
    if (childRef.current) {
      childRef.current.stdin.write(`${JSON.stringify({action: 'exit'})}\n`);
      childRef.current.kill();
    }
    notifyExit();
  }, [notifyExit]);

  return {state, submit, exit, setState};
}

export {useBridgeSession};
